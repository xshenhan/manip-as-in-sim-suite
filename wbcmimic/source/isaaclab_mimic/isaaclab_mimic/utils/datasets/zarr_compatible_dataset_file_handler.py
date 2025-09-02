# Copyright 2025 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import concurrent.futures
import json
import math
import numbers
import os
import time
from collections.abc import Iterable
from typing import Dict, Optional, Union

import numcodecs
import numpy as np
import torch
import zarr

try:
    from loguru import logger as lgr

    warn_fn = lgr.warning
    info_fn = lgr.info
except ImportError:
    # Fallback to print if loguru is not available
    lgr = None
    warn_fn = info_fn = print

from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase
from isaaclab.utils.datasets.episode_data import EpisodeData


def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0


def rechunk_recompress_array(
    group, name, chunks=None, chunk_length=None, compressor=None, tmp_key="_temp"
):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)

    if compressor is None:
        compressor = old_arr.compressor

    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr


def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6, max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if (
            this_chunk_bytes <= target_chunk_bytes
            and next_chunk_bytes > target_chunk_bytes
        ):
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(
        this_max_chunk_length, math.ceil(target_chunk_bytes / item_chunk_bytes)
    )
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    return chunks


class ZarrCompatibleDatasetFileHandler(DatasetFileHandlerBase):
    """Zarr dataset file handler for storing and loading episode data."""

    def __init__(self):
        """Initializes the Zarr dataset file handler."""
        self._zarr_store = None
        self._zarr_root = None
        self._data_group = None
        self._meta_group = None
        self._demo_count = 0
        self._env_args = {}

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing Zarr dataset file."""
        if self._zarr_root is not None:
            raise RuntimeError("Zarr dataset store is already in use")
        self._zarr_store = zarr.DirectoryStore(os.path.expanduser(file_path))
        self._zarr_root = zarr.open(self._zarr_store, mode=mode)
        self._data_group = self._zarr_root["data"]
        self._meta_group = self._zarr_root["meta"]
        self._demo_count = len(self._meta_group["episode_ends"])

    def create(self, file_path: str, env_name: str = None):
        """Create a new Zarr dataset file."""
        if self._zarr_root is not None:
            raise RuntimeError("Zarr dataset store is already in use")
        if not file_path.endswith(".zarr"):
            file_path += ".zarr"
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        self._zarr_store = zarr.DirectoryStore(os.path.expanduser(file_path))
        self._zarr_root = zarr.group(store=self._zarr_store)
        self._data_group = self._zarr_root.require_group("data", overwrite=False)
        self._meta_group = self._zarr_root.require_group("meta", overwrite=False)

        # Initialize metadata arrays
        if "episode_ends" not in self._meta_group:
            self._meta_group.zeros(
                "episode_ends", shape=(0,), dtype=np.int64, compressor=None
            )
        if "episode_descriptions" not in self._meta_group:
            self._meta_group.zeros(
                "episode_descriptions", shape=(1,), dtype="U100", compressor=None
            )
            episode_descriptions = self._meta_group["episode_descriptions"]
            episode_descriptions[-1] = env_name or ""
        if "env_names" not in self._meta_group:
            self._meta_group.zeros(
                "env_names", shape=(1,), dtype="U50", compressor=None
            )
            env_names = self._meta_group["env_names"]
            env_names[-1] = env_name or ""

        self._meta_group.attrs["total"] = 0
        self._demo_count = 0

        # Set environment arguments
        env_name = env_name if env_name is not None else ""
        self.add_env_args({"env_name": env_name, "type": 2})

    def __del__(self):
        """Destructor for the file handler."""
        self.close()

    """
    Properties
    """

    def add_env_args(self, env_args: dict):
        """Add environment arguments to the dataset."""
        self._raise_if_not_initialized()
        self._env_args.update(env_args)
        self._meta_group.attrs["env_args"] = json.dumps(self._env_args)

    def set_env_name(self, env_name: str):
        """Set the environment name."""
        self._raise_if_not_initialized()
        self.add_env_args({"env_name": env_name})

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        self._raise_if_not_initialized()
        try:
            env_args = json.loads(self._meta_group.attrs["env_args"])
        except KeyError:
            env_args = {}
        return env_args.get("env_name", None)

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        self._raise_if_not_initialized()
        return [f"demo_{i}" for i in range(self._demo_count)]

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return self._demo_count

    @property
    def demo_count(self) -> int:
        """The number of demos collected so far."""
        return self._demo_count

    """
    Operations
    """

    def load_episode(
        self, episode_name: str, device: str, once_per_episode_keys=None
    ) -> EpisodeData | None:
        """Load episode data from the file."""
        if once_per_episode_keys is None:
            once_per_episode_keys = ["initial_state"]

        self._raise_if_not_initialized()
        idx = int(episode_name.split("_")[-1])
        if idx >= self._demo_count:
            return None

        episode = EpisodeData()
        start_idx = 0 if idx == 0 else self._meta_group["episode_ends"][idx - 1]
        end_idx = self._meta_group["episode_ends"][idx]

        def load_dataset_helper(group, slice_obj, prefix_key=""):
            """Helper method to load dataset that contains recursive dict objects."""
            data = {}
            for key in group:
                if isinstance(group[key], zarr.Group):
                    data[key] = load_dataset_helper(
                        group[key], slice_obj, prefix_key=f"{prefix_key}/{key}"
                    )
                else:
                    once_per_episode = False
                    for once_per_episode_key in once_per_episode_keys:
                        if once_per_episode_key in f"{prefix_key}/{key}":
                            once_per_episode = True
                    if once_per_episode:
                        data[key] = torch.tensor(
                            group[key][idx : idx + 1], device=device
                        )
                    else:
                        data[key] = torch.tensor(group[key][slice_obj], device=device)
            return data

        episode.data = load_dataset_helper(self._data_group, slice(start_idx, end_idx))

        if "seed" in self._meta_group.attrs:
            episode.seed = self._meta_group.attrs.get(f"seed_{idx}", None)

        if "success" in self._meta_group.attrs:
            episode.success = self._meta_group.attrs.get(f"success_{idx}", None)
        episode.env_id = self.get_env_name()

        return episode

    def write_episode(
        self, episode: EpisodeData, chunks: Optional[Dict[str, tuple]] = None
    ):
        """Add an episode to the dataset."""
        start_time = time.time()
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        if "wbc_step" in episode.data:
            wbc_step = episode.data["wbc_step"]
        else:
            if "actions" in episode.data:
                wbc_step = torch.ones(
                    (episode.data["actions"].shape[0]),
                    device=episode.data["actions"].device,
                )
            else:
                wbc_step = torch.tensor([])

        curr_len = self._meta_group.attrs["total"]
        episode_length = wbc_step.sum().item()
        new_traj_len = curr_len + episode_length
        if episode_length == 0:
            warn_fn("Episode length is 0, skipping write.")
            return

        def create_dataset_helper(group, key, value, parent_path=""):
            """Helper method to create dataset with recursive dict objects."""
            full_key = f"{parent_path}/{key}" if parent_path else key
            if "initial_state" in full_key or "camera_info" in full_key:
                new_len = self._demo_count + 1
            else:
                new_len = new_traj_len
            if isinstance(value, dict):
                key_group = group.require_group(key)
                for sub_key, sub_value in value.items():
                    create_dataset_helper(key_group, sub_key, sub_value, full_key)
            else:
                new_shape = (new_len,) + value.shape[1:]
                if key not in group:
                    chunk_size = (
                        chunks.get(
                            full_key,
                            get_optimal_chunks(
                                shape=new_shape, dtype=value.cpu().numpy().dtype
                            ),
                        )
                        if chunks
                        else get_optimal_chunks(
                            shape=new_shape, dtype=value.cpu().numpy().dtype
                        )
                    )
                    arr = group.zeros(
                        name=key,
                        shape=new_shape,
                        chunks=chunk_size,
                        dtype=value.cpu().numpy().dtype,
                        compressor=self.resolve_compressor("default"),
                    )
                else:
                    arr = group[key]
                    arr.resize(new_shape)
                if "initial_state" in full_key:
                    arr[-1:] = value.cpu().numpy()
                elif "camera_info" in full_key:
                    arr[-1:] = value[0:1].cpu().numpy()
                else:
                    arr[-episode_length:] = value[wbc_step].cpu().numpy()

        for key, value in episode.data.items():
            if "camera" in key:
                if key not in self._meta_group:  # only once
                    create_dataset_helper(self._meta_group, key, value)
            else:
                create_dataset_helper(self._data_group, key, value)

        # Update metadata
        episode_ends = self._meta_group["episode_ends"]
        episode_ends.resize(self._demo_count + 1)
        episode_ends[-1] = new_traj_len

        if episode.seed is not None:
            self._meta_group.attrs[f"seed_{self._demo_count}"] = episode.seed

        if episode.success is not None:
            self._meta_group.attrs[f"success_{self._demo_count}"] = episode.success

        self._meta_group.attrs["total"] = new_traj_len
        self._demo_count += 1

        # Rechunk metadata if necessary
        for key in ["episode_ends", "episode_descriptions", "env_names"]:
            arr = self._meta_group[key]
            if arr.chunks[0] < arr.shape[0]:
                rechunk_recompress_array(
                    self._meta_group,
                    key,
                    chunk_length=int(arr.shape[0] * 1.5),
                )
        info_fn(
            f"Write episode {self._demo_count} took {time.time() - start_time:.2f} seconds"
        )

    def flush(self):
        """Flush the episode data to disk."""
        self._raise_if_not_initialized()

    def close(self):
        """Close the dataset file handler."""
        if self._zarr_root is not None:
            self._zarr_root = None
            self._zarr_store = None

    def _raise_if_not_initialized(self):
        """Raise an error if the dataset file handler is not initialized."""
        if self._zarr_root is None:
            raise RuntimeError("Zarr dataset store is not initialized")

    @staticmethod
    def resolve_compressor(compressor="default"):
        if compressor == "default":
            compressor = numcodecs.Blosc(
                cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
            )
        elif compressor == "disk":
            compressor = numcodecs.Blosc(
                "zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
            )
        return compressor

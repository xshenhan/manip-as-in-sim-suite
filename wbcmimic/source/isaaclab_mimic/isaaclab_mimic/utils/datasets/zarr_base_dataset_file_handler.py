# Copyright 2025 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import os
from collections.abc import Iterable

import numpy as np
import torch
import zarr
from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase
from isaaclab.utils.datasets.episode_data import EpisodeData


class ZarrDatasetFileHandler(DatasetFileHandlerBase):
    """Zarr dataset file handler for storing and loading episode data."""

    def __init__(self):
        """Initializes the Zarr dataset file handler."""
        self._zarr_store = None
        self._zarr_data_group = None
        self._demo_count = 0
        self._env_args = {}

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        if self._zarr_store is not None:
            raise RuntimeError("Zarr dataset store is already in use")
        self._zarr_store = zarr.open(file_path, mode=mode)
        self._zarr_data_group = self._zarr_store["data"]
        self._demo_count = len(self._zarr_data_group)

    def create(self, file_path: str, env_name: str = None):
        """Create a new dataset file."""
        if self._zarr_store is not None:
            raise RuntimeError("Zarr dataset store is already in use")
        if not file_path.endswith(".zarr"):
            file_path += ".zarr"
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        # Initialize Zarr store with directory storage
        self._zarr_store = zarr.open(file_path, mode="w")

        # Set up a data group in the store
        self._zarr_data_group = self._zarr_store.create_group("data")
        self._zarr_data_group.attrs["total"] = 0
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
        self._zarr_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def set_env_name(self, env_name: str):
        """Set the environment name."""
        self._raise_if_not_initialized()
        self.add_env_args({"env_name": env_name})

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        self._raise_if_not_initialized()
        env_args = json.loads(self._zarr_data_group.attrs["env_args"])
        if "env_name" in env_args:
            return env_args["env_name"]
        return None

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        self._raise_if_not_initialized()
        return self._zarr_data_group.keys()

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

    def load_episode(self, episode_name: str, device: str) -> EpisodeData | None:
        """Load episode data from the file."""
        self._raise_if_not_initialized()
        if episode_name not in self._zarr_data_group:
            return None
        episode = EpisodeData()
        zarr_episode_group = self._zarr_data_group[episode_name]

        def load_dataset_helper(group):
            """Helper method to load dataset that contains recursive dict objects."""
            data = {}
            for key in group:
                if isinstance(group[key], zarr.Group):
                    data[key] = load_dataset_helper(group[key])
                else:
                    # Convert Zarr array to numpy then to torch tensor
                    data[key] = torch.tensor(np.array(group[key]), device=device)
            return data

        episode.data = load_dataset_helper(zarr_episode_group)

        if "seed" in zarr_episode_group.attrs:
            episode.seed = zarr_episode_group.attrs["seed"]

        if "success" in zarr_episode_group.attrs:
            episode.success = zarr_episode_group.attrs["success"]

        episode.env_id = self.get_env_name()

        return episode

    def write_episode(self, episode: EpisodeData):
        """Add an episode to the dataset.

        Args:
            episode: The episode data to add.
        """
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        # Create episode group based on demo count
        zarr_episode_group = self._zarr_data_group.create_group(
            f"demo_{self._demo_count}"
        )

        # Store number of steps taken
        if "actions" in episode.data:
            zarr_episode_group.attrs["num_samples"] = len(episode.data["actions"])
        else:
            zarr_episode_group.attrs["num_samples"] = 0

        if episode.seed is not None:
            zarr_episode_group.attrs["seed"] = episode.seed

        if episode.success is not None:
            zarr_episode_group.attrs["success"] = episode.success

        def create_dataset_helper(group, key, value):
            """Helper method to create dataset that contains recursive dict objects."""
            if isinstance(value, dict):
                key_group = group.create_group(key)
                for sub_key, sub_value in value.items():
                    create_dataset_helper(key_group, sub_key, sub_value)
            else:
                # Store as Zarr array with default compressor
                group.create_dataset(
                    key, data=value.cpu().numpy(), compressor=zarr.Blosc()
                )

        for key, value in episode.data.items():
            create_dataset_helper(zarr_episode_group, key, value)

        # Increment total step counts
        self._zarr_data_group.attrs["total"] += zarr_episode_group.attrs["num_samples"]

        # Increment total demo counts
        self._demo_count += 1

    def flush(self):
        """Flush the episode data to disk."""
        self._raise_if_not_initialized()
        # Zarr handles writes immediately, but we can ensure sync
        self._zarr_store.store.sync()

    def close(self):
        """Close the dataset file handler."""
        if self._zarr_store is not None:
            # Zarr store is closed automatically when using zarr.open
            self._zarr_store = None

    def _raise_if_not_initialized(self):
        """Raise an error if the dataset file handler is not initialized."""
        if self._zarr_store is None:
            raise RuntimeError("Zarr dataset store is not initialized")

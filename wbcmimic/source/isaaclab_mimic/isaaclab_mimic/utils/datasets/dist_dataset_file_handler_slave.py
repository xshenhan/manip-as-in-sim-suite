# Copyright 2025 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import math
import numbers
import os
from collections.abc import Iterable
from typing import Dict, Optional, Union

import numcodecs
import numpy as np
import torch
import zarr
from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase
from isaaclab.utils.datasets.episode_data import EpisodeData


class DistDatasetFileHandlerSlave(DatasetFileHandlerBase):
    """Zarr dataset file handler for storing and loading episode data."""

    def __init__(self):
        pass

    def set_data_queue(self, episode_data_queue):
        self.data_queue = episode_data_queue

    def open(self, file_path: str, mode: str = "r"):
        pass

    def create(self, file_path: str, env_name: str = None):
        pass

    def __del__(self):
        """Destructor for the file handler."""
        pass

    """
    Properties
    """

    def add_env_args(self, env_args: dict):
        """Add environment arguments to the dataset."""
        pass

    def set_env_name(self, env_name: str):
        """Set the environment name."""
        pass

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        pass

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        pass

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        pass

    @property
    def demo_count(self) -> int:
        """The number of demos collected so far."""
        pass

    """
    Operations
    """

    def load_episode(self, episode_name: str, device: str) -> EpisodeData | None:
        """Load episode data from the file."""
        pass

    def write_episode(self, episode: EpisodeData):
        """Add an episode to the dataset."""
        # episode.data = to_device(episode.data, device="cpu")
        self.data_queue.put(episode)

    def flush(self):
        """Flush the episode data to disk."""
        pass

    def close(self):
        """Close the dataset file handler."""
        pass

    def _raise_if_not_initialized(self):
        """Raise an error if the dataset file handler is not initialized."""
        pass

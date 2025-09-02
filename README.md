# Manipulation As in Simulation Suite

This repository contains the open-source implementation of the paper **"Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots"**. The suite provides tools for bridging the sim-to-real gap in robotic manipulation through high-quality depth perception and automated demonstration generation.

## 📋 Table of Contents

- [🎯 Overview](#-overview)
  - [🔍 CDM (Camera Depth Models)](#-cdm-camera-depth-models)
  - [🤖 WBCMimic](#-wbcmimic)
- [✨ Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [🔧 Environment Setup](#-environment-setup)
  - [Environment Variables](#environment-variables)
  - [Setup Options](#setup-options)
  - [Verification](#verification)
  - [Troubleshooting](#troubleshooting)
- [🚀 Usage](#-usage)
  - [CDM Usage](#cdm-usage)
  - [WBCMimic Usage](#wbcmimic-usage)
- [🔬 Research Contributions](#-research-contributions)
  - [Camera Depth Models (CDMs)](#camera-depth-models-cdms)
  - [WBCMimic Enhancements](#wbcmimic-enhancements)
- [🎯 Supported Tasks & Hardware](#-supported-tasks--hardware)
  - [Robotic Tasks](#robotic-tasks)
  - [Supported Cameras](#supported-cameras)
- [📚 Documentation](#-documentation)
- [📄 Citation](#-citation)
- [📝 License](#-license)
- [🔗 Links](#-links)
- [📧 Contact](#-contact)

## 🎯 Overview

The suite consists of two main components that enable seamless sim-to-real transfer for robotic manipulation:

### 🔍 CDM (Camera Depth Models)
A depth estimation package that produces clean, simulation-like depth maps from noisy real-world camera data. CDMs enable policies trained purely in simulation to transfer directly to real robots by providing perfect depth perception.

### 🤖 WBCMimic 
An enhanced version of MimicGen that extends autonomous data generation to mobile manipulators with whole-body control. It enables efficient generation of high-quality manipulation demonstrations through automated pipelines with multi-GPU parallel simulation.

## ✨ Key Features

- **Sim-to-Real Depth Transfer**: Clean, metric depth estimation that matches simulation quality
- **Multi-Camera Support**: Pre-trained models for various depth sensors (RealSense, ZED, Kinect)
- **Automated Data Generation**: Scalable demonstration generation using enhanced MimicGen
- **Whole-Body Control**: Unified control for mobile manipulators for mimicgen
- **Multi-GPU Parallelization**: Distributed simulation for faster data collection
- **VR Teleoperation**: Intuitive demonstration recording using Meta Quest controllers

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- [Isaac Lab 2.1](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html)
- [CuRobo](https://curobo.org/get_started/1_install_instructions.html) for motion planning

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/manip-as-in-sim-suite.git
   cd manip-as-in-sim-suite
   ```

2. **Install CDM**:
   ```bash
   cd cdm
   pip install -e .
   ```

3. **Install WBCMimic**:
   ```bash
   cd ../wbcmimic
   pip install -e source/isaaclab_mimic
   ```

4. **Configure Environment Variables** (Required for WBCMimic):
   ```bash
   # Set paths to your Isaac Lab and CuRobo installations
   export ISAACLAB_DIR="/path/to/IsaacLab"
   export CUROBO_DIR="/path/to/curobo"
   
   # Set paths to robot assets
   export X7_URDF_PATH="/path/to/X7/urdf/X7_2.urdf"
   export UR5_URDF_PATH="/path/to/UR5/ur5_isaac_simulation/robot.urdf"
   export ROOM_USD_PATH="/path/to/Room_empty_table.usdc"
   export ISAAC_SIM_FRANKA_PATH="/path/to/isaac-sim/src/franka"
   ```

## 🔧 Environment Setup

This section explains how to configure the environment variables needed for the manipulation-as-in-simulation suite. The codebase uses configurable environment variables and fallback paths to make the code portable and maintainable.

### Environment Variables

#### Robot URDF Paths

##### X7_URDF_PATH
- **Purpose**: Path to the ARX-X7 robot URDF file
- **Used in**: `wbc_controller_dual.py`
- **Default fallback**: `../../../../../assets/X7/urdf/X7_2.urdf` (relative to the controller file)
- **Example**:
  ```bash
  export X7_URDF_PATH="/path/to/your/X7/urdf/X7_2.urdf"
  ```

##### UR5_URDF_PATH  
- **Purpose**: Path to the UR5 robot URDF file
- **Used in**: `wbc_controller.py`, `wbc_controller_cuda.py`, `test_robot_ik.ipynb`
- **Default fallback**: `../../../../../assets/UR5/ur5_isaac_simulation/robot.urdf` (relative to the controller file)
- **Example**:
  ```bash
  export UR5_URDF_PATH="/path/to/your/UR5/ur5_isaac_simulation/robot.urdf"
  ```

#### Scene Assets

##### ROOM_USD_PATH
- **Purpose**: Path to the room USD scene file
- **Used in**: `record_demos_quest.py`
- **Default fallback**: `../../../assets/Room_empty_table.usdc` (relative to the script)
- **Example**:
  ```bash
  export ROOM_USD_PATH="/path/to/your/Room_empty_table.usdc"
  ```

#### Isaac Sim Integration

##### ISAAC_SIM_FRANKA_PATH
- **Purpose**: Path to Isaac Sim's Franka source directory
- **Used in**: `vr_policy.py` files
- **Default fallback**: Tries common installation paths:
  - `~/.local/share/ov/pkg/isaac-sim-4.0.0/src/franka`
  - `/opt/nvidia/isaac-sim/src/franka`
  - `/usr/local/isaac-sim/src/franka`
- **Example**:
  ```bash
  export ISAAC_SIM_FRANKA_PATH="/your/isaac-sim/installation/src/franka"
  ```

### Setup Options

#### Option 1: Set Environment Variables (Recommended)

Create a shell script or add to your `.bashrc`/`.zshrc`:

```bash
# Robot assets - adjust paths according to your setup
export X7_URDF_PATH="/path/to/IsaacLab/source/arxx7_assets/X7/urdf/X7_2.urdf"
export UR5_URDF_PATH="/path/to/IsaacLab/source/arxx7_assets/UR5/ur5_isaac_simulation/robot.urdf"

# Scene assets
export ROOM_USD_PATH="/path/to/your/scene/Room_empty_table.usdc"

# Isaac Sim (if not in standard location)
export ISAAC_SIM_FRANKA_PATH="/path/to/isaac-sim/src/franka"

# Source the environment
source ~/.bashrc  # or ~/.zshrc
```

#### Option 2: Use Default Fallback Paths

If you don't set environment variables, the code will use relative paths. Ensure your asset directory structure follows this layout:

```
manip-as-in-sim-suite/
├── assets/
│   ├── X7/urdf/X7_2.urdf
│   ├── UR5/ur5_isaac_simulation/robot.urdf
│   └── Room_empty_table.usdc
└── wbcmimic/
    └── source/isaaclab_mimic/...
```

### Verification

To verify your setup works correctly:

1. **Check environment variables**:
   ```bash
   echo $X7_URDF_PATH
   echo $UR5_URDF_PATH
   echo $ROOM_USD_PATH
   echo $ISAAC_SIM_FRANKA_PATH
   ```

2. **Test file access**:
   ```bash
   # Check if files exist at the specified paths
   ls -la $X7_URDF_PATH
   ls -la $UR5_URDF_PATH
   ls -la $ROOM_USD_PATH
   ```

3. **Run a simple test**:
   ```python
   import os
   
   # Test X7 path resolution
   x7_path = os.environ.get('X7_URDF_PATH', 'fallback/path')
   print(f"X7 URDF path: {x7_path}")
   print(f"X7 URDF exists: {os.path.exists(x7_path)}")
   
   # Test UR5 path resolution
   ur5_path = os.environ.get('UR5_URDF_PATH', 'fallback/path')
   print(f"UR5 URDF path: {ur5_path}")
   print(f"UR5 URDF exists: {os.path.exists(ur5_path)}")
   ```

### Troubleshooting

#### Common Issues

1. **FileNotFoundError**: Check that the environment variable points to an existing file
2. **Import errors in VR policy**: Ensure `ISAAC_SIM_FRANKA_PATH` is set correctly or Isaac Sim is in a standard location
3. **Relative path issues**: Make sure you're running scripts from the expected directory

#### Debug Commands

```bash
# Check all environment variables
env | grep -E "(X7_URDF_PATH|UR5_URDF_PATH|ROOM_USD_PATH|ISAAC_SIM_FRANKA_PATH)"

# Test path resolution in Python
python -c "
import os
print('X7:', os.environ.get('X7_URDF_PATH', 'Not set'))
print('UR5:', os.environ.get('UR5_URDF_PATH', 'Not set'))
print('Room:', os.environ.get('ROOM_USD_PATH', 'Not set'))
print('Isaac Sim:', os.environ.get('ISAAC_SIM_FRANKA_PATH', 'Not set'))
"
```

## 🚀 Usage

### CDM Usage

Run depth inference on RGB-D camera data:

```bash
cd cdm
python infer.py \
    --encoder vitl \
    --model-path /path/to/model.pth \
    --rgb-image /path/to/rgb.jpg \
    --depth-image /path/to/depth.png \
    --output result.png
```

### WBCMimic Usage

Generate manipulation demonstrations using the three-step workflow:

```bash
cd wbcmimic

# 1. Record VR demonstrations
python scripts/basic/record_demos_ur5_quest.py \
    --task Isaac-UR5-CloseMicrowave-Joint-Mimic-v0 \
    --dataset_file ./demos/close_microwave.hdf5 \
    --num_demos 5

# 2. Annotate subtasks
python scripts/mimicgen/annotate_demos.py \
    --task Isaac-UR5-CloseMicrowave-Joint-Mimic-Annotate-v0 \
    --input_file ./demos/close_microwave.hdf5 \
    --output_file ./demos/close_microwave_annotated.hdf5 \
    --auto

# 3. Generate large-scale dataset
python scripts/mimicgen/generate_dataset_parallel_all.py \
    --task Isaac-UR5-CloseMicrowave-Joint-GoHome-OneCamera-Mimic-MP-v0 \
    --input_file ./demos/close_microwave_annotated.hdf5 \
    --output_file ./datasets/close_microwave_10k.zarr \
    --generation_num_trials 125 \
    --num_envs 4 \
    --n_procs 8 \
    --distributed \
    --enable_cameras \
    --mimic_mode uni \
    --headless
```

## 🔬 Research Contributions

### Camera Depth Models (CDMs)
- **Sensor-Specific Training**: Models trained on synthetic data with camera-specific noise patterns
- **Dual-Branch Architecture**: RGB semantics + depth scale fusion using Vision Transformers
- **Real-Time Performance**: Lightweight inference suitable for robot control
- **Zero-Shot Generalization**: Cross-camera performance without fine-tuning

### WBCMimic Enhancements
- **Whole-Body Control**: Unified control for mobile manipulators
- **Smooth Motion Generation**: Improved trajectory quality over original MimicGen
- **Multi-GPU Scaling**: Distributed simulation for faster data collection
- **Mobile Manipulation**: Support for tasks requiring base movement

## 🎯 Supported Tasks & Hardware

### Robotic Tasks
- **UR5 Tasks**: Clean plate, pick bowl, put bowl in microwave, close microwave
- **ARX-X7 Tasks**: Pick toothpaste into cup and push (mobile manipulation)

### Supported Cameras
- Intel RealSense D405/D415/D435/D455/L515
- Stereolabs ZED 2i (4 modes: Performance, Ultra, Quality, Neural)  
- Microsoft Azure Kinect

## 📚 Documentation

- **CDM Documentation**: See [cdm/README.md](cdm/README.md) for detailed usage
- **WBCMimic Documentation**: See [wbcmimic/README.md](wbcmimic/README.md) for setup and examples

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{liu2025manipulation,
  title={Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots},
  author={Liu, Minghuan and Zhu, Zhengbang and Han, Xiaoshen and Hu, Peng and Lin, Haotong and 
          Li, Xinyao and Chen, Jingxiao and Xu, Jiafeng and Yang, Yichu and Lin, Yunfeng and 
          Li, Xinghang and Yu, Yong and Zhang, Weinan and Kong, Tao and Kang, Bingyi},
  journal={arXiv preprint},
  year={2025}
}
```

## 📝 License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## 🔗 Links

- **Project Page**: [Website](https://manipulation-as-in-simulation.github.io/)
- **Dataset**: [ByteCameraDepth](https://huggingface.co/datasets/ByteDance-Seed/ByteCameraDepth)
- **Isaac Lab**: [Documentation](https://isaac-sim.github.io/IsaacLab/)
- **CuRobo**: [Documentation](https://curobo.org/)

## 📧 Contact

For questions and support, please open an issue in this repository.

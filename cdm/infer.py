#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from rgbddepth.dpt import RGBDDepth

# Automatically select the best available device for inference
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Model configurations for different Vision Transformer (ViT) encoder sizes
# Each config specifies the encoder type, feature dimensions, and output channels
model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },  # Small
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },  # Base
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },  # Large
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },  # Giant
}


def colorize(value, vmin=None, vmax=None, cmap="Spectral"):
    """Convert depth values to colorized visualization.

    Args:
        value: Input depth array
        vmin: Minimum value for normalization (auto-computed if None)
        vmax: Maximum value for normalization (auto-computed if None)
        cmap: Colormap name for visualization

    Returns:
        RGB image array for visualization
    """
    # Skip processing if input is already RGB
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]  # Extract single channel

    # Mark invalid/zero depth values
    invalid_mask = value < 0.0001

    # Normalize values to [0, 1] range
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = ((value - vmin) / (vmax - vmin)).clip(0, 1)

    # Apply colormap to create colored visualization
    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # Returns RGBA values

    # Set invalid regions to black
    value[invalid_mask] = 0

    # Return only RGB channels (drop alpha)
    img = value[..., :3]
    return img


def image_grid(imgs, rows, cols):
    """Create a grid layout from a list of images.

    Args:
        imgs: List of image arrays to arrange in grid
        rows: Number of rows in the grid
        cols: Number of columns in the grid

    Returns:
        Combined image array with grid layout
    """
    if not len(imgs):
        return None

    # Ensure we have the right number of images for the grid
    assert len(imgs) == rows * cols

    # Use dimensions from first image as reference
    h, w = imgs[0].shape[:2]

    # Create empty canvas for the grid
    grid = Image.new("RGB", size=(cols * w, rows * h))

    # Place each image in its grid position
    for i, img in enumerate(imgs):
        # Calculate grid position (row, col) from linear index
        col_idx = i % cols
        row_idx = i // cols

        # Paste image at calculated position
        grid.paste(
            Image.fromarray(img.astype(np.uint8)).resize(
                (w, h), resample=Image.BILINEAR
            ),
            box=(col_idx * w, row_idx * h),
        )

    return np.array(grid)


def parse_arguments():
    """Parse command line arguments for RGBD depth inference.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="RGBD Depth Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        default="vitl",
        help="Model encoder type",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--rgb-image", type=str, required=True, help="Path to the RGB input image"
    )
    parser.add_argument(
        "--depth-image", type=str, required=True, help="Path to the depth input image"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output visualization image path",
    )
    parser.add_argument(
        "--input-size", type=int, default=518, help="Input size for inference"
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for depth values",
    )
    parser.add_argument(
        "--max-depth", type=float, default=6.0, help="Maximum valid depth value"
    )
    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments and create output directory if needed.

    Args:
        args: Parsed command line arguments

    Raises:
        SystemExit: If any required input files don't exist
    """
    # Check if model checkpoint file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        sys.exit(1)

    # Check if input RGB image exists
    if not os.path.exists(args.rgb_image):
        print(f"Error: RGB image '{args.rgb_image}' does not exist")
        sys.exit(1)

    # Check if input depth image exists
    if not os.path.exists(args.depth_image):
        print(f"Error: Depth image '{args.depth_image}' does not exist")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def load_model(encoder, model_path):
    """Load and initialize the RGBD depth estimation model.

    Args:
        encoder: Model encoder type ('vits', 'vitb', 'vitl', 'vitg')
        model_path: Path to the model checkpoint file

    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    # Initialize model with configuration for specified encoder
    model = RGBDDepth(**model_configs[encoder])

    # Load checkpoint and extract state dict
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model" in checkpoint:
        # Handle checkpoints that wrap state dict in 'model' key
        # Remove 'module.' prefix if present (from DataParallel training)
        states = {k[7:]: v for k, v in checkpoint["model"].items()}
    elif "state_dict" in checkpoint:
        states = checkpoint["state_dict"]
        states = {k[9:]: v for k, v in states.items()}
    else:
        # Direct state dict checkpoint
        states = checkpoint

    # Load weights and move to device
    model.load_state_dict(states, strict=False)
    model = model.to(DEVICE).eval()

    print(f"Model loaded: {encoder} from {model_path}")
    return model


def load_images(rgb_path, depth_path, depth_scale, max_depth):
    """Load and preprocess RGB and depth images.

    Args:
        rgb_path: Path to RGB image file
        depth_path: Path to depth image file
        depth_scale: Scale factor to convert depth values to meters
        max_depth: Maximum valid depth value (values above this are set to 0)

    Returns:
        tuple: (rgb_image, depth_low_res, similarity_depth_low_res)
            - rgb_image: RGB image in numpy array format (BGR -> RGB)
            - depth_low_res: Depth values in meters
            - similarity_depth_low_res: Inverse depth values (1/depth) for model input
    """
    # Load RGB image and convert from BGR to RGB
    rgb_src = np.asarray(cv2.imread(rgb_path)[:, :, ::-1])
    if rgb_src is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")

    # Load depth image (usually 16-bit)
    depth_low_res = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_low_res is None:
        raise ValueError(f"Could not load depth image from {depth_path}")

    # Convert depth to meters and clamp invalid values
    depth_low_res = np.asarray(depth_low_res).astype(np.float32) / depth_scale
    depth_low_res[depth_low_res > max_depth] = 0.0  # Remove values beyond max range

    # Create similarity depth (inverse depth) for model input
    # Only compute inverse for valid depth values
    simi_depth_low_res = np.zeros_like(depth_low_res)
    simi_depth_low_res[depth_low_res > 0] = 1 / depth_low_res[depth_low_res > 0]

    print(f"Images loaded: RGB {rgb_src.shape}, Depth {depth_low_res.shape}")
    return rgb_src, depth_low_res, simi_depth_low_res


def create_visualization(rgb_src, depth_rs, simi_depth, pred):
    """Create a 2x2 grid visualization comparing input and predicted depth.

    Args:
        rgb_src: Original RGB image
        depth_rs: Ground truth depth image
        simi_depth: Similarity (inverse) depth input
        pred: Predicted depth from model

    Returns:
        numpy.ndarray: Combined visualization image with 2x2 grid layout:
            - Top-left: Original RGB image
            - Top-right: Input depth (colorized)
            - Bottom-left: Predicted depth (colorized)
            - Bottom-right: Relative error map (colorized)
    """
    # Colorize predicted depth
    depth_pred_abs_col = colorize(pred, vmin=1.0, vmax=15.0, cmap="Spectral")
    depth_pred_abs_col = (depth_pred_abs_col * 255).astype(np.uint8)
    depth_pred_abs_col = cv2.cvtColor(depth_pred_abs_col, cv2.COLOR_RGB2BGR)

    # Convert similarity depth back to regular depth for visualization
    depth_low_res_col = colorize(depth_rs, vmin=1.0, vmax=15.0, cmap="Spectral")
    depth_low_res_col = (depth_low_res_col * 255).astype(np.uint8)
    depth_low_res_col = cv2.cvtColor(depth_low_res_col, cv2.COLOR_RGB2BGR)

    # Calculate relative error between ground truth and prediction
    depth_arel_abs = np.zeros_like(depth_rs)
    valid = depth_rs > 0  # Only compute error for valid depth pixels
    depth_arel_abs[valid] = np.abs(depth_rs[valid] - pred[valid]) / depth_rs[valid]
    depth_error_abs_col = colorize(depth_arel_abs, vmin=0.0, vmax=0.2, cmap="coolwarm")
    depth_arel_abs = (depth_arel_abs * 255).astype(np.uint8)
    depth_arel_abs = cv2.cvtColor(depth_arel_abs, cv2.COLOR_RGB2BGR)

    # Arrange all visualizations in a 2x2 grid
    return image_grid(
        [rgb_src, depth_low_res_col, depth_pred_abs_col, depth_error_abs_col], 2, 2
    )


def inference(args):
    """Run complete depth inference pipeline.

    Args:
        args: Parsed command line arguments containing all parameters
    """
    # Validate all input files and create output directory
    validate_inputs(args)

    # Load the trained model
    model = load_model(args.encoder, args.model_path)

    # Load and preprocess input images
    rgb_src, depth_low_res, simi_depth_low_res = load_images(
        args.rgb_image, args.depth_image, args.depth_scale, 25.0
    )

    # Run model inference
    pred = model.infer_image(rgb_src, simi_depth_low_res, input_size=args.input_size)
    print(
        f"Prediction info: shape={pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}"
    )

    # Convert from inverse depth back to regular depth
    pred = 1 / pred

    # Create visualization comparing input, prediction, and error
    artifact = create_visualization(rgb_src, depth_low_res, simi_depth_low_res, pred)

    # Save the visualization
    Image.fromarray(artifact).save(args.output)
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    # Parse command line arguments and run inference
    args = parse_arguments()
    inference(args)

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import argparse
import math
import os

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Interpolate position embeddings in the checkpoint for highres FT")

    # Required arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the input checkpoint file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the modified checkpoint",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--target_img_size",
        type=int,
        default=518,
        help="Target image size (default: 518)",
    )
    parser.add_argument(
        "--target_patch_size",
        type=int,
        default=14,
        help="Target patch size (default: 14)",
    )
    parser.add_argument(
        "--pos_embed_key",
        type=str,
        default="backbone.pos_embed",
        help="Key for position embeddings in state dict (default: backbone.pos_embed)",
    )
    parser.add_argument(
        "--teacher_key",
        type=str,
        default="teacher",
        help="Key for teacher model in checkpoint (default: teacher)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint[args.teacher_key]

    # Calculate target grid size
    target_grid_size = args.target_img_size // args.target_patch_size
    print(f"Target grid size: {target_grid_size}")

    # Extract position embeddings
    pos_embed = state_dict[args.pos_embed_key]
    print(f"Original position embedding shape: {pos_embed.shape}")

    # Separate class and patch position embeddings
    cls_pos_embed = pos_embed[:, :1]
    patch_pos_embed = pos_embed[:, 1:]
    embed_dim = pos_embed.shape[-1]

    # Calculate original grid size
    original_num_patches = patch_pos_embed.shape[1]
    original_grid_size = int(math.sqrt(original_num_patches))
    print(f"Original grid size: {original_grid_size}")

    # Reshape for interpolation
    patch_pos_embed = patch_pos_embed.reshape(1, original_grid_size, original_grid_size, embed_dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

    # Interpolate patch position embeddings
    interpolated_patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(target_grid_size, target_grid_size),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    )
    interpolated_patch_pos_embed = interpolated_patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, embed_dim)

    # Combine class and patch embeddings
    new_pos_embed = torch.cat((cls_pos_embed, interpolated_patch_pos_embed), dim=1)
    print(f"New position embedding shape: {new_pos_embed.shape}")

    # Update checkpoint
    state_dict[args.pos_embed_key] = new_pos_embed
    checkpoint[args.teacher_key] = state_dict

    # Save modified checkpoint
    print(f"Saving modified checkpoint to: {args.output_path}")
    torch.save(checkpoint, args.output_path)
    print("Done!")


if __name__ == "__main__":
    main()

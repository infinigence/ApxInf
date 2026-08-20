#!/usr/bin/env python3
"""Dump intermediate HF vision tower states for debugging apxinf's vision port."""
import torch
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "/hanjinchen/models/Qwen3-VL-2B-Instruct"

# Load reference pixel_values + grid_thw from the Phase 0 dump.
ref = np.load("tests/qwen3vl_reference/qwen3vl_image.npz")
pixel_values = torch.from_numpy(ref["image_pixel_values"]).to(torch.bfloat16).cuda()
grid_thw = torch.from_numpy(ref["image_grid_thw"]).cuda()
print("pixel_values:", pixel_values.shape, pixel_values.dtype)
print("grid_thw:", grid_thw.tolist())

model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
visual = model.model.visual

with torch.no_grad():
    # Patch embed
    x = visual.patch_embed(pixel_values)
    print(f"post_patch_embed: shape={x.shape} first3={x[0,:3].float().tolist()}")
    np.save("/tmp/hf_post_patch_embed.npy", x.float().cpu().numpy())

    # Pos embed
    pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
    print(f"post_pos_embed_interpolate: shape={pos_embeds.shape} first3={pos_embeds[0,:3].float().tolist()}")
    np.save("/tmp/hf_pos_embeds.npy", pos_embeds.float().cpu().numpy())

    x = x + pos_embeds
    print(f"post_add_pos: shape={x.shape} first3={x[0,:3].float().tolist()}")
    np.save("/tmp/hf_post_add_pos.npy", x.float().cpu().numpy())

    # Rotary pos emb — HF does cat((rotary, rotary)) then cos/sin
    rotary = visual.rot_pos_emb(grid_thw)
    print(f"rotary_pos_emb: shape={rotary.shape} first3={rotary[0,:3].float().tolist()}")
    emb = torch.cat((rotary, rotary), dim=-1)
    pos_emb_tuple = (emb.cos(), emb.sin())
    print(f"cos shape: {pos_emb_tuple[0].shape}")

    # Run blocks 0 and 1
    cu_seqlens = torch.tensor([0, x.shape[0]], device=x.device)

    for i, blk in enumerate(visual.blocks[:2]):
        x = blk(x, cu_seqlens, position_embeddings=pos_emb_tuple)
        print(f"post_block_{i}: shape={x.shape} first3={x[0,:3].float().tolist()}")
        np.save(f"/tmp/hf_post_block_{i}.npy", x.float().cpu().numpy())

    # Run the merger on the final block output
    # First run all 24 blocks
    for i, blk in enumerate(visual.blocks[2:], start=2):
        x = blk(x, cu_seqlens, position_embeddings=pos_emb_tuple)
    print(f"post_all_blocks: shape={x.shape} first3={x[0,:3].float().tolist()}")
    np.save("/tmp/hf_post_all_blocks.npy", x.float().cpu().numpy())

    primary = visual.merger(x)
    print(f"primary: shape={primary.shape} first3={primary[0,:3].float().tolist()}")
    np.save("/tmp/hf_primary.npy", primary.float().cpu().numpy())

print("Done — dumps in /tmp/hf_*.npy")

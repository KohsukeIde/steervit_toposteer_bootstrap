#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from toposteer.config import deep_update, load_yaml, parse_override_pairs
from toposteer.datasets import UnifiedRefExpDataset, collate_refexp
from toposteer.evaluation import compute_flip_accuracy
from toposteer.losses import background_cosine_drift, counterfactual_margin_loss, soft_patch_ce
from toposteer.models import SteerViTTrainable
from toposteer.utils import ensure_dir, patchify_soft_mask, seed_everything, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm-start training scaffold for SteerViT-based dense prompting.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def save_checkpoint(path: Path, model: SteerViTTrainable, optimizer, step: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "config": cfg,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, path)


@torch.no_grad()
def evaluate(model: SteerViTTrainable, loader: DataLoader, cfg: dict) -> dict[str, float]:
    model.eval()
    pos_scores_all = []
    neg_scores_all = []
    losses = []

    for batch in loader:
        images = batch["images"].to(model.device_name)
        masks_pos = batch["masks_pos"].to(model.device_name)
        valid_neg_mask = batch["valid_neg_mask"].to(model.device_name)

        out = model.forward_outputs(images, texts=batch["prompts_pos"])
        patch_target = patchify_soft_mask(masks_pos, model.patch_grid, normalize=True)
        losses.append(float(soft_patch_ce(out.logits, patch_target).item()))

        if valid_neg_mask.any():
            masks_neg = batch["masks_neg"].to(model.device_name)
            patch_pos = patchify_soft_mask(masks_pos, model.patch_grid, normalize=False)
            patch_neg = patchify_soft_mask(masks_neg, model.patch_grid, normalize=False)
            valid_idx = valid_neg_mask.nonzero(as_tuple=False).squeeze(1)
            pos_scores = (out.probs[valid_idx] * patch_pos[valid_idx]).sum(dim=1) / patch_pos[valid_idx].sum(dim=1).clamp_min(1e-6)
            neg_scores = (out.probs[valid_idx] * patch_neg[valid_idx]).sum(dim=1) / patch_neg[valid_idx].sum(dim=1).clamp_min(1e-6)
            pos_scores_all.append(pos_scores.detach().cpu())
            neg_scores_all.append(neg_scores.detach().cpu())

    metrics = {"val_refseg_loss": sum(losses) / max(len(losses), 1)}
    if pos_scores_all:
        pos_scores = torch.cat(pos_scores_all)
        neg_scores = torch.cat(neg_scores_all)
        metrics["val_flip_accuracy"] = compute_flip_accuracy(pos_scores, neg_scores)
        metrics["val_gap"] = float((pos_scores - neg_scores).mean().item())
    return metrics


def _select_texts(texts: list[str | None], indices: torch.Tensor) -> list[str]:
    return [texts[int(i)] for i in indices.detach().cpu().tolist()]


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = deep_update(cfg, parse_override_pairs(args.override))
    if args.device is not None:
        cfg["device"] = args.device

    seed_everything(cfg.get("seed", 42))
    output_dir = ensure_dir(args.output_dir)
    write_json(output_dir / "resolved_config.json", cfg)

    trainable_modules = []
    freeze_cfg = cfg.get("freeze", {})
    if not freeze_cfg.get("connector", False):
        trainable_modules.append("connector")
    if not freeze_cfg.get("segmentation_head", False):
        trainable_modules.append("lin_seg_head")
    if not freeze_cfg.get("gated_cross_attn", False):
        trainable_modules.append("gated_cross_attn")
    if not freeze_cfg.get("vision_backbone", True):
        trainable_modules.append("vision_backbone")
    if not freeze_cfg.get("text_model", True):
        trainable_modules.append("text_model")

    model = SteerViTTrainable(
        checkpoint=args.checkpoint,
        device=cfg.get("device", "cuda"),
        trainable_modules=trainable_modules,
    )
    transform = model.get_transforms()

    family_whitelist = cfg.get("data", {}).get("family_whitelist")
    train_ds = UnifiedRefExpDataset(
        args.train_manifest,
        image_transform=transform,
        family_whitelist=family_whitelist,
    )
    val_ds = UnifiedRefExpDataset(
        args.val_manifest,
        image_transform=transform,
        family_whitelist=family_whitelist,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_refexp,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_refexp,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.get("lr", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True) and "cuda" in str(model.device_name)))

    step = 0
    max_steps = int(cfg.get("max_steps", 20000))
    log_rows = []

    while step < max_steps:
        model.train()
        for batch in tqdm(train_loader, desc=f"Training step {step}/{max_steps}", leave=False):
            if step >= max_steps:
                break

            images = batch["images"].to(model.device_name)
            masks_pos = batch["masks_pos"].to(model.device_name)
            masks_neg = batch["masks_neg"].to(model.device_name)
            valid_neg_mask = batch["valid_neg_mask"].to(model.device_name)
            valid_prompt_neg_mask = batch["valid_prompt_neg_mask"].to(model.device_name)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out_pos = model.forward_outputs(images, texts=batch["prompts_pos"])
                patch_target = patchify_soft_mask(masks_pos, model.patch_grid, normalize=True)
                loss_refseg = soft_patch_ce(out_pos.logits, patch_target)

                total_loss = float(cfg["loss"].get("refseg_weight", 1.0)) * loss_refseg
                loss_cf = torch.tensor(0.0, device=model.device_name)
                loss_bg = torch.tensor(0.0, device=model.device_name)

                if cfg["loss"].get("use_counterfactual", False) and valid_neg_mask.any():
                    patch_pos = patchify_soft_mask(masks_pos, model.patch_grid, normalize=False)
                    patch_neg = patchify_soft_mask(masks_neg, model.patch_grid, normalize=False)

                    valid_idx = valid_neg_mask.nonzero(as_tuple=False).squeeze(1)
                    loss_cf = counterfactual_margin_loss(
                        patch_probs_pos_prompt=out_pos.probs[valid_idx],
                        patch_mask_pos=patch_pos[valid_idx],
                        patch_mask_neg=patch_neg[valid_idx],
                        margin=float(cfg["loss"].get("cf_margin", 0.10)),
                    )

                    if valid_prompt_neg_mask.any():
                        reverse_idx = valid_prompt_neg_mask.nonzero(as_tuple=False).squeeze(1)
                        reverse_out = model.forward_outputs(images[reverse_idx], texts=_select_texts(batch["prompts_neg"], reverse_idx))
                        loss_cf = loss_cf + counterfactual_margin_loss(
                            patch_probs_pos_prompt=reverse_out.probs,
                            patch_mask_pos=patch_neg[reverse_idx],
                            patch_mask_neg=patch_pos[reverse_idx],
                            margin=float(cfg["loss"].get("cf_margin", 0.10)),
                        )

                    total_loss = total_loss + float(cfg["loss"].get("cf_weight", 1.0)) * loss_cf

                if cfg["loss"].get("use_background", False):
                    base_out = model.forward_outputs(images, texts=None)
                    patch_pos = patchify_soft_mask(masks_pos, model.patch_grid, normalize=False)
                    protect = 1.0 - torch.clamp(patch_pos + patchify_soft_mask(masks_neg, model.patch_grid, normalize=False), 0.0, 1.0)
                    loss_bg = background_cosine_drift(out_pos.dense_tokens, base_out.dense_tokens.detach(), protect)
                    total_loss = total_loss + float(cfg["loss"].get("bg_weight", 0.25)) * loss_bg

            scaler.scale(total_loss).backward()

            grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            row = {
                "step": step,
                "loss_total": float(total_loss.item()),
                "loss_refseg": float(loss_refseg.item()),
                "loss_cf": float(loss_cf.item()),
                "loss_bg": float(loss_bg.item()),
                "num_valid_neg": int(valid_neg_mask.sum().item()),
                "num_valid_prompt_neg": int(valid_prompt_neg_mask.sum().item()),
            }
            log_rows.append(row)

            if step % int(cfg.get("log_every", 20)) == 0:
                print(row)

            if step > 0 and step % int(cfg.get("eval_every", 250)) == 0:
                metrics = evaluate(model, val_loader, cfg)
                metrics["step"] = step
                log_rows.append(metrics)
                print(metrics)
                write_jsonl(output_dir / "metrics.jsonl", log_rows)

            if step > 0 and step % int(cfg.get("save_every", 1000)) == 0:
                save_checkpoint(output_dir / f"checkpoint_step_{step}.pt", model, optimizer, step, cfg)

            step += 1

    save_checkpoint(output_dir / "checkpoint_final.pt", model, optimizer, step, cfg)
    write_jsonl(output_dir / "metrics.jsonl", log_rows)
    print(f"Training complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()

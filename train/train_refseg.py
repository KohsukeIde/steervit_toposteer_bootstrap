#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from toposteer.config import deep_update, load_yaml, parse_override_pairs
from toposteer.datasets import UnifiedRefExpDataset, collate_refexp
from toposteer.evaluation import compute_flip_accuracy, heatmap_mass_gap
from toposteer.losses import background_cosine_drift, counterfactual_margin_loss, soft_patch_ce
from toposteer.losses.counterfactual import pairwise_mask_scores
from toposteer.models import SteerViTTrainable
from toposteer.utils import ensure_dir, patchify_soft_mask, seed_everything, write_json, write_jsonl

METRIC_ALIASES = {
    "flip_acc": "locked_flip_accuracy",
    "flip_accuracy": "locked_flip_accuracy",
    "mean_gap": "locked_mean_gap",
    "gap": "locked_mean_gap",
    "refseg_loss": "val_refseg_loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm-start training scaffold for SteerViT-based dense prompting.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=False, default=None)
    parser.add_argument(
        "--eval-manifest",
        required=False,
        default=None,
        help="Optional locked paired benchmark used only for model selection / monitoring.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Required when --checkpoint is a TopoSteer training checkpoint with model_state_dict.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--best-metric", type=str, default=None)
    parser.add_argument("--best-tiebreak", type=str, default=None)
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


@torch.inference_mode()
def evaluate_refseg_loader(
    model: SteerViTTrainable,
    loader: DataLoader,
    gate_factor: float,
    score_mode: str = "mean",
) -> dict[str, float]:
    model.eval()
    model.set_gate_factor(float(gate_factor))
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
            pos_scores, neg_scores = pairwise_mask_scores(
                out.probs[valid_idx],
                patch_mask_pos=patch_pos[valid_idx],
                patch_mask_neg=patch_neg[valid_idx],
                mode=score_mode,
            )
            pos_scores_all.append(pos_scores.detach().cpu())
            neg_scores_all.append(neg_scores.detach().cpu())

    metrics = {"val_refseg_loss": sum(losses) / max(len(losses), 1), "val_gate_factor": float(gate_factor)}
    if pos_scores_all:
        pos_scores = torch.cat(pos_scores_all)
        neg_scores = torch.cat(neg_scores_all)
        metrics["val_flip_accuracy"] = compute_flip_accuracy(pos_scores, neg_scores)
        metrics["val_gap"] = float((pos_scores - neg_scores).mean().item())
    return metrics


@torch.inference_mode()
def evaluate_locked_loader(
    model: SteerViTTrainable,
    loader: DataLoader,
    gate_factor: float,
    score_mode: str = "mean",
) -> dict[str, float]:
    model.eval()
    model.set_gate_factor(float(gate_factor))
    pos_scores_all = []
    neg_scores_all = []

    for batch in loader:
        images = batch["images"].to(model.device_name)
        masks_pos = batch["masks_pos"].to(model.device_name)
        masks_neg = batch["masks_neg"].to(model.device_name)
        valid_neg_mask = batch["valid_neg_mask"].to(model.device_name)

        if not valid_neg_mask.any():
            continue

        out = model.forward_outputs(images, texts=batch["prompts_pos"])
        patch_pos = patchify_soft_mask(masks_pos, model.patch_grid, normalize=False)
        patch_neg = patchify_soft_mask(masks_neg, model.patch_grid, normalize=False)
        valid_idx = valid_neg_mask.nonzero(as_tuple=False).squeeze(1)
        pos_scores, neg_scores = pairwise_mask_scores(
            out.probs[valid_idx],
            patch_mask_pos=patch_pos[valid_idx],
            patch_mask_neg=patch_neg[valid_idx],
            mode=score_mode,
        )
        pos_scores_all.append(pos_scores.detach().cpu())
        neg_scores_all.append(neg_scores.detach().cpu())

    metrics = {"locked_gate_factor": float(gate_factor)}
    if pos_scores_all:
        pos_scores = torch.cat(pos_scores_all)
        neg_scores = torch.cat(neg_scores_all)
        metrics["locked_num_pairs"] = int(pos_scores.numel())
        metrics["locked_flip_accuracy"] = compute_flip_accuracy(pos_scores, neg_scores)
        metrics["locked_mean_gap"] = heatmap_mass_gap(pos_scores, neg_scores)
    return metrics


def _select_texts(texts: list[str | None], indices: torch.Tensor) -> list[str]:
    out: list[str] = []
    for i in indices.detach().cpu().tolist():
        text = texts[int(i)]
        if text is None:
            raise ValueError("Encountered None in prompt_neg for a reverse-counterfactual subset.")
        out.append(text)
    return out


def _resolve_metric_name(name: str | None) -> str | None:
    if name is None:
        return None
    return METRIC_ALIASES.get(name, name)


def _infer_mode(metric_name: str | None, explicit_mode: str | None = None) -> str:
    if explicit_mode in {"min", "max"}:
        return explicit_mode
    if metric_name is None:
        return "max"
    return "min" if "loss" in metric_name.lower() else "max"


def _metric_value(metrics: dict[str, Any], name: str | None) -> float | None:
    if name is None:
        return None
    value = metrics.get(name)
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _is_better(new: float, old: float, mode: str, tol: float = 1e-12) -> bool:
    if mode == "max":
        return new > old + tol
    if mode == "min":
        return new < old - tol
    raise ValueError(f"Unknown comparison mode: {mode}")


def _should_update_best(
    metrics: dict[str, Any],
    best_metrics: dict[str, Any] | None,
    primary_metric: str | None,
    primary_mode: str,
    tiebreak_metric: str | None,
    tiebreak_mode: str,
) -> bool:
    if primary_metric is None:
        return False

    new_primary = _metric_value(metrics, primary_metric)
    if new_primary is None:
        return False
    if best_metrics is None:
        return True

    old_primary = _metric_value(best_metrics, primary_metric)
    if old_primary is None:
        return True
    if _is_better(new_primary, old_primary, primary_mode):
        return True
    if abs(new_primary - old_primary) > 1e-12:
        return False

    if tiebreak_metric is None:
        return False
    new_tie = _metric_value(metrics, tiebreak_metric)
    old_tie = _metric_value(best_metrics, tiebreak_metric)
    if new_tie is None:
        return False
    if old_tie is None:
        return True
    return _is_better(new_tie, old_tie, tiebreak_mode)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = deep_update(cfg, parse_override_pairs(args.override))
    if args.device is not None:
        cfg["device"] = args.device

    selection_cfg = cfg.get("selection", {})
    best_metric = _resolve_metric_name(args.best_metric or selection_cfg.get("best_metric"))
    best_tiebreak = _resolve_metric_name(args.best_tiebreak or selection_cfg.get("best_tiebreak"))

    seed_everything(cfg.get("seed", 42))
    output_dir = ensure_dir(args.output_dir)

    gate_cfg = cfg.get("gate", {})
    train_gate_factor = float(gate_cfg.get("train_factor", 1.0))
    val_gate_factor = float(gate_cfg.get("val_factor", train_gate_factor))
    locked_gate_factor = float(gate_cfg.get("locked_eval_factor", 1.0))
    score_mode = str(cfg.get("score_mode", "mean"))

    if best_metric is None:
        if args.eval_manifest is not None:
            best_metric = "locked_flip_accuracy"
            best_tiebreak = best_tiebreak or "locked_mean_gap"
        elif args.val_manifest is not None:
            best_metric = "val_refseg_loss"
        else:
            best_metric = None

    best_mode = _infer_mode(best_metric, selection_cfg.get("best_mode"))
    best_tiebreak_mode = _infer_mode(best_tiebreak, selection_cfg.get("best_tiebreak_mode"))

    resolved_config = dict(cfg)
    resolved_config["selection"] = {
        "best_metric": best_metric,
        "best_mode": best_mode,
        "best_tiebreak": best_tiebreak,
        "best_tiebreak_mode": best_tiebreak_mode,
    }
    resolved_config["gate"] = {
        "train_factor": train_gate_factor,
        "val_factor": val_gate_factor,
        "locked_eval_factor": locked_gate_factor,
    }
    resolved_config["base_checkpoint"] = args.base_checkpoint or args.checkpoint
    write_json(output_dir / "resolved_config.json", resolved_config)

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

    model = SteerViTTrainable.from_any_checkpoint(
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        device=cfg.get("device", "cuda"),
        trainable_modules=trainable_modules,
    )
    model.set_gate_factor(train_gate_factor)
    transform = model.get_transforms()
    write_json(output_dir / "checkpoint_load_info.json", getattr(model, "load_info", {}))

    family_whitelist = cfg.get("data", {}).get("family_whitelist")
    train_ds = UnifiedRefExpDataset(
        args.train_manifest,
        image_transform=transform,
        family_whitelist=family_whitelist,
    )
    val_loader = None
    locked_loader = None
    if args.val_manifest is not None:
        val_ds = UnifiedRefExpDataset(
            args.val_manifest,
            image_transform=transform,
            family_whitelist=family_whitelist,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.get("batch_size", 8),
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            collate_fn=collate_refexp,
            drop_last=False,
        )
    if args.eval_manifest is not None:
        locked_ds = UnifiedRefExpDataset(
            args.eval_manifest,
            image_transform=transform,
            family_whitelist=family_whitelist,
        )
        locked_loader = DataLoader(
            locked_ds,
            batch_size=cfg.get("eval_batch_size", cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            collate_fn=collate_refexp,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
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
    best_metrics: dict[str, Any] | None = None

    while step < max_steps:
        model.train()
        model.set_gate_factor(train_gate_factor)
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
                        mode=score_mode,
                    )

                    if valid_prompt_neg_mask.any():
                        reverse_idx = valid_prompt_neg_mask.nonzero(as_tuple=False).squeeze(1)
                        reverse_out = model.forward_outputs(images[reverse_idx], texts=_select_texts(batch["prompts_neg"], reverse_idx))
                        loss_cf = loss_cf + counterfactual_margin_loss(
                            patch_probs_pos_prompt=reverse_out.probs,
                            patch_mask_pos=patch_neg[reverse_idx],
                            patch_mask_neg=patch_pos[reverse_idx],
                            margin=float(cfg["loss"].get("cf_margin", 0.10)),
                            mode=score_mode,
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
                metrics: dict[str, Any] = {"step": step}
                if val_loader is not None:
                    metrics.update(evaluate_refseg_loader(model, val_loader, gate_factor=val_gate_factor, score_mode=score_mode))
                if locked_loader is not None:
                    metrics.update(evaluate_locked_loader(model, locked_loader, gate_factor=locked_gate_factor, score_mode=score_mode))
                log_rows.append(metrics)
                print(metrics)
                write_jsonl(output_dir / "metrics.jsonl", log_rows)

                if _should_update_best(
                    metrics,
                    best_metrics=best_metrics,
                    primary_metric=best_metric,
                    primary_mode=best_mode,
                    tiebreak_metric=best_tiebreak,
                    tiebreak_mode=best_tiebreak_mode,
                ):
                    best_metrics = dict(metrics)
                    save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, step, resolved_config)
                    write_json(output_dir / "best_metrics.json", best_metrics)
                    print({"best_checkpoint_updated": True, **best_metrics})

                model.train()
                model.set_gate_factor(train_gate_factor)

            if step > 0 and step % int(cfg.get("save_every", 1000)) == 0:
                save_checkpoint(output_dir / f"checkpoint_step_{step}.pt", model, optimizer, step, resolved_config)

            step += 1

    save_checkpoint(output_dir / "checkpoint_final.pt", model, optimizer, step, resolved_config)
    if best_metrics is not None:
        write_json(output_dir / "best_metrics.json", best_metrics)
    write_jsonl(output_dir / "metrics.jsonl", log_rows)
    print(f"Training complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()

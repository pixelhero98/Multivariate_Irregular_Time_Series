import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from Dataset.fin_dataset import run_experiment
from Latent_Space.latent_vae import LatentVAE
from Model.llapdit import LLapDiT
from Model.llapdit_utils import (
    EMA,
    build_context,
    compute_latent_stats,
    decode_latents_with_vae,
    flatten_targets,
    set_torch,
)
from Model.summarizer import LaplaceAE


@dataclass
class PredictionBundle:
    dataset: str
    pred_len: int
    context: np.ndarray
    ground_truth: np.ndarray
    llapdit_mean: np.ndarray
    llapdit_samples: np.ndarray


@dataclass
class DatasetSpec:
    name: str
    config_module: str


# ----------------------------- configuration helpers -----------------------------


def _as_namespace(module) -> SimpleNamespace:
    cfg = {k: getattr(module, k) for k in dir(module) if k.isupper()}
    return SimpleNamespace(**cfg)


def _clone_for_pred(base_cfg: SimpleNamespace, pred: int) -> SimpleNamespace:
    cfg = SimpleNamespace(**vars(base_cfg))
    cfg.PRED = pred
    cfg.SUM_CONTEXT_LEN = getattr(cfg, "SUM_CONTEXT_LEN", pred)
    cfg.VAE_CKPT = getattr(cfg, "VAE_DIR", "./vae") + f"/pred-{pred}_ch-{cfg.VAE_LATENT_CHANNELS}_elbo.pt"
    cfg.SUM_CKPT = getattr(cfg, "SUM_DIR", "./summarizer") + f"/{pred}-{cfg.VAE_LATENT_CHANNELS}-summarizer.pt"
    cfg.CKPT_DIR = getattr(cfg, "CKPT_DIR", "./checkpoints")
    cfg.OUT_DIR = getattr(cfg, "OUT_DIR", "./output")
    cfg.POLE_PLOT_DIR = getattr(cfg, "POLE_PLOT_DIR", str(Path(cfg.OUT_DIR) / "pole_plots"))
    return cfg


def _resolve_llapdit_checkpoint(cfg: SimpleNamespace, pred: int, override: Optional[str]) -> Path:
    if override:
        cand = Path(override.format(pred=pred))
        if cand.exists():
            return cand
        raise FileNotFoundError(f"LLapDiT checkpoint not found at {cand}")

    candidates: List[Path] = []
    trained = getattr(cfg, "TRAINED_LLapDiT", "")
    if trained:
        p = Path(str(trained).format(pred=pred))
        if p.exists():
            candidates.append(p)
    ckpt_dir = Path(cfg.CKPT_DIR)
    candidates.extend(sorted(ckpt_dir.glob(f"*pred-{pred}*.pt")))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(
        "No LLapDiT checkpoint found. Provide --llapdit-checkpoint or set TRAINED_LLapDiT."
    )


# ----------------------------- model loaders -----------------------------


def _load_components(
    cfg: SimpleNamespace,
    device: torch.device,
    *,
    llapdit_ckpt: Optional[str],
) -> Tuple[torch.utils.data.DataLoader, LaplaceAE, LatentVAE, LLapDiT, torch.Tensor, torch.Tensor]:
    # Dataloaders
    train_dl, val_dl, test_dl, sizes = run_experiment(
        data_dir=cfg.DATA_DIR,
        date_batching=cfg.date_batching,
        dates_per_batch=cfg.BATCH_SIZE,
        K=cfg.WINDOW,
        H=cfg.PRED,
        coverage=cfg.COVERAGE,
        batch_size=cfg.BATCH_SIZE,
    )
    if sizes:
        print(f"sizes: {sizes}")
    (xb0, yb0, meta0) = next(iter(train_dl))
    V0, T0 = xb0
    B0, N0, K0, Fv = V0.shape
    H = yb0.shape[-1]
    print("V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)

    # Summarizer
    summarizer = LaplaceAE(
        num_entities=N0,
        feat_dim=Fv,
        window_size=cfg.WINDOW,
        lap_k=cfg.SUM_LAPLACE_K,
        tv_hidden=cfg.SUM_TV_HIDDEN,
        out_len=cfg.SUM_CONTEXT_LEN,
        context_dim=cfg.SUM_CONTEXT_DIM,
        dropout=cfg.SUM_DROPOUT,
    ).to(device)
    sum_path = Path(cfg.SUM_CKPT)
    if sum_path.exists():
        state = torch.load(sum_path, map_location="cpu")
        summarizer.load_state_dict(state.get("model", state))
        print(f"Loaded summarizer checkpoint: {sum_path}")
    summarizer.eval()
    for p in summarizer.parameters():
        p.requires_grad = False

    # VAE
    vae = LatentVAE(
        seq_len=cfg.PRED,
        latent_dim=cfg.VAE_LATENT_DIM,
        latent_channel=cfg.VAE_LATENT_CHANNELS,
        enc_layers=cfg.VAE_LAYERS,
        enc_heads=cfg.VAE_HEADS,
        enc_ff=cfg.VAE_FF,
        dec_layers=cfg.VAE_LAYERS,
        dec_heads=cfg.VAE_HEADS,
        dec_ff=cfg.VAE_FF,
    ).to(device)
    vae_path = Path(cfg.VAE_CKPT)
    if vae_path.exists():
        ckpt = torch.load(vae_path, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        vae.load_state_dict(sd)
        print(f"Loaded VAE checkpoint: {vae_path}")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    mu_mean, mu_std = compute_latent_stats(vae, train_dl, device)

    # LLapDiT
    diff_model = LLapDiT(
        data_dim=cfg.VAE_LATENT_CHANNELS,
        hidden_dim=cfg.MODEL_WIDTH,
        num_layers=cfg.NUM_LAYERS,
        num_heads=cfg.NUM_HEADS,
        predict_type=cfg.PREDICT_TYPE,
        laplace_k=cfg.LAPLACE_K,
        timesteps=cfg.TIMESTEPS,
        schedule=cfg.SCHEDULE,
        dropout=cfg.DROPOUT,
        attn_dropout=cfg.ATTN_DROPOUT,
        self_conditioning=cfg.SELF_COND,
        lap_mode_main=cfg.LAP_MODE,
    ).to(device)

    ckpt_path = _resolve_llapdit_checkpoint(cfg, cfg.PRED, llapdit_ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    diff_model.load_state_dict(ckpt["model_state"])
    print(f"Loaded LLapDiT checkpoint: {ckpt_path}")

    if getattr(cfg, "USE_EMA_EVAL", False) and "ema_state" in ckpt:
        ema = EMA(diff_model, decay=cfg.EMA_DECAY)
        ema.load_state_dict(ckpt["ema_state"])
        ema.copy_to(diff_model)
    diff_model.eval()

    return val_dl, summarizer, vae, diff_model, mu_mean, mu_std


# ----------------------------- forecasting utilities -----------------------------
def _select_first_valid(mask: torch.Tensor) -> Tuple[int, int]:
    idx = mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        raise RuntimeError("No valid entities found in validation batch.")
    return int(idx[0, 0]), int(idx[0, 1])


def _gather_forecast(
    val_dl,
    summarizer,
    vae,
    diff_model,
    mu_mean,
    mu_std,
    cfg: SimpleNamespace,
    dataset_name: str,
    device: torch.device,
    num_samples: int,
) -> PredictionBundle:
    with torch.inference_mode():
        for xb, yb, meta in val_dl:
            V, T = xb
            mask_bn = meta["entity_mask"]
            if not mask_bn.any():
                continue
            cond_summary = build_context(summarizer, V, T, mask_bn, device)
            y_in, batch_ids = flatten_targets(yb, mask_bn, device)
            if y_in is None:
                continue

            B, N, K, F = V.shape
            valid = mask_bn.nonzero(as_tuple=False)
            ent_ids = valid[:, 1]
            b_ids = valid[:, 0]

            sample_idx = 0
            b_sel = int(b_ids[sample_idx])
            n_sel = int(ent_ids[sample_idx])

            cs = cond_summary[batch_ids[sample_idx : sample_idx + 1]]

            samples = []
            for _ in range(num_samples):
                x0_norm = diff_model.generate(
                    shape=(1, cfg.PRED, cfg.VAE_LATENT_CHANNELS),
                    steps=getattr(cfg, "GEN_STEPS", 42),
                    guidance_strength=float(getattr(cfg, "GUIDANCE_STRENGTH", (1.0, 2.0))[0]),
                    guidance_power=float(getattr(cfg, "GUIDANCE_POWER", 1.0)),
                    cond_summary=cs,
                    self_cond=cfg.SELF_COND,
                    cfg_rescale=True,
                )
                y_hat = decode_latents_with_vae(vae, x0_norm, mu_mean=mu_mean, mu_std=mu_std)
                samples.append(y_hat.squeeze(-1))

            all_samples = torch.stack(samples, dim=0).cpu().numpy()  # [S,1,H]
            mean_forecast = all_samples.mean(axis=0)[0]
            y_true = y_in[sample_idx].squeeze(-1).cpu().numpy()

            return PredictionBundle(
                dataset=dataset_name,
                pred_len=cfg.PRED,
                context=V[b_sel, n_sel, :, 0].cpu().numpy(),
                ground_truth=y_true,
                llapdit_mean=mean_forecast,
                llapdit_samples=all_samples[:, 0, :],
            )
    raise RuntimeError("Validation loader yielded no usable samples.")


# ----------------------------- plotting -----------------------------


def _plot_grid(
    bundles: Sequence[PredictionBundle],
    dataset_names: Sequence[str],
    output_path: Path,
    *,
    samples_to_show: int = 3,
):
    rows = len(dataset_names)
    cols = len({b.pred_len for b in bundles})
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), sharey=False)
    # Normalize axes to a 2D array for consistent indexing
    if isinstance(axes, plt.Axes):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(rows, cols)

    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    palette = color_cycle.by_key()["color"] if color_cycle is not None else None

    pred_order = sorted({b.pred_len for b in bundles})
    for row_idx, dataset in enumerate(dataset_names):
        row_bundles = [b for b in bundles if b.dataset == dataset and b.pred_len in pred_order]
        row_bundles = sorted(row_bundles, key=lambda b: pred_order.index(b.pred_len))
        for col_idx in range(cols):
            if col_idx >= len(row_bundles):
                axes[row_idx, col_idx].axis("off")
                continue
            bundle = row_bundles[col_idx]
            ax = axes[row_idx, col_idx]
            t_future = np.arange(bundle.pred_len)

            # Ground truth
            ax.plot(t_future, bundle.ground_truth, color="black", linewidth=2.0, label="Ground truth")

            # LLapDiT samples for uncertainty
            for s in bundle.llapdit_samples[:samples_to_show]:
                ax.plot(t_future, s, color="#6ca0dc", alpha=0.35, linewidth=1.0, linestyle="-")

            # LLapDiT mean
            ax.plot(
                t_future,
                bundle.llapdit_mean,
                color="#1f77b4" if not palette else palette[0],
                linewidth=2.2,
                label="LLapDiT (mean)",
            )

            ax.set_title(f"{dataset} | H={bundle.pred_len}")
            ax.set_xlabel("Forecast step")
            ax.set_ylabel("Normalized value")
            ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
            if col_idx == cols - 1:
                ax.legend(loc="upper left", fontsize="small")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved qualitative forecast grid to {output_path}")


# ----------------------------- main -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualitative forecast plotting for LLapDiT")
    p.add_argument("--datasets", nargs="+", default=["crypto_config"], help="Config modules for each dataset row")
    p.add_argument(
        "--prediction-lengths", nargs="+", type=int, default=[20, 40, 60], help="Prediction horizons to plot"
    )
    p.add_argument("--llapdit-checkpoint", type=str, default=None, help="Optional format string for LLapDiT ckpt (use {pred})")
    p.add_argument("--num-samples", type=int, default=6, help="Number of LLapDiT samples for uncertainty")
    p.add_argument("--output", type=Path, default=Path("qualitative_forecasts.pdf"), help="Where to save the plot")
    return p.parse_args()


def main():
    args = parse_args()
    device = set_torch()

    dataset_specs = [DatasetSpec(name=mod.split(".")[-1], config_module=mod) for mod in args.datasets]
    bundles: List[PredictionBundle] = []

    for spec in dataset_specs:
        module = importlib.import_module(spec.config_module)
        base_cfg = _as_namespace(module)
        for pred in args.prediction_lengths:
            cfg = _clone_for_pred(base_cfg, pred)
            val_dl, summarizer, vae, diff_model, mu_mean, mu_std = _load_components(
                cfg, device, llapdit_ckpt=args.llapdit_checkpoint
            )
            bundle = _gather_forecast(
                val_dl,
                summarizer,
                vae,
                diff_model,
                mu_mean,
                mu_std,
                cfg,
                spec.name,
                device,
                num_samples=args.num_samples,
            )
            bundles.append(bundle)

    _plot_grid(bundles, [spec.name for spec in dataset_specs], args.output)


if __name__ == "__main__":
    main()

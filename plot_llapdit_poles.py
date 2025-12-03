"""Plot Laplace poles for a trained LLapDiT checkpoint."""

import argparse
from pathlib import Path
from typing import Optional

import torch

import crypto_config as config
from Model.llapdit import LLapDiT
from Model.llapdit_utils import EMA, plot_laplace_poles, set_torch


def _resolve_checkpoint(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    trained = getattr(config, "TRAINED_LLapDiT", None) or ""
    if trained:
        return Path(trained).expanduser().resolve()

    ckpt_dir = Path(getattr(config, "CKPT_DIR", "./ldt/checkpoints"))
    candidates = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No LLapDiT checkpoint found. Provide --checkpoint or set TRAINED_LLapDiT in crypto_config.py."
    )


def _load_model(device: torch.device) -> LLapDiT:
    return LLapDiT(
        data_dim=config.VAE_LATENT_CHANNELS,
        hidden_dim=config.MODEL_WIDTH,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        predict_type=config.PREDICT_TYPE,
        laplace_k=config.LAPLACE_K,
        timesteps=config.TIMESTEPS,
        schedule=config.SCHEDULE,
        dropout=config.DROPOUT,
        attn_dropout=config.ATTN_DROPOUT,
        self_conditioning=config.SELF_COND,
        lap_mode_main=config.LAP_MODE,
    ).to(device)


def _apply_checkpoint(model: LLapDiT, checkpoint: Path, *, device: torch.device, use_ema: bool) -> None:
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state_dict)

    if use_ema and "ema_state" in ckpt:
        ema = EMA(model, decay=ckpt.get("ema_decay", getattr(config, "EMA_DECAY", 0.999)))
        ema.load_state_dict(ckpt["ema_state"])
        ema.copy_to(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Laplace poles for a trained LLapDiT model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained LLapDiT checkpoint")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=getattr(config, "POLE_PLOT_DIR", Path(config.OUT_DIR) / "pole_plots"),
        help="Directory to save the generated pole plot",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="checkpoint-",
        help="Prefix applied to the legend labels",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=False,
        help="Apply EMA weights when available in the checkpoint",
    )
    args = parser.parse_args()

    device = set_torch()
    checkpoint = _resolve_checkpoint(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = _load_model(device)
    _apply_checkpoint(model, checkpoint, device=device, use_ema=args.use_ema)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"llapdit_poles_pred{config.PRED}_{checkpoint.stem}.pdf"

    plotted = plot_laplace_poles(
        [model.model],
        save_path,
        tag_prefix=args.tag_prefix,
        prediction_length=getattr(config, "PRED", None),
    )
    if plotted is None:
        print("No Laplace encoders were found; no plot was written.")
    else:
        print(f"Saved pole plot to: {plotted}")


if __name__ == "__main__":
    main()

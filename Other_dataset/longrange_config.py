"""Configuration defaults for long-range forecasting benchmarks.

Edit ``DATASET_NAME`` or override individual settings before running
``train_val_llapdit_longrange.py``.  The configuration mirrors the crypto
setup but with values that are more appropriate for public datasets like
ETT, Solar Energy, Electricity, and PEMS-BAY Traffic.
"""

from __future__ import annotations

import os

from Data_Prep.long_range_datasets import DATASET_REGISTRY


DATA_ROOT = "./ldt/long_range_data"
DATASET_NAME = "ettm1"


def _select_dataset_cfg(name: str):
    spec = DATASET_REGISTRY.get(name.lower())
    if spec is None:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {', '.join(sorted(DATASET_REGISTRY))}"
        )
    return spec


_SPEC = _select_dataset_cfg(DATASET_NAME)

DATA_PATH = _SPEC.file
WINDOW = _SPEC.window
PRED = _SPEC.horizon
RATIOS = _SPEC.ratios
STRIDE = _SPEC.stride
BATCH_SIZE = _SPEC.batch_size
SHUFFLE_TRAIN = _SPEC.shuffle_train
NUM_WORKERS = _SPEC.num_workers
DROP_LAST = _SPEC.drop_last
SCALING = _SPEC.scaling


# ======================= VAE Architecture =======================
VAE_LATENT_CHANNELS = _SPEC.vae_channels
VAE_LATENT_DIM = 128
VAE_LAYERS = 3
VAE_HEADS = 4
VAE_FF = 256
VAE_DROPOUT = 0.05

MKT = f"{DATASET_NAME}_{WINDOW}x{PRED}"
VAE_DIR = os.path.join("./ldt/saved_model", MKT)
VAE_CKPT = os.path.join(VAE_DIR, f"pred-{PRED}_ch-{VAE_LATENT_CHANNELS}_elbo.pt")

VAE_LEARNING_RATE = 2e-4
VAE_WEIGHT_DECAY = 1e-4
VAE_WARMUP_EPOCHS = 5
VAE_BETA = 5e-3
VAE_MAX_PATIENCE = 10
DECODER_FT_EPOCHS = 10
DECODER_FT_LR = 2e-4


# ============================ Diffusion Model (LLapDiT) ============================
CKPT_DIR = os.path.join("./ldt/checkpoints", MKT)

TIMESTEPS = 1000
SCHEDULE = "cosine"
PREDICT_TYPE = "v"

LOSS_WEIGHT_SCHEME = "weighted_min_snr"
MINSNR_GAMMA = 5.0

MODEL_WIDTH = _SPEC.model_width
NUM_LAYERS = 5
NUM_HEADS = 4
LAPLACE_K = 64
GLOBAL_K = 256
LAP_MODE_main = "recurrent"
LAP_MODE_cond = "recurrent"
zero_first_step = False
add_guidance_tokens = True
CONTEXT_LEN = GLOBAL_K // 2 if add_guidance_tokens else GLOBAL_K


# ============================ Training Hyperparameters ============================
EPOCHS = 400
BASE_LR = 5e-4
MIN_LR = 5e-6
WARMUP_FRAC = 0.1
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EARLY_STOP = 50
DROPOUT = 0.1
ATTN_DROPOUT = 0.1
DROP_COND_P = 0.2

SELF_COND = False
SELF_COND_P = 0.5
SELF_COND_START_EPOCH = 150

TRAINED_LLapDiT = ""
downstream = False

FT_WARMUP_FRAC = 0.05
FT_BASE_LR = 1e-4
FT_MIN_LR = 5e-6


# ============================ Evaluation & Sampling ============================
USE_EMA_EVAL = True
EMA_DECAY = 0.999

GEN_STEPS = 36
NUM_EVAL_SAMPLES = 10
GUIDANCE_STRENGTH = (1.5, 3.0)
GUIDANCE_POWER = 1.0
DECODER_FT_ANCHOR = 0.1

OUT_DIR = "./ldt/output"

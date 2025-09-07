import json

# ============================ Data & Preprocessing ============================
DATA_DIR = "./ldt/LLapDiT_Data/Crypto_100/crypto_data"
with open(f"{DATA_DIR}/cache_ratio_index/meta.json", "r") as f:
    assets = json.load(f)["assets"]

# --- Data Parameters ---
WINDOW = 60           # Input sequence length (K)
PRED = 20             # Target sequence length to predict (H)
COVERAGE = 0.8
date_batching=True

# --- Dataloader Parameters ---
BATCH_SIZE = 20
train_ratio=0.7
val_ratio=0.1
test_ratio=0.2

# ============================ VAE (Encoder/Decoder) ============================
VAE_DIR = './ldt/saved_model'
VAE_CKPT = "./ldt/saved_model/16_0.00124_2.16059_best_recon.pt"

# --- VAE Architecture ---
VAE_LATENT_DIM = 16
VAE_LAYERS = 3
VAE_HEADS = 4
VAE_FF = 256

# --- VAE Fine-Tuning (Optional, after diffusion training) ---
# Set DECODER_FT_EPOCHS = 0 to disable this step.
DECODER_FT_EPOCHS = 6
DECODER_FT_LR = 1e-4

# ============================ Diffusion Model (LLapDiT) ============================
CKPT_DIR = "./ldt/checkpoints"

# --- Diffusion Process ---
TIMESTEPS     = 1200
# Recommended to try "cosine", as it pairs well with v-prediction.
SCHEDULE      = "linear"     # ["cosine", "linear"]
PREDICT_TYPE  = "v"          # ["v", "eps"]

# --- Loss Function ---
# 'weighted_min_snr' is highly recommended. Set to 'none' to disable.
LOSS_WEIGHT_SCHEME = 'weighted_min_snr'
# The gamma parameter for min-SNR weighting. A value of 5.0 is a common starting point.
MINSNR_GAMMA = 5.0

# --- LLapDiT Architecture ---
MODEL_WIDTH   = 256
NUM_LAYERS    = 5
NUM_HEADS     = 4
CONTEXT_LEN   = 2 * PRED      # Learned summary tokens
LAPLACE_K     = 64
GLOBAL_K      = 128
LAP_MODE      = 'parallel'    # or 'recurrent'

# ============================ Training Hyperparameters ============================
EPOCHS = 1500
BASE_LR = 8e-4
WARMUP_FRAC = 0.055
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 1.0
EARLY_STOP = 100

# --- Regularization & Conditioning ---
DROPOUT       = 0.0
ATTN_DROPOUT  = 0.0
# Probability of dropping conditioning for Classifier-Free Guidance.
DROP_COND_P   = 0.1

# --- Self-Conditioning ---
SELF_COND     = True
SELF_COND_P   = 0.5
# Recommended to lower this to see benefits earlier in training.
SELF_COND_START_EPOCH = 300

# --- Latent Normalization ---
# two-stage latent whitening: per-window EWMA, then global z-score
USE_EWMA      = True
EWMA_LAMBDA   = 0.99

# ============================ Evaluation & Sampling ============================
# Use Exponential Moving Average of model weights for evaluation.
USE_EMA_EVAL = True
EMA_DECAY    = 0.999
DECODE_USE_GT_SCALE = True
# --- Generation Parameters ---
GEN_STEPS = 36
NUM_EVAL_SAMPLES = 25
GUIDANCE_STRENGTH = 2.0
GUIDANCE_POWER = 0.3
DECODER_FT_ANCHOR = 0.25

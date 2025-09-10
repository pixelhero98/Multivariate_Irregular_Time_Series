
# ============================ Data & Preprocessing ============================
DATA_DIR = "./ldt/LLapDiT_Data/CRYPTO_130_data"


# --- Data Parameters ---
WINDOW = 200           # Input sequence length (K)
PRED = 20             # Target sequence length to predict (H)
COVERAGE = 0.8
date_batching=True

# --- Dataloader Parameters ---
BATCH_SIZE = 5
train_ratio=0.7
val_ratio=0.1
test_ratio=0.2

# ============================ VAE (Encoder/Decoder) ============================
VAE_DIR = './ldt/saved_model'
VAE_CKPT = "./ldt/saved_model/PRED|HDIM_20|24_recon.pt"

# --- VAE Architecture ---
VAE_PATCH_N = 1
VAE_PATCH_H = 5
VAE_LATENT_CHANNELS = 24
VAE_LATENT_DIM = 64
VAE_LAYERS = 3
VAE_HEADS = 4
VAE_FF = 256
VAE_DROPOUT = 0.0
# --- VAE Fine-Tuning (Optional, after diffusion training) ---
# Set DECODER_FT_EPOCHS = 0 to disable this step.
VAE_LEARNING_RATE = 2e-4
VAE_WEIGHT_DECAY = 5e-4
VAE_WARMUP_EPOCHS = 8
VAE_BETA = 0.8
VAE_MAX_PATIENCE = 10
DECODER_FT_EPOCHS = 20
DECODER_FT_LR = 2e-4

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
MINSNR_GAMMA = 3.0

# --- LLapDiT Architecture ---
MODEL_WIDTH   = 256
NUM_LAYERS    = 5
NUM_HEADS     = 4
CONTEXT_LEN   = 2 * PRED  if PRED <= 100 else 200    # Learned summary tokens
LAPLACE_K     = 64
GLOBAL_K      = 128
LAP_MODE      = 'parallel'    # or 'recurrent'

# ============================ Training Hyperparameters ============================
EPOCHS = 1500
BASE_LR = 6e-4
MIN_LR = 1e-5
WARMUP_FRAC = 0.06
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 1.0
EARLY_STOP = 80

# --- Regularization & Conditioning ---
DROPOUT       = 0.0
ATTN_DROPOUT  = 0.0
# Probability of dropping conditioning for Classifier-Free Guidance.
DROP_COND_P   = 0.1

# --- Self-Conditioning ---
SELF_COND     = False
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
DECODE_USE_GT_SCALE = False
# --- Generation Parameters ---
GEN_STEPS = 100
NUM_EVAL_SAMPLES = 10
GUIDANCE_STRENGTH = 2.0
GUIDANCE_POWER = 0.3
DECODER_FT_ANCHOR = 0.01 # 0.1

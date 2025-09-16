
# ============================ Data & Preprocessing ============================
DATA_DIR = "/home/pyh/Documents/school/projects/yzn_ts/ldt/LLapDiT_Data/CRYPTO_130_data_ca"
WINDOW = 200           # Input sequence length (K)
PRED = 5             # Target sequence length to predict (H)
COVERAGE = 0.8
date_batching=True

# --- Dataloader Parameters ---
BATCH_SIZE = 5
train_ratio=0.7
val_ratio=0.1
test_ratio=0.2

# ============================ VAE (Encoder/Decoder) ============================
VAE_DIR = './ldt/saved_model'
VAE_CKPT = "./ldt/saved_model/PRED|CHANNEL_5|36_elbo.pt"

# ======================= VAE Architecture =======================
VAE_LATENT_CHANNELS = 36 # compress N (e.g., 130, 150, 200) to C (e.g., 16, 20, 24, 36)
VAE_LATENT_DIM = 64       # d_model of Transformer-backbone vae
VAE_LAYERS = 3
VAE_HEADS = 4
VAE_FF = 256              # feed-forward dim of the Transformers
VAE_DROPOUT = 0.0
# --- VAE Fine-Tuning (Optional, after diffusion training) ---
# Set DECODER_FT_EPOCHS = 0 to disable this step.
VAE_LEARNING_RATE = 2e-4
VAE_WEIGHT_DECAY = 5e-4
VAE_WARMUP_EPOCHS = 5
VAE_BETA = 0.005
VAE_MAX_PATIENCE = 6
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
MINSNR_GAMMA = 5.0

# --- LLapDiT Architecture ---
MODEL_WIDTH   = 256
NUM_LAYERS    = 5
NUM_HEADS     = 4
LAPLACE_K     = 64
GLOBAL_K      = 256
CONTEXT_LEN   = 2 * PRED if PRED <= 100 else 200 # Learned summary tokens
LAP_MODE      = 'parallel'    # or 'recurrent'

# ============================ Training Hyperparameters ============================
EPOCHS = 1500
BASE_LR = 6e-4
MIN_LR = 0.9e-4
WARMUP_FRAC = 0.06
WEIGHT_DECAY = 9e-4
GRAD_CLIP = 1.0
EARLY_STOP = 100

# --- Regularization & Conditioning ---
DROPOUT       = 0.1
ATTN_DROPOUT  = 0.1
# Probability of dropping conditioning for Classifier-Free Guidance.
DROP_COND_P   = 0.11

# --- Self-Conditioning ---
SELF_COND     = False
SELF_COND_P   = 0.5
# Recommended to lower this to see benefits earlier in training.
SELF_COND_START_EPOCH = 210
TRAINED_LLapDiT = ""
downstream = False
# --- Latent Normalization ---
# Currently all disabled, no need to consider
USE_EWMA      = False
EWMA_LAMBDA   = 0.99
DECODE_USE_GT_SCALE = False
# ============================ Evaluation & Sampling ============================
# Use Exponential Moving Average of model weights for evaluation.
USE_EMA_EVAL = True
EMA_DECAY    = 0.999

# --- Generation Parameters ---
GEN_STEPS = 36
NUM_EVAL_SAMPLES = 10
GUIDANCE_STRENGTH = (1.5, 3.0)
GUIDANCE_POWER = 1.0
DECODER_FT_ANCHOR = 0.05 # 0.1

OUT_DIR = "./ldt/output"

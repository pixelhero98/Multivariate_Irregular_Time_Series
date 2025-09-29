
# ============================ Data & Preprocessing ============================
DATA_DIR = "/home/pyh/Documents/school/projects/yzn_ts/ldt/LLapDiT_Data/CRYPTO_130_data"
MKT = DATA_DIR[60:-5]

# --- Data Parameters ---
PRED = 20             # Target sequence length to predict (H)
WINDOW = 200           # Conditional input sequence length (K)
COVERAGE = 0.8
date_batching=True

# --- Dataloader Parameters ---
BATCH_SIZE = 20
train_ratio=0.7
val_ratio=0.1
test_ratio=0.2

# ======================= VAE Architecture =======================
VAE_LATENT_CHANNELS = 24 # compress N (e.g., 130, 150, 200) to C (e.g., 16, 20, 24, 36)
VAE_LATENT_DIM = 128       # d_model of Transformer-backbone vae
VAE_LAYERS = 3
VAE_HEADS = 4
VAE_FF = 256              # feed-forward dim of the Transformers
VAE_DROPOUT = 0.0
VAE_DIR = './ldt/saved_model/' + MKT
VAE_CKPT = "./ldt/saved_model/" + MKT + f"/pred-{PRED}_ch-{VAE_LATENT_CHANNELS}_elbo.pt"
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
CKPT_DIR = "./ldt/checkpoints/" + MKT

# --- Diffusion Process ---
TIMESTEPS     = 1200
# Recommended to try "cosine", as it pairs well with v-prediction.
SCHEDULE      = "cosine"     # ["cosine", "linear"]
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
LAP_MODE      = 'recurrent'   # 'parallel' or 'recurrent (support irregular sampling interval, with time-varying Lap basis updates)'
zero_first_step = False
add_guidance_tokens = True
CONTEXT_LEN   = GLOBAL_K // 2 if add_guidance_tokens else GLOBAL_K # Learned summary tokens length
# ============================ Training Hyperparameters ============================
EPOCHS = 900
BASE_LR = 8e-4
MIN_LR = 8e-6
WARMUP_FRAC = 0.095
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 1.0
EARLY_STOP = 100
DROPOUT       = 0.07
ATTN_DROPOUT  = 0.07
DROP_COND_P   = 0.16

# --- Self-Conditioning ---
SELF_COND     = False
SELF_COND_P   = 0.5
# Recommended to lower this to see benefits earlier in training.
SELF_COND_START_EPOCH = 210
TRAINED_LLapDiT = "/home/pyh/Documents/school/projects/yzn_ts/ldt/checkpoints/CRYPTO_130/mode-recurrent-pred-20_ch-24_val_0.190331_cond_0.268251.pt"
downstream = False
# --- Latent Normalization ---
FT_WARMUP_FRAC=0.035
FT_BASE_LR=1e-4
FT_MIN_LR=5e-6
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


# 160109, 22787, 45911
# 65097, 9241, 18786
OUT_DIR = "./ldt/output"

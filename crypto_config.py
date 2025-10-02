# ============================ Data & Preprocessing ============================
DATA_DIR = "/home/pyh/Documents/school/projects/yzn_ts/ldt/LLapDiT_Data/CRYPTO_130_data"
MKT = DATA_DIR[60:-5]
SEED = 42
# --- Data Parameters ---
PRED = 20            # Target sequence length to predict (H)
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
VAE_DIR = './ldt/vae/saved_model/' + MKT
VAE_CKPT = VAE_DIR + f"/pred-{PRED}_ch-{VAE_LATENT_CHANNELS}_elbo.pt"
# --- VAE Fine-Tuning (Optional, after diffusion training) ---
# Set DECODER_FT_EPOCHS = 0 to disable this step.
VAE_LEARNING_RATE = 2e-4
VAE_WEIGHT_DECAY = 5e-4
VAE_WARMUP_EPOCHS = 5
VAE_BETA = 0.005
VAE_MAX_PATIENCE = 6
DECODER_FT_EPOCHS = 30
DECODER_FT_LR = 2e-4
# ======================= Summarizer (LaplaceAE) =======================
SUM_DIR = "./ldt/SUMMARIZER_EFF/saved_model/" + MKT
SUM_LAPLACE_K = 512
SUM_CONTEXT_LEN = PRED
SUM_CONTEXT_DIM = 256
SUM_TV_HIDDEN = 16
SUM_DROPOUT = 0.0
SUM_LR = 2e-4
SUM_WEIGHT_DECAY = 1e-4
SUM_EPOCHS = 200
SUM_GRAD_CLIP = 1.0
SUM_AMP = True
SUM_PATIENCE = 25
SUM_MIN_DELTA = 1e-6
SUM_CKPT = SUM_DIR + f"/{PRED}-{VAE_LATENT_CHANNELS}-summarizer.pt"
# ============================ Diffusion Model (LLapDiT) ============================
CKPT_DIR = "./ldt/checkpoints/" + MKT

# --- Diffusion Process ---
TIMESTEPS     = 1000
# Recommended to try "cosine", as it pairs well with v-prediction.
SCHEDULE      = "cosine"     # ["cosine", "linear"]
PREDICT_TYPE  = "v"          # ["v", "eps"]

# --- Loss Function ---
# 'weighted_min_snr' is highly recommended. Set to 'none' to disable.
LOSS_WEIGHT_SCHEME = 'weighted_min_snr'
# The gamma parameter for min-SNR weighting. A value of 5.0 is a common starting point.
MINSNR_GAMMA = 5.0

# --- LLapDiT Architecture ---
MODEL_WIDTH   = SUM_CONTEXT_DIM
NUM_LAYERS    = 5
NUM_HEADS     = 4
LAPLACE_K     = 256
LAP_MODE     = 'parallel'   # 'parallel' or 'recurrent (support irregular sampling interval, with time-varying Lap basis updates)'
CONTEXT_LEN   = SUM_CONTEXT_LEN
# ============================ Training Hyperparameters ============================
EPOCHS = 650
BASE_LR = 6e-4
MIN_LR = 5e-6
WARMUP_FRAC = 0.09
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EARLY_STOP = 100
DROPOUT       = 0.0
ATTN_DROPOUT  = 0.0
DROP_COND_P   = 0.19
# --- Self-Conditioning ---
SELF_COND     = False
SELF_COND_P   = 0.5
# Recommended to lower this to see benefits earlier in training.
SELF_COND_START_EPOCH = 450
TRAINED_LLapDiT = "" #"./ldt/checkpoints/CRYPTO_130/"
downstream = False
# ============================ Evaluation & Sampling ============================
# Use Exponential Moving Average of model weights for evaluation.
USE_EMA_EVAL = True
EMA_DECAY    = 0.999

# --- Generation Parameters ---
GEN_STEPS = 64
NUM_EVAL_SAMPLES = 10
GUIDANCE_STRENGTH = (1.0, 2.0)
GUIDANCE_POWER = 1.0
DECODER_FT_ANCHOR = 0.1 # 0.1

OUT_DIR = "./ldt/output"

# ============================ Data & Preprocessing ============================
DATA_DIR = "./ldt/LLapDiT_Data/bms_air_quality_data/bms_air_cache"
MKT = "bms_air_quality"
SEED = 42
# --- Data Parameters ---
PRED = 20            # Target sequence length to predict (H) fin: 5/20/60/100
WINDOW = 336           # Conditional input sequence length (K) fin: 200
COVERAGE = 0.8
date_batching=True

# --- Dataloader Parameters ---
BATCH_SIZE = 10
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
VAE_LEARNING_RATE = 1e-4
VAE_WEIGHT_DECAY = 1e-4
VAE_WARMUP_EPOCHS = 5
VAE_BETA = 0.00035
VAE_MAX_PATIENCE = 6
DECODER_FT_EPOCHS = 40
DECODER_FT_LR = 2e-4
DECODER_FT_ANCHOR = 0.08 # 0.1
# ======================= Summarizer (LaplaceAE) =======================
SUM_DIR = "./ldt/summarizer/saved_model/" + MKT
SUM_LAPLACE_K = 512
SUM_CONTEXT_LEN = PRED
SUM_CONTEXT_DIM = 256
SUM_TV_HIDDEN = 16
SUM_DROPOUT = 0.0
SUM_LR = 5e-4
SUM_WEIGHT_DECAY = 1e-4
SUM_EPOCHS = 200
SUM_GRAD_CLIP = 1.0
SUM_AMP = True
SUM_PATIENCE = 10
SUM_MIN_DELTA = 1e-6
SUM_GAMMA = 1.0
SUM_CKPT = SUM_DIR + f"/{PRED}-{VAE_LATENT_CHANNELS}-summarizer.pt"
# ============================ Diffusion Model (LLapDiT) ============================
CKPT_DIR = "./ldt/checkpoints/" + MKT

# --- Diffusion Process ---
TIMESTEPS     = 200 # 100 - 1000
# Recommended to try "cosine", as it pairs well with v-prediction.
SCHEDULE      = "linear"     # ["cosine", "linear"]
PREDICT_TYPE  = "x0"          # ["x0", "v", "eps"]

# --- Loss Function ---
# 'weighted_min_snr' is highly recommended. Set to 'none' to disable.
LOSS_WEIGHT_SCHEME = 'weighted_min_snr'
# The gamma parameter for min-SNR weighting. A value of 5.0 is a common starting point.
MINSNR_GAMMA = 4.5

# --- LLapDiT Architecture ---
MODEL_WIDTH   = SUM_CONTEXT_DIM
NUM_LAYERS    = 5
NUM_HEADS     = 4
LAPLACE_K     = 256
LAP_MODE     = 'effective'   # 'parallel' or 'recurrent (support irregular sampling interval, with time-varying Lap basis updates)'
CONTEXT_LEN   = SUM_CONTEXT_LEN
# PATCH_SIZE = 2
# ============================ Training Hyperparameters ============================
EPOCHS = 650
BASE_LR = 5e-4
MIN_LR = 5e-6
WARMUP_FRAC = 0.095
WEIGHT_DECAY = 5e-4
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
TRAINED_LLapDiT = "./ldt/checkpoints/CRYPTO_130/mode-effective-pred-100-val-0.036304-cond-0.000534-ratio-0.041827.pt" #"./ldt/checkpoints/CRYPTO_130/"
downstream = False
# ============================ Evaluation & Sampling ============================
# Use Exponential Moving Average of model weights for evaluation.
USE_EMA_EVAL = True
EMA_DECAY    = 0.999

# --- Generation Parameters ---
GEN_STEPS = 64
NUM_EVAL_SAMPLES = 20
GUIDANCE_STRENGTH = (1.0, 2.0)
GUIDANCE_POWER = 1.0
DECODER_FT_ANCHOR = 0.1 # 0.1

OUT_DIR = "./ldt/output"
SAVE_POLE_PLOTS = True
POLE_PLOT_DIR = OUT_DIR + "/pole_plots"

OUT_DIR = "./ldt/output"
POLE_PLOT_ONLY = False  # Set True to load TRAINED_LLapDiT and export pole plots without retraining

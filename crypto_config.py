import json

# --- Data & Model Parameters ---
DATA_MODULE   = "Dataset.fin_data_prep_ratiosp_indexcache"
DATA_DIR = "./ldt/LLapDiT_Data/Crypto_100/crypto_data"
VAE_DIR = './ldt/saved_model'
FEATURES_DIR = f"{DATA_DIR}/features"  # your per-ticker parquet/pickle files live here
with open(f"{DATA_DIR}/cache_ratio_index/meta.json", "r") as f:
    assets = json.load(f)["assets"]
NUM_ENTITIES = len(assets)
WINDOW = 128               # panel context length K
PRED = 20                  # target sequence length H
COVERAGE = 0.85
norm_scope = "train_only"
date_batching=True
shuffle_train=False
train_ratio=0.7
val_ratio=0.1
test_ratio=0.2

# --- VAE Architecture ---
VAE_LATENT_DIM = 64
VAE_LAYERS = 3
VAE_HEADS = 4
VAE_FF = 256

# --- Training Hyperparameters ---
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_PATIENCE = 20

# --- VAE Loss Parameters ---
FREE_BITS_PER_ELEM = 0.5
SOFTNESS = 0.15

# --- LLapDiT Parameters ---
HORIZON       = PRED           # target length H
BASE_LR       = 1e-4
WARMUP_FRAC   = 0.05
WEIGHT_DECAY  = 5e-4
GRAD_CLIP     = 1.0
EARLY_STOP    = 50
TIMESTEPS     = 1500
SCHEDULE      = "cosine"     # ["cosine","linear","sigmoid"]
PREDICT_TYPE  = "v"          # ["v","eps"]
DROP_COND_P   = 0.25         # classifier-free guidance (drop conditioning prob)
SELF_COND     = True
SELF_COND_P   = 0.50
SNR_CLIP      = 5.0

MODEL_WIDTH   = 256
NUM_LAYERS    = 4
NUM_HEADS     = 4
LAPLACE_K     = [24, 20, 20, 16]
GLOBAL_K      = 64
DROPOUT       = 0.0
ATTN_DROPOUT  = 0.0
CONTEXT_LEN   = HORIZON      # learned summary tokens

USE_EWMA      = True         # two-stage latent whitening: per-window EWMA, then global
EWMA_LAMBDA   = 0.94

CKPT_DIR      = "./ldt/checkpoints"
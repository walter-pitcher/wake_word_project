"""
MagisAI Wake Word Training Pipeline v6 — Centralized Configuration
===================================================================
All paths, hyperparameters, feature-extraction constants, and
augmentation settings live here.  Every other script imports from
this module so there is a single source of truth.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # wake_word_project/
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
NEW_DIR      = PROJECT_ROOT / "new"
NEW_MODEL_DIR = NEW_DIR / "model"

# Original positives
POS_TRIMMED_DIR = DATA_DIR / "positive_trimmed_1p5_A"
POS_FRIENDS_DIR = DATA_DIR / "positive_new_friends"

# Augmented positives (produced by augment_positives.py)
POS_AUGMENTED_DIR = DATA_DIR / "positive_augmented"

# Negative manifests
NEG_MANIFEST             = DATA_DIR / "negative_manifest.txt"
HARD_NEG_MANIFEST        = DATA_DIR / "negative_manifest_hard_merged.txt"
HARD_NEG_TOP_DIR         = DATA_DIR / "hard_negatives_top"
HARD_NEG_MINED_MANIFEST  = DATA_DIR / "negative_manifest_hard_mined_v6.txt"

# Phonetic confusers (produced by generate_phonetic_negatives.py)
PHONETIC_NEG_DIR = DATA_DIR / "negative" / "phonetic_confusers"

# Background noise
BACKGROUND_NOISE_DIR  = DATA_DIR / "negative" / "_background_noise_"
EXPANDED_NOISE_DIR    = DATA_DIR / "negative" / "_background_noise_expanded_"

# ──────────────────────────────────────────────────────────────────
# AUDIO / FEATURE EXTRACTION  (must match Dart inference exactly)
# ──────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16_000
CLIP_SECONDS   = 1.5
NUM_SAMPLES    = int(SAMPLE_RATE * CLIP_SECONDS)   # 24 000

FRAME_LENGTH   = 400     # 25 ms
FRAME_STEP     = 160     # 10 ms
FFT_LENGTH     = 512
NUM_MELS       = 40
LOWER_HZ       = 80.0
UPPER_HZ       = 7600.0
TARGET_FRAMES  = 148

INPUT_SHAPE    = (TARGET_FRAMES, NUM_MELS, 1)      # (148, 40, 1)

# ──────────────────────────────────────────────────────────────────
# TRAINING HYPER-PARAMETERS
# ──────────────────────────────────────────────────────────────────
BATCH_SIZE        = 32
EPOCHS            = 60
LEARNING_RATE     = 5e-4
MIN_LR            = 1e-6
PATIENCE          = 10        # early-stopping patience (epochs)
LR_PATIENCE       = 3         # reduce-LR patience
LR_FACTOR         = 0.5

NEG_MULTIPLIER    = 3         # negatives per positive each epoch
HARD_NEG_FRACTION = 0.60      # share of hard negatives in each batch

# Focal loss
FOCAL_GAMMA       = 2.0
FOCAL_ALPHA       = 0.75      # weight for the positive (minority) class

# Label smoothing
LABEL_SMOOTHING   = 0.05

# Model architecture
USE_SE_ATTENTION  = True       # squeeze-and-excitation on last DS block
DROPOUT_RATE      = 0.30

# ──────────────────────────────────────────────────────────────────
# ONLINE AUGMENTATION  (applied in the tf.data pipeline)
# ──────────────────────────────────────────────────────────────────
MAX_GAIN_DB       = 8.0
NOISE_MIX_PROB    = 0.40
NOISE_MIX_SNR_DB  = (0.0, 18.0)
TIME_SHIFT_MAX    = 800        # ±50 ms at 16 kHz

# SpecAugment
SPEC_AUGMENT_PROB   = 0.50
TIME_MASK_PARAM     = 20       # max frames to mask
FREQ_MASK_PARAM     = 5        # max mel bins to mask
NUM_TIME_MASKS      = 2
NUM_FREQ_MASKS      = 2

# ──────────────────────────────────────────────────────────────────
# OFFLINE AUGMENTATION  (augment_positives.py)
# ──────────────────────────────────────────────────────────────────
PITCH_SHIFTS   = [-2, -1, 1, 2]           # semitones
SPEED_FACTORS  = [0.9, 1.1]               # rate multiplier
NOISE_SNRS_AUG = [5, 10, 15]              # SNR in dB
TARGET_AUGMENTED_COUNT = 5000

# ──────────────────────────────────────────────────────────────────
# HARD-NEGATIVE MINING
# ──────────────────────────────────────────────────────────────────
HARD_NEG_SCORE_THRESHOLD = 0.3
HARD_NEG_OVERSAMPLE      = 4

# ──────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────
THRESHOLDS_TO_TEST = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.92, 0.95, 0.97, 0.99,
]

EVAL_SNR_LEVELS = [float("inf"), 10.0, 5.0, 0.0]   # clean, 10 dB, 5 dB, 0 dB
POSTERIOR_SMOOTHING_COUNTS = [1, 2, 3, 4, 5]

# ──────────────────────────────────────────────────────────────────
# RUN NAMING
# ──────────────────────────────────────────────────────────────────
RUN_NAME = "hey_magis_v6"

"""NeuroVoice configuration."""

# -- EEG (Muse S via BrainFlow) --
BOARD_ID = 39  # BoardIds.MUSE_S_BOARD
EEG_SAMPLE_RATE = 256
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
EEG_BUFFER_SECONDS = 10

FREQ_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

# -- Signal Processing --
NOTCH_FREQ = 60.0
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 50.0
FILTER_ORDER = 4

# -- Brain State --
BRAIN_STATE_HZ = 10
EMA_ALPHA = 0.05

# -- NeuroRL --
STYLE_PARAMS = [
    "formality", "enthusiasm", "verbosity", "empathy",
    "humor", "assertiveness", "emotional_expressiveness", "warmth",
]
STYLE_DIM = len(STYLE_PARAMS)
BRAIN_STATE_DIM = 12
RL_LEARNING_RATE = 0.15
RL_EXPLORATION = 0.25
REWARD_WEIGHTS = {
    "engagement_delta": 0.4,
    "valence_shift": 0.3,
    "no_error": 0.2,
    "focus": 0.1,
}
CLENCH_BONUS = 0.5
ERRP_PENALTY = -0.5

# -- Communication --
CLAUDE_MODEL = "claude-sonnet-4-6-20250514"
NUM_CANDIDATES = 4
INTENT_CLASSES = [
    "agree", "disagree", "elaborate", "acknowledge",
    "question", "express_emotion", "correct", "greeting",
]

# -- Voice --
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "coral"

# -- Server --
WS_HOST = "127.0.0.1"
WS_PORT = 8765

# -- iMessage --
IMESSAGE_DB = "~/Library/Messages/chat.db"

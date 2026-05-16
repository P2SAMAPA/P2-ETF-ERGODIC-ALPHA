"""
Configuration for P2-ETF-ERGODIC-ALPHA engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-ergodic-alpha-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM", "IWD"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}
ACTIVE_UNIVERSE = "COMBINED"

# --- Ergodic Test Parameters ---
LOOKBACK_WINDOW = 252          # days for rolling test
NUM_BLOCKS = 10                 # for ensemble averaging (sub-samples)
MIN_OBS_FOR_TEST = 100

# --- Decomposition Parameters ---
NUM_ERGODIC_COMPONENTS = 3      # number of ergodic components to extract

# --- Signal Parameters ---
NON_ERGODICITY_MA_WINDOW = 21   # smoothing window for signal

# --- Backtest Parameters ---
TRAIN_WINDOW = 504               # days to estimate ergodic signal
REBALANCE_FREQ = 5               # days (weekly rebalancing)
COMMISSION = 0.001

# --- Results ---
LOCAL_RESULTS_DIR = "results"

# --- Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# --- Run ID ---
TODAY = datetime.now().strftime("%Y-%m-%d")

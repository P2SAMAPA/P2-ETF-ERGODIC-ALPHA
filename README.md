# Ergodic Theory Alpha Engine

Test ETF return processes for ergodicity using Birkhoff ergodic theorem.  
Non‑ergodic returns imply path dependency → momentum signal.

## Features
- Birkhoff ergodicity test (time vs ensemble average)
- Ergodic decomposition via PCA
- Rolling non‑ergodicity signal (market timing)
- Backtest: long when signal > 0, else cash
- Daily automated runs via GitHub Actions
- Streamlit dashboard for signal and performance

## Setup
1. Clone repo
2. Install dependencies: `pip install -r requirements.txt`
3. Set `HF_TOKEN` environment variable or GitHub secret
4. Run backtest: `python run_backtest.py`
5. Launch dashboard: `streamlit run app.py`

## Interpretation
- Signal > 0 → path dependent (momentum) → go long
- Signal ≤ 0 → ergodic (mean-reverting) → stay in cash

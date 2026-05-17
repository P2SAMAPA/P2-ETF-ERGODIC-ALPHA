from backtest import run_backtest_for_universe
from results_uploader import upload_results
from config import UNIVERSES

def main():
    print("Running Ergodic Theory backtest for all universes...")
    all_results = {}
    for universe_name, tickers in UNIVERSES.items():
        print(f"\n{'='*50}\nRunning for universe: {universe_name}\n{'='*50}")
        results = run_backtest_for_universe(universe_name, tickers)
        all_results.update(results)
    
    # Upload everything under one run folder
    upload_results(all_results)
    print("\nAll backtests completed and uploaded to a single run folder.")

if __name__ == "__main__":
    main()

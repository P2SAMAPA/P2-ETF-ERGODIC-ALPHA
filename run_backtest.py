from backtest import run_backtest_all_universes
from results_uploader import upload_results
from config import UNIVERSES

def main():
    print(f"Running Ergodic Theory backtest for all universes: {list(UNIVERSES.keys())}")
    all_results = run_backtest_all_universes()
    
    for universe_name, results_dict in all_results.items():
        # Add universe prefix to each key
        prefixed_dict = {f"{universe_name}_{k}": v for k, v in results_dict.items()}
        upload_results(prefixed_dict, run_id=None)
        print(f"Uploaded results for universe: {universe_name}")
    
    print("All backtests completed.")

if __name__ == "__main__":
    main()

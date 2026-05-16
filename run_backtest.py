from backtest import run_backtest
from results_uploader import upload_results
from config import ACTIVE_UNIVERSE, UNIVERSES

def main():
    print(f"Running Ergodic Theory backtest for universe: {ACTIVE_UNIVERSE} ({len(UNIVERSES[ACTIVE_UNIVERSE])} tickers)")
    results_df, cum_strat, cum_bench = run_backtest()
    if results_df is not None and not results_df.empty:
        # Save results
        upload_results({
            "daily_returns": results_df,
            "cumulative_strategy": cum_strat.to_frame(name="strategy"),
            "cumulative_benchmark": cum_bench.to_frame(name="benchmark")
        })
        print("Backtest completed and uploaded.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

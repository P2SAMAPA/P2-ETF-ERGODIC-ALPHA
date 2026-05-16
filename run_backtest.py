from backtest import run_backtest
from results_uploader import upload_results
from config import ACTIVE_UNIVERSE, UNIVERSES

def main():
    print(f"Running Ergodic Theory backtest for universe: {ACTIVE_UNIVERSE} ({len(UNIVERSES[ACTIVE_UNIVERSE])} tickers)")
    results_df, cum_strat, cum_bench, latest_scores, per_etf_scores_history = run_backtest()
    
    if results_df is not None and not results_df.empty:
        upload_dict = {
            "daily_returns": results_df,
            "cumulative_strategy": cum_strat.to_frame(name="strategy"),
            "cumulative_benchmark": cum_bench.to_frame(name="benchmark"),
            "latest_etf_scores": latest_scores.to_frame(name="non_ergodicity_score"),
            "per_etf_scores_history": per_etf_scores_history
        }
        upload_results(upload_dict)
        print("Backtest completed and uploaded.")
        
        final_strat = cum_strat.iloc[-1] - 1
        final_bench = cum_bench.iloc[-1] - 1
        print(f"Strategy total return: {final_strat:.2%}")
        print(f"Benchmark total return: {final_bench:.2%}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

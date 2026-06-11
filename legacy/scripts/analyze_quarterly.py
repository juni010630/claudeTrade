
import sys
import yaml
import pandas as pd
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DataLoader
from scripts.run_backtest import build_engine
from metrics.returns import sharpe

def analyze_quarterly(params_path: str):
    with open(params_path) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    initial_capital = bt.get("initial_capital", 100)
    since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
    until = pd.Timestamp(bt.get("end"), tz="UTC") if bt.get("end") else None

    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )

    engine = build_engine(p, initial_capital)
    
    sig_path = Path("data/signals_dump.parquet")
    if sig_path.exists():
        print(f"Using fast replay mode with {sig_path}...")
        signals_df = pd.read_parquet(sig_path)
        # Ensure timestamp is datetime
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        report = engine.run_replay(signals_df, loader.iterate(since=since, until=until))
    else:
        print(f"Running full backtest (no signals dump found)...")
        report = engine.run(loader.iterate(since=since, until=until))
    
    eq = engine.equity_curve.to_series()
    if eq.empty:
        print("No equity data.")
        return

    # Quarterly Resampling
    # We want: Start Equity, End Equity, Max DD within quarter, Trade Count
    
    # 1. Equity stats
    # Resample to Daily first to make it cleaner, then Quarter
    daily_eq = eq.resample('D').last().ffill()
    q_end_eq = daily_eq.resample('QE').last()
    q_start_eq = daily_eq.resample('QE').first() # This is roughly the start of the quarter
    
    # Better way: group by year and quarter
    eq_df = eq.to_frame()
    eq_df['year'] = eq_df.index.year
    eq_df['quarter'] = eq_df.index.quarter
    
    results = []
    groups = eq_df.groupby(['year', 'quarter'])
    
    prev_end_equity = initial_capital
    
    # Ledger for trade counts
    ledger_df = engine.ledger.to_dataframe()
    if not ledger_df.empty:
        ledger_df['year'] = ledger_df['exit_time'].dt.year
        ledger_df['quarter'] = ledger_df['exit_time'].dt.quarter
        trade_counts = ledger_df.groupby(['year', 'quarter']).size()
    else:
        trade_counts = pd.Series()

    for (year, quarter), group in groups:
        start_equity = prev_end_equity
        end_equity = group['equity'].iloc[-1]
        
        # Intra-quarter MDD
        peak = group['equity'].cummax()
        dd = (group['equity'] - peak) / peak
        max_dd = dd.min()
        
        q_return = (end_equity - start_equity) / start_equity * 100
        trades = trade_counts.get((year, quarter), 0)
        
        results.append({
            "Period": f"{year} Q{quarter}",
            "Start": start_equity,
            "End": end_equity,
            "Return (%)": q_return,
            "Max DD (%)": max_dd * 100,
            "Trades": trades
        })
        prev_end_equity = end_equity

    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print(f"QUARTERLY PERFORMANCE: {params_path}")
    print("="*80)
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("="*80)
    print(f"Total Return: {(prev_end_equity - initial_capital)/initial_capital*100:,.2f}%")
    print(f"Final Equity: ${prev_end_equity:,.2f}")

if __name__ == "__main__":
    analyze_quarterly("config/final_v13_eth.yaml")

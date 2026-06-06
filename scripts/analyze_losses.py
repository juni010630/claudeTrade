
import sys
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DataLoader
from scripts.run_backtest import build_engine

def analyze_losses(params_path: str):
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
        signals_df = pd.read_parquet(sig_path)
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        engine.run_replay(signals_df, loader.iterate(since=since, until=until))
    else:
        engine.run(loader.iterate(since=since, until=until))
    
    ledger_df = engine.ledger.to_dataframe()
    if ledger_df.empty:
        print("No trades found.")
        return

    ledger_df['exit_time'] = pd.to_datetime(ledger_df['exit_time'])
    ledger_df['year'] = ledger_df['exit_time'].dt.year
    ledger_df['quarter'] = ledger_df['exit_time'].dt.quarter

    # Target periods: 2022 Q3, 2023 Q4
    loss_periods = [(2022, 3), (2023, 4)]
    
    for year, quarter in loss_periods:
        print("\n" + "="*80)
        print(f"ANALYSIS FOR {year} Q{quarter}")
        print("="*80)
        
        q_trades = ledger_df[(ledger_df['year'] == year) & (ledger_df['quarter'] == quarter)].copy()
        if q_trades.empty:
            print("No trades in this period.")
            continue
            
        print(f"Total Trades: {len(q_trades)}")
        print(f"Total PnL: ${q_trades['pnl'].sum():,.2f}")
        print(f"Win Rate: {(q_trades['pnl'] > 0).mean()*100:.1f}%")
        
        print("\n[Strategy Breakdown]")
        strat_perf = q_trades.groupby('strategy')['pnl'].agg(['sum', 'count', 'mean'])
        print(strat_perf)
        
        print("\n[Symbol Breakdown]")
        sym_perf = q_trades.groupby('symbol')['pnl'].agg(['sum', 'count', 'mean']).sort_values('sum')
        print(sym_perf)
        
        print("\n[Exit Reason Breakdown]")
        exit_perf = q_trades.groupby('exit_reason')['pnl'].agg(['sum', 'count'])
        print(exit_perf)

        # Look for consecutive losses
        q_trades = q_trades.sort_values('exit_time')
        q_trades['is_win'] = q_trades['pnl'] > 0
        
        print("\n[Regime/Confluence Analysis]")
        if 'confluence_score' in q_trades.columns:
            score_perf = q_trades.groupby('confluence_score')['pnl'].agg(['sum', 'count'])
            print(score_perf)

if __name__ == "__main__":
    analyze_losses("config/final_v13_eth.yaml")


import sys
import yaml
import pandas as pd
from pathlib import Path
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DataLoader
from scripts.run_backtest import build_engine

def run_sweep(params_path: str):
    with open(params_path) as f:
        base_p = yaml.safe_load(f)

    bt = base_p.get("backtest", {})
    initial_capital = bt.get("initial_capital", 100)
    since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
    until = pd.Timestamp(bt.get("end"), tz="UTC") if bt.get("end") else None

    loader = DataLoader(
        symbols=base_p["symbols"],
        timeframes=base_p["timeframes"],
        primary_tf=base_p.get("primary_timeframe", "1h"),
        cache_dir=base_p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=base_p.get("data", {}).get("lookback_bars", 300),
    )
    
    # Full backtest to see TP effect
    test_cases = [
        (3.5, 4.0), # Original
        (2.5, 3.0),
        (1.5, 2.0),
    ]

    results = []

    for ema_tp, multi_tp in test_cases:
        p = copy.deepcopy(base_p)
        p["strategies"]["ema_cross"]["atr_tp_mult"] = ema_tp
        p["strategies"]["multi_tf_breakout"]["atr_tp_mult"] = multi_tp
        
        engine = build_engine(p, initial_capital)
        print(f"Testing TP: ({ema_tp}, {multi_tp})...")
        # Reuse loader to avoid re-loading data if possible, but loader.iterate is a generator
        report = engine.run(loader.iterate(since=since, until=until))
        
        eq = engine.equity_curve.to_series()
        
        # Quarterly stability calculation
        daily_eq = eq.resample('D').last().ffill()
        q_returns = daily_eq.resample('QE').last().pct_change().dropna()
        q_std = q_returns.std() * 100
        
        results.append({
            "EMA_TP": ema_tp,
            "Multi_TP": multi_tp,
            "WinRate(%)": report.win_rate,
            "Sharpe": report.sharpe,
            "MaxDD(%)": report.max_drawdown,
            "FinalEquity": report.final_equity,
            "Q_StdDev(%)": q_std,
            "TotalTrades": report.total_trades
        })
        print(f"DONE: ({ema_tp}, {multi_tp}) -> WinRate: {report.win_rate:.1f}%, Sharpe: {report.sharpe:.2f}, Final: ${report.final_equity:,.0f}")

    df_results = pd.DataFrame(results)
    print("\n" + "="*100)
    print("TP MULTIPLIER SWEEP RESULTS (FULL BACKTEST)")
    print("="*100)
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("="*100)

if __name__ == "__main__":
    run_sweep("config/final_v13_eth.yaml")

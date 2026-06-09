# MarginTierTable 배선 측정 (2026-06-10)

`engine.use_margin_tiers` 키로 Binance 브래킷 MMR 주입 (run_backtest/live_trade 양쪽 배선, 기본 off).

## 결과: off == on 완전 동일 (v17 FULL·OOS, 거래/MDD/Sharpe 전부)
배선 smoke test 통과 (ALT $200K notional → MMR 10% = flat 0.5%의 20배). 그런데도 무영향
→ **v17은 어떤 MMR 기준으로도 청산 임계 근처에 도달하지 않음** (고정 SL 1.8 ATR이 항상 선행 청산).
flat 0.5% 폴백의 "낙관" 우려는 현 전략/스케일에서 실측 0. 시드 커져 notional이 브래킷 상단 갈 때 재측정.

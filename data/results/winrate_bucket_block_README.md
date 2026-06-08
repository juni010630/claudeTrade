# 승률/버킷 차단 실험 결과 (2026-06-07)

데이터: [`winrate_bucket_block_results.csv`](winrate_bucket_block_results.csv) (42행)

## 목적
final_v13_eth 전략의 손실 버킷을 진단(`scripts/bucket_diagnostic.py`)하고, 차단 후보를
단독으로 차단한 풀 백테스트(`scripts/block_sweep.py`)로 비교. 승률(WR) 개선이 목표.

## CSV 스키마
- `experiment`: single_block(전기간+OOS) / by_year(연도별) / wednesday_on_btcshort(수요일, btc_short 위 추가)
- `config`: baseline=차단없음, block_*=해당 버킷 단독 차단, Sema+*=교차, btc_short(+wednesday)=배포config 기준
- `period`: full / 2025-26 / 2022 / 2023 / 2024
- `baseline_ref`: **none**=차단없는 원본 기준 / **btc_short**=이미 btc_short 차단된 config 기준 (수요일 런)
- 지표: final_equity($100 시작 복리), sharpe, calmar, max_dd_pct, win_rate_pct, profit_factor, trades

## 결론

**✅ 채택: BTC숏 차단** (`symbol_block_directions: {BTCUSDT: [short]}`) — config/final_v13_eth.yaml 반영 + 실계좌 배포.
- 연도별 강건: 2022 +21%/WR+5pp, 2023 ~중립, 2024 +8.5%, 2025-26 +29%/WR+1pp. 4년 중 3년+ 1년중립.
- Sharpe·Calmar·MDD도 매년 동반 개선. 인과: BTC숏 PF 2.51 < 포트폴리오 평균 3.49 → 슬롯/자본 기회비용.
- ⚠️ 전기간 자산 +66.7%는 복리증폭(실엣지 Sharpe +4.5%). "방향·일관성"이 신호지 배율 아님.

**❌ 기각:**
- **block_S_emacross** — 2025-26만 환상적(PF 11), 2023·2024엔 WR·수익 다 악화. 단일구간 과최적화.
- **block_wednesday** — 3/4년 WR 포함 전 지표 악화, 2025-26은 WR만 +1.6pp인데 Calmar 절반 붕괴.
- **block_S_all** — 전기간 MDD -66.7% 악화(multi_tf S티어는 PF 14 우량). S차단은 ema_cross 한정.
- **교차(Sema+btc_short, Sema+eth_long)** — S_emacross 성분이 2023/24 끌어내려 비강건.
- block_eth_long / block_arb_long — 2기간(full+OOS)서 baseline 열위 (연도별 미실시).

## 방법론 교훈
- 버킷 차단 검증은 **반드시 연도별 4분할**. 전기간/단일 OOS만 보면 과최적화를 통과시킴(S_emacross가 실례).
- 흑자 버킷이라도 PF가 포트폴리오 평균 미만이면 기회비용으로 차단 가치 검토(btc_short).
- WR<40% 단독은 차단 근거 아님(BTC숏은 WR 39%지만 흑자) — PF·연도별 일관성과 함께 판단.

재현: `python3 -u scripts/block_sweep.py --by-year --only <configs>` / 진단 `python3 scripts/bucket_diagnostic.py`

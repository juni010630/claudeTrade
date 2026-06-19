The verification confirms the flagged correction. The actual Bitget numbers are: Binance ref $2,722/Sh2.74(bar)/MDD-28.0% vs Bitget $1,144/Sh2.04(bar)/MDD-51.0%, 2024 Sharpe 1.62→0.86 / MDD -26.2→-41.8, book-level PnL/trade multi_tf 12.55→4.33, ema 6.53→2.84. The "결과 로그 미생성(config만 빌드)" in the original summary is wrong. I'll correct this item and produce the final document.

# 자동매매 봇 매매기법 파라미터 튜닝 전수 완성본

---

## 1. 버전 계보 한눈에 (직전 대비 delta)

| 버전 | 날짜 | 직전 대비 핵심 변경 (delta) | 백테 헤드라인 | 비고 |
|---|---|---|---|---|
| **final_fixed_v2** (소실) | 구버전 | BTC regime, rpt 0.07, max_pos 5, trailing/breakeven 포함 | — | 원본 (파일 소실, 서술만) |
| **v13 (final_v13_eth)** | ~2026-04 | regime BTC→**ETH** (3x 수익↑), rpt **0.07→0.099**, max_pos 5→7, sd→5, TP 2.0/SL 1.0→**ema TP3.5/SL1.8·multi TP4.0/SL1.8**, max_hold→336h, 티어 SSS6/SS5/S4/A3, hour-block **26개**(ema13+multi13), tier_block ADA S차단, BTC숏 차단 | 완전데이터 $5.28M/Sh2.72/MDD**-88.6%**/2023 **-78%** | 2023 취약 (백필로 드러남) |
| **v14 (final_v14_pyramid)** | ~2026-06-07 | v13 + **피라미딩** +1.0R/25%/max_adds1/ema_cross 한정 | 좁은데이터 +66% | 완전데이터선 MDD 악화로 off |
| **v15 (final_v15_gate)** | 2026-06-07 | tier_block **ADA→ADA/FIL/ARB A·S 게이트**(SS티어+만 진입), **SSS 티어 신규**(lev50/score6, 커밋 fe12195), 피라미딩 off | full $10.6M/Sh2.921/Calmar21.1/MDD-65.8%/2023 **-13% 생존** | 2023 취약 해결 |
| **v16 (final_v16_slwide)** | 2026-06-08 | **multi SL 1.8→2.1**(유일 변경, 커밋 1d58400) | full $19.3M/Sh3.049/Calmar26.7/MDD-60.5%/2023 흑자 | 단독 정직값 $27K/Sh1.67/MDD-89% (헤드라인 과적합) |
| **merged_v16_sleeve** | 2026-06-08 | v16 + **평균회귀 슬리브** 도입 + **capital_fraction 50:50**, max_pos 7→10, sd 5→8, daily DD **OFF**(-1.0), hour-block **funding3 {0,8,16}** | $4,442/Sh1.69/MDD봉-49.7%/833거래 | funding3 차단 버전 |
| **merged_noblock_sleeve** | 2026-06-09 | funding3 **제거(무차단)** | $6,079/Sh1.66/MDD-61.0%/940거래 | v17 베이스 |
| **v17 (final_v17)** | 2026-06-09 | **SS5→6**(SS6) + **sd8→4**(sd4) [강화클러스터] + **rpt 0.099→0.07** + **deep_floor -0.55** 신규 + **maker_entry** 신규(06-10) + **timeout청산**(06-12) | $8,991/Sh1.94/MDD봉-42.4%/805거래/2023+2% | IS·OOS 6셀 전부↑ |
| **v18 (final_v18_triple)** | 2026-06-11 | **macross_d 슬리브 신규**(1d EMA20×100 알트78 숏수확, 격리북) + **트리플 40:30:30** + 격리북 CB포함 버그수정 | ×45.7/Sh일1.86/MDD일-30.8%/2023+11% | 메인넷 라이브 |
| **v19 (final_v19_dvol)** | 2026-06-17 | **DVOL 글로벌 변동성타게팅 신규**(target 55→45, 커밋 039549e→423000f) | t45 ×35.4/Sh~1.975/MDD-24.6% | per-book로 즉시 진화 |
| **v20 (final_v20_dvol_perbook)** | 2026-06-17 | DVOL **글로벌→per-book**(추세t40/MR·mac t55) + **배분 40:30:30→50:25:25**(t50s25n25 공격) | ×48/Sh2.013/봉MDD-30.0~36.71% | **현 라이브** |
| (v19_fngtilt) | 2026-06-17 | FNG 레짐틸트(δ0.10) — **기각**, enabled:false 코드보존 | no-op (Sh-0.002/MDD-0.4pp) | 미배포, v18/v20 유지 |

**현 라이브 = `config/final_v20_dvol_perbook.yaml`** (per-book DVOL t50s25n25 공격, 커밋 30785fe). 계보 핵심값(rpt 0.07, max_pos 10, sd4, SS6, corr 0.8, 무차단, deep_floor -0.55, maker_entry)은 v17 이래 전부 상속.

---

## 2. 카테고리별 전수 목록

표기: `[파라미터 — from→to — 효과 — 판정 — 근거]`

### (a) 지표 파라미터 (EMA/MACD/BB/RSI/ATR/ADX)

**ema_cross**
- EMA fast — 8 고정 — 8→9 = -19%/MDD-61.7(악화) — **현라이브** — config/OAT
- EMA slow — 21 고정 — 21→20 = **+72%($11,349)/Sh1.92(표 최고)이나 거래+15뿐=경로運 의심** — **기각(노이즈, walk-forward 미신뢰)** — OAT_ABLATION_FINDINGS
- MACD (fast/slow/signal) — 12/21/7 고정 — MACD fast 12→14 = +20%/MDD-55.0(약한+, 채택 미달) — **현라이브; 14 기각(약함)** — OAT
- 4h EMA 방향필터 — 150 고정 — 150→200 = +21%/MDD-53.0(강화클러스터 후보였으나 미채택); 150→100 = -23% — **현라이브; 200 off(미검증), 100 기각** — OAT
- ATR period — 14 고정(전 전략·전 버전) — ATR14=$67K vs ATR20=$56K(둘 다 ETH regime서 BTC 대비 3x) — **현라이브; ATR20 유효하나 미채택** — strategy_experiments
- 다중속도 ema_slow(20/52, MACD30/52/18 ×2.5클럭) — 추가 시도 — full Sh2.921→2.843, Calmar/MDD 악화, 개선 1/4년(2024만) — **기각(off)** — v15_gate_champion

**multi_tf_breakout**
- BB period — 25 고정 — — **현라이브** — config
- BB std 4h — 2.0 고정 — — **현라이브** — config
- BB std 1h — 2.2σ 고정 — 2.2→2.0 = **-68%/Sh1.37 파국(multi 핵심나사)**; 2.2→2.5 = -7%이나 MDD+9.6pp(-47.9)·2023+134(강화클러스터 최강 2023); ±0.2 = -96% 날카로운 정점 — **현라이브 2.2; 2.0 기각(파국), 2.5 기각** — OAT/v16_revalidation
- RSI period — 10 고정 — 10→14 = 베이스라인 완전동일(진입 0건 변화=죽은나사); RSI 7/14/21 스윕 전부 v16 미달 — **현라이브; 14 off(죽은나사)** — OAT/v16_rsi
- RSI 임계 long_min/short_max — 55/45 고정 — 50/50 완화 = 진입 무변화(죽은나사); 45/55/65 변형 lo65만 +3.7%(거래1개差=노이즈); **65/45 강화 = +33%/MDD+5.4pp(강화클러스터, 미채택)** — **현라이브 55/45; 65/45 off(미채택)** — OAT/v16_rsi
- volume_multiplier — 1.8x 고정 — 1.8→1.5 = -28%(악화); **1.8→2.2 = +27%/MDD+4.4pp/2023+18이나 들어간 조합 3개 전부 OOS -180~240pp 패배=과적합 폐기** — **현라이브 1.8; vol2.2 기각(과적합 OOS 전패)** — OAT_ISOOS
- volume_lookback — 20 고정 — — **현라이브** — config

**scorer/기타 지표**
- scorer volume_ratio_threshold — 1.8(final 전버전) — — **현라이브** — config
- scorer rsi_long_max/short_min — 70.0/40.0 고정 — — **현라이브** — config
- scorer funding_long_max/short_min — 0.0003/-0.0003 고정(펀딩 점수게이트) — — **현라이브** — config
- scorer daily_ema_period — 200 고정 — — **현라이브** — config
- 4h offset(봉정렬) — 0h~3h 테스트 — 모든 offset Sh>1.0, 0h 최적(2.16)/2h 최약(1.23) — **현라이브 0h(바이낸스 기본)** — CLAUDE.md
- ADX 스코어링 차등가점 — ADX>40→+2/>50차단/조합 — 전부 baseline 미달(Sh3.005→2.86~2.96), ADX20~25 최고 WR61.8%/50+ 최악 — **기각** — ml_adx
- momentum_breakout 지표(15m, bb_std1.8, RSI 등) — 신규 — 비용 비생존 — **기각** — MOMENTUM_BREAKOUT

### (b) TP/SL · 보유 · 쿨다운

**TP/SL (ema_cross)**
- atr_tp_mult — 3.5 고정 — **WF: IS pick 4.0(IS+616/Sh1.32, OOS+729/Sh2.79) vs base 3.5(IS+599, OOS+1276/Sh3.44)=4.0 OOS 붕괴**; 2.5(IS+434/Sh1.20 꼴찌, OOS+1689/Sh3.83 최고)=test-peeking; 3.0(IS·OOS 모두 base 열위, 순위역전); OAT 3.5→3.0=+10%, →4.0=-36%; ema-only 짧은TP 2.5=+44%(최근구간 착시) — **현라이브 3.5; 4.0/3.0/2.5 전부 기각** — EMA_TP_WF/OAT
- atr_sl_mult — 1.8 고정 — 1.8→1.5=-18%, →2.1=-25%(양방향 악화), SL1.8 능선 확정 최적 — **현라이브 1.8** — OAT/FAST_SWEEP

**TP/SL (multi_tf_breakout)**
- atr_tp_mult — 4.0 고정 — 4.0→3.0=-35%, →2.5=-39%(2022/23/25 전부 악화), fast_sweep 정점 2.5는 ema가 끌어올린 착시 — **현라이브 4.0; 3.0/2.5 기각** — OAT_FOLLOWUP
- atr_sl_mult — **1.8→2.1 (v16, 커밋 1d58400)** — full $10.6M→$19.3M/Sh2.921→3.049/MDD-65.8→-60.5/2023 흑자전환, 이웃 1.95~2.25 plateau; **2.1→1.8 역시도=+4%이나 MDD-59.8 악화/2023-25** — **현라이브 2.1(v16~v20)** — v16_revalidation/OAT

**uniform/fast_sweep (방법론)**
- uniform TP2.5/SL1.8 — fast_sweep 정점 Sh2.08/MDD-50.8% — **아티팩트(uniform·진입고정·풀샘플 한계, MDD ×100 버그)** — **기각(점추천 금지)** — FAST_SWEEP
- uniform SL 능선 — SL1.2/1.5/1.8/2.2/2.5 — SL1.8 행이 능선(OAT ema SL1.8→1.5 -18%·→2.1 -25%와 일치) — **방향 채택(1.8)** — FAST_SWEEP
- uniform TP 능선 — TP2.0~5.0 — TP2.5 정점·이후 단조악화, TP5.0/SL2.2=Sh0.94/$633 최악 — **방향 experimental(WF로 3.5 확정)** — FAST_SWEEP

**적응형 TP/SL**
- 국면조건부 동적 TP/SL — 가설 8종 — 전부 Calmar 하락, global WF 모든 폴드 in-sample 최적=현행값(항등) — **기각** — adaptive_tpsl
- SL 적응형 좁힘(loADX) — 1.8 고정 vs 좁힘 — MDD-55→-51% 소폭이나 우측꼬리 절단→Calmar/수익 더 하락 — **기각** — adaptive_tpsl
- multi TP 넓힘(적응형) — — MDD -65~-72% 악화 — **기각** — adaptive_tpsl

**보유 (max_hold_hours)**
- max_hold — 336h(14일) 고정 — 336→168=-4%/MDD-65.0(악화); 336→720=-37%/MDD-63.5(악화) 양방향 악화 — **현라이브 336** — OAT

**쿨다운 (tp_cooldown_hours)**
- 현재 TP만 6h — 6h 고정 — $6,602/Sh1.75/MDD-57.5%(스위트스팟)/2023·2024 최선 — **현라이브** — COOLDOWN_VARIANTS
- 쿨다운 없음 0h — 6h→0h — **전지표 최악** $5,499/Sh1.66/MDD-62.6%/PF1.52 — **기각** — COOLDOWN_VARIANTS
- TP+SL 6h(SL도 쿨다운) — TP만 6h→TP+SL — $7,234/Sh1.79/MDD-61.2%(수익최고이나 MDD 3.7pp 깊음, 2023 못살림)=수익레버 — **off(MDD↑)** — COOLDOWN_VARIANTS
- 쿨다운 12h — 6→12 — -36%/MDD-57.5(경로노이즈 의심) — **기각** — OAT
- (3h와 6h 비트동일=비바인딩, 504h 보유와도 비트동일) — — — 재검증 정점 — v16_revalidation

**동적 SL (제거됨)**
- breakeven_trigger_r — null(비활성) — breakeven 1.0R = **1h vs 5m 괴리 94%(뻥튀기)** — **기각/제거** — config/CLAUDE.md
- trailing_r_mult — null(비활성) — trailing = 1h 성과 뻥튀기(고정 TP/SL은 1h=5m 괴리 0%) — **기각/제거(final_aggressive_v1 유물)** — config/CLAUDE.md

### (c) 리스크/사이징 (rpt, size_fraction, leverage 티어)

**risk_per_trade (계좌)**
- **0.099→0.07 (v17, 커밋 806c80b)** — 파레토지배: $5,896→$6,602, MDD-61→-57.5%, Sh1.654→1.752, Calmar2.60→2.89, 연도별 MDD 4년 전부↓, 2024 +7→+31% — **현라이브 0.07(v17~v20)** — RISK_DEPLOYMENT_FRONTIER
- 0.07→0.05 — $6,602→$5,110/MDD-56.0(0.07보다 수익↓ 과하향) — **기각(0.07 열위)** — RISK_FRONTIER/OAT
- 0.07→0.09 — -6%/MDD-59.7(악화) — **기각** — OAT
- 0.099→0.12+ — 진짜 악화 신호(Kelly 초과) — **기각** — sizing_universe
- 0.07~0.11 글로벌 스윕 — Sharpe 평탄(2.99~3.07), equity ±30% 경로노이즈 — **0.099 유지(당시)/이후 0.07** — sizing_universe
- 0.099→0.30 상방 — MDD-89→-91 완전정체(**max_notional_equity_mult=3.0 하드캡이 rpt~0.15 위 바인딩**) rpt 크랭크=리스크환상 — **기각(추천 철회)** — blend_sizing_reality
- 0.099→0.035/0.015 하방 — 0.035 아래여야 진짜 MDD 반응(0.015→-22% 블렌드), 0.099→0.05 MDD-89→-86 정체 — **experimental(MDD레버 sticky)** — blend_sizing_reality
- 초기 최적화 0.08→0.099/0.10 수렴 — 0.08=$115K/0.099=$200K, MDD 거의 불변(-48~-50%) — **adopted(당시 수렴)** — strategy_experiments
- multi 한정 rpt 0.10/0.14, lev×1.5 — 0.10=$5,015(-44%)/0.14=$6,795, lev×1.5 무변화(캡 흡수) — **기각(변동성드래그)** — sizing_universe

**SSS 티어 rpt 오버라이드**
- SSS rpt — 0.20 고정(v13~v20) — 0.25+는 자산↑/Sharpe↓(레버리지일 뿐) — **현라이브** — config/v16_revalidation

**레버리지/size_fraction 티어 (추세전략)**
- SSS — lev50 / sf0.30 / rpt0.20 (v15+ 신규, 커밋 fe12195) — 최고 conviction 분리 — **현라이브** — config
- SS — lev40 / sf0.30 (score5+) — v13 254거래 WR48% PF3.43 +$440,753 — **현라이브** — config/README
- S — lev25 / sf0.22 (score4) — v13 114거래 WR53.5% PF0.95 -$9,976(ADA차단 정화) — **현라이브** — config/README
- A — lev10 / sf0.10 (score3) — v13 65거래 WR41.5% PF3.09 +$33,201 — **현라이브** — config/README
- B/C — lev1 / sf0.01 (min_score 99=비활성) — — **off-default(의도적)** — config
- leverage_tiers 축소 — 티어별 낮춤 — **MDD 불변(rpt 캡이 leverage 지배=죽은 노브)** — **기각** — v16_rsi
- RSI 모멘텀 가중 사이징(강모멘텀×1.5~2.0) — — 자산 $530M 폭증이나 Sh3.049→2.8↓/MDD-60→-78% — **기각(레버리지함정)** — v16_rsi
- 변동성타게팅 vol_target_ann 50~90% — — 목표↓=수익↓/목표↑=레버리지(MDD↑), rpt캡이 이미 변동성 반영 — **off(엔진 보존)** — v16_rsi

**size_bonus**
- strategy_size_bonus_mult — 1.25, ema[0,5,12,15,22]/multi[0,1,12,14,22] 고정 — **제거시 -47%/Sh1.61→1.45/MDD불변=양성(A2)**, 단 자산만 증폭(레버리지성, MDD민감 계좌엔 제거 고려) — **현라이브 유지** — OAT/overfit_audit

**circuit_breaker**
- CB pause/stop/pause_hours — 5/10/48 고정 — CB제거=+23%/MDD동일(인샘플 거의 일 안함, 꼬리보험), **강화 5→3=-70% 파국** — **현라이브 ON; 제거 off(보험), 강화 기각** — OAT

### (d) 포지션 한도/상관

- max_positions — **7→10 (merged/v17)** — max_pos 5=-22%/MDD-63.9(집중·분산상실 악화); 10→15=완전동일(상한 미도달); 7→8/9=비트동일(슬롯 안참) — **현라이브 10; 5 기각, 15 off(무영향)** — OAT/sizing_universe
- max_same_direction — **5(v13~v16)→8(merged)→4(v17, sd4, 커밋 e93c061)** — sd8→4=+17%($7,755)/MDD+2.1pp/2023+11, SS6와 결합 IS·OOS 6셀 전부↑ — **현라이브 4** — OAT_ISOOS
- correlation_block_threshold — 0.8 고정(params.yaml만 0.9) — corr 0.8→1.0(제거)=+209%($20,428)이나 2022/23/24 전부 악화·MDD-72.1·Sh+0.01뿐=2025 집중복리 가짜; 0.8→0.6=-27%/MDD-48.7(거래급감 과필터) — **현라이브 0.8; 제거·0.6 기각** — OAT
- correlation_lookback — 100 고정 — 슬리브 동시 상관진입을 1~2개로 자동제한 — **현라이브** — config

### (e) 티어 게이트 (score, 위성티어)

- tier_sss_min_score — 6 고정 — sss7 = -80% — **현라이브** — config/v16_revalidation
- tier_ss_min_score — **5→6 (v17, SS6, 커밋 e93c061)** — SS5→6=+23%($8,096)/MDD+11.4pp(-46.1)/2023+14/2024+88, **IS·OOS 6셀 전부 baseline 파레토지배(IS+599 vs +455, OOS+1276/Sh3.44)=유일 견고 생존**; 5→4 완화=-19%/MDD-69.9 악화 — **현라이브 6; 4 기각** — OAT_ISOOS
- tier_s_min_score — 4 고정 — 4→3=-2%(무변화); 4→5=-46%/MDD-61.5(악화, 경로노이즈) — **현라이브 4** — OAT
- tier_a_min_score — 3 고정(confluence score 3+ 진입) — — **현라이브** — config
- tier_b/c_min_score — 99(B/C 차단) — — **off-default** — config
- **위성심볼 티어게이트 (tier_block_symbols)** — v13: S:[ADA] → **v15+: A·S 둘다 [ADA,ARB,FIL] SS+만 진입** — v15 full $10.6M/Calmar21.1/MDD-65.8/2023-13% 생존(v13 -78%→); OAT 게이트제거=-18%/MDD-71.2/2023-34; 게이트약화(gateA)/확대(+DOGE)/SSS만 사방 하락 봉우리 — **현라이브(v15~v20)** — v15_gate/OAT
- multi×SSS 셀 조건부 게이트(2023 출혈셀 PF0.29 차단) — 1d ADX/Kaufman ER/ML — 차단시 full-99.5%, ex-ante 분리 불가 — **기각(-13%는 로켓연료 유지비)** — v15_gate

### (f) 시간대 차단 (전 이력)

- **26개 데이터마이닝 차단** (ema13[6,7,8,9,10,11,13,14,16,18,19,21,23] + multi13[3,4,5,7,8,9,10,13,16,19,20,21,23]) — v13~v15 — **과적합: 1h시프트 Sh2.79→1.43 붕괴, per-hour IS↔OOS 상관≈0(ema+.06/multi+.01), WF OOS 일반화 +0.3 Sharpe뿐, 무차단 MDD-92%** — **기각/폐기(v16서 교체)** — overfit_audit
- 구버전 UTC[6,7,8,9,16] — 초기 — Sharpe 2.15→2.61(구기록), 이후 26개로 확장 — **past-live** — CLAUDE.md
- **funding3 {0,8,16}** (ema/multi 공통, v16, 커밋 7248bdb) — 26개→3개 인과교체(펀딩정산 휩쏘) — 블렌드 Sh1.66→1.71/MDD-56→-49/2023+10→+39, v16단독 $27K/Sh1.67/MDD-89%/2023-32→+15; **무차단(Sh1.71/$47K)보다 못함(1.67/$27K), 랜덤3시간 8/13위=엣지 아님** — **v16 adopted → v17서 제거(past-live)** — funding3_verification
- 정산직후 {1,9,17} — funding3 대안 — **전 변형 최악 $4,948/Sh1.33/MDD-95%**(='막으면 폭망'=funding3 인과 정밀성 증거) — **기각** — funding3_verification
- {2,10,18} — — $194K 압살(풀샘플)이나 데이터마이닝 — **기각** — funding3_verification
- funding_post {1,9,17}/us_open/thin_overnight — — funding_post Sh1.0 폭망, us_open/thin_overnight H2 악화 — **기각** — overfit_audit
- ema 14·19시 해제 (unb14_19) — — +25%($24.2M)/3-4년개선이나 2023 $105→$92·MDD-60.5→-63.2 후퇴 — **off/보류(미채택)** — v16_revalidation
- **무차단 (block hours 전체 제거, v17, 커밋 6aa0d0b)** — funding3→무 — merged_noblock_sleeve $6,079/MDD-61.0/Sh1.66/940 — **현라이브(v17~v20)** — config
- hour-block + size보너스(초기) — 차단+x1.25 — Sharpe 2.15→2.61, 역추세 오버레이는 전부 마이너스 — **past-live(초기)** — winrate_plan

### (g) 레짐/필터

- **regime 기준 심볼 BTC→ETH** — final_fixed_v2→v13 — $200,122/Sh2.245/MDD-50/Calmar9.78, **BTC 대비 3x 수익**; DOGE regime $211K(MDD높음), ADA Sh2.80, SOL MDD-90% 사용불가 — **현라이브(ETH 첫번째)** — strategy_experiments
- regime ADX trending/ranging — 30.0/12.0, period14 고정(params.yaml만 ranging22) — ranging 높이면 유해 — **현라이브** — config/v16_revalidation
- regime BB — period20/std2.0/width_lookback50/squeeze_pct0.2 고정 — — **현라이브** — config
- **BTC숏 차단 (symbol_block_directions BTCUSDT:[short])** — 채택 — 2022+21%/WR+5pp, 2023~중립(-2%), 2024+8.5%, 2025-26+29%, 전기간 +66.7%/Sh2.875→3.005, BTC숏 PF2.51<평균3.49; OAT 풀데이터 BTC숏허용=+26%/MDD-55.0개선이나 **2023 +5→-2 악화(loan-seed 걸림)** — **현라이브; 허용 기각** — bucket_block/OAT
- S|ema_cross 버킷차단 — 시도 — 집계/OOS 환상(PF3.5→11)이나 2023 Sh0.72→-0.00/2024 자산반토막 — **기각(과최적화)** — bucket_block
- block_S_all/arb_long/eth_long/wednesday/교차 — 시도 — baseline 열위, multi S티어 PF14 같이 잘림 — **기각** — bucket_block
- BTC 거시레짐 게이트/가중(btc_mom_gate/opposite_weight, PF0.43 독성셀) — — 차단=경로카오스(자산1/5), 가중=PF↑이나 Sh<3.049 — **off(엔진 보존)** — v16_rsi
- RSI 모멘텀 게이트(차단, 임계55/60+) — — 55까지 v16동일, 60+폭락(약모멘텀도 흑자라 차단할 게 없음) — **off(코드 보존)** — v16_rsi
- multi 진입 RSI 45/55/65 — — 전부 미달, lo65 노이즈 — **기각** — v16_rsi
- ema_cross 롱 차단 — 시도 — PF0.95이나 3/5년 흑자(단일구간 핏) — **기각** — winrate_plan
- 거래량 필터(ema min_volume_ratio) — 저유동 차단 — 진단 통과(WR36 vs 52)이나 full Sh1.61→1.64뿐+국면의존 — **off(코드 보존)** — overfit_audit
- 펀딩 정렬 필터(scorer 가드) — — 펀딩 역방향 진입도 PF2.46 흑자 — **기각(후보 아님)** — sizing_universe
- 티어 필터 추가(A티어 차단/축소) — — 연도별 n5~40, 패턴 난폭(A티어 PF0.09→13.71) — **기각(노이즈)** — sizing_universe
- 국면전환 메타컨트롤러(주간 모드선택 K3~8/60~180일) — — 제로섬(2023절약≈2024회복손실), 2024 연패=2023 횡보와 통계적 동일 — **기각** — v15_gate

**펀딩-z 게이트 (funding_gate, multi_tf, 2026-06-19)**
- funding_gate on/off (z_max 0.5, 8h펀딩 60일 롤링z, 스킵률 32%) — off→on — **gate Sh1.986 < random 2.132(대조군 패배)**, full Sh평평(1.93 vs 1.96)·수익-37%, 2022★1.95/2023★0.81/2024 1.57→1.19(불장 손상)/25-26 2.94, daily MDD 동일(-26.9%) — **기각(off-default 코드보존, v20 무영향)** — FUNDING_GATE
- z_max 임계 (WF 미튜닝) — 0.5 — 2023 gate가 baseline·random 둘다 능가(0.81 vs 0.51/0.46)=진짜효과 1개이나 2024 손상 순상쇄 — **기각** — FUNDING_GATE

**FNG/브레드스 레짐 틸트 (regime_tilt)**
- FNG δ0.1 G→mom — post-hoc Sh1.894→1.906/MDD-36.7, **IS/OOS 반기 부호반전(1차 -0.045/2차 +0.036)=노이즈** — **기각** — REGIME_TILT
- 브레드스(% 1d>EMA50, 83심볼) δ0.1 G→mom — Sh1.912(최고+0.018)/MDD-36.7, 반기 부호반전(-0.047/+0.081) — **기각** — REGIME_TILT
- 반대방향 G→mr — 전부 ↓(경제적으로 틀림) — **기각** — REGIME_TILT
- δ 크기 0.1→0.15/0.2 — δ↑=포화·MDD악화(약신호), δ0.15 1차반기-0.071 — **기각** — REGIME_TILT
- 실엔진 배선(v19_fngtilt, capital_fraction_schedule δ0.10 dir+1) — v18→v19 Sh1.864→1.862/MDD-30.8→-31.2=**완전 no-op**(스크린 +0.018조차 증발) — **기각(미배포, enabled:false 코드보존)** — REGIME_TILT
- BTC도미넌스/알트시즌 지수 신호 — — 2022+ 정직 히스토리 무료소스 없음(404) — **기각(보류)** — REGIME_TILT

**TRI 추세지수 (Trend-Range Index)**
- TRI 라벨컷 ≥65추세/≤35횡보 + 히스테리시스 — IS분위 33/67%→고정컷 — OOS주 AUC0.746 vs 베이스0.744 동급, 플립7.7 vs 12.6/년(-39%), 월별 Spearman0.694(OOS0.748) — **adopted(정보용 CLI만)** — TREND_INDEX
- TRI 성분선택 (11종→6종 등가중, AUC≥0.60) — 채택6(er.744/tstat.743/r2.737/bbwpct.714/dircon.663/chop.643), 탈락 vr5/vr10/acsum/adx_w/hurst(.46역방향) — 고정앵커 시그모이드(가중치 학습 없음) — **adopted** — TREND_INDEX
- TRI 기반 스위칭/틸트 — — **전방 2주 AUC 0.46~0.49=예측력 없음 → 스위칭 배선 금지** — **기각(스위칭)/adopted(정보용)** — trend_index

**MVRV 타이밍 (외부신호)**
- MVRV 롱숏 z-score 역발상 (z_win365/lag1/cost10bps) — 저MVRV롱/고MVRV숏 — 약함/음수(롱숏-0.56) — **기각** — mvrv_timing
- MVRV 고정임계 (<1.0롱/>2.2숏/0.25) — — 롱편향=BTC바이홀드+0.72 위장 롱베타(바이홀드0.75보다 못함), 사이클신호=일/스윙 부적합 — **기각** — mvrv_timing

### (h) 슬리브/배분

**평균회귀 슬리브 (mean_reversion)**
- 슬리브 도입 — merged_v16~v20 — LTC/UNI/STORJ/ARPA/BAND, signal_tf 1d, RSI(14) oversold30/overbought70, max_adx99(게이트무력), require_bb false, use_regime_gate false, ATR TP3.0/SL2.0; **v16과 무상관(-0.14)→합산 MDD-55→-34%, Sh2.60→2.71**, 추세장엔 발목; OAT 슬리브 OFF=-30%/MDD-61.3(하중지지) — **현라이브** — config/altseung
- 슬리브 lev/sf — lev3/sf0.08(익스포저0.24, sleeve_lev_grid lev2~5×sf0.06~0.15서 확정) — forced_stop 31%→6%, sf0.06~0.32 plateau — **현라이브** — altseung/SLEEVE_TP
- 슬리브 primary_timeframe — 1h→1d — 1h 마찰(진입가/intrabar/레버리지/daily stop) 해소, +262%R 재현 — **adopted** — altseung
- 슬리브 TP 스윕 — 3.0 고정 — TP1.5=$7,715/WR60.6/PnL540, 2.0=$7,605/PnL442, 2.5=$7,178/PnL472, **3.0=$8,991/Sh1.94/PnL733**, TP<3.0 단조악화(WR↑이나 건당익절폭축소 압도) — **현라이브 3.0; 저TP 기각** — SLEEVE_TP
- 슬리브 대칭 1:1 — 3.0/2.0→1.0/1.0=$5,378/PnL99 참사, 1.5/1.5=$8,287(1:1중 최선이나 전지표열위), 2.0/2.0=$7,605 — **기각** — SLEEVE_TP
- 슬리브 2:1 — 2.0/1.0=$7,321/MDD-47.8(SL조임 휩쏘), 3.0/1.5=$9,943/Sh1.98이나 MDD-45.2, **4.0/2.0=$12,914/Sh2.07(1단계통과 플라토✓연도별✓)이나 IS우위·OOS열위(OOS+1276→+1166/Sh3.44→3.32)** — **기각(OOS)** — SLEEVE_TP
- 슬리브 TP 플라토(3.5~5.0) — 3.0→스윕 — Sh 3.5=1.98/4.0=2.07/4.5=2.06/5.0=1.99(완만한 언덕) — **experimental/off(OOS 증거 없이 변경 불가)** — SLEEVE_TP
- 슬리브 신호빈도 — 1d 고정 — 12h=Sh1.93/WR46.5(약신호 자주노출), **8h=$215/Sh0.80 참사**(4h 재현), 12h+1:1 1.5=$9,202이나 MDD-49.9(+7.5pp 레버변환), 8h+1:1=$6,343/n1252 — **현라이브 1d; 전 변형 기각** — SLEEVE_TP
- 슬리브 청산개선(시간청산10일/RSI50복귀) — — 둘다 열위(반등이 ATR3배까지 큼) — **기각** — altseung
- 슬리브 추세필터(EMA50/100/200 순응) — — 거래 185→3~33 증발(구조적 양립불가) — **기각** — altseung
- 슬리브 ADX 국면 스위칭(임계22/25/28, 3/7일리밸) — — 스위칭 Sh2.0~2.5/MDD-52~62% vs 혼합2.71/-34% — **기각(상시병행 확정)** — altseung
- 슬리브 max_positions 5→2~3 — — 단독 MDD-48→-33%/4년양수이나 블렌드선 희석(75:25 MDD동일-45%) — **기각(미반영)** — sleeve_trend_switch
- **슬리브 심볼확장 5→8(MTL/SNX/ONT)** — — MDD-35→-26이나 수익/Sharpe↓, 추가 슬리피지46-60bp — **기각** — altseung
- **슬리브 심볼확장 5→10(WF, SFP/AXS/BCH/ZEC/MTL)** — 84심볼 스캔 후보79중 72개 현역중앙값(Sh0.412)미달, 통과7(SFP1.17/AXS0.82/BCH0.70/ZEC0.57/MTL0.52/LPT0.48/ENS0.47) — IS에서조차 전지표악화(IS+599→+397/Sh1.32→1.16/MDD-42→-60), 크래시일 클러스터→동시적층 폭발 — **기각(3차, 영구종결)** — SLEEVE_EXPANSION
- 1차 슬리브 변형(BB20터치+RSI, 1h/4h, RSI25~30, TP1.0~1.5) — 4변형 — 홈그라운드(2023) PF0.63~0.92 적자(밴드터치 MR은 비용후 엣지 없음) — **기각(off)** — v15_gate

**macross_d 슬리브 (1d EMA20×100 알트 숏수확)**
- macross_d 도입 (v18, ~700변형/7반복 유일생존) — fast20/slow100, signal_tf 1d, ATR TP6.0/SL3.0, max_hold 1440h(60d), liq_min $3M, liq_window30, 티어 고정A, lev3/sf0.5/rpt0.02, 격리북 max_pos8/sd8+CB격리, 알트78(메이저top12 제외) — 엣지=알트만성하락 숏수확+펀딩수취, OOS Sh0.7~1.1, 30bp 둔감, v17상관0.10, 5m교차 깨끗 — **현라이브(v18~v20)** — NEWEDGE_GREEDY
- macross EMA쌍 스윕 — 9쌍(5×20/10×50/10×100/20×100) — **20×100(쌍둥이10×100) 챔피언**(고원=슬로우쌍, 빠른쌍 hold60붕괴), WF IS최적셀=OOS최적 — **adopted** — NEWEDGE_GREEDY
- macross 출구 — 6×ATR/3×ATR/hold60d 고정 — tp10/3 OOS+592<6/3 +730, 채널청산 OOS-405bp, 역크로스 2024-52/2026-111bp, hold120 IS PF1.52이나 와이드 6/3만 생존 — **adopted(6/3)** — NEWEDGE_GREEDY
- macross 유동성필터 — liq0→liq>$10M(config $3M) — 메이저top12 PF0.97(-27bp 엣지전무)/알트 PF1.60(+621bp), 알트-only OOS+726bp — **adopted(알트78)** — NEWEDGE_GREEDY
- macross 방향 — both(롱+숏) vs short-only — short-only FULL×2.22(숏=만성하락 2022+2097/2025+1771/2026+1459bp)이나 2023~24 2년 플랫; **both=전연도 흑자** — **adopted(both)** — NEWEDGE_GREEDY
- macross TF — 1d 고정 — 12h 기각(2024적자), 7d 기각(2022적자), 2d/3d PF1.34~1.93이나 포트Sh0.28/2022-11% — **adopted(1d만 생존)** — NEWEDGE_GREEDY
- macross 컨플루언스필터(+macd/+vol/+rsi_mid 2팩터게이트) — — 전부 PF≤1.3 무가치 — **기각(무필터)** — NEWEDGE_GREEDY
- macross supertrend/aroon/OBV/ichimoku/CCI/W%R/keltner 14종 — — keltner만 IS통과(PF1.35)이나 2025-200bp 사망(돌파형 2025 횡보 전멸) — **기각** — NEWEDGE_GREEDY
- macross 심볼프루닝 78→41 — — full standalone $165/Sh0.561 → pruned $106/Sh0.189(Sharpe·수익 둘다 악화) — **기각(78 유지)** — DVOL research/mac
- macross 단독 capfrac 1.0(_mac_full/_mac_pruned) — — 스크래치 비교 — **experimental** — _mac config

**newedge 패밀리 (전부 기각, F3·macross만 생존)**
- F1 Donchian 1d (N20채널) — IS PF1.29/+206bp/2022적자, OOS 2025-163bp — **기각(F3흡수)** — NEWEDGE
- F2 추세-눌림목 1d (EMA50+RSI3) — IS PF0.77~1.04 — **기각** — NEWEDGE
- F3 압축→돌파 1d (NATR압축+채널, pct0.3/N10/TP5/SL2.5/hold30/liq>$10M) — IS PF1.34/+223bp/3년흑자, 포트 greedy×2.42/OOS+1.8%, **IS유일생존이나 OOS(2025~) 엣지소멸(2025-50bp)=2023횡보돌파 아티팩트, 23~36bp서 적자** — **기각** — NEWEDGE
- F4 쇼크바 1h — IS PF0.95~0.99 — **기각** — NEWEDGE
- F5 BTC→알트 리드래그 — follow대적자/fade 2024소멸 — **기각** — NEWEDGE
- F6 일중 변동성돌파 1h — -17bp/건(비용사망) — **기각** — NEWEDGE
- F7 월간 TSMOM — Sh~1.2/연16%/MDD-50 — **기각(미달)** — NEWEDGE
- 2차 그리드 1h/4h(540변형, n≥200/avg≥40bp/PF≥1.25/3년흑자) — 1h 0통과(최고1.10), 4h 0통과(최고1.21), 1d 7통과(6=EMA크로스) — **기각(1h/4h 비용벽)** — NEWEDGE

**capital_fraction 배분 (추세:슬리브 2-way)**
- 50:50 — merged/v17 — $4,442~$6,602/MDD봉-49.7~-49.6/Sh1.68~1.75 — **past-live(사용자 채택)** — config
- 45:55 — — MDD봉-44.9/Sh1.70/$3,612 (rpt0.099판: -57.0/$4,774) — **experimental(미채택)** — frontier
- **40:60** — — **MDD봉-40.9/Sh1.71/$2,870 (MDD 스위트스팟·효율 끝)** (rpt0.099판: -53.5/$3,758); OAT 50:50→40:60=-40%/MDD+6.6pp 재확인 — **권장/experimental(사용자 50:50 채택)** — frontier/OAT
- 35:65 — — MDD봉-37.2/Sh1.70/$2,229 (rpt0.099판: -49.7/Sh1.699 최고급/$2,874) — **experimental** — frontier
- 30:70 — — MDD봉-33.4/Sh1.67/$1,690 (rpt0.099판: -45.8 최저/$2,135 Sharpe꺾임) — **experimental** — frontier
- 55:45 (정적, sizing_pools 대안) — 50→55 — $11,589/MDD봉-44.8/Sh1.912(게이트턱걸이, OOS미검증) — **off-default(미채택)** — SIZING_POOLS
- 60:40 (OAT shift) — — $9,930(+50%)/MDD-64.0(악화)=엣지 아닌 비중레버 — **기각** — OAT
- 40:60+rpt0.07 (결합 안전후보) — — $3,975/MDD-50.9/Sh1.781(스윕최고)/연도별 4년 전부개선/수익~⅔ — **off-default 권장(대출시드, 미배포)** — frontier
- 2026YTD 5:5/4:6 (rpt0.07) — 10:0→ — 5:5 +128.7%/MDD-19.9(반토막)/Sh3.337; 4:6 +91.75%/MDD-15.83(최저) — **experimental(부분년 표본, 영구결정 근거 아님)** — frontier
- 리밸런싱 빈도 — 매일→월간→무리밸 — 매일-49%/월간-50%(실행가능)/무리밸-69~79%, **슬리브 분산은 월간리밸 필수전제** — **experimental** — blend_sizing_reality

**트리플/3-way 배분 (추세:슬리브:macross)**
- **40:30:30 (v18, 커밋 724cd8d)** — v17 단독→트리플 — ×45.7/Sh일1.86/MDD일-30.8%(v17-42.2% 대비 14pp↓)/2023+11%, macross 한계기여 T조건부(T60:0→T40:-6.5→T34:-8pp) — **past-live(v18~v19 base)** — NEWEDGE/config
- 34:33:33 (v18 후보) — — v18a(capital_fraction)×31.5/Sh1.91/MDD봉-28.1/2023+13 vs v18b(pools월간)×41.0/Sh1.88/-33.5 — **experimental(사용자 40:30:30 채택)** — NEWEDGE
- 3-way 프론티어(T:S:N) — 33:33:33/40:30:30/20:40:40/60:20:20 등 — 33:33:33 Sh1.90/MDD-34.7, 40:30:30 Sh1.85/MDD-41.0, **20:40:40 Sh2.05/MDD-24.6(Sharpe최대)**, 60:20:20 Sh1.73/MDD-57.0(딥플로어권), 슬리브↔신규 상관-0.40(최강분산쌍), 추세비중=수익·MDD 단조다이얼 — **adopted(40:30:30); 기타 off-default(다이얼)** — NEWEDGE
- 2-way v17:newedge (100:0~50:50) — — 100:0×84.7/Sh1.80/MDD-41.5, 70:30×43.4/Sh1.93/-28.3, 50:50×24.5/Sh1.98/-23.9, 일별상관0.10~0.12, 30% 편입만으로 MDD 13pp↓ — **experimental(3-way로 발전)** — NEWEDGE
- **v20: 50:25:25 (t50s25n25, 커밋 30785fe)** — v18 40:30:30→ema0.5/multi0.5/MR0.25/mac0.25 — DVOL per-book이 고추세 길들여 추세비중↑ 허용(t40 무DVOL은 MDD폭증 기각이었음), t50+perbook Sh1.962/봉MDD-30.0/×48(v19×35의 1.4배), 사용자 '공격' 선택 — **현라이브** — config
- 60:20:20 스크래치(_htrend_t60) — — 고추세 비중 — **experimental** — _htrend config

**TRI-MR / 기타 슬리브 후보 (기각)**
- TRI<35 횡보게이트 광역MR 슬리브(5번째 격리북) — v18 4슬리브→+TRI-MR — standalone Sh0.777/4년양수/v18상관+0.028 유망(post-hoc 블렌드 w0.3=Sh1.999/MDD-29.2), **정식통합서 기각: rsi25/65=+47PnL·rsi30/70=-413PnL 과적합, 격리북 cross-margin=레버리지로 MDD-30.8→-46.8% 폭증** — **기각** — TRI-MR
- TRI-MR 그리디 IS우승(thr_t60/thr_r35/ema20/rsi25_65/tp4/sl2/hold40/cost15bps) — IS Sh0.482→OOS0.63→FULL0.529, 모드분해 trend절반 2024-0.14 음수=노이즈/range절반0.65, 격자경계=과적합 — **기각(추세-모멘텀 절반)** — tri_strategy
- v19 5슬리브 통합(rsi25/65 vs 30/70) — — rsi25/65 $6,831/PnL+47.45(MDD악화-46.75), rsi30/70 $2,170/PnL-413.52 참사 — **기각** — tri_strategy
- 전략추가 volume_imbalance(CVD) — 14건/건당$72.67 — +4% 수익이나 MDD 6.9pp악화→Calmar기준 제거 나음(final_v11=without VI) — **기각(off)** — strategy_experiments
- 20+ 전략추가(rsi_trend/breakout_confirm/supertrend/ichimoku/keltner/stoch/vwap/cci/heikin/pivot 등) — — 전부 포트폴리오 악화, 100건+ 시그널=WR35-39% 수렴 실패 — **기각** — strategy_experiments
- 추세 6심볼 확장 9/12/24/48/92 — — 단조파괴(2024: 6심볼$5,350→12심볼$2,233→24$608→48$9.74→92 -84%), 6심볼 최적 확정 — **기각** — strategy_experiments/sizing_universe

### (i) 변동성타게팅 DVOL

- **DVOL 메커니즘 도입(글로벌 인버스)** — f=clip(target/DVOL[D-1], 0.3, 2.0), lag1 — **세션 유일 엔진검증 알파, 신호/종목/진입 불변·사이즈만**, IV≫RV(실현변동성 타게팅 MDD-47% 더 나쁨=IV 선행), 고DVOL서 엣지약화→자본재배치 — **adopted(메커니즘)** — DVOL
- DVOL 글로벌 target — **55→45 (v19, 커밋 039539e→423000f)** — t55 노출103% MDD방어미미(-29.2)/×70, **t45 MDD-24.6%(전구성 최고, 4년 전부개선 2022-22→-14)/Sh1.975/노출85%** — **past-live(v19, v20으로 교체)** — config
- DVOL target 다이얼 t35~t55 — — **Sharpe~1.96 평평(과적합반대=robust 알파)**, 평균노출 t40=0.75/t45=0.85/t55=1.03, 최종수익 ×24.8/×35.4/×70(노출복리), target=노출/레버리지 다이얼 — **off-default(다이얼)** — DVOL
- DVOL clip_lo/clip_hi — 0.3/2.0 고정(전 변형 공통) — 디레버 최대70%↓/가산 최대2x — **현라이브** — config
- DVOL lag_days — 1 (DVOL[D-1]) — look-ahead 차단 — **현라이브** — config
- **per-book DVOL (v20, 추세t40/MR·mac t55, 커밋 30785fe)** — 글로벌→책별차등 — 고DVOL 엣지저하 차등(추세 저하1.23~2.38 최대/macross 1d슬로우 0.23 최소), 글로벌t45 Sh1.975/MDD-24.6→per-book 40/55 Sh2.024/MDD-22.3·38/60 Sh2.044/MDD-21.4, **4년 전부 Sharpe↑MDD↓, post-hoc 9격자 전부 글로벌 상회**; htrend_t50 글로벌$6,368/-34.46→per-book$4,816/-29.96(4.5pp↓); htrend_t60 글로벌-41.42→per-book-36.71(4.7pp↓) — **현라이브** — config/DVOL
- per-book target 스윕 (38/60 vs 40/55) — — 38_60 $2,625/Sh2.163/MDD-22.16(최고급), 40_55 $2,850/Sh2.141/MDD-23.12, 더 공격적 추세디레버(t38) 소폭우위 — **experimental** — DVOL research
- per-book t60 — t50→t60 — ×78/봉MDD-36.7(공격끝), 사용자가 t60→t50 완화 — **기각(너무 공격적)** — DVOL
- DVOL 비대칭(상한1.0) — — 노이즈(+0.006) — **기각** — DVOL
- VIX(FRED) 단독 — DVOL→VIX — v18보다↓/MDD-40.9/결합희석(DVOL이 크립토 직접 우월) — **기각** — DVOL
- 실현변동성(RV) 타게팅 — IV→RV — MDD-47% 더 나쁨(사후·회복놓침) — **기각** — DVOL
- DVOL서 macross 제거(v17 0.5/0.5/0.5) — — 수익↑이나 MDD-42.2(macross=MDD보험) — **기각(macross 유지)** — DVOL
- v20 base_frac (per-book 시변) — v18 {0.4/0.4/0.3/0.3}→v20 {0.5/0.5/0.25/0.25} — base_frac × clip(target/DVOL) 시변적용 — **현라이브** — config

### (j) 집행 (maker-first, timeout청산, 비용/슬리피지, gap SL, margin tier)

**maker-first 진입**
- maker_entry 도입 (v17, 커밋 92b4b84) — execution.maker_entry enabled:true, timeout_sec300, poll_sec3, 시그널close 지정가+5분 시장가폴백, 백테 무영향 — 체결률91~98%, 순절감+6~35bps/노셔널, 미체결2~9% — **현라이브(v17~v20)** — MAKER_ENTRY/config
- maker 타임아웃 윈도 — 5분 채택 — 5분 체결91.4~97.8/S25시+6.2~25.4bps; 15분 95.1~98.9/드리프트+73bp; 30분 96.2~99.1/드리프트+55bp — **현라이브 5분; 15/30분 off(드리프트↑)** — MAKER_ENTRY
- maker-only (폴백없음) — strict: $2,110(-77%)/Sh1.450(미체결9% 러너놓침), touch: $4,656(-48%) — **기각(폴백 필수)** — MAKER_ENTRY
- retest 지정가(deeper) — — 깊을수록 단조악화, 체결분 WR 시장가보다↓(45→33% 역선택) — **기각** — winrate_plan
- macross maker-first (W4h~24h, limit TP+STOP_MARKET SL) — — 체결률98.5~99.7%, 절감+10~23bp/건, 역선택 무해 — **adopted(권장 W12~24h)** — NEWEDGE

**timeout 청산**
- timeout 청산 maker-first (v17, 커밋 fdc3e9b) — exit_reason==timeout만 allow_maker(postOnly+reduceOnly+잔량시장가폴백), SL/TP미스/딥플로어/emergency=시장가, 전량판정 status==closed만+포지션 재조회 — mean_reversion(336h)·macross_d(1440h) timeout 비중 커 실절감 유효, 백테 회귀 바이트동일 — **현라이브** — maker_entry_todo

**비용/슬리피지**
- default_slippage_bps — 5.0 고정(백테 가정, params.yaml만 10.0) — 실측 23~36bps, **15bps만 돼도 2023 +9%→-47% 붕괴=딥플로어 사망($138), 25bps→$116, 35bps→$100(전구간 사망); OOS(2025~)는 15→$1,061/25→$817/35→$629/Sh2.42(35bps도 생존, 헤드룸충분)**; 비용분해 gross$9,882→commission$541+slippage$473+funding$13→net$8,854 — **현라이브(5bps 가정 명기 필수, 비용=생존레버)** — COST_REALITY
- commission maker/taker — 0.0002/0.0005 고정(무할인) — BNB feeBurn ON이나 엔진 config 무할인 보수적 유지 — **현라이브** — config/bnb_fee
- funding_interval_hours — 8 고정 — — **현라이브** — config
- BNB 수수료할인(feeBurn 10%) — 이미 ON — BNB 입금=키 이체권한 대기, 잔고모니터 배포(9facf37) — **live(feeBurn ON, 엔진 무할인)** — bnb_fee
- SL 청산 슬리피지 모델 — backtest.py L987 MARKET 5bps — 이미 모델링(잔여리스크=급락시 5bps보다 나쁠 수, 1h OHLC 모델링불가 테일인지) — **현라이브** — GAP_SL

**gap SL / margin tier (옵션, off)**
- engine.gap_sl_pessimistic — 옵션 추가 기본 off — **808거래 중 봉경계 갭 SL 관통 0건(1h 갭 중앙값0.0003%/99.9분위0.057% ETHUSDT)**, off==on 완전동일(FULL$9,576·OOS$1,376) — **off-default(실측0건, 보험)** — GAP_SL
- engine.use_margin_tiers — 옵션 배선 기본 off — **off==on 완전동일**, v17은 어떤 MMR로도 청산임계 미도달(고정SL 1.8ATR 항상 선행), smoke ALT$200K notional MMR10%=flat0.5%의 20배인데도 무영향 — **off-default(무영향)** — MARGIN_TIER
- 엔진 maker-queue 한도 버그수정(per-fill 게이트, 커밋 559ae33) — — maker_entry off시 무영향(v17/taker 레저 bit-identical) — **adopted(버그수정)** — MOMENTUM_BREAKOUT

**거래소 교차검증 (Bitget, 백테 데이터 검증 — 파라미터 튜닝 아님)**
- v20 config Bitget 데이터 백테 (동일 기간 2024-01~2026-06, 동일 70종, 거래소 가격계열만 차이, 슬리피지 5bps 양쪽 동일) — **결과 산출·종결(2026-06-19): Binance ref $2,722/Sh2.74(봉)·2.63(일)/MDD봉-28.0%·일-27.2%/601거래/PF1.98 → Bitget $1,144/Sh2.04(봉)·2.02(일)/MDD봉-51.0%·일-41.8%/556거래/PF1.63**; 2024 Sharpe 1.62→0.86·MDD-26.2→-41.8, 25-26 Sharpe 3.22→2.78·MDD-27.2→-20.2; 책별 거래당 PnL multi_tf 12.55→4.33·ema 6.53→2.84(추세책 ⅓ 수준)·mean_rev 0.25→0.43(둔감)·macross 0.44→0.44(둔감) — 판정 = **엣지 ~절반·MDD ~2배, 헤드라인 v20 수치는 거래소 비전이**(추세/돌파책이 바이낸스 미시구조에 핏). Bitget perp 알트 히스토리 부재로 2022 베어·2023 횡보 검증 불가, 펀딩=0 근사(Bitget 결과는 낙관 상한). 실거래 집행 포팅(live_broker STOP_MARKET→Bitget 트리거, build_exchange)은 미완 — **검증완료(친구 Bitget 배포용 백테, 배포금지·실거래 미완)** — research/bitget_backtest/RESULTS.md, bitget_backtest

### (k) 리스크가드

- **deep_floor_dd — 도입 -0.55 (v17, 커밋 7a40719)** — running peak 대비 -55% 초과시 백테=early-stop·라이브=전량청산+정지, peak state.json 영속화(앵커$9,456.15), abort_mdd 재사용 — **인샘플 0발동=무비용 파산보험, 패리티동일($8,991/805)**, 단 15bps 스트레스서 2023 -55% 관통 발동 — **현라이브(v17~v20)** — edge_monitor/COST_REALITY
- daily_drawdown_pause — **-0.04(v13~v16)→-1.0(OFF, v17~v20)** — 단일계좌 공유DD가 분산죽임, OAT ON(-4/-10)=-30%/2023-21(회복구간 손절드래그, 봉MDD 안줄고 수익만 깎음) — **현라이브 OFF(딥플로어로 대체); ON 기각** — config/OAT
- daily_drawdown_stop — **-0.1(v13~v16)→-1.0(OFF, v17~v20)** — 딥플로어로 대체 — **현라이브 OFF** — config
- tail-cut (계좌단위/v16만 컷 -45%) — 시도 — **발동하면 처참($4,442→~$280, 회복죽임), v16만 컷-45%=MDD 오히려 악화(-49.7→-55.9)**, 드로다운이 최대상승 직전 — **기각(전수, 딥플로어만 허용)** — 2026_06_09_session/gap_sl
- 킬스위치 (emergency_stop.py) — 구현 — 봇정지 후 독립실행, dry-run기본/--yes 실집행, 텔레그램기록 — **adopted(수기청산 대신)** — axis_sweep
- 봉기준 MDD 낙관(갭 SL 고정체결) — — 모든 OAT MDD 낙관(실MDD 더 깊음), 강화클러스터 MDD 15pp축소 주장 보정 필요 — **방법론 경고** — OAT_FINDINGS

### (l) 기각된 발전축

**피라미딩**
- ema_cross 한정 +1.0R/25%/max_adds1 (v14 도입) — v13 full $8.4M→$13.9M(+66%)/Sh3.005→3.023/3-4년개선/EV 전연도+/5m검증통과 — **adopted(v14)→완전데이터 MDD악화로 off** — pyramid_result
- v17 재검증 — base$8,991/Sh1.937/MDD-42.4 → pyr$7,927/Sh1.845(**-12% 역전**), IS$699→$628/OOS$1,376→$1,362 전부 열위, sd4 슬롯경쟁+SS6 진입질↑로 증액 한계효용 음수, 5m교차 -10.3% 완전일치 — **기각(G-B 위반)** — PYRAMID_V17
- 전 전략 피라미딩 pyr10_f25 — full $8.4M→$15.3M(+82%)/Calmar25→31.2이나 2022-5%/2023-15%(횡보/베어 악화) — **기각/대체** — pyramid_result
- 변형 adds2/t0.75/t1.25/f0.35/ss_plus/multi_only — adds2=2023 PF0.92 전멸, plateau는 t1.0~1.25×f0.15~0.25 — **기각** — pyramid_result
- 전략×심볼 차단(block_emaETH/drop_FIL) — full -81%/-85%(낮은PF셀도 슬롯/복리 기여) — **기각** — pyramid_result
- 라이브 STOP_MARKET 증액 구현 완료(testnet 미검증) — — — final_v14_pyramid는 v13 유물로만 유지 — pyramid

**ML 메타필터**
- ml_soft_scoring hardcut (cut0.45) — 1차 v13 293건 AUC0.487 / 2차 v17 452건 LGBM WF AUC0.615 — **OOS 거래205→139, PF2.23→2.08, Sh3.44→3.12, 수익$1,376→$484(⅓)**, 약신호(~0.55)가 거래컷 복리손실 못이김 — **기각(off-default 보존)** — ML_FILTER_V17
- bonus 모드 (0.6/0.75 tier승급) — — v17 OOS prob 최대~0.57<0.6 → **발동0**, baseline동일 — **off-default** — ML_FILTER_V17
- 모델선택 LR vs LGBM — WF OOS LR0.544/LGBM0.615 — LGBM Purged CV0.508과 WF0.615 불일치(test113건노이즈, G3위반), permutation 1위 hour_cos(과적합축) — **기각** — ML_FILTER_V17
- 데이터셋(추세전략만 452건, 슬리브 가드) — — WR46.2%/NaN0/셔플AUC0.490(누수없음) — **재시도용 유지하나 기각** — ML_FILTER_V17
- 검증설계 Purged K-Fold(5)+Embargo14d+WF — — 보유겹침 누수차단(trailing 뻥튀기와 동일), 음성테스트 통과 — **adopted(방법론 채택, 결과 기각)** — ML_PLAN
- 피처(adx14/bb_width/rsi_1h/vol_ratio/atr_pct/dist_ema200/funding/hour sin·cos/direction) — 이진임계 연속값화 — direction_long(bull편향) OOS소멸, hour 피처 과적합 — **기각(재시도=신호레벨 n수천+hour제외)** — ML_PLAN/ml_adx

**momentum_breakout (15m Scalp)**
- 포팅 (고정티어A, 6심볼 BTC/ETH/SOL/BNB/XRP/DOGE, maker) — VIP9/1bps +60.19%/Sh0.765/MDD-9.27%/13212거래 헤드라인은 **진입집행 아티팩트**; VIP0/5bps -42.78%, 15bps -80.80%, 25bps -93.56%, 36bps -98.06%(실측9.3~36.2bps서 딥플로어관통 사망), **83종목 20종목 Sh~2.0→83종목 MDD-112% 파산** — **기각(비용 비생존)** — MOMENTUM_BREAKOUT
- SL 배수 0.5×ATR(6:1 R:R) — — 청산 ~81%가 SL=테이커(10742SL/2307TP/163timeout), 수수료·슬리피지 민감도가 v17보다 한자릿수 큼 — **기각(구조적 비용원인)** — MOMENTUM_BREAKOUT
- 비용 스윕 NET bps/trade — VIP9/1bps +3.666 → VIP0/5bps -4.123(swing -7.8bps 엣지소멸) → VIP0/36bps -29.722 — **기각** — MOMENTUM_BREAKOUT

**캐리/시장중립 (제3엣지 완전종결)**
- 현물 캐리 상시 (11심볼, 펀딩8h, 왕복0.4%) — 메이저 연5%(2024만12%/2022·2026음수, 판단선10%미달), 11심볼평균 연~5% — **기각(전연도양수 불충족, 봇부진과 겹쳐 분산가치약)** — CARRY_YIELD
- 현물 캐리 동적 (rate>0만) — 평균 **-27%/yr 참사**(펀딩 부호반전→전환 왕복비용 압살) — **기각** — CARRY_YIELD
- 크로스섹션 모멘텀/반전 (18메이저, 주간리밸 상위3롱/하위3숏) — 비용전+1.03%/주 Sh0.86, 비용(15bp) 반영 WR48-50% 소멸 — **기각** — market_neutral
- 페어 트레이딩 (6쌍 z-score, |z|>2진입) — WR57-68%이나 총수익 대부분 음수(LTC-BCH 67%/-119R), SOL-AVAX만+84(노이즈) — **기각(평균회귀 꼬리함정)** — market_neutral
- 펀딩극단 역포지셔닝 standalone (1,517이벤트) — 방향성 3d -0.34%(IS·OOS음수), 롱과밀 숏 -2.23%/3d, 임계강화 악화 — **기각(크라우딩 역베팅=모멘텀에 짐)** — market_neutral
- OI/테이커비율 — — 바이낸스 API ~30일만(2022 startTime invalid), oi_collector 자체축적(1년+ 후) — **기각(30일 벽, 검증불가)** — return_axes

**스위치 (추세↔슬리브 동적, 전 격자 소진)**
- 실시간 5지표 국면라우팅(추세→v16/횡보→슬리브) — full $5,896→$539(-91%)/Sh1.65→0.93/MDD-61→-77, 2023+7→-31/2024+7→-37 — **기각(전연도 압도)** — sleeve_trend_switch
- 주간 국면스위치 (ETH 1d 5지표, 매주 월요일 G0) — 추세장141주+2.87%(4/5년 방향OK), **횡보장82주+6.24%(슬리브 이겨야하는데 추세승, 0/4년 역방향)** — **기각(G0 횡보측 전연도 역방향, 백테 없이 종결)** — WEEKLY_REGIME
- 일별 ER 동적배분(100:0↔50:50) — — 기각 — **기각** — WEEKLY_REGIME
- 월간 ER 동적전환(틸트) — — H1/H2 WF 기각 — **기각** — WEEKLY_REGIME
- 슬리브 EMA 추세필터(추세시 진입차단) — — 신호고사(마스크가 회복분 삭제 아티팩트) — **기각** — WEEKLY_REGIME/sleeve_trend_switch
- 슬리브 trend_gate(basket ER>thr, 임계0.18/0.22/0.28) — — 플랫Sh-0.40/진입차단Sh-0.09~0.34(<0.65), 손실구간=미래수익구간 회복놓침, 마스크Sh3.0=회복분삭제 아티팩트 — **기각(off, inert보존)** — sleeve_trend_switch
- 성과 스트릭(핫핸드) 5변형 (K=2 고정) — A1뮤텍스 $637/MDD-27.4(수익94%증발=열등 디레버), A2 $5,294, **B1 핫핸드교대 $166/MDD-58.6/Sh0.49 참사(스위치73회=회복직전 그룹꺼버림)**, B2 $3,164, C1 소프트0.7/0.3 $8,102(가장온건이나 전지표악화) — **기각(전수, 연패=회복신호)** — STREAK_SWITCH
- TRI 한쪽틸트 (횡보50:50/추세만70:30, T73/T64) — BASE$8,991/Sh1.937 → T73$10,670/Sh1.854/MDD-51.0(IS$699→$630↓ 최근구간착시, OOS$1,376→$1,816↑), T64$9,900/-45.1, 전환34회 — **기각(IS악화·MDD-51·Sharpe열위, 배분동적전환 완전종결)** — TREND_INDEX/sleeve_trend_switch
- 광역MR ADX<25 게이트 (TRI대리) — — $58/Sh-1.806/MDD-55.23 딥플로어 사망(RSI극단↔ADX<25 비양립, 26거래) — **기각** — tri_strategy
- 광역MR 무게이트 (max_adx99) — — $64/Sh-0.098/MDD-55.15 딥플로어 사망 — **기각(심볼확장 영구기각 재확인)** — tri_strategy

**sizing_pools**
- 가상서브계좌 + 월간리밸 (config 기본 off) — off=v17 비트동일($8,991/805); **ON(월간50:50)=수익+22%($10,970)/Sh1.937→1.926(비열위)/MDD-45.1%(0.1pp초과)** IS Sharpe1.321→1.343/OOS$1,376→$1,490(+8%), 거래수805동일(사이징만), 정적55:45($11,589/-44.8)가 더 단순지배 — **기각(G-C 위반, off-default inert보존)** — SIZING_POOLS

**심볼확장 / max_pos / 펀딩필터 (sizing_universe)**
- 추세 6→12/24/48/92심볼 — 단조파괴 — **기각(6심볼 과적합 확정)** — sizing_universe
- max_pos 7→8/9, >10 — 슬롯 안참(비트동일) — **기각(무의미)** — sizing_universe
- 펀딩 정렬 필터 — 역방향도 PF2.46 흑자 — **기각** — sizing_universe

**MVRV / 외부신호** — (g) MVRV 항목 참조, 전부 기각 (단일사이클 방향베팅, 히스토리 부재)

---

## 3. 기각 사유 요약 (공통 교훈)

1. **과적합 (full-sample 최적화의 함정)** — 가장 빈번한 기각 사유. 시간대 26개 차단(1h시프트 Sh2.79→1.43 붕괴, per-hour IS↔OOS 상관≈0), vol_thr 2.2(IS최강이나 OOS -180~240pp 전패), ema slow 21→20(+72%이나 거래+15뿐), TRI-MR rsi30/70(-413). 교훈: **full-sample 헤드라인 금지, walk-forward(IS도출→OOS검증)·격자경계 회피 필수**. 적용 후 v17 강화클러스터(SS6+sd4)만 IS·OOS 6셀 전부 통과해 유일 생존.

2. **경로 카오스 (path-dependent 노이즈)** — 거래 ~10건짜리 미세조정은 슬롯/쿨다운/복리 연쇄로 결과가 ±수배 출렁임. 셀 차단(나쁜셀 제거)·티어필터·MACD fast 12→14·쿨다운 12h 등이 여기 해당. 교훈: **연도별 분리검증 필수, 거래수 거의 안 변하는데 수익만 출렁이면 경로運 의심**.

3. **레버리지 환상 (디레버를 알파로 오인)** — rpt 크랭크(0.099→0.30 MDD-89→-91 정체, max_notional 하드캡 바인딩), RSI 가중·leverage 티어·size_bonus·배분 60:40·TP+SL 쿨다운·뮤텍스 스위치 모두 "수익·MDD 동반이동"하는 순수 비중/레버 다이얼이지 엣지가 아님. 교훈: **MDD 레버는 사이징(rpt·배분)이지 TP/SL·필터가 아니다. Sharpe 비열위가 진짜 개선의 기준**. (단, DVOL은 사이즈만 바꾸면서도 고DVOL 구간 엣지약화에 자본을 재배치하는 점에서 다이얼이 아닌 엔진검증 알파로 분류 — 4년 전부 Sharpe↑MDD↓.)

4. **비용 = 생존 레버 (저빈도가 곧 생존)** — 슬리피지 15bps만 돼도 2023형 횡보장에서 딥플로어 -55% 관통 사망. momentum_breakout(15m, 81% 테이커), F6 일중돌파(-17bp), 캐리 동적(왕복비용 압살), 크로스섹션(15bp서 소멸), 슬리브 저TP/고빈도(8h $215 참사) 전부 회전율↑로 비용에 죽음. 또한 같은 전략·기간·심볼도 **거래소 미시구조가 바뀌면(Bitget) 엣지 ~절반·MDD ~2배**로 비전이 — 추세/돌파책일수록 바이낸스 가격계열에 핏. 교훈: **타이트 SL·고빈도·짧은 TP는 테이커 비중↑로 비용 비생존, maker-first(폴백 필수)가 생존 집행. 거래소 전환 = config 스왑이 아닌 재검증 프로젝트**.

5. **국면 조건부 끄기 불가 (손실구간 = 미래수익구간)** — 평균회귀 손실구간이 곧 미래 회복구간이라 추세지표·스트릭·주간/월간 ER로 on/off하면 회복분을 삭제하는 아티팩트. B1 핫핸드교대 $166 참사, 횡보판정주 슬리브전환 -6.2%p/주. 교훈: **연패=회복 신호, 상시 병행(정적 배분)이 정답. TRI는 전방 예측력 AUC0.47=스위칭 금지(정보용만)**.

6. **약신호의 복리 손실 (필터의 거래컷 비용)** — ML 메타필터(AUC~0.55)·funding_gate(random에 패배)·RSI 게이트는 "평균보다 좋은 거래"까지 잘라 복리 손실이 약한 선택 이득을 압도. 교훈: **거래를 자르는 필터는 random 대조군을 이기고 OOS PF·Sharpe 비열위여야 채택**.

7. **제3엣지 부재 (독립엣지 = 추세 + 평균회귀 둘뿐 확정)** — 캐리·페어·크로스섹션·펀딩극단·OI·MVRV 등 시장중립/외부신호 전부 기각. 크라우딩 역베팅은 모멘텀에 짐, 외부신호는 단일사이클 방향베팅이거나 2022+ 정직 히스토리 부재. (단 macross_d=알트 만성하락 숏수확은 추세 계열 내 별도 슬리브로 v18 채택, 트리플 분산의 한 축.)

8. **서브셋/헤드라인 착시** — 6심볼/20종목 결과로 사이징 금지(83종목 MDD-112% 파산), VIP9/0슬리피지 헤드라인 금지, 봉기준 MDD는 갭 SL 고정체결로 낙관(실MDD 더 깊을 수). 거래소 헤드라인도 비전이(Binance $2,722/Sh2.74 → Bitget $1,144/Sh2.04). 교훈: **항상 풀 유니버스·실측 비용·OOS로 검증, 헤드라인 인용 시 가정(5bps/풀샘플/거래소) 명기**.

---

### 부기: [누락검증] 반영 내역
- **(j) Bitget 항목 수정** — 정리본의 "결과 로그 미생성(config만 빌드), experimental(진행중)"은 오류였음. `research/bitget_backtest/RESULTS.md`(2026-06-19 종결) 기준으로 실제 결과 수치 전부 반영: Binance ref $2,722/Sh2.74(봉)/MDD봉-28.0% → Bitget $1,144/Sh2.04(봉)/MDD봉-51.0%, 2024 Sharpe 1.62→0.86, 책별 PnL/거래(multi_tf 12.55→4.33·ema 6.53→2.84·mean_rev/macross 둔감), 판정="엣지 ~절반·MDD ~2배, 헤드라인 비전이"로 교체. 단 이는 v20 config를 다른 거래소 데이터에 돌린 **검증**이지 파라미터 튜닝 자체가 아니므로 (j)의 별도 소항목으로 표기하고, 기각 사유 요약 4번(비용=생존레버)에 거래소 비전이 교훈을 통합.
- **누락검증 결론(그 외 누락·값오류 없음)** 확인: capital_fraction(ema0.5/multi0.5/MR0.25/mac0.25), rpt0.07, max_pos10, sd4, deep_floor-0.55, corr0.8, cooldown6h, daily DD-1.0(OFF), dvol_perbook(ema/multi40·MR/mac55), macross_d(fast20/slow100·ATR TP6/SL3·hold1440·liq$3M·liq_window30·lev3/sf0.5/rpt0.02·tierA·격리북 max_pos8/sd8), maker_entry·commission_maker0.0002(무할인)·size_bonus1.25 모두 라이브 v20 config와 일치. 코드 버그수정(loader look-ahead 4772ab6 등)은 파라미터 튜닝 범주가 아니므로 비포함(maker-queue 559ae33은 부산물로 (j)에 기재 유지).
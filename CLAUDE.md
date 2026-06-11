# claudeTrade — 자동매매 봇

## 프로젝트 개요
바이낸스 USDM 선물 자동매매 봇. 백테스트 엔진 + 라이브 트레이딩 인프라.

## 작업 원칙

신중함 > 속도. 사소한 작업은 판단껏 생략 가능.

### 1. 코딩 전에 생각하기
- 가정을 명시적으로 밝힐 것. 불확실하면 물어볼 것.
- 해석이 여러 가지면 제시할 것 — 조용히 하나를 고르지 말 것.
- 더 단순한 방법이 있으면 말할 것. 필요하면 반박할 것.
- 불명확하면 멈추고, 뭐가 헷갈리는지 명명하고, 질문할 것.

### 2. 단순함 우선
- 요청받은 것 이상의 기능 금지. 추측성 코드 금지.
- 한 번 쓰는 코드에 추상화 금지. 요청 없는 "유연성/설정 가능성" 금지.
- 불가능한 시나리오에 대한 에러 처리 금지.
- 200줄이 50줄로 가능하면 다시 쓸 것.
- 자문: "시니어 엔지니어가 과하다고 할까?" → 그렇다면 단순화.

### 3. 외과적 수정
- 꼭 필요한 부분만 건드릴 것. 인접 코드/주석/포맷 "개선" 금지.
- 안 깨진 것 리팩토링 금지. 기존 스타일을 따를 것.
- 무관한 dead code 발견 시 언급만 하고 삭제하지 말 것.
- 내 수정으로 orphan이 된 import/변수/함수는 제거할 것.
- 테스트: 변경된 모든 줄이 사용자 요청으로 직접 추적 가능해야 함.

### 4. 목표 기반 실행
- 작업을 검증 가능한 목표로 변환:
  - "버그 수정" → 재현 케이스 만들고 통과시키기
  - "전략/엔진 수정" → 수정 전후 백테스트 결과 비교 (의도한 차이만 발생 확인)
  - "리팩토링" → 전후 백테스트 결과 완전 동일 확인
- 다단계 작업은 `단계 → 검증 방법` 형태로 간단히 계획 제시.

### 5. 프로젝트 절대 규칙
- **look-ahead bias 절대 불허** — 전략/엔진 수정 시 미래 데이터 참조 여부 항상 확인
- **백테스트 = 라이브 일치** — config 새 키 추가 시 `run_backtest.py`와 `live_trade.py`의 build_engine 양쪽 모두 반영
- trailing/breakeven류 동적 SL 도입 시 5m 교차검증 필수 (1h 단독 결과 신뢰 금지)

## 백테스트 실행 규칙

- 여러 백테스트는 항상 병렬로 실행할 것 (subprocess, asyncio, concurrent.futures 등 상황에 맞게)
- 폴링 루프 금지 — 완료 신호(콜백, Future, Event, returncode 등)를 기다릴 것
- 결과는 완료 이벤트 수신 후 일괄 확인
- 최적화 결과는 연도별(2022~2025) 분리 검증 — 특정 연도 과최적화 기각

## 최종 전략 (final_v15_gate)

### ⚠️ 데이터 백필 사건 (2026-06-07)
기존 캐시의 ADA/FIL은 2024-02, ARB는 2024-02부터만 존재했다. 백필(ADA/FIL 2022-01~, ARB 2023-03~) 후
v13의 진짜 성과 = **2023년 -78%, full MDD -88.6%** (기존 문서 수치는 데이터 공백 착시).
→ 위성심볼 티어 게이트(v15)로 해결. **백테스트 수치 인용 전 캐시 데이터 범위 확인 필수.**

### 설정 파일
- `config/final_v18_triple.yaml` — **현 라이브 (2026-06-11~, 커밋 724cd8d)** = v17 + macross_d(1d EMA20×100 알트78 숏수확 슬리브, 격리 북) **40:30:30** 배분. 백테 ×45.7/Sh일 1.86/**MDD일 -30.8%**(v17 -42.2%)/2023 +11%. 검증·프론티어 = `NEWEDGE_GREEDY_RESULTS.md`. ⚠️ edge_monitor 앵커/베이스라인 v18 갱신 미완 (83심볼 edge_cache 필요)
- `config/final_v17.yaml` — 직전 라이브 (2026-06-09~06-11, 메인넷 systemd `trade-bot.service`, 봇 운용자본 ~$9,450 — 2026-06-10 사용자 시드 충전, 정상) = merged_noblock(시간차단 funding3 제거 + 슬리브 50:50) + **SS6+sd4**(tier_ss_min_score 5→6, max_same_direction 8→4). IS/OOS 홀드아웃 통과 — 직전 baseline을 IS·OOS 양 기간 파레토 지배(전구간 봉기준 $8,991/Sh1.94/MDD봉-42.4%/805거래; OOS 2025~ MDD-35.5/Sh3.44). vol_thr 2.2는 과적합으로 폐기. ⚠️ daily DD stop OFF — 대신 **딥플로어 -55% 장착(2026-06-10)**: `risk.deep_floor_dd: -0.55`, running peak 대비 DD 초과 시 백테=early-stop·라이브=전량청산+정지(`data/deep_floor_halt.json` 생성, 해제=rm 후 재시작). peak는 state.json 영속화(재기동이 피크 못 낮춤). 인샘플 0발동·패리티 동일($8,991/1.937/805) 검증. (봉MDD 갭 SL 낙관 우려는 2026-06-10 실측 0건으로 해소 — `GAP_SL_RESULTS.md`.) 봇 운용자본: 초기 $2,451(시드 $7,451 중 $5K spot 분리) → **2026-06-10 충전 후 ~$9,450**. 봇 제어=`sudo systemctl restart trade-bot`만(수동실행=중복인스턴스 금지). 직전 라이브 merged_noblock_sleeve=아래 표 베이스. [[project_oat_ablation_2026_06]] [[project_remote_deploy]] [[user_funding_caution]]
- `config/merged_v16_sleeve.yaml` — 직전 라이브(funding3, 06-08~09). 백테 $4,442/Sh1.69(bar)/MDD-49.7%(bar)·-45%(daily)/833거래. funding3가 무차단보다 MDD 얕음(완충) — 대출시드엔 이쪽이 더 안전했으나 사용자 선택으로 교체. [[project_funding3_verification]]
- `config/final_v16_slwide.yaml` — v16 단독 (funding3 차단). **헤드라인 $19.3M/Sh3.05는 26시간 과적합 폐기**, 정직값 $27K/Sh1.67/MDD-89%. 슬리브 없이 단독은 대출시드 부적합
- `config/final_v15_gate.yaml` — v13 + ADA/FIL/ARB SS티어(score 5+) 게이트
- `config/final_v13_eth.yaml` — 구버전 — 완전 데이터 기준 2023 -78% 취약
- `config/final_v14_pyramid.yaml` — v13 + ema_cross 한정 피라미딩. 완전 데이터에선 MDD 악화로 기본 비활성. 라이브 STOP_MARKET 증액 구현 완료 (testnet 미검증)

### 완전 데이터 재검증 (2026-06-08 전수 완료)
모든 상속 파라미터를 백필 데이터로 재검증함. 유지 확정:
TP 3.5/4.0, 티어 임계(SSS6/SS5/S4), 쿨다운 6h, 보유 336h, SSS rpt 0.20, corr 0.8, regime 임계,
ema 8/21, multi BB 25/2.2σ/vol 1.8x. 유일한 변경 = multi SL 2.1 (v16).
교훈: 거래 ~10건짜리 미세조정은 경로 카오스(슬롯/쿨다운 연쇄)로 결과가 ±수배 출렁임 — 연도별 검증 필수.

### ⚠️ 시간대 차단 과적합 (2026-06-08 검증, [[project_v16_overfit_audit]])
**시간대 차단 26개(ema 13 + multi 13)는 과적합.** 기존 기록 "제거 시 -99%(진짜 OOS 통과)"는 오도:
- "-99%"=무차단 시 **최종자산**만 99%↓(v16 $21.5M→$49K, v13 $5.9M→$33.7K). **무차단도 흑자**(Sh~1.5, +수만%).
- **"OOS 통과"는 거짓**: 1h 시프트하면 Sharpe 2.79→1.43 붕괴, 동일개수 랜덤도 못 재현, per-hour 손익 IS↔OOS 상관 ≈0(ema+.06/multi+.01), 연 3건/시간 표본. walk-forward 정직 도출 시 OOS 일반화는 **+0.3 Sharpe뿐**(나머지 ~99% 자산은 노이즈 핏).
- **시사:** 헤드라인 Sharpe 3.05/MDD-55%는 신뢰 불가. 차단은 수익뿐 아니라 **드로다운도 과적합 억제**(무차단 MDD-92%) → **라이브 실제 MDD가 백테보다 깊을 위험**. 정직 기대치 ≈ Sh 1.6~2.0/MDD-60~90%. **대출 시드 사이징은 -55%가 아니라 깊은 MDD 기준으로.**
- look-ahead/결정론/TP·SL(plateau)/슬리브는 **깨끗**(과적합 아님). 과적합은 hour-of-day 차단 레이어에 국한. size_bonus는 양성(제거 시 Sharpe↓·MDD불변 → 유지 확정, A2).
- **향후 모든 파라미터 변경은 full-sample 최적화 금지, walk-forward(IS도출→OOS검증)로만.**

### ✅ 조치: 차단 26개 → 인과 3개 교체 (2026-06-08, A4)
**데이터마이닝 26시간 삭제 → 인과 `funding3 {0,8,16}` 채택** (ema/multi 공통). 바이낸스 펀딩정산 UTC 시각 = 정산 휩쏘 회피. 사전선언 메커니즘 + 양쪽 반쪽(H1/H2) robust + 정산직후(1,9,17) 막으면 폭망(=인과 정밀성 증거).
- 효과(블렌드 50:50): Sharpe 1.66→1.71, MDD -56→-49%, **2023 횡보 +10→+39%**(취약국면 보강). v16 단독: $27K/Sh1.67/MDD-89%/2023 -32→+15%.
- **현 라이브 config(final_v16_slwide.yaml) = funding3.** 단독 MDD 여전히 -89% → A1(rpt 축소)+A3(슬리브 50:50) 동반 배포 필수.

### 2026-06-09 검증 세션 — 백테 기록 (version / 특성 / 결과)
전구간 2022-01-01~2026-04-23, $100 시작, 봉기준(엔진 equity_curve). 격리 스크립트: `scripts/{tailcut_backtest,recent_oos_check,funding3_robust,alloc_frontier,sleeve_dryrun,sleeve_dryrun_engine}.py` (프로덕션 무수정).

**핵심 백테 (config별 특성·결과):**
| config | 특성 | 최종$ | MDD봉 | Sharpe | 거래 |
|---|---|---|---|---|---|
| merged_v16_sleeve | v16+슬리브 50:50, funding3 차단 | 4,442 | -49.7% | 1.69 | 833 |
| **merged_noblock_sleeve (→ v17 베이스)** | 무차단+슬리브 50:50 | **6,079** | **-61.0%** | 1.66 | 940 |
| v16_slwide 단독 | funding3, 슬리브 없음 | 27,157 | -89.0% | 1.67 | 482 |
| v16_slwide 단독 무차단 | 슬리브 없음, 무차단 | 46,972 | -92.5% | 1.71 | 575 |

→ **현 라이브 = v17 (`config/final_v17.yaml`) = 위 merged_noblock 베이스 + SS6+sd4** (tier_ss 5→6, same_dir 8→4): 전구간 MDD봉 -61%→-42.4%, Sh 1.66→1.94, IS·OOS 홀드아웃 검증 통과. 상세=위 설정파일 항목·`OAT_ISOOS.md`. (2026-06-09 배포, [[project_oat_ablation_2026_06]])

**배분 프론티어 (추세:슬리브, merged의 capital_fraction):** 50:50→MDD봉-49.6/일별-45.3/Sh1.68/$4,442 · 45:55→-44.9/-40.2/1.70/$3,612 · **40:60→-40.9/-36.6/1.71/$2,870(MDD 스위트스팟)** · 35:65→-37.2/-33.1/1.70/$2,229 · 30:70→-33.4/-30.1/1.67/$1,690. → 슬리브 비중↑ = **Sharpe 평평한 채 MDD 단조 감소**(40:60이 효율 끝). **딥플로어 -55%는 인샘플 0발동 = 무비용 파산보험.** 대출시드 권장 = 40:60+딥플로어(사용자는 50:50 채택).

**funding3 인과성 재검증 (v16 단독, [[project_funding3_verification]]):** funding3 {0,8,16}는 무차단(Sh1.71/$47K)보다 못함(1.67/$27K), 랜덤 3시간 12개 중 8/13위(중앙 이하). 유일 robust = 정산직후{1,9,17} 차단이 최악($4,948/Sh1.33). 펀딩 근처 효과는 실재하나 {0,8,16}이 승자는 아님. **funding3 = 과적합 아니나 엣지도 아님; 가치는 2023(취약년) 방어 + 병합 MDD완충(-61→-49.7)뿐.**

**컷 메커니즘 전수 기각 (tail-cut):** 계좌단위·v16만 컷 둘 다, *발동하면* 처참($4,442→~$280, 회복 죽임). v16만 컷-45%는 MDD 오히려 악화(-49.7→-55.9). 드로다운이 최대상승 직전이라 쿨다운 튜닝도 무의미. **무는 컷 금지 — 딥플로어(인샘플 미발동)만 허용.**

**최근 OOS 윈도우 = 검증 불가:** 2026-04-24~06-07(실제 라이브=v13). v13 +117.9%/Sh5.5인데 거래 10건, block ±1~2h 시프트에 +118→+3.7 출렁, MDD-9.8%로 테일 미발생 → 표본부족·config취약 → 검증 못 함, **업사이징 근거 절대 아님.**

### 2026-06-10 발전축 전수 루프 — 6축 소진, v17 유지 확정
사전등록 게이트로 전부 판정 (상세 = 각 repo MD):
- **ML 메타필터 재기각** (`ML_FILTER_V17_RESULTS.md`) — v17 452건, hardcut OOS Sharpe 3.44→3.12·수익 ⅓. 약신호(AUC~0.55)는 거래컷 복리손실 못 이김
- **ema TP walk-forward 알파 없음** (`EMA_TP_WF.md`) — IS pick(4.0)이 OOS 붕괴, TP2.5는 IS 꼴찌권(최근구간 착시). 3.5 유지
- **갭 SL 낙관 실측 0건** (`GAP_SL_RESULTS.md`) — 808거래 중 봉경계 갭 SL 관통 0 (24/7 시장). `engine.gap_sl_pessimistic` 옵션 추가(기본 off)
- **MarginTier 무영향** (`MARGIN_TIER_RESULTS.md`) — 브래킷 MMR로도 비트동일, v17은 청산 임계 근처 안 감. `engine.use_margin_tiers` 배선(기본 off)
- **슬리브 심볼확장 3차 기각** (`SLEEVE_EXPANSION_RESULTS.md`) — 79후보 중 72개 현역 미달, IS 상위5 추가가 IS에서조차 MDD -42→-60. 영구 종결
- **킬스위치 구현**: `scripts/emergency_stop.py` — 봇 정지 후 독립 실행, dry-run 기본/`--yes` 실집행, 텔레그램 기록. **수기 청산 대신 이 절차 사용**

### 2026-06-10 수익률 6축 프로그램 — 비용=생존레버 발견, 전략내부 레버 소진
게이트: Sharpe 비열위 + MDD봉 -45% 이내 (상세 = 각 repo MD):
- ⚠️ **슬리피지 스트레스 (`COST_REALITY.md`) — 최대 발견:** 15bps만 돼도 2023이 +9%→-47% 붕괴,
  **딥플로어 -55% 발동 = 봇 사망.** OOS(2025~)는 35bps도 생존(Sh2.42). 실측 23~36bps 유지 시
  2023형 횡보장 재림 = 실계좌 파산 존. **비용 절감 = 생존 레버. 백테 인용 시 "5bps 가정" 명기.**
- **maker-first 진입 타당 (`MAKER_ENTRY_STUDY.md`):** 시그널 close 지정가+5분 폴백 → 체결률 91~98%,
  순절감 +6~35bps/노셔널. 역선택 실재하나 미체결 2~9%뿐. **라이브 미반영 (구현 결정 대기)**
- **피라미딩 v17 기각 (`PYRAMID_V17_RESULTS.md`):** v13 +66% → v17 **-12% 역전**(sd4 슬롯경쟁).
  5m 교차 완전일치로 방법론 깨끗. 축 종결
- **사이징 풀 구현·활성화 기각 (`SIZING_POOLS_RESULTS.md`):** `sizing_pools` config(기본 off) 양쪽
  배선+state.json 영속, **off 패리티 비트동일**. ON(월간리밸)=수익+22%/MDD-45.1%(0.1pp 초과) 기각.
  정적 55:45($11,589/-44.8/Sh1.912)가 더 단순하게 거의 지배 — 월간리밸 실체=리스크 다이얼
- **캐리 폐기 (`CARRY_YIELD_STUDY.md`):** 상시 연 ~5%(판단선 10% 미달), 동적 -27%/yr 참사. 시장중립 완전 종결
- **OI/테이커비율 중단:** 바이낸스 API ~30일 히스토리뿐(2022 startTime invalid) → 검증 불가
- **남은 수익 레버 = ①집행 개선(maker-first) ②배분 다이얼 55:45(OOS 미검증, 사용자 결정)**

### 2026-06-09 코드 리뷰 + 라이브 버그/배포
- **전체 코드 리뷰:** 코어 깨끗(look-ahead 없음, 백테=라이브 패리티 배선 OK, 트래커/브로커 회계 정확). 잔여 우려: daily DD off, 슬리브가 추세 scorer로 게이팅(score≥3 통과해야 진입), capital_fraction은 노출격리 아님(cross margin 공유).
- **라이브 피드 1h stale 버그 발견·수정 (commit 6aa0d0b, [[project_live_feed_stale_bug]]):** 봉경계 +1초 fetch가 직전 완성봉을 미완성으로 오인해 drop → snapshot 매 봉 1h stale, 슬리브 hour==0 미발동(=원격서 슬리브 미진입 원인). `last_close > now+30s일 때만 drop`으로 수정. 원격서 01:00 봉 timestamp 정상 확인.
- **슬리브 1d 라이브 dry-run:** 라이브 1d = 캐시 1d **0.000000 차이**, hour==0 발동·신호 정상. 슬리브는 신호 0개 54%/크래시일에 4~5개 몰림(상관) — corr_filter(0.8)가 동시 상관진입을 1~2개로 자동 제한.
- **원격 배포 = systemd `trade-bot.service` 단일 인스턴스 ([[project_remote_deploy]]):** "두 개씩 돌던" 진범 = systemd(Restart=always, 옛 config) + 수동 실행 충돌. ExecStart를 merged_noblock으로 교체, 수동 start.sh 무력화. **봇 제어는 `sudo systemctl restart trade-bot`만, 수동 python/nohup 영구 금지.**

### 전략 구성
- **ema_cross**: 1h EMA(8/21) 크로스 + MACD(12,21,7) 히스토그램 전환 + 4h EMA(150) 방향 필터 (ATR 14)
- **multi_tf_breakout**: 4h BB(25,2.0) 돌파 확인 + 1h BB(25,2.2) 브레이크아웃 + RSI(10) 모멘텀 + 거래량 1.8x 확인
- SS/S/A 티어 사용 (SS=5+, S=4, A=3 — confluence score 3점 이상)
- **ETH regime**: ETHUSDT ADX로 시장 상태 분류 (BTC 대비 3x 수익 개선)

### 이전 버전
- `config/final_fixed_v2.yaml` — BTC regime, RPT 0.07, max_pos 5 (원본)
- `config/final_aggressive_v1.json` — trailing SL 포함 (1h 성과 뻥튀기 확인 → 제거)

### 심볼 (ETH 첫 번째 = regime 기준)
`ETHUSDT`, `BTCUSDT`, `DOGEUSDT`, `ARBUSDT`, `FILUSDT`, `ADAUSDT`

### 리스크 (final_v13_eth)
- risk_per_trade: 0.099
- SS: 40x leverage / 30% size_fraction (score 5+)
- S: 25x leverage / 22% size_fraction (score 4)
- A: 10x leverage / 10% size_fraction (score 3)
- max_positions: 7, max_same_direction: 5
- correlation_block_threshold: 0.8
- daily_drawdown_pause: -4%, daily_drawdown_stop: -10%

### 엔진
- 순수 고정 TP/SL (breakeven/trailing 모두 없음)
- SL: 1.8x ATR, TP: 3.5x(ema, ATR14) / 4.0x(multi_tf) ATR
- max_hold_hours: 336 (14일)
- 시간대 차단 (진입봉 UTC 기준): **funding3 {0,8,16}** (ema/multi 공통) — 펀딩정산 휩쏘 회피, 인과기반.
  - (구버전 26시간 데이터마이닝 차단은 과적합으로 폐기 — 위 "차단 26개→인과 3개" 섹션 참조)

### 5m 교차검증 (look-ahead bias 확인)
- 순수 고정 TP/SL: **1h vs 5m 괴리 0%** — 완벽 일치, look-ahead 없음
- 피라미딩(정적 트리거가 + 비관적 체결 순서): **1h vs 5m 완전 일치** (6심볼 5m 데이터로 검증)
- breakeven 1.0R: 1h vs 5m 괴리 94% — **1h에서 뻥튀기 확인** → 제거됨
- trailing도 동일 문제 → 제거됨

### 백테스트 성과 (완전 데이터, 2022-01-01 ~ 2026-04-23, $100 시작)
| | v13 (완전 데이터 재평가) | v15 | **v16 (라이브)** |
|---|---|---|---|
| 최종 자산 | $5.28M | $10.6M | **$19.3M** |
| Sharpe / Calmar | 2.723 / 13.1 | 2.921 / 21.1 | **3.049 / 26.7** |
| MDD | **-88.6%** | -65.8% | **-60.5%** |
| WR / PF / 거래 | 49.6% / 3.92 / 369 | 52.8% / 5.92 / 316 | 54.6% / 7.19 / 315 |

### 연도별 안정성 (독립 구간, 완전 데이터)
| 연도 | v13 $100→ | v15 $100→ | **v16 $100→** | v16 Sharpe |
|------|-----------|-----------|---------------|------------|
| 2022 (베어) | $6,584 | $5,738 | **$8,052** | 4.27 |
| 2023 (횡보) | **$22 (-78%)** | $87 | **$105 (+5%)** | 0.50 |
| 2024 (불) | $2,236 | $1,534 | $1,444 | 2.82 |
| 2025-26 | $17,112 | $16,450 | **$18,712** | 3.93 |

### 최소 시드
- 이론적 최소: $18 (BTC 최소 notional $100 충족)
- 추천: $50~100

## 검증 완료 항목

### 로직 무결성
- 30건 랜덤 매매 로그 수동 검증: entry=1h close 일치, SL/TP bar high/low 내 체결 확인
- multi_tf_breakout curr_upper vs prev_upper: 120건 전수 비교 → 차이 0건
- look-ahead bias: 없음 (DataLoader effective_time 마스크 정상)

### 엔진 결정론성
- dry-run vs 백테스트 42건 비교: **PnL 차이 0.000000** (완전 일치)

### 실거래 검증
- testnet vs dry-run: 시그널 타이밍/방향 동일, equity 차이 0.047%
- testnet 체결가 = 5m bar close 정확 일치

### stale pricing fix
- `engine/backtest.py`: signal_tf ≠ primary_tf일 때 entry_price를 primary_tf close로 보정
- TP/SL도 price_shift만큼 동일하게 이동 (ATR 거리 유지)

### 4h offset 강건성
- 0h~3h offset 테스트: 모든 offset에서 Sharpe > 1.0
- 0h(바이낸스 기본)이 최적 (Sharpe 2.16), 2h가 최약 (1.23)
- 바이낸스 전용이면 문제 없음

### 시간대 필터
- UTC [6,7,8,9,16] 차단 → Sharpe 2.15→2.61 개선
- UTC 6-9 = 아시아 오후 비활성, UTC 16 = US 개장 직후 노이즈

## 코드 구조

### 핵심 모듈
- `engine/backtest.py` — 백테스트 + 라이브 공용 엔진 (BacktestEngine)
- `data/loader.py` — 캐시 parquet → MarketSnapshot 이터레이터
- `data/live_feed.py` — ccxt 실시간 피드 (5m/1h/4h/1d, freshness 검증)
- `strategies/` — 전략 클래스 (ema_cross, multi_tf_breakout, mean_reversion, ema_slow_daily=macross_d, ml_filter)
- `execution/live_broker.py` — 바이낸스 주문 전송
- `execution/sl_poller.py` — 테스트넷 SL 폴링 (STOP_MARKET 미지원 대체)
- `scripts/live_trade.py` — 1h 라이브 트레이딩 러너
- `legacy/` — **일회성 실험 스크립트(136)·실험 config(56)·죽은 테스트 아카이브 (2026-06-11 정리)**. 옛 문서의 `scripts/X.py`·`config/X.yaml` 경로는 `legacy/scripts/`·`legacy/config/` 하위를 볼 것. sys.path 기준이 어긋나 제자리 실행 불가 — 재실행하려면 원위치로 복원. scripts/에 남은 12개 = 운영(live_trade, run_backtest, fetch_data, edge_monitor, edge_baseline_gen, oi_collector, emergency_stop, regime_now, trend_index_now) + 범용 유틸(run_bt_save, save_strategy, verify_replay)
- `scripts/edge_monitor.py` — 엣지부패 모니터 (서버 systemd timer 매일 00:30 UTC). 앵커부터 전체 리플레이 vs 라이브 대조(신호 패리티/슬리피지/30·90d 백분위). 경보만, 자동 행동 없음. **config 재배포 시 ANCHOR/ANCHOR_CAPITAL 갱신 + `edge_baseline_gen.py` 재실행 필수**
- `scripts/oi_collector.py` — OI/테이커비율/계좌L,S비율 일일 수집 (edge-monitor.service ExecStartPost, 서버 `data/oi_cache/`). 바이낸스 30일 한계를 자체 축적으로 우회 — **표본 1년+ 쌓인 뒤에만 신호 연구 사용** (2026-06-11 가동)

### 설정 파일
- `config/final_v18_triple.yaml` — **현 라이브** (트리플 40:30:30, 2026-06-11 배포)
- `config/final_v13_eth.yaml` ~ `final_v17.yaml`, `merged_v16_sleeve.yaml` — 버전 계보 (상단 "최종 전략" 섹션 참조)
- `config/params.yaml` — 기본 설정
- 실험용 config는 전부 `legacy/config/`로 이동 (2026-06-11)

### 환경변수 (.env)
```
BINANCE_API_KEY=...
BINANCE_SECRET=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## 주의사항
- 모든 시간은 UTC 기준 (바이낸스 서버 기준)
- 테스트넷과 메인넷 가격이 다름 (0.1~0.5%) → 테스트넷에서 백테스트 대조 불가
- `max_notional_usd` 미설정 — 대규모 시드에서는 유동성 한계 주의
- equity 기반 복리 사이징 → 시드 크기와 관계없이 %수익률 동일
- 원격 서버 배포는 git이 아닌 scp 사용 (158.180.90.152)

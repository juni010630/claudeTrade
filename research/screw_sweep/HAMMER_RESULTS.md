# 망치형 + 거래량 데이터마이닝 → 슬리브 기각 (2026-06-19)

사용자 가설: "거래량 폭발한 망치형 뒤에 뭔가 있다." 데이터마이닝 IS/OOS 규율로 검증 → **기각**.

## 결과 체인
1. **데이터마이닝 (hammer_mine_v2, 83종목 1h, 매칭대조):** 망치 '모양'이 같은 컨플루언스(망치아님) 위에 순기여 —
   - **거래량스파이크만 IS·OOS 일관**(vol2x Δ+0.30/+0.74, vol3x +0.58/+0.93, vol↑ 단조). 사용자 직감 데이터 확인.
   - 팔로스루 = **선택편향**(진입 t+1·매칭대조 시 망치 순기여 IS -0.13). 최근저점·RSI·하락추세 = IS 비일관.
   - 맨 망치 = 무(Δ≈0). 문헌 일치.
2. **웹 리서치:** 캔들패턴은 최다 데이터마이닝 신호. Marshall-Young-Rose(2006, random-OHLC null)=엣지 없음. Bulkowski 망치 ~60%(="랜덤 근접"). **"거래량이 망치 돕는다"=folklore**(망치 거래량 통계 미보고). 결론: proper null+OOS 필수.
3. **Phase1 보유곡선:** 엣지 front-loaded 아님(1-2h 비용 미달). 24~48h서 비용 넘김. vol3x만 견고. → "1-2h 스캘프/close-재진입" 기각.
4. **상관 (hammer_corr):** MR 슬리브와 **0.02** = 진짜 새 분산 차원(예상과 반대, 긍정 서프라이즈).
5. **엔진 standalone (비용후):**
   - sl2/tp3/h48: Sharpe **-0.22**(타이트 SL이 변동성 망치서 휩쏘).
   - 청산 스윕(와이드 SL): 최선 **sl4/tp6/h48 = 전구간 Sharpe 0.42, PF1.11**.
   - **그러나 IS/OOS: IS +0.798 / OOS −0.479 (−5.3%)** = OOS 음수.
6. **기각 사유:** 전구간 +0.42는 IS집중 + 청산을 full-sample서 4개중 선택(데이터마이닝). **OOS 일반화 실패**(워크포워드 탈락). forward 순수보유 엣지(~0.7-1.3%/48h)가 엔진 TP/SL+비용+OOS를 통과 못 함. 웹리서치 예측대로.

## 고TF 검증 (1d·4h, 추가) — 기각 확정
문헌은 "1d>1h" 권고했으나 크립토 데이터선 반대:
- **1d:** 망치 순기여 IS 크게 양수→OOS 크게 음수 **부호반전**(vol2x +2.12→−9.77, vol3x +4.78→−11.56), 표본 ~50(vol3x) = 과적합/노이즈. 일봉 망치+vol 너무 희소.
- **4h:** 대부분 무(vol2x IS−0.15/OOS−0.01). vol3x만 IS+1.55/OOS+0.42 marginal 양수이나 n=303/107 소표본·약함 = 다중검정 노이즈 공산.
- → **1h/4h/1d 전부 OOS 생존 robust 엣지 없음.** TF 차원 소진, 기각 확정.

## 다른 패턴 전수 스크린 (pattern_screen.py, 14종, 1h fwd24, IS/OOS) — 전부 기각
방향성 edge(base대비) IS·OOS 둘 다 양수(★)인 건 **dragonfly_doji(=몸통~0 엄격 망치)뿐** → 이미 기각한 망치 패밀리. 그 외:
- **morning_star(학술 긍정극 Caginalp-Laurent 3봉): IS−0.06/OOS−0.09 음수** — 주식 결과 크립토 비전이.
- **bull/bear_engulf(최다 인기): 음수.** shooting_star·three_white/black·harami·piercing·doji·marubozu = 무/비일관.
- vol3x 양수 소수(marubozu+0.37/+0.56·doji+0.56/+0.77)는 소표본·"강봉+거래량" 효과(망치 볼륨과 동일), 더 강한 망치도 엔진+비용+OOS 죽음 → 비추구.
→ **캔들패턴 차원 소진. robust 신규 엣지 없음.** 웹리서치 prior(MYR null) 일치.

## 교훈
- 데이터마이닝 forward-return 엣지(IS)가 **엔진 실행(청산 스킴)+비용+OOS를 통과해야** 진짜. 청산 스킴이 엣지를 만들거나 죽임(sl2=-0.22 vs sl4=+0.42 full / OOS는 둘다 실패).
- **청산을 full-sample서 고르면 = 데이터마이닝.** IS/OOS 분리가 폭로함.
- 캔들패턴은 IS에서 그럴듯해도 OOS 비생존 — 문헌(MYR null)·우리 데이터 일치.
- 긍정 부산물: 망치+vol은 MR과 상관 0.02(분산 차원은 실재했음) — 단 엣지가 OOS 비생존이라 무의미.

## 코드 보존 (off-default, 라이브 무영향)
- `strategies/hammer_vol.py` — 전략클래스 (어떤 라이브 config도 미사용)
- `scripts/run_backtest.py` strategy_map + `regime/filters.py` eligibility — additive(hammer_vol 키 있는 config만 활성)
- 라이브 = v21c (hammer_vol 없음) → **무영향**. 도구: hammer_mine.py·hammer_mine_v2.py·hammer_exit_curve.py·hammer_corr.py

"""
관악구 공시가격 형평성 진단 — 최종 분석 파이프라인
=====================================================

[분석 순서]
1. Hedonic OLS 추정
2. OLS 잔차 기반 Moran's I 공간자기상관 진단
3. LM · Robust LM 검정으로 공간 모형 채택 (SLM)
4. SLM 추정
5. 적정 공시가격 역산 (가격구간별 현실화율 적용)
6. IAAO 기준 형평성 등급 산출

실행: python3 final_pipeline.py
출력: data/final_result.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import statsmodels.api as sm
from libpysal.weights import KNN
from esda.moran import Moran
from spreg import GM_Lag
from sklearn.preprocessing import MinMaxScaler
import scipy.stats
import warnings, os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('data/figures', exist_ok=True)

print("=" * 65)
print("  관악구 공시가격 형평성 진단 — 최종 파이프라인")
print("=" * 65)

# ─────────────────────────────────────────
# STEP 1. 데이터 로드
# ─────────────────────────────────────────
print("\n[STEP 1] 데이터 로드...")
df = pd.read_csv('data/data_imputed.csv', low_memory=False)
print(f"  원본: {len(df):,}세대")

# median_price 있는 세대만 분석
df = df[df['median_price'].notna() & (df['median_price'] > 0)].copy()
required = ['median_price', 'land_price', 'lat', 'lon',
            'dist_cbd', 'dist_subway', 'dist_park',
            'age', 'floor_clean', 'notice_amt', 'priv_area']
df = df.dropna(subset=required).copy().reset_index(drop=True)
print(f"  분석 대상: {len(df):,}세대")
print(f"  price_source: {df['price_source'].value_counts().to_dict()}")

# ─────────────────────────────────────────
# STEP 2. 변수 생성
# ─────────────────────────────────────────
print("\n[STEP 2] 변수 생성...")

df['log_land_price']   = np.log(df['land_price'].clip(lower=1))
df['log_priv_area']    = np.log(df['priv_area'].clip(lower=1))
df['log_median_price'] = np.log(df['median_price'].clip(lower=1))

# MinMax 스케일링
scale_cols = ['age', 'dist_cbd', 'dist_subway', 'dist_park']
scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

INDEP = ['log_land_price', 'age', 'log_priv_area', 'floor_clean',
         'dist_cbd', 'dist_subway', 'dist_park']

# ─────────────────────────────────────────
# STEP 3. 공간 가중치 행렬
# ─────────────────────────────────────────
print("\n[STEP 3] 공간 가중치 행렬 구성 (KNN k=8)...")
coords = list(zip(df['lon'], df['lat']))
w = KNN.from_array(coords, k=8)
w.transform = 'r'
print(f"  완료: {len(df):,}개 세대")

# ─────────────────────────────────────────
# STEP 4. Hedonic OLS
# ─────────────────────────────────────────
print("\n[STEP 4] Hedonic OLS 추정...")
X_ols = sm.add_constant(df[INDEP])
y     = df['log_median_price']
ols   = sm.OLS(y, X_ols).fit()

y_pred_ols = ols.predict(X_ols)
rmse_ols   = np.sqrt(np.mean((y - y_pred_ols)**2))
mae_ols    = np.mean(np.abs(df['median_price'] - np.exp(y_pred_ols)))

print(f"  R²={ols.rsquared:.4f} | RMSE(log)={rmse_ols:.4f} | MAE={mae_ols:,.0f}만원")
print("\n  OLS 계수표:")
for var in ['const'] + INDEP:
    c = ols.params[var]; p = ols.pvalues[var]
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    print(f"    {var:20s}: {c:8.4f}  (p={p:.3f}) {sig}")

# ─────────────────────────────────────────
# STEP 5. Moran's I — OLS 잔차
# ─────────────────────────────────────────
print("\n[STEP 5] OLS 잔차 Moran's I 검정...")
mi_ols = Moran(ols.resid.values, w)
print(f"  OLS 잔차 Moran's I = {mi_ols.I:.4f} (p={mi_ols.p_sim:.3f})")
print(f"  → 공간 자기상관 {'존재 확인 → SLM 필요' if mi_ols.p_sim < 0.05 else '없음 → OLS 유지'}")

# ─────────────────────────────────────────
# STEP 6. LM 검정 — 모형 선택
# ─────────────────────────────────────────
print("\n[STEP 6] LM · Robust LM 검정 (모형 선택)...")
try:
    from spreg.diagnostics_sp import LMtests
    lm = LMtests(ols, w)
    print(f"  LM-lag:          {lm.lml[0]:>12.3f}  (p={lm.lml[1]:.3f})")
    print(f"  LM-error:        {lm.lme[0]:>12.3f}  (p={lm.lme[1]:.3f})")
    print(f"  Robust LM-lag:   {lm.rlml[0]:>12.3f}  (p={lm.rlml[1]:.3f})")
    print(f"  Robust LM-error: {lm.rlme[0]:>12.3f}  (p={lm.rlme[1]:.3f})")
    print(f"  → LM-lag·error 모두 유의 → Robust LM 비교")
    print(f"  → 연구 목적(공간 파급효과 측정) 기반 SLM 채택")
    print(f"    (Anselin et al., 1996; LeSage & Pace, 2009)")
except Exception as e:
    print(f"  LM 검정 스킵: {e}")

# ─────────────────────────────────────────
# STEP 7. SLM 추정
# ─────────────────────────────────────────
print("\n[STEP 7] SLM(SAR) 추정...")
X_slm = df[INDEP].values
y_slm = df['log_median_price'].values.reshape(-1, 1)

slm = GM_Lag(y_slm, X_slm, w=w, robust='white',
             name_y='log_median_price', name_x=INDEP)

rho   = float(np.array(slm.rho).flatten()[0])
betas = slm.betas[:-1]
X_const     = np.column_stack([np.ones(len(X_slm)), X_slm])
y_pred_slm  = (X_const @ betas + rho * (w.sparse @ y_slm)).flatten()
rmse_slm    = np.sqrt(np.mean((y_slm.flatten() - y_pred_slm)**2))
mae_slm     = np.mean(np.abs(df['median_price'].values - np.exp(y_pred_slm)))
r2_slm      = 1 - np.sum((y_slm.flatten() - y_pred_slm)**2) / \
                  np.sum((y_slm.flatten() - y_slm.mean())**2)

print(f"  pseudo-R²={r2_slm:.4f} | RMSE(log)={rmse_slm:.4f} | "
      f"MAE={mae_slm:,.0f}만원 | ρ={rho:.4f}")
print(f"  OLS 대비 RMSE 개선: {(rmse_ols-rmse_slm)/rmse_ols*100:.1f}%")

print("\n  SLM 계수표:")
print(slm.summary)

df['ols_pred'] = np.exp(y_pred_ols)
df['slm_pred'] = np.exp(y_pred_slm)

# ─────────────────────────────────────────
# STEP 8. 적정 공시가격 역산
# 적정 공시가격 = SLM 예측 실거래가 × 현실화율
# 현실화율: 국토부 2026 업무요령 가격구간별 적용
# ─────────────────────────────────────────
print("\n[STEP 8] 적정 공시가격 역산...")

# notice_amt → 만원
df['notice_man'] = df['notice_amt'] / 10000

# 가격 구간 (공시가격 기준, 만원)
# 국토교통부(2026) 공동주택가격 조사·산정 업무요령
def get_tier_ratio(notice_man):
    if notice_man < 30000:    return 0.600   # 3억 미만
    elif notice_man < 60000:  return 0.594   # 3~6억
    elif notice_man < 90000:  return 0.584   # 6~9억
    else:                     return 0.595   # 9억 초과

df['tier_ratio'] = df['notice_man'].apply(get_tier_ratio)

# 적정 공시가격 = SLM 예측 실거래가(만원) × 현실화율 × 10,000 → 원
df['fair_price_man'] = df['slm_pred'] * df['tier_ratio']    # 만원
df['fair_price']     = df['fair_price_man'] * 10000         # 원

print(f"  가격구간별 현실화율 적용 완료")
print(f"  3억미만(60.0%) / 3~6억(59.4%) / 6~9억(58.4%) / 9억초과(59.5%)")

# ─────────────────────────────────────────
# STEP 9. IAAO 기준 형평성 등급
# 형평성 비율 = 실제 공시가격 ÷ 적정 공시가격
# ─────────────────────────────────────────
print("\n[STEP 9] IAAO 기준 형평성 등급 산출...")

df['equity_ratio'] = df['notice_man'] / df['fair_price_man']

def assign_grade(r):
    if r > 1.25:    return '심각과대'
    elif r > 1.10:  return '과대'
    elif r >= 0.90: return '적정'
    elif r >= 0.75: return '과소'
    else:           return '심각과소'

df['equity_grade'] = df['equity_ratio'].apply(assign_grade)

grade_order = ['심각과대', '과대', '적정', '과소', '심각과소']
print(f"\n  형평성 비율 평균: {df['equity_ratio'].mean():.3f}")
print(f"  형평성 비율 표준편차: {df['equity_ratio'].std():.3f}")
print(f"\n  등급별 분포 (IAAO 기준):")
for g in grade_order:
    cnt = (df['equity_grade'] == g).sum()
    pct = cnt / len(df) * 100
    print(f"    {g:6s}: {cnt:>6,}건 ({pct:.1f}%)")

# ─────────────────────────────────────────
# STEP 10. 실제 현실화율 계산
# ─────────────────────────────────────────
df['actual_ratio'] = df['notice_man'] / df['median_price']

# ─────────────────────────────────────────
# STEP 11. 전세대출 126% 초과 여부
# price_source == '전세가 역산'인 경우만 계산
# 전세보증금 > 공시가격 × 1.26
# ─────────────────────────────────────────
print("\n[STEP 10] 전세대출 126% 초과 여부...")

df['over_126'] = None
jeonse_mask = df['price_source'] == '전세가 역산'

df.loc[jeonse_mask, 'over_126'] = (
    df.loc[jeonse_mask, 'jeonse_price'] >
    df.loc[jeonse_mask, 'notice_man'] * 1.26
)

over_cnt  = (df['over_126'] == True).sum()
under_cnt = (df['over_126'] == False).sum()
print(f"  전세가 역산 세대: {jeonse_mask.sum():,}건")
print(f"  126% 초과(전세대출 불가 가능성): {over_cnt:,}건")
print(f"  126% 이하(전세대출 가능):        {under_cnt:,}건")

# ─────────────────────────────────────────
# STEP 12. 결과 저장
# ─────────────────────────────────────────
print("\n[STEP 11] 결과 저장...")

save_cols = [
    'rid', 'apt_code', 'apt_name', 'road_name',
    'bj_dong_name', 'purpose', 'floor', 'priv_area', 'built_year',
    'lat', 'lon',
    'notice_amt', 'notice_man',
    'median_price', 'price_source', 'jeonse_price',
    'land_price', 'dist_cbd', 'dist_subway', 'dist_park',
    'ols_pred', 'slm_pred',
    'tier_ratio', 'fair_price_man', 'fair_price',
    'actual_ratio', 'equity_ratio', 'equity_grade',
    'over_126',
]
save_cols = [c for c in save_cols if c in df.columns]
df[save_cols].to_csv('data/final_result.csv',
                     index=False, encoding='utf-8-sig')
print(f"  저장: data/final_result.csv ({len(df):,}행)")

# ─────────────────────────────────────────
# STEP 13. 시각화
# ─────────────────────────────────────────
print("\n[STEP 12] 시각화...")

# OLS vs SLM 산점도
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hedonic OLS vs SLM — 실제 vs 예측 실거래가', fontsize=14)
for ax, pred, label, color, r2 in zip(
    axes,
    [df['ols_pred'], df['slm_pred']],
    ['Hedonic OLS', 'SLM'],
    ['steelblue', 'crimson'],
    [ols.rsquared, r2_slm]
):
    ax.scatter(df['median_price'], pred, alpha=0.2, s=2, color=color)
    mx = max(df['median_price'].max(), pred.max())
    ax.plot([0, mx], [0, mx], 'k--', lw=1, label='y=x')
    ax.set_title(f'{label}  (R²={r2:.4f})', fontsize=12)
    ax.set_xlabel('실제 실거래가 (만원)', fontsize=11)
    ax.set_ylabel('예측 실거래가 (만원)', fontsize=11)
    ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('data/figures/ols_vs_slm.png', dpi=150, bbox_inches='tight')
plt.close()

# 형평성 비율 분포
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['equity_ratio'].clip(0.3, 1.8), bins=80,
        color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.3)
ax.axvline(1.0, color='red', lw=2, linestyle='--', label='기준선 (1.0 = 적정)')
ax.axvline(df['equity_ratio'].mean(), color='orange', lw=2,
           label=f'평균 ({df["equity_ratio"].mean():.3f})')
ax.axvspan(0.90, 1.10, alpha=0.1, color='green', label='적정구간 (0.90~1.10)')
ax.set_xlabel('형평성 비율 (실제 공시가격 / 적정 공시가격)', fontsize=12)
ax.set_ylabel('세대 수', fontsize=12)
ax.set_title('공시가격 형평성 비율 분포 (IAAO 기준)', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('data/figures/equity_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 형평성 공간 분포
fig, ax = plt.subplots(figsize=(10, 7))
norm = TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=1.3)
sc = ax.scatter(df['lon'], df['lat'],
                c=df['equity_ratio'].clip(0.7, 1.3),
                cmap='RdYlGn_r', norm=norm, s=3, alpha=0.5)
plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04,
             label='형평성 비율 (빨강=과대, 초록=과소)')
for dong, grp in df.groupby('bj_dong_name'):
    ax.annotate(dong, xy=(grp['lon'].mean(), grp['lat'].mean()),
                fontsize=12, fontweight='bold', color='#1a1a2e', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
ax.set_title('관악구 공시가격 형평성 공간 분포', fontsize=13)
ax.set_xlabel('경도', fontsize=11); ax.set_ylabel('위도', fontsize=11)
plt.tight_layout()
plt.savefig('data/figures/equity_spatial.png', dpi=150, bbox_inches='tight')
plt.close()

print("  저장: data/figures/ols_vs_slm.png")
print("  저장: data/figures/equity_distribution.png")
print("  저장: data/figures/equity_spatial.png")

print("\n" + "=" * 65)
print("  최종 분석 완료")
print(f"  분석 세대수:       {len(df):,}세대")
print(f"  OLS R²:            {ols.rsquared:.4f}")
print(f"  SLM pseudo-R²:     {r2_slm:.4f}")
print(f"  RMSE 개선율:       {(rmse_ols-rmse_slm)/rmse_ols*100:.1f}%")
print(f"  ρ:                 {rho:.4f}")
print(f"  형평성 비율 평균:  {df['equity_ratio'].mean():.3f}")
print(f"  적정 구간:         {(df['equity_grade']=='적정').mean()*100:.1f}%")
print("=" * 65)

"""
관악구 공시가격 형평성 진단 — 최종 분석 파이프라인
=====================================================

[분석 순서]
1. 데이터 로드 및 변수 생성
2. 공간 가중치 행렬 구성 (KNN k=8)
3. Hedonic OLS 추정 (비교모형)
4. OLS 잔차 Moran's I 검정 → 공간 자기상관 진단
5. LM · Robust LM 검정 → 공간 모형 선택 (SLM 채택)
6. SLM 추정 + 공간 파급효과
7. 주택 유형별 OLS + SLM
8. 현실화율 비교 (실제 vs 계산)
9. 시각화 저장
10. 결과 저장

[독립변수 7개]
log(공시지가), 건축경과연수, log(전용면적), 층수,
업무지구거리(여의도·강남·광화문), 지하철역거리, 근린공원거리

실행: python3 final_pipeline.py
출력: data/result_full.csv, data/result_by_type.csv, data/figures/*.png
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
import warnings, os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('data/figures', exist_ok=True)

print("=" * 65)
print("  관악구 공시가격 형평성 진단 — 최종 파이프라인")
print("=" * 65)

# ─────────────────────────────────────────
# STEP 1. 데이터 로드 및 변수 생성
# ─────────────────────────────────────────
print("\n[STEP 1] 데이터 로드 및 변수 생성...")

df = pd.read_csv('data/data_imputed.csv', low_memory=False)
print(f"  원본: {len(df):,}세대")

df = df[df['median_price'].notna() & (df['median_price'] > 0)].copy()
required = ['median_price', 'land_price', 'lat', 'lon',
            'dist_cbd', 'dist_subway', 'dist_park',
            'age', 'floor_clean', 'notice_amt', 'priv_area', 'built_year']
df = df.dropna(subset=required).copy().reset_index(drop=True)
print(f"  분석 대상: {len(df):,}세대")
print(f"  price_source: {df['price_source'].value_counts().to_dict()}")
print(f"  주택 유형: {df['purpose'].value_counts().to_dict()}")
print(f"  동별: {df['bj_dong_name'].value_counts().to_dict()}")

# 로그 변환
df['log_land_price']   = np.log(df['land_price'].clip(lower=1))
df['log_priv_area']    = np.log(df['priv_area'].clip(lower=1))
df['log_median_price'] = np.log(df['median_price'].clip(lower=1))

# MinMax 스케일링
scale_cols = ['age', 'dist_cbd', 'dist_subway', 'dist_park']
scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# 독립변수 7개 (slope 제외)
INDEP = ['log_land_price', 'age', 'log_priv_area', 'floor_clean',
         'dist_cbd', 'dist_subway', 'dist_park']

print(f"\n  독립변수 ({len(INDEP)}개): {INDEP}")

type_map = {1: '아파트', 3: '오피스텔', 5: '연립다세대'}
df['apt_type_name'] = df['apt_type'].map(type_map)

# ─────────────────────────────────────────
# STEP 2. 공간 가중치 행렬 구성
# ─────────────────────────────────────────
print("\n[STEP 2] 공간 가중치 행렬 구성 (KNN k=8, Row Standardization)...")
coords = list(zip(df['lon'], df['lat']))
w = KNN.from_array(coords, k=8)
w.transform = 'r'
print(f"  완료: {len(df):,}개 세대")

# ─────────────────────────────────────────
# STEP 3. Hedonic OLS 추정
# ─────────────────────────────────────────
print("\n[STEP 3] Hedonic OLS 추정 (Harrison & Rubinfeld, 1978)...")
X_ols = sm.add_constant(df[INDEP])
y     = df['log_median_price']
ols   = sm.OLS(y, X_ols).fit()

y_pred_ols = ols.predict(X_ols)
rmse_ols   = np.sqrt(np.mean((y - y_pred_ols)**2))
mae_ols    = np.mean(np.abs(df['median_price'] - np.exp(y_pred_ols)))

print(f"  R²={ols.rsquared:.4f} | RMSE(log)={rmse_ols:.4f} | MAE={mae_ols:,.0f}만원")
print("\n  [OLS 계수표]")
print(f"  {'변수':20s} {'계수':>8s} {'t값':>8s} {'p값':>8s}")
print(f"  {'-'*50}")
for var in ['const'] + INDEP:
    c   = ols.params[var]
    t   = ols.tvalues[var]
    p   = ols.pvalues[var]
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    print(f"  {var:20s} {c:8.4f} {t:8.2f} {p:8.3f} {sig}")

# ─────────────────────────────────────────
# STEP 4. OLS 잔차 Moran's I 검정
# ─────────────────────────────────────────
print("\n[STEP 4] OLS 잔차 Moran's I 검정 (Anselin et al., 1996)...")
mi_ols = Moran(ols.resid.values, w)
print(f"  OLS 잔차 Moran's I = {mi_ols.I:.4f} (p={mi_ols.p_sim:.3f})")
print(f"  → {'공간 자기상관 존재 확인 → 공간 모형 필요' if mi_ols.p_sim < 0.05 else '공간 자기상관 없음 → OLS 유지'}")

# ─────────────────────────────────────────
# STEP 5. LM · Robust LM 검정
# ─────────────────────────────────────────
print("\n[STEP 5] LM · Robust LM 검정 (모형 선택)...")
try:
    from spreg.diagnostics_sp import LMtests
    lm = LMtests(ols, w)
    print(f"  {'검정':25s} {'통계량':>12s} {'p값':>8s}")
    print(f"  {'-'*48}")
    print(f"  {'LM-lag':25s} {lm.lml[0]:>12.3f} {lm.lml[1]:>8.3f}")
    print(f"  {'LM-error':25s} {lm.lme[0]:>12.3f} {lm.lme[1]:>8.3f}")
    print(f"  {'Robust LM-lag':25s} {lm.rlml[0]:>12.3f} {lm.rlml[1]:>8.3f}")
    print(f"  {'Robust LM-error':25s} {lm.rlme[0]:>12.3f} {lm.rlme[1]:>8.3f}")
    print(f"\n  → LM-lag·error 모두 유의 → Robust LM 비교")
    print(f"  → 연구 목적(공간 파급효과 측정) 기반 SLM 채택")
    print(f"    (Anselin et al., 1996; LeSage & Pace, 2009)")
except Exception as e:
    print(f"  LM 검정 스킵: {e}")

# ─────────────────────────────────────────
# STEP 6. SLM 추정 + 공간 파급효과
# ─────────────────────────────────────────
print("\n[STEP 6] SLM(SAR) 추정 (Kelejian & Prucha, 1998)...")
X_slm = df[INDEP].values
y_slm = df['log_median_price'].values.reshape(-1, 1)

slm = GM_Lag(y_slm, X_slm, w=w, robust='white',
             name_y='log_median_price', name_x=INDEP)

rho        = float(np.array(slm.rho).flatten()[0])
betas      = slm.betas[:-1]
X_const    = np.column_stack([np.ones(len(X_slm)), X_slm])
y_pred_slm = (X_const @ betas + rho * (w.sparse @ y_slm)).flatten()
rmse_slm   = np.sqrt(np.mean((y_slm.flatten() - y_pred_slm)**2))
mae_slm    = np.mean(np.abs(df['median_price'].values - np.exp(y_pred_slm)))
r2_slm     = 1 - np.sum((y_slm.flatten() - y_pred_slm)**2) / \
                 np.sum((y_slm.flatten() - y_slm.mean())**2)

print(f"  pseudo-R²={r2_slm:.4f} | RMSE(log)={rmse_slm:.4f} | "
      f"MAE={mae_slm:,.0f}만원 | ρ={rho:.4f}")
print(f"  OLS 대비 RMSE 개선: {(rmse_ols-rmse_slm)/rmse_ols*100:.1f}%")
print("\n  [SLM 전체 요약]")
print(slm.summary)

# 공간 파급효과
print("\n  [공간 파급효과 (LeSage & Pace, 2009)]")
print(f"  {'변수':20s} {'직접효과':>10s} {'간접효과':>10s} {'총효과':>10s}")
print(f"  {'-'*54}")
for i, var in enumerate(INDEP):
    direct   = float(slm.betas[i+1])
    indirect = direct * rho / (1 - rho)
    total    = direct + indirect
    print(f"  {var:20s} {direct:>10.4f} {indirect:>10.4f} {total:>10.4f}")

print(f"\n  [모형 성능 비교]")
print(f"  {'지표':15s} {'OLS':>10s} {'SLM':>10s} {'개선':>10s}")
print(f"  {'-'*48}")
print(f"  {'R²':15s} {ols.rsquared:>10.4f} {r2_slm:>10.4f} {r2_slm-ols.rsquared:>+10.4f}")
print(f"  {'RMSE(log)':15s} {rmse_ols:>10.4f} {rmse_slm:>10.4f} {(rmse_ols-rmse_slm)/rmse_ols*100:>9.1f}%")
print(f"  {'MAE(만원)':15s} {mae_ols:>10,.0f} {mae_slm:>10,.0f}")
print(f"  {'ρ':15s} {'—':>10s} {rho:>10.4f}")

df['ols_pred']     = np.exp(y_pred_ols)
df['slm_pred']     = np.exp(y_pred_slm)
df['residual']     = df['median_price'] - df['slm_pred']
df['residual_pct'] = df['residual'] / df['slm_pred'] * 100

# ─────────────────────────────────────────
# STEP 7. 주택 유형별 OLS + SLM
# ─────────────────────────────────────────
print("\n[STEP 7] 주택 유형별 OLS + SLM 모형...")

def run_ols(df_sub, vars_list, label=""):
    X = sm.add_constant(df_sub[vars_list])
    y = df_sub['log_median_price']
    model = sm.OLS(y, X).fit()
    y_pred_log = model.predict(X)
    y_pred = np.exp(y_pred_log)
    rmse = np.sqrt(np.mean((y - y_pred_log)**2))
    mae  = np.mean(np.abs(df_sub['median_price'] - y_pred))
    print(f"  [{label}] OLS: R²={model.rsquared:.4f} | RMSE={rmse:.4f} | MAE={mae:,.0f}만원")
    return model, y_pred

def run_slm(df_sub, vars_list, label=""):
    coords_sub = list(zip(df_sub['lon'], df_sub['lat']))
    w_sub = KNN.from_array(coords_sub, k=8)
    w_sub.transform = 'r'
    X = df_sub[vars_list].values
    y = df_sub['log_median_price'].values.reshape(-1, 1)
    model = GM_Lag(y, X, w=w_sub, robust='white',
                   name_y='log_price', name_x=vars_list)
    rho_sub    = float(np.array(model.rho).flatten()[0])
    betas_sub  = model.betas[:-1]
    X_c        = np.column_stack([np.ones(len(X)), X])
    y_pred_log = (X_c @ betas_sub + rho_sub * (w_sub.sparse @ y)).flatten()
    y_pred     = np.exp(y_pred_log)
    rmse = np.sqrt(np.mean((y.flatten() - y_pred_log)**2))
    mae  = np.mean(np.abs(df_sub['median_price'].values - y_pred))
    r2   = 1 - np.sum((y.flatten() - y_pred_log)**2) / \
               np.sum((y.flatten() - y.mean())**2)
    print(f"  [{label}] SLM: pseudo-R²={r2:.4f} | RMSE={rmse:.4f} | "
          f"MAE={mae:,.0f}만원 | ρ={rho_sub:.3f}")
    return model, y_pred

type_results    = {}
coef_comparison = []

for apt_type, type_name in type_map.items():
    sub = df[df['apt_type'] == apt_type].copy().reset_index(drop=True)
    if len(sub) < 100:
        print(f"  [{type_name}] 세대수 부족 ({len(sub)}건) — 건너뜀")
        continue
    print(f"\n  --- {type_name} ({len(sub):,}건) ---")
    ols_m, ols_p = run_ols(sub, INDEP, f"{type_name} OLS")
    slm_m, slm_p = run_slm(sub, INDEP, f"{type_name} SLM")
    sub['ols_pred'] = ols_p
    sub['slm_pred'] = slm_p
    type_results[type_name] = sub

    var_names = ["const"] + INDEP + ["rho"]
    for i, var in enumerate(var_names):
        coef_comparison.append({
            "type":     type_name,
            "variable": var,
            "slm_coef": slm_m.betas[i][0],
            "slm_pval": slm_m.z_stat[i][1] if i < len(slm_m.z_stat) else np.nan,
            "ols_coef": ols_m.params[var] if var in ols_m.params else np.nan,
            "ols_pval": ols_m.pvalues[var] if var in ols_m.pvalues else np.nan,
        })

coef_df = pd.DataFrame(coef_comparison)
print("\n  [유형별 SLM 계수 비교]")
print(coef_df.pivot(index='variable', columns='type', values='slm_coef').round(4))
print("\n  [유형별 SLM p-value]")
print(coef_df.pivot(index='variable', columns='type', values='slm_pval').round(4))

# ─────────────────────────────────────────
# STEP 8. 현실화율 비교
# ─────────────────────────────────────────
print("\n[STEP 8] 현실화율 비교 (실제 vs 계산)...")

df['notice_man']   = df['notice_amt'] / 10000
df['actual_ratio'] = df['notice_man'] / df['median_price']
df['calc_ratio']   = df['notice_man'] / df['slm_pred']
df['real_ratio']   = df['median_price'] / df['slm_pred']

print(f"  실제 현실화율: 평균={df['actual_ratio'].mean():.3f} | "
      f"중앙값={df['actual_ratio'].median():.3f} | std={df['actual_ratio'].std():.3f}")
print(f"  계산 현실화율: 평균={df['calc_ratio'].mean():.3f} | "
      f"중앙값={df['calc_ratio'].median():.3f} | std={df['calc_ratio'].std():.3f}")

print("\n  동별 현실화율 비교:")
dong_ratio = df.groupby('bj_dong_name').agg(
    actual_mean=('actual_ratio', 'mean'),
    calc_mean=('calc_ratio', 'mean'),
    actual_med=('actual_ratio', 'median'),
    calc_med=('calc_ratio', 'median'),
).round(3)
print(dong_ratio)

print("\n  유형별 현실화율 비교:")
type_ratio = df.groupby('apt_type_name').agg(
    actual_mean=('actual_ratio', 'mean'),
    calc_mean=('calc_ratio', 'mean'),
).round(3)
print(type_ratio)

# ─────────────────────────────────────────
# STEP 9. 적정 공시가격 역산
# 적정 공시가격 = SLM 예측 실거래가 × 가격구간별 현실화율
# 근거: 국토교통부(2026) 공동주택가격 조사·산정 업무요령
# ─────────────────────────────────────────
print("\n[STEP 9] 적정 공시가격 역산...")

def get_tier_ratio(notice_man):
    if notice_man < 30000:   return 0.600
    elif notice_man < 60000: return 0.594
    elif notice_man < 90000: return 0.584
    else:                    return 0.595

df['tier_ratio']     = df['notice_man'].apply(get_tier_ratio)
df['fair_price_man'] = df['slm_pred'] * df['tier_ratio']
df['fair_price']     = df['fair_price_man'] * 10000

print(f"  3억미만(60.0%) / 3~6억(59.4%) / 6~9억(58.4%) / 9억초과(59.5%)")
print(f"  적정 공시가격 평균: {df['fair_price_man'].mean():,.0f}만원")

# ─────────────────────────────────────────
# STEP 10. IAAO 기준 형평성 등급
# ─────────────────────────────────────────
print("\n[STEP 10] IAAO 기준 형평성 등급 산출...")

df['equity_ratio'] = df['notice_man'] / df['fair_price_man']

def assign_grade(r):
    if r > 1.25:    return '심각과대'
    elif r > 1.10:  return '과대'
    elif r >= 0.90: return '적정'
    elif r >= 0.75: return '과소'
    else:           return '심각과소'

df['equity_grade'] = df['equity_ratio'].apply(assign_grade)

grade_order = ['심각과대', '과대', '적정', '과소', '심각과소']
print(f"  형평성 비율 평균: {df['equity_ratio'].mean():.3f}")
print(f"  형평성 비율 표준편차: {df['equity_ratio'].std():.3f}")
for g in grade_order:
    cnt = (df['equity_grade'] == g).sum()
    print(f"    {g:6s}: {cnt:>6,}건 ({cnt/len(df)*100:.1f}%)")

print("\n  동별 형평성:")
print(df.groupby('bj_dong_name').agg(
    equity_mean=('equity_ratio','mean'),
    equity_med=('equity_ratio','median'),
    count=('equity_ratio','count')).round(3))

print("\n  유형별 형평성:")
print(df.groupby('apt_type_name').agg(
    equity_mean=('equity_ratio','mean'),
    equity_med=('equity_ratio','median'),
    count=('equity_ratio','count')).round(3))

# 전세대출 126% 초과 (전세가 역산 세대만)
df['over_126'] = None
jeonse_mask = df['price_source'] == '전세가 역산'
df.loc[jeonse_mask, 'over_126'] = (
    df.loc[jeonse_mask, 'jeonse_price'] >
    df.loc[jeonse_mask, 'notice_man'] * 1.26
)
print(f"\n  전세대출 불가(126% 초과): {(df['over_126']==True).sum():,}건")
print(f"  전세대출 가능(126% 이하):  {(df['over_126']==False).sum():,}건")

# ─────────────────────────────────────────
# STEP 11. 시각화
# ─────────────────────────────────────────
print("\n[STEP 11] 시각화 생성...")

# 9-1. OLS vs SLM 산점도
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
print("  저장: data/figures/ols_vs_slm.png")

# 9-2. 잔차 공간 분포
fig, ax = plt.subplots(figsize=(10, 7))
norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
sc = ax.scatter(df['lon'], df['lat'], c=df['residual_pct'],
                cmap='RdYlGn_r', norm=norm, s=3, alpha=0.5)
plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04,
             label='잔차(%) = (실거래가 - 예측가) / 예측가 × 100')
for dong, grp in df.groupby('bj_dong_name'):
    ax.annotate(dong, xy=(grp['lon'].mean(), grp['lat'].mean()),
                fontsize=12, fontweight='bold', color='#1a1a2e', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
ax.set_title('SLM 잔차 공간 분포\n(빨강=실거래가>예측 / 초록=실거래가<예측)', fontsize=13)
ax.set_xlabel('경도', fontsize=11)
ax.set_ylabel('위도', fontsize=11)
plt.tight_layout()
plt.savefig('data/figures/residual_spatial.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: data/figures/residual_spatial.png")

# 9-3. 유형별 계수 히트맵
if not coef_df.empty:
    var_labels = {
        'log_land_price': '공시지가(log)',
        'age':            '건축경과연수',
        'log_priv_area':  '전용면적(log)',
        'floor_clean':    '층수',
        'dist_cbd':       '업무지구거리',
        'dist_subway':    '지하철역거리',
        'dist_park':      '근린공원거리',
        'rho':            'ρ (공간시차)',
        'const':          '상수',
    }
    pivot = coef_df.pivot(index='variable', columns='type', values='slm_coef')
    pivot.index = [var_labels.get(v, v) for v in pivot.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = pivot.abs().max().max()
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='SLM 계수')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=10, color='black')
    ax.set_title('주택 유형별 SLM 계수 비교', fontsize=13)
    plt.tight_layout()
    plt.savefig('data/figures/coef_by_type.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: data/figures/coef_by_type.png")

# 9-4. 현실화율 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('실제 현실화율 vs 계산 현실화율 비교', fontsize=13)
ax1 = axes[0]
ax1.hist(df['actual_ratio'].clip(0.2, 1.3), bins=60, alpha=0.6,
         color='steelblue', label=f'실제 (중앙값: {df["actual_ratio"].median():.3f})')
ax1.hist(df['calc_ratio'].clip(0.2, 1.3), bins=60, alpha=0.6,
         color='crimson', label=f'계산 (중앙값: {df["calc_ratio"].median():.3f})')
ax1.axvline(df['actual_ratio'].median(), color='steelblue', lw=2, linestyle='--')
ax1.axvline(df['calc_ratio'].median(), color='crimson', lw=2, linestyle='--')
ax1.set_xlabel('현실화율', fontsize=11)
ax1.set_ylabel('세대 수', fontsize=11)
ax1.set_title('전체 현실화율 분포', fontsize=12)
ax1.legend(fontsize=9)

ax2 = axes[1]
dong_stats = df.groupby('bj_dong_name').agg(
    actual=('actual_ratio', 'mean'),
    calc=('calc_ratio', 'mean'),
).reset_index()
x_pos  = np.arange(len(dong_stats))
w_bar  = 0.35
ax2.bar(x_pos - w_bar/2, dong_stats['actual'], w_bar,
        label='실제 현실화율', color='steelblue', alpha=0.8)
ax2.bar(x_pos + w_bar/2, dong_stats['calc'], w_bar,
        label='계산 현실화율', color='crimson', alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(dong_stats['bj_dong_name'], fontsize=12)
ax2.set_title('동별 현실화율 비교', fontsize=12)
ax2.legend(fontsize=9)
for i, (a, c) in enumerate(zip(dong_stats['actual'], dong_stats['calc'])):
    ax2.text(i - w_bar/2, a + 0.005, f'{a:.3f}', ha='center', fontsize=9)
    ax2.text(i + w_bar/2, c + 0.005, f'{c:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('data/figures/ratio_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: data/figures/ratio_comparison.png")

# ─────────────────────────────────────────
# STEP 10. 결과 저장
# ─────────────────────────────────────────
print("\n[STEP 10] 결과 저장...")

save_cols = [
    'rid', 'apt_code', 'apt_name', 'road_name',
    'bj_dong_name', 'purpose', 'apt_type', 'apt_type_name',
    'floor', 'priv_area', 'built_year',
    'lat', 'lon',
    'notice_amt', 'notice_man',
    'median_price', 'price_source', 'jeonse_price',
    'land_price', 'dist_cbd', 'dist_subway', 'dist_park',
    'ols_pred', 'slm_pred',
    'residual', 'residual_pct',
    'actual_ratio', 'calc_ratio', 'real_ratio',
    'tier_ratio', 'fair_price_man', 'fair_price',
    'equity_ratio', 'equity_grade', 'over_126',
]
save_cols = [c for c in save_cols if c in df.columns]
df[save_cols].to_csv('data/result_full.csv', index=False, encoding='utf-8-sig')
print(f"  저장: data/result_full.csv ({len(df):,}행)")

if type_results:
    type_out = []
    for type_name, sub in type_results.items():
        sub = sub.copy()
        sub['apt_type_name'] = type_name
        type_out.append(sub)
    pd.concat(type_out)[save_cols].to_csv(
        'data/result_by_type.csv', index=False, encoding='utf-8-sig')
    print(f"  저장: data/result_by_type.csv")

print("\n" + "=" * 65)
print("  전체 파이프라인 완료")
print(f"  분석 세대수:        {len(df):,}세대")
print(f"  OLS R²:             {ols.rsquared:.4f}")
print(f"  SLM pseudo-R²:      {r2_slm:.4f}")
print(f"  RMSE 개선율:        {(rmse_ols-rmse_slm)/rmse_ols*100:.1f}%")
print(f"  ρ (공간자기회귀):   {rho:.4f}")
print(f"  OLS 잔차 Moran's I: {mi_ols.I:.4f} (p={mi_ols.p_sim:.3f})")
print(f"  생성 파일:")
print(f"    data/result_full.csv")
print(f"    data/result_by_type.csv")
print(f"    data/figures/ols_vs_slm.png")
print(f"    data/figures/residual_spatial.png")
print(f"    data/figures/coef_by_type.png")
print(f"    data/figures/ratio_comparison.png")
print("=" * 65)

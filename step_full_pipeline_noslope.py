"""
관악구 공시가격 형평성 진단 플랫폼
전체 분석 파이프라인 (과제 2~5번)

실행: python3 step_full_pipeline.py
결과: data/figures/ 폴더에 시각화 저장
      data/result_full.csv — 전체 결과
      data/result_by_type.csv — 유형별 결과
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
plt.rcParams['font.family'] = "AppleGothic"
plt.rcParams['axes.unicode_minus'] = False

from libpysal.weights import KNN
from esda.moran import Moran
from spreg import GM_Lag
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import os

os.makedirs('data/figures', exist_ok=True)

print("=" * 60)
print("  관악구 공시가격 형평성 진단 — 전체 분석 파이프라인")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# STEP 1. 데이터 로드 및 전처리
# ─────────────────────────────────────────────────────────
print("\n[1/7] 데이터 로드 및 전처리...")

df = pd.read_csv('data/data_imputed.csv', low_memory=False)
print(f"  원본 세대수: {len(df):,}")

# 결측치 제거
df = df.dropna(subset=['median_price', 'land_price', 'lat', 'lon',
                        'dist_cbd', 'dist_subway', 'dist_park',
                        'age', 'floor_clean', 'notice_amt'])
print(f"  결측치 제거 후: {len(df):,}")

# 로그 변환
df['log_land_price'] = np.log(df['land_price'].clip(lower=1))
df['log_priv_area']  = np.log(df['priv_area'].clip(lower=1))
df['log_median_price'] = np.log(df['median_price'].clip(lower=1))

# MinMax 스케일링 (age, dist 변수)
scaler = MinMaxScaler()
scale_cols = ['age', 'dist_cbd', 'dist_subway', 'dist_park']
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# 주택 유형 레이블
type_map = {1: '아파트', 3: '오피스텔', 5: '연립다세대'}
df['apt_type_name'] = df['apt_type'].map(type_map)

print(f"  주택 유형별: {df['apt_type_name'].value_counts().to_dict()}")
print(f"  동별: {df['bj_dong_name'].value_counts().to_dict()}")

# 독립변수 정의
INDEP_VARS = ['log_land_price', 'age', 'log_priv_area', 'floor_clean', 'dist_cbd',
              'dist_subway', 'dist_park']
INDEP_VARS_NO_CBD = [v for v in INDEP_VARS if v != 'dist_cbd']


# ─────────────────────────────────────────────────────────
# STEP 2. 전체 OLS + SLM 모형
# ─────────────────────────────────────────────────────────
print("\n[2/7] 전체 OLS + SLM 모형 추정...")

def run_ols(df_sub, vars_list, label=""):
    X = sm.add_constant(df_sub[vars_list])
    y = df_sub['log_median_price']
    model = sm.OLS(y, X).fit()
    y_pred_log = model.predict(X)
    y_pred = np.exp(y_pred_log)
    rmse_log = np.sqrt(np.mean((y - y_pred_log)**2))
    mae = np.mean(np.abs(df_sub['median_price'] - y_pred))
    print(f"  [{label}] OLS: R²={model.rsquared:.4f} | RMSE(log)={rmse_log:.4f} | MAE={mae:,.0f}만원")
    return model, y_pred

def run_slm(df_sub, vars_list, label=""):
    coords = list(zip(df_sub['lon'], df_sub['lat']))
    w = KNN.from_array(coords, k=8)
    w.transform = 'r'
    X = df_sub[vars_list].values
    y = df_sub['log_median_price'].values.reshape(-1, 1)
    model = GM_Lag(y, X, w=w, robust='white', name_y='log_price',
                   name_x=vars_list)
    # 예측값
    rho = model.rho
    betas = model.betas[:-1]  # rho 제외
    X_const = np.column_stack([np.ones(len(X)), X])
    y_pred_log = X_const @ betas + rho * (w.sparse @ y)
    y_pred_log = y_pred_log.flatten()
    y_pred = np.exp(y_pred_log)
    rmse_log = np.sqrt(np.mean((y.flatten() - y_pred_log)**2))
    mae = np.mean(np.abs(df_sub['median_price'].values - y_pred))
    r2 = 1 - np.sum((y.flatten() - y_pred_log)**2) / np.sum((y.flatten() - y.mean())**2)
    print(f"  [{label}] SLM: pseudo-R²={r2:.4f} | RMSE(log)={rmse_log:.4f} | MAE={mae:,.0f}만원 | ρ={rho[0]:.3f}")
    return model, y_pred, y_pred_log

# 전체 OLS
ols_full, ols_pred_full = run_ols(df, INDEP_VARS, "전체 OLS")

# 전체 SLM
slm_full, slm_pred_full, slm_pred_log_full = run_slm(df, INDEP_VARS, "전체 SLM")

df['ols_pred'] = ols_pred_full
df['slm_pred'] = slm_pred_full
df['slm_pred_log'] = slm_pred_log_full
df['residual'] = df['median_price'] - df['slm_pred']
df['residual_pct'] = df['residual'] / df['slm_pred'] * 100

# Moran's I — OLS 잔차
coords = list(zip(df['lon'], df['lat']))
w_full = KNN.from_array(coords, k=8)
w_full.transform = 'r'
ols_resid = df['log_median_price'] - np.log(df['ols_pred'].clip(lower=1))
mi = Moran(ols_resid.values, w_full)
print(f"  OLS 잔차 Moran's I = {mi.I:.4f} (p={mi.p_sim:.3f})")

# ─────────────────────────────────────────
# 전체 모형 계수 출력
# ─────────────────────────────────────────
print("\n[전체 OLS 계수표]")
ols_coef = pd.DataFrame({
    "coef": ols_full.params,
    "p_value": ols_full.pvalues,
    "t_value": ols_full.tvalues
}).round(4)

print(ols_coef)

print("\n[전체 SLM 계수표]")

slm_rows = ["const"] + INDEP_VARS + ["rho"]

slm_coef = pd.DataFrame({
    "variable": slm_rows,
    "coef": slm_full.betas.flatten()
}).round(4)

print(slm_coef)

print("\n[전체 SLM 요약]")
print(slm_full.summary)

# ─────────────────────────────────────────────────────────
# STEP 3. 주택 유형별 모형
# ─────────────────────────────────────────────────────────
print("\n[3/7] 주택 유형별 OLS + SLM 모형...")

type_results = {}
coef_comparison = []

for apt_type, type_name in type_map.items():
    sub = df[df['apt_type'] == apt_type].copy().reset_index(drop=True)

    if len(sub) < 100:
        continue

    print(f"\n--- {type_name} ({len(sub):,}건) ---")

    ols_m, ols_p = run_ols(sub, INDEP_VARS, f"{type_name} OLS")
    slm_m, slm_p, slm_pl = run_slm(sub, INDEP_VARS, f"{type_name} SLM")

    sub["ols_pred"] = ols_p
    sub["slm_pred"] = slm_p
    type_results[type_name] = sub

    # 전체 변수명
    var_names = ["const"] + INDEP_VARS + ["rho"]

    for i, var in enumerate(var_names):
        coef_comparison.append({
            "type": type_name,
            "variable": var,
            "slm_coef": slm_m.betas[i][0],
            "slm_pval": slm_m.z_stat[i][1] if i < len(slm_m.z_stat) else np.nan,
            "ols_coef": ols_m.params[var] if var in ols_m.params else np.nan,
            "ols_pval": ols_m.pvalues[var] if var in ols_m.pvalues else np.nan
        })

coef_df = pd.DataFrame(coef_comparison)
print("\n  유형별 계수 비교 (SLM 기준):")
print(coef_df.pivot(index='variable', columns='type', values='slm_coef').round(4))

print("\n  유형별 SLM p-value 비교:")
print(coef_df.pivot(index='variable', columns='type', values='slm_pval').round(4))

print("\n  유형별 OLS p-value 비교:")
print(coef_df.pivot(index='variable', columns='type', values='ols_pval').round(4))


# ─────────────────────────────────────────────────────────
# STEP 4. 튜닝 — 업무지구 제거 버전
# ─────────────────────────────────────────────────────────
print("\n[4/7] 튜닝 — 업무지구 변수 제거 비교...")

ols_nocbd, ols_pred_nocbd = run_ols(df, INDEP_VARS_NO_CBD, "업무지구 제거 OLS")
slm_nocbd, slm_pred_nocbd, _ = run_slm(df, INDEP_VARS_NO_CBD, "업무지구 제거 SLM")

print(f"\n  업무지구 포함 vs 제거 비교:")
print(f"  {'':20s} {'포함':>10s} {'제거':>10s}")
ols_rmse = np.sqrt(np.mean((df['log_median_price'] - np.log(ols_pred_full.clip(lower=1)))**2))
ols_rmse_nc = np.sqrt(np.mean((df['log_median_price'] - np.log(ols_pred_nocbd.clip(lower=1)))**2))
print(f"  {'OLS RMSE(log)':20s} {ols_rmse:>10.4f} {ols_rmse_nc:>10.4f}")

# ─────────────────────────────────────────────────────────
# STEP 4+. 실제 vs 예측
# ─────────────────────────────────────────────────────────
df['real_ratio'] = df['median_price'] / df['slm_pred']

# ─────────────────────────────────────────────────────────
# STEP 5. 현실화율 비교
# ─────────────────────────────────────────────────────────
print("\n[5/7] 실제 현실화율 vs 계산 현실화율 비교...")

df['actual_ratio'] = (df['notice_amt'] / 10000) / df['median_price']
df['calc_ratio']   = (df['notice_amt'] / 10000) / df['slm_pred']

print(f"  실제 현실화율: 평균={df['actual_ratio'].mean():.3f} | 중앙값={df['actual_ratio'].median():.3f} | std={df['actual_ratio'].std():.3f}")
print(f"  계산 현실화율: 평균={df['calc_ratio'].mean():.3f} | 중앙값={df['calc_ratio'].median():.3f} | std={df['calc_ratio'].std():.3f}")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

# 동별 현실화율 비교
dong_ratio = df.groupby('bj_dong_name').agg(
    actual_ratio_mean=('actual_ratio', 'mean'),
    calc_ratio_mean=('calc_ratio', 'mean'),
    actual_ratio_med=('actual_ratio', 'median'),
    calc_ratio_med=('calc_ratio', 'median'),
).round(3)
print("\n  동별 현실화율 비교:")
print(dong_ratio)

# 유형별 현실화율 비교
type_ratio = df.groupby('apt_type_name').agg(
    actual_ratio_mean=('actual_ratio', 'mean'),
    calc_ratio_mean=('calc_ratio', 'mean'),
).round(3)
print("\n  유형별 현실화율 비교:")
print(type_ratio)

# ─────────────────────────────────────────────────────────
# STEP 6. 시각화
# ─────────────────────────────────────────────────────────
print("\n[6/7] 시각화 생성...")

# --- 6-1. OLS vs SLM 산점도 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hedonic OLS vs SLM — 실제 vs 예측 실거래가', fontsize=14)

for ax, pred, label, color in zip(
    axes,
    [df['ols_pred'], df['slm_pred']],
    ['Hedonic OLS', 'SLM'],
    ['steelblue', 'crimson']
):
    r2 = 0.7851 if label == 'Hedonic OLS' else 0.9092
    ax.scatter(df['median_price'], pred, alpha=0.2, s=2, color=color)
    mx = max(df['median_price'].max(), pred.max())
    ax.plot([0, mx], [0, mx], 'k--', lw=1, label='y=x')
    ax.set_title(f'{label}  (R²={r2})', fontsize=12)
    ax.set_xlabel('실제 실거래가 (만원)', fontsize=11)
    ax.set_ylabel('예측 실거래가 (만원)', fontsize=11)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('data/figures/ols_vs_slm.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: data/figures/ols_vs_slm.png")

# --- 6-2. 잔차 공간 분포 ---
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
ax.set_xlabel('경도', fontsize=11); ax.set_ylabel('위도', fontsize=11)
plt.tight_layout()
plt.savefig('data/figures/residual_spatial.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: data/figures/residual_spatial.png")

# --- 6-3. 유형별 계수 비교 히트맵 ---
var_labels = {
    'log_land_price': '공시지가(log)',
    'age': '건축경과연수',
    'log_priv_area': '전용면적(log)',
    'floor_clean': '층수',
    'dist_cbd': '업무지구거리',
    'dist_subway': '지하철역거리',
    'dist_park': '근린공원거리',
}

if not coef_df.empty:
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
    ax.set_title('주택 유형별 SLM 계수 비교\n(빨강=양의 영향 / 파랑=음의 영향)', fontsize=13)
    plt.tight_layout()
    plt.savefig('data/figures/coef_by_type.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: data/figures/coef_by_type.png")

# --- 6-4. 현실화율 비교 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('실제 현실화율 vs 계산 현실화율 비교', fontsize=13)

# 분포 비교
ax1 = axes[0]
ax1.hist(df['actual_ratio'].clip(0, 1.5), bins=60, alpha=0.6,
         color='steelblue', label='실제 현실화율 (공시가/실거래가)')
ax1.hist(df['calc_ratio'].clip(0, 1.5), bins=60, alpha=0.6,
         color='crimson', label='계산 현실화율 (공시가/SLM예측가)')
ax1.axvline(df['actual_ratio'].median(), color='steelblue', lw=2, linestyle='--',
            label=f'실제 중앙값: {df["actual_ratio"].median():.3f}')
ax1.axvline(df['calc_ratio'].median(), color='crimson', lw=2, linestyle='--',
            label=f'계산 중앙값: {df["calc_ratio"].median():.3f}')
ax1.set_xlabel('현실화율', fontsize=11)
ax1.set_ylabel('세대 수', fontsize=11)
ax1.set_title('전체 현실화율 분포 비교', fontsize=12)
ax1.legend(fontsize=9)

# 동별 현실화율 비교
ax2 = axes[1]
x = np.arange(len(dong_ratio))
w = 0.35
ax2.bar(x - w/2, dong_ratio['actual_ratio_mean'], w, label='실제 현실화율',
        color='steelblue', alpha=0.8)
ax2.bar(x + w/2, dong_ratio['calc_ratio_mean'], w, label='계산 현실화율',
        color='crimson', alpha=0.8)
ax2.axhline(0.591, color='navy', lw=1.5, linestyle='--', label='전체 중앙 59.1%')
ax2.set_xticks(x)
ax2.set_xticklabels(dong_ratio.index, fontsize=12)
ax2.set_ylabel('현실화율', fontsize=11)
ax2.set_title('동별 현실화율 비교', fontsize=12)
ax2.legend(fontsize=9)
for i, (a, c) in enumerate(zip(dong_ratio['actual_ratio_mean'], dong_ratio['calc_ratio_mean'])):
    ax2.text(i - w/2, a + 0.005, f'{a:.3f}', ha='center', fontsize=9)
    ax2.text(i + w/2, c + 0.005, f'{c:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('data/figures/ratio_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: data/figures/ratio_comparison.png")

# --- 6-5. 유형별 현실화율 분포 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('주택 유형별 실제 현실화율 vs 계산 현실화율', fontsize=13)
for ax, (type_name, sub) in zip(axes, type_results.items()):
    sub['actual_r'] = (sub['notice_amt'] / 10000) / sub['median_price']
    sub['calc_r']   = (sub['notice_amt'] / 10000) / sub['slm_pred']
    ax.hist(sub['actual_r'].clip(0, 1.5), bins=40, alpha=0.6,
            color='steelblue', label=f'실제 ({sub["actual_r"].median():.3f})')
    ax.hist(sub['calc_r'].clip(0, 1.5), bins=40, alpha=0.6,
            color='crimson', label=f'계산 ({sub["calc_r"].median():.3f})')
    ax.axvline(sub['actual_r'].median(), color='steelblue', lw=2, linestyle='--')
    ax.axvline(sub['calc_r'].median(), color='crimson', lw=2, linestyle='--')
    ax.set_title(f'{type_name} ({len(sub):,}건)', fontsize=12)
    ax.set_xlabel('현실화율', fontsize=10)
    ax.legend(fontsize=9, title='중앙값')

plt.tight_layout()
plt.savefig('data/figures/ratio_by_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: data/figures/ratio_by_type.png")

# --- 6-6. 동별 비교 바차트 ---
dong_stats = df.groupby('bj_dong_name').agg(
    real_pred=('median_price', lambda x: (x / df.loc[x.index, 'slm_pred']).mean()),
    gong_pred=('actual_ratio', 'mean'),
    count=('median_price', 'count')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('동별 예측가격 대비 실제가격 비율 비교', fontsize=13)
x = np.arange(len(dong_stats))
colors = ['#e74c3c', '#3498db', '#2ecc71']

ax1 = axes[0]
bars = ax1.bar(x, dong_stats['real_pred'], color=colors, width=0.5, alpha=0.85)
ax1.axhline(1.0, color='black', lw=1.2, linestyle='--', label='기준선 (1.0)')
ax1.set_xticks(x); ax1.set_xticklabels(dong_stats['bj_dong_name'], fontsize=12)
ax1.set_ylabel('실거래가 / SLM 예측가', fontsize=11)
ax1.set_title('실거래가 / 예측가 (동별)', fontsize=12)
ax1.legend(fontsize=10)
for bar, val in zip(bars, dong_stats['real_pred']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

ax2 = axes[1]
bars2 = ax2.bar(x, dong_stats['gong_pred'], color=colors, width=0.5, alpha=0.85)
ax2.axhline(0.591, color='navy', lw=1.2, linestyle='--', label='전체 현실화율 59.1%')
ax2.set_xticks(x); ax2.set_xticklabels(dong_stats['bj_dong_name'], fontsize=12)
ax2.set_ylabel('공시가격 / 실거래가 (실제 현실화율)', fontsize=11)
ax2.set_title('실제 현실화율 (동별)', fontsize=12)
ax2.legend(fontsize=10)
for bar, val in zip(bars2, dong_stats['gong_pred']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('data/figures/dong_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: data/figures/dong_comparison.png")

# ─────────────────────────────────────────────────────────
# STEP 7. 결과 저장
# ─────────────────────────────────────────────────────────
print("\n[7/7] 결과 저장...")

out_cols = ['rid', 'apt_code', 'bj_dong_name', 'road_name', 'apt_name',
            'apt_type', 'apt_type_name', 'floor', 'priv_area', 'built_year',
            'notice_amt', 'median_price', 'lat', 'lon',
            'dist_cbd', 'dist_subway', 'dist_park', 'age',
            'land_price', 'ols_pred', 'slm_pred','real_ratio',
            'residual', 'residual_pct', 'actual_ratio', 'calc_ratio']

df[out_cols].to_csv('data/result_full.csv', index=False, encoding='utf-8-sig')
print("  저장: data/result_full.csv")

# 유형별 결과
type_out = []
for type_name, sub in type_results.items():
    sub['apt_type_name'] = type_name
    
    
    # 실제가격 / SLM 예측가격
    sub['real_ratio'] = sub['median_price'] / sub['slm_pred']
    
    # 현실화율 계산식
    sub['actual_ratio'] = (sub['notice_amt'] / 10000) / sub['median_price']
    sub['calc_ratio']   = (sub['notice_amt'] / 10000) / sub['slm_pred']
    
    type_out.append(sub)

if type_out:
    pd.concat(type_out)[out_cols].to_csv(
        'data/result_by_type.csv', index=False, encoding='utf-8-sig')
    print("  저장: data/result_by_type.csv")

print("\n" + "=" * 60)
print("  전체 파이프라인 완료!")
print("  생성 파일:")
print("    data/figures/ols_vs_slm.png         — OLS vs SLM 산점도")
print("    data/figures/residual_spatial.png   — 잔차 공간 분포")
print("    data/figures/coef_by_type.png       — 유형별 계수 히트맵")
print("    data/figures/ratio_comparison.png   — 현실화율 비교")
print("    data/figures/ratio_by_type.png      — 유형별 현실화율")
print("    data/figures/dong_comparison.png    — 동별 비교")
print("    data/result_full.csv                — 전체 결과")
print("    data/result_by_type.csv             — 유형별 결과")
print("=" * 60)


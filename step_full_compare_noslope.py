# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:43:22 2026

@author: LYE
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
INDEP_VARS = ['log_land_price', 'age', 'log_priv_area', 'floor_clean','dist_cbd',
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


# ------------------------------------------------------------
# 1. 유형별 결과 합치기
# ------------------------------------------------------------
type_out = []

for type_name, sub in type_results.items():
    sub = sub.copy()
    sub["apt_type_name"] = type_name

    # --------------------------------------------------------
    # 2. 유형별 최종 예측값 선택
    # --------------------------------------------------------
    # 아파트: SLM 성능 좋음
    # 연립다세대: SLM이 약간 더 좋음
    # 오피스텔: OLS와 SLM 차이 거의 없으므로 OLS 사용 추천
    if type_name == "오피스텔":
        sub["final_pred"] = sub["ols_pred"]
        sub["final_model"] = "OLS"
    else:
        sub["final_pred"] = sub["slm_pred"]
        sub["final_model"] = "SLM"

    type_out.append(sub)

df_type_final = pd.concat(type_out, ignore_index=True)

# ─────────────────────────────────────────
# 추가 분석. 전체모형 기준 실제가격 대비 예측가격
# ─────────────────────────────────────────
print("\n[비교 분석] 전체모형 기준 실제가격 대비 예측가격 등급화...")

# 전체 SLM 예측가 기준
df["real_pred_ratio_total"] = df["median_price"] / df["slm_pred"]

print(f"  평균: {df['real_pred_ratio_total'].mean():.4f}")
print(f"  중앙값: {df['real_pred_ratio_total'].median():.4f}")
print(f"  표준편차: {df['real_pred_ratio_total'].std():.4f}")

df["real_compare_total"] = pd.cut(
    df["real_pred_ratio_total"],
    bins=[0, 0.75, 0.90, 1.10, 1.25, 99],
    labels=[
        "기준 대비 매우 낮음",
        "기준 대비 낮음",
        "기준과 유사",
        "기준 대비 높음",
        "기준 대비 매우 높음"
    ]
)

print("\n  전체모형 기준 등급 분포:")
for grade, cnt in df["real_compare_total"].value_counts().sort_index().items():
    bar = "█" * int(cnt / len(df) * 50)
    print(f"    {str(grade):14s}: {cnt:6,}건 ({cnt/len(df)*100:5.1f}%) {bar}")

# ─────────────────────────────────────────
# 추가 분석. 실제가격 대비 예측가격 등급화
# ─────────────────────────────────────────
print("\n[추가 분석] 실제가격 대비 예측가격 등급화...")

# notice_amt : 원 → notice_price : 만원
df_type_final["notice_price"] = df_type_final["notice_amt"] / 10000

# 실제 실거래가 / 유형별 최종 예측가격
# final_pred = 아파트/연립다세대 SLM, 오피스텔 OLS
df_type_final["real_pred_ratio"] = (
    df_type_final["median_price"] / df_type_final["final_pred"]
)

print(f"  실제/예측 비율 평균: {df_type_final['real_pred_ratio'].mean():.4f}")
print(f"  실제/예측 비율 중앙값: {df_type_final['real_pred_ratio'].median():.4f}")
print(f"  실제/예측 비율 표준편차: {df_type_final['real_pred_ratio'].std():.4f}")

df_type_final["real_compare_level"] = pd.cut(
    df_type_final["real_pred_ratio"],
    bins=[0, 0.75, 0.90, 1.10, 1.25, 99],
    labels=[
        "기준 대비 매우 낮음",
        "기준 대비 낮음",
        "기준과 유사",
        "기준 대비 높음",
        "기준 대비 매우 높음"
    ]
)

df_type_final["real_compare_level_num"] = pd.cut(
    df_type_final["real_pred_ratio"],
    bins=[0, 0.75, 0.90, 1.10, 1.25, 99],
    labels=[0, 1, 2, 3, 4]
)

print("\n  실제가격 대비 예측가격 등급 분포:")
for grade, cnt in df_type_final["real_compare_level"].value_counts().sort_index().items():
    bar = "█" * int(cnt / len(df_type_final) * 50)
    print(f"    {str(grade):14s}: {cnt:6,}건 ({cnt/len(df_type_final)*100:5.1f}%) {bar}")

# ─────────────────────────────────────────
# 추가 분석. 주택 유형별 실제/예측 등급 비율
# ─────────────────────────────────────────
type_real_grade_t = pd.crosstab(
    df_type_final["apt_type_name"],
    df_type_final["real_compare_level"],
    normalize="index"
) * 100

type_real_grade_t = type_real_grade_t.reindex(columns=[
    "기준 대비 매우 낮음",
    "기준 대비 낮음",
    "기준과 유사",
    "기준 대비 높음",
    "기준 대비 매우 높음"
]).round(2)

print("\n[주택 유형별 실제가격 대비 예측가격 등급 비율(%)_유형별 SLM]")
print(type_real_grade_t)

type_real_grade = pd.crosstab(
    df["apt_type_name"],
    df["real_compare_total"],
    normalize="index"
) * 100

type_real_grade = type_real_grade.reindex(columns=[
    "기준 대비 매우 낮음",
    "기준 대비 낮음",
    "기준과 유사",
    "기준 대비 높음",
    "기준 대비 매우 높음"
]).round(2)

print("\n[주택 유형별 실제가격 대비 예측가격 등급 비율(%)_전체]")
print(type_real_grade)

# ─────────────────────────────────────────
# 추가 분석. 전체모형 기준 동별 요약표
# ─────────────────────────────────────────

# 등급 숫자화
compare_num_map = {
    "기준 대비 매우 낮음": 0,
    "기준 대비 낮음": 1,
    "기준과 유사": 2,
    "기준 대비 높음": 3,
    "기준 대비 매우 높음": 4
}

df["real_compare_total_num"] = df["real_compare_total"].astype(str).map(compare_num_map)

dong_summary_total = df.groupby("bj_dong_name").agg(
    # 핵심 비교지표
    real_pred_ratio_mean=("real_pred_ratio_total", "mean"),
    real_pred_ratio_median=("real_pred_ratio_total", "median"),
    real_pred_ratio_std=("real_pred_ratio_total", "std"),
    real_compare_level_mean=("real_compare_total_num", "mean"),

    # 가격 수준
    slm_pred_price_mean=("slm_pred", "mean"),
    median_price_mean=("median_price", "mean"),

    # 표본수
    count=("real_pred_ratio_total", "count"),

    # 변수 평균
    dist_subway_mean=("dist_subway", "mean"),
    dist_cbd_mean=("dist_cbd", "mean"),
    dist_park_mean=("dist_park", "mean"),
    slope_mean=("slope", "mean"),
    age_mean=("age", "mean"),
    priv_area_mean=("log_priv_area", "mean")
).reset_index()

dong_summary_total = dong_summary_total.sort_values(
    "real_pred_ratio_mean", ascending=False
)

dong_summary_total.to_csv(
    "data/dong_feature_summary_total_model.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== 동별 요약표: 전체모형 기준 ===")
print(dong_summary_total.round(4))


# ------------------------------------------------------------
# 저장
# ------------------------------------------------------------
df_type_final.to_csv(
    "data/result_type_based_grade.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:")
print("  data/result_type_based_grade.csv")
print("  data/dong_grade_type_based.csv")
print("  data/apt_type_grade_type_based.csv")
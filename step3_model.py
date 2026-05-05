"""
Step 3: 공간 더빈 모형(SDM) 기반 적정 공시가격 추정
=====================================================
방법론:
  - 전체 데이터에 SDM(공간 더빈 모형) 단일 적용
  - Hedonic OLS와 성능 비교 (선행연구 대비 개선 검증)
  - SDM 추정 실거래가 × 현실화율 → 적정 공시가격 역산
  - LISA 클러스터는 공간 패턴 설명 정보로 활용

독립변수 (7개):
  log(공시지가), 건축경과연수, log(전용면적),
  층수, 업무지구거리, 지하철역거리, 근린공원거리
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from libpysal.weights import KNN
from spreg import OLS, GM_Lag

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("data/figures", exist_ok=True)

# ─────────────────────────────────────────
# 1. 데이터 로드 및 준비
# ─────────────────────────────────────────
print("▶ 데이터 로드 중...")
df = pd.read_csv("data/Int_Data_geo.csv", low_memory=False)
df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

FEATURES = [
    "land_price", "age", "priv_area",
    "floor_clean", "dist_cbd", "dist_subway", "dist_park",
]
TARGET = "median_price"  # 만원

df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

# 이상치 제거 (평가비율 0.1~2.0)
df["assess_ratio"] = df["notice_amt"] / (df["median_price"] * 10000)
df = df[(df["assess_ratio"] > 0.1) & (df["assess_ratio"] < 2.0)].reset_index(drop=True)
print(f"  분석 대상: {len(df):,}행")

# ─────────────────────────────────────────
# 2. 로그 변환
# ─────────────────────────────────────────
print("\n▶ 변수 변환 중...")
df["log_price"]      = np.log(df[TARGET])
df["log_land_price"] = np.log(df["land_price"].clip(lower=1))
df["log_priv_area"]  = np.log(df["priv_area"].clip(lower=1))

# 동 고정효과 더미변수 (봉천동 기준, 신림동/남현동 더미 2개)
df = pd.get_dummies(df, columns=["block_name"], drop_first=True)
dummy_cols = [c for c in df.columns if c.startswith("block_name_")]
df[dummy_cols] = df[dummy_cols].astype(float)  # bool → float 변환
dummy_labels = [c.replace("block_name_", "") + "(더미)" for c in dummy_cols]
print(f"  동 더미변수: {dummy_cols}")

LOG_FEATURES = [
    "log_land_price", "age", "log_priv_area",
    "floor_clean", "dist_cbd", "dist_subway", "dist_park",
] + dummy_cols

LABELS = [
    "log(공시지가)", "건축경과연수", "log(전용면적)",
    "층수", "업무지구거리", "지하철역거리", "근린공원거리",
] + dummy_labels

X = df[LOG_FEATURES].values
y = df["log_price"].values.reshape(-1, 1)
print(f"  X: {X.shape}, y: {y.shape}")

# ─────────────────────────────────────────
# 3. 공간 가중치 행렬
# ─────────────────────────────────────────
print("\n▶ 공간 가중치 행렬 구성 중... (수 분 소요)")
coords = list(zip(df["lon"], df["lat"]))
w = KNN.from_array(coords, k=8)
w.transform = "R"
print(f"  KNN(k=8) 완료")

# ─────────────────────────────────────────
# 4. 비교 기준: Hedonic OLS
# ─────────────────────────────────────────
print("\n▶ [비교모형] Hedonic OLS 추정 중...")
ols = OLS(
    y, X,
    w=w,
    name_y="log(실거래가)",
    name_x=LABELS,
    name_ds="관악구 Hedonic OLS",
    spat_diag=True,
    moran=True,
)
print(ols.summary)

df["ols_pred_price"] = np.exp(ols.predy.flatten())
ols_r2   = ols.r2
ols_rmse = np.sqrt(np.mean((y.flatten() - ols.predy.flatten())**2))
ols_mae  = np.mean(np.abs(df[TARGET].values - df["ols_pred_price"].values))
print(f"\n  OLS R²: {ols_r2:.4f}  RMSE(log): {ols_rmse:.4f}  MAE(만원): {ols_mae:,.0f}")

# ─────────────────────────────────────────
# 5. 주 모형 SAR
# ─────────────────────────────────────────
print("\n▶ [주 모형] SAR(공간자기회귀모형) 추정 중... (수 분 소요)")
"""
OLS 진단 결과:
  Lagrange Multiplier (lag) p=0.000 → SAR 채택
  SAR은 SDM의 특수 케이스로, 종속변수의 공간 시차(Wy)를
  설명변수로 포함하여 인접 주택 가격의 영향을 명시적으로 반영.
  WX를 추가하면 도구변수 행렬이 특이행렬이 되므로 SAR로 추정.
"""
sdm = GM_Lag(
    y, X,
    w=w,
    name_y="log(실거래가)",
    name_x=LABELS,
    name_ds="관악구 SAR(공간자기회귀모형)",
    robust="white",
)
print(sdm.summary)

df["sdm_pred_price"] = np.exp(sdm.predy.flatten())
sdm_r2   = sdm.pr2
sdm_rmse = np.sqrt(np.mean((y.flatten() - sdm.predy.flatten())**2))
sdm_mae  = np.mean(np.abs(df[TARGET].values - df["sdm_pred_price"].values))
print(f"\n  SAR pseudo-R²: {sdm_r2:.4f}  RMSE(log): {sdm_rmse:.4f}  MAE(만원): {sdm_mae:,.0f}")

# ─────────────────────────────────────────
# 6. 모형 비교 요약
# ─────────────────────────────────────────
print("\n" + "="*60)
print(" 모형 비교 — Hedonic OLS vs SAR")
print("="*60)
print(f"  {'지표':<20} {'Hedonic OLS':>14} {'SDM':>14}")
print(f"  {'-'*50}")
print(f"  {'R²':<20} {ols_r2:>14.4f} {sdm_r2:>14.4f}")
print(f"  {'RMSE(log)':<20} {ols_rmse:>14.4f} {sdm_rmse:>14.4f}")
print(f"  {'MAE(만원)':<20} {ols_mae:>14,.0f} {sdm_mae:>14,.0f}")
improvement = (ols_rmse - sdm_rmse) / ols_rmse * 100
print(f"\n  RMSE 개선율: {improvement:.1f}% (OLS 대비 SDM)")
print("="*60)

# ─────────────────────────────────────────
# 7. 적정 공시가격 역산
# ─────────────────────────────────────────
print("\n▶ 적정 공시가격 역산 중...")
"""
현실화율 = 실제 공시가격 / 실제 실거래가
적정 공시가격 = SDM 추정 실거래가 × 현실화율

그룹(동일 건물 × 층 구간) 중앙 현실화율 사용
→ 개별 이상치 영향 최소화
"""
# ── 1단계: 가격 구간별 현실화율 (지침서 기준)
bins   = [0, 30000, 60000, 90000, float("inf")]
labels = ["3억미만", "3~6억", "6~9억", "9억초과"]
df["price_tier"] = pd.cut(df["median_price"], bins=bins, labels=labels)

tier_ratio = df.groupby("price_tier", observed=True)["assess_ratio"].median()
print("\n  가격 구간별 현실화율 (지침서 기준):")
for tier, ratio in tier_ratio.items():
    print(f"    {tier}: {ratio:.4f} ({ratio*100:.1f}%)")
df["tier_ratio"] = df["price_tier"].map(tier_ratio)

# ── 2단계: 그룹 현실화율 (동일 건물 × 층 구간)
df["group_key"] = df["apt_code"].astype(str) + "_" + df["floor_group"].astype(str)
group_ratio     = df.groupby("group_key")["assess_ratio"].median().rename("group_ratio")
df              = df.join(group_ratio, on="group_key")

# ── 3단계: 최종 현실화율 결정
# 그룹 내 세대가 5개 이상이면 그룹 현실화율 우선 적용 (충분한 표본)
# 5개 미만이면 가격 구간별 현실화율로 대체
group_size      = df.groupby("group_key")["group_key"].transform("count")
df["final_ratio"] = df["group_ratio"].where(group_size >= 5, df["tier_ratio"])

# 그래도 결측이면 전체 중앙값으로 대체
overall_ratio     = df["assess_ratio"].median()
df["final_ratio"] = df["final_ratio"].fillna(overall_ratio)

print(f"\n  전체 중앙 현실화율: {overall_ratio:.4f} ({overall_ratio*100:.1f}%)")
print(f"  그룹 현실화율 적용: {(group_size >= 5).sum():,}건")
print(f"  구간 현실화율 적용: {(group_size < 5).sum():,}건")

# ── 4단계: 적정 공시가격 산출
df["fair_price"] = df["sdm_pred_price"] * df["final_ratio"] * 10000

# 형평성 비율: 실제 공시가격 / 적정 공시가격
df["equity_ratio"] = df["notice_amt"] / df["fair_price"]

print(f"  형평성 비율 평균:    {df['equity_ratio'].mean():.4f}")
print(f"  형평성 비율 중앙값:  {df['equity_ratio'].median():.4f}")
print(f"  형평성 비율 표준편차: {df['equity_ratio'].std():.4f}")

# 형평성 등급
# IAAO 국제감정평가사협회 기준
# 적정: ±10% (0.90~1.10), 심각: ±25% 초과
df["equity_grade"] = pd.cut(
    df["equity_ratio"],
    bins=[0, 0.75, 0.90, 1.10, 1.25, 99],
    labels=["심각과소", "과소", "적정", "과대", "심각과대"]
)

print("\n  형평성 등급 분포:")
for grade, cnt in df["equity_grade"].value_counts().sort_index().items():
    bar = "█" * int(cnt / len(df) * 50)
    print(f"    {grade:6s}: {cnt:6,}건 ({cnt/len(df)*100:5.1f}%) {bar}")

# 126% 초과 여부
# 실거래가가 공시가격의 126%를 초과하는 경우
df["over_126pct"] = (df["median_price"] * 10000) > (df["notice_amt"] * 1.26)
over_cnt = df["over_126pct"].sum()
print(f"\n  공시가격 126% 초과: {over_cnt:,}건 ({over_cnt/len(df)*100:.1f}%)")

# ─────────────────────────────────────────
# 8. 시각화
# ─────────────────────────────────────────
print("\n▶ 시각화 저장 중...")

# (a) OLS vs SDM 예측 비교
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Hedonic OLS vs SDM — 실제 vs 예측 실거래가", fontsize=13)
for ax, col, title, color in zip(
    axes,
    ["ols_pred_price", "sdm_pred_price"],
    [f"Hedonic OLS  (R²={ols_r2:.3f})", f"SDM  (R²={sdm_r2:.3f})"],
    ["#378ADD", "#E24B4A"]
):
    ax.scatter(df[TARGET], df[col], alpha=0.1, s=2, color=color)
    mn = min(df[TARGET].min(), df[col].min())
    mx = max(df[TARGET].max(), df[col].max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y=x")
    ax.set_xlabel("실제 실거래가 (만원)")
    ax.set_ylabel("예측 실거래가 (만원)")
    ax.set_title(title)
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("data/figures/ols_vs_sdm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  data/figures/ols_vs_sdm.png 저장 완료")

# (b) 형평성 비율 분포
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["equity_ratio"].clip(0.5, 1.5), bins=80,
        color="#378ADD", alpha=0.75, edgecolor="white", linewidth=0.3)
ax.axvline(1.0, color="#E24B4A", linewidth=2,
           linestyle="--", label="기준선 (1.0 = 적정)")
ax.axvline(df["equity_ratio"].mean(), color="#EF9F27", linewidth=1.5,
           label=f"평균 ({df['equity_ratio'].mean():.3f})")
ax.axvspan(0.95, 1.05, alpha=0.1, color="green", label="적정 구간 (0.95~1.05)")
ax.set_xlabel("형평성 비율 (실제 공시가격 / 적정 공시가격)")
ax.set_ylabel("세대 수")
ax.set_title("공시가격 형평성 비율 분포 (SDM 기준)")
ax.legend()
plt.tight_layout()
plt.savefig("data/figures/equity_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  data/figures/equity_distribution.png 저장 완료")

# (c) 형평성 공간 분포
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(
    df["lon"], df["lat"],
    c=df["equity_ratio"].clip(0.7, 1.3),
    cmap="RdYlGn_r", alpha=0.5, s=3,
    vmin=0.7, vmax=1.3
)
plt.colorbar(sc, ax=ax, label="형평성 비율 (빨강=과대, 초록=과소)")
ax.set_title("관악구 공시가격 형평성 공간 분포 (SDM 기준)", fontsize=11)
ax.set_xlabel("경도")
ax.set_ylabel("위도")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("data/figures/equity_spatial.png", dpi=150, bbox_inches="tight")
plt.close()
print("  data/figures/equity_spatial.png 저장 완료")

# (d) LISA 클러스터별 형평성 비율 박스플롯
fig, ax = plt.subplots(figsize=(10, 5))
cluster_order = ["HH(과대집중)", "HL(주변과소)", "비유의", "LH(주변과대)", "LL(과소집중)"]
cluster_data  = [
    df[df["lisa_cluster"] == c]["equity_ratio"].clip(0.5, 1.5).values
    for c in cluster_order
]
bp = ax.boxplot(cluster_data, labels=cluster_order, patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
colors = ["#E24B4A", "#F09595", "#D3D1C7", "#85B7EB", "#378ADD"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="기준(1.0)")
ax.set_xlabel("LISA 클러스터")
ax.set_ylabel("형평성 비율")
ax.set_title("LISA 클러스터별 형평성 비율 분포")
ax.legend()
plt.tight_layout()
plt.savefig("data/figures/equity_by_lisa.png", dpi=150, bbox_inches="tight")
plt.close()
print("  data/figures/equity_by_lisa.png 저장 완료")

# ─────────────────────────────────────────
# 9. 최종 결과 저장
# ─────────────────────────────────────────
print("\n▶ 최종 결과 저장 중...")

result_df = df[[
    "apt_code", "apt_name", "road_name", "dong_name",
    "floor", "priv_area", "built_year", "age",
    "lat", "lon",
    "notice_amt",        # 실제 공시가격 (원)
    "median_price",      # 실거래가 중앙값 (만원)
    "land_price",        # 공시지가 (원/㎡)
    "dist_cbd", "dist_subway", "dist_park",
    "lisa_cluster",      # LISA 클러스터 유형 (정보 제공용)
    "ols_pred_price",    # OLS 예측 실거래가 (만원, 비교용)
    "sdm_pred_price",    # SDM 예측 실거래가 (만원, 주 모형)
    "fair_price",        # 적정 공시가격 (원)
    "equity_ratio",      # 형평성 비율
    "equity_grade",      # 형평성 등급
    "over_126pct",       # 126% 초과 여부
    "assess_ratio",      # 실제 현실화율
    "final_ratio",       # 최종 적용 현실화율 (그룹 or 구간)
    "price_tier",        # 가격 구간
]].copy()

result_df.to_csv("data/Int_Data_result.csv", index=False, encoding="utf-8-sig")
print(f"  저장 완료: data/Int_Data_result.csv ({len(result_df):,}행)")

# 모형 성능 요약 저장
with open("data/model_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== 모형 성능 비교 ===\n\n")
    f.write(f"Hedonic OLS\n")
    f.write(f"  R²        : {ols_r2:.4f}\n")
    f.write(f"  RMSE(log) : {ols_rmse:.4f}\n")
    f.write(f"  MAE(만원) : {ols_mae:,.0f}\n\n")
    f.write(f"SDM(공간 더빈 모형)\n")
    f.write(f"  pseudo-R² : {sdm_r2:.4f}\n")
    f.write(f"  RMSE(log) : {sdm_rmse:.4f}\n")
    f.write(f"  MAE(만원) : {sdm_mae:,.0f}\n\n")
    f.write(f"RMSE 개선율 : {improvement:.1f}%\n\n")
    f.write(f"=== 형평성 진단 결과 ===\n\n")
    f.write(f"전체 중앙 현실화율  : {overall_ratio:.4f}\n")
    f.write(f"형평성 비율 평균    : {df['equity_ratio'].mean():.4f}\n")
    f.write(f"형평성 비율 중앙값  : {df['equity_ratio'].median():.4f}\n")
    f.write(f"126% 초과           : {over_cnt:,}건 ({over_cnt/len(df)*100:.1f}%)\n\n")
    f.write("형평성 등급 분포:\n")
    for grade, cnt in df["equity_grade"].value_counts().sort_index().items():
        f.write(f"  {grade}: {cnt:,}건 ({cnt/len(df)*100:.1f}%)\n")

print("  저장 완료: data/model_summary.txt")
print("\n✅ Step 3 완료")
print(f"   적정 공시가격 평균: {result_df['fair_price'].mean()/10000:,.0f}만원")
print(f"   형평성 비율 평균:   {result_df['equity_ratio'].mean():.4f}")
print("\n→ 다음 단계: Step 4 (PostgreSQL DB 구축 + FastAPI 백엔드)")

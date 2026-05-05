"""
Step 2: 공간적 자기상관 진단 (Moran's I)
==========================================
목적:
  - 공시가격과 실거래가의 공간적 자기상관성 진단
  - Moran's I 통계량 및 p-value 산출
  - 결과에 따라 Step 3에서 Hedonic(OLS) 또는 SDM 선택

판단 기준:
  - p-value < 0.05 이고 Moran's I > 0  → 양의 공간 자기상관 → SDM 채택
  - p-value >= 0.05                     → 공간 자기상관 없음  → Hedonic(OLS) 채택

사전 준비:
  pip install libpysal esda splot matplotlib seaborn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 서버/터미널 환경 대응
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from libpysal.weights import KNN, Queen
from libpysal.weights.util import fill_diagonal
import esda
from esda.moran import Moran, Moran_Local

# ─────────────────────────────────────────
# 0. 한글 폰트 설정 (macOS)
# ─────────────────────────────────────────
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────
# 1. 데이터 로드 및 전처리
# ─────────────────────────────────────────
print("▶ 데이터 로드 중...")
df = pd.read_csv("data/Int_Data_geo.csv")

# 위경도 결측 제거
df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
print(f"  유효 데이터: {len(df):,}행")

# ─────────────────────────────────────────
# 2. 분석 변수 준비
# ─────────────────────────────────────────
# 공시가격 (원) → 만원 단위로 통일
df["notice_amt_man"] = df["notice_amt"] / 10000

# land_price 대체: 단위면적당 공시가격 (만원/㎡)
# 실제 공시지가 수집 전까지 proxy로 사용
df["land_price_proxy"] = df["notice_amt_man"] / df["priv_area"]

# 평가비율: 공시가격 / 실거래가 (핵심 분석 대상)
# notice_amt(원) / median_price_won(원)
df["assess_ratio"] = df["notice_amt"] / df["median_price_won"]

# 이상치 제거 (평가비율 0.1~2.0 범위만 유효)
df = df[(df["assess_ratio"] > 0.1) & (df["assess_ratio"] < 2.0)].reset_index(drop=True)
print(f"  이상치 제거 후: {len(df):,}행")
print(f"  평가비율 평균: {df['assess_ratio'].mean():.4f}")
print(f"  평가비율 표준편차: {df['assess_ratio'].std():.4f}")

# ─────────────────────────────────────────
# 3. 공간 가중치 행렬 구성
# ─────────────────────────────────────────
print("\n▶ 공간 가중치 행렬 구성 중...")

# 좌표 배열
coords = list(zip(df["lon"], df["lat"]))

# KNN (k=8) — 가장 가까운 8개 이웃
# 관악구처럼 행정구역이 좁은 경우 KNN이 Queen보다 안정적
print("  KNN(k=8) 가중치 행렬 생성 중... (수 분 소요)")
w_knn = KNN.from_array(coords, k=8)
w_knn.transform = "R"   # Row-standardization (행 합계 = 1)
print(f"  KNN 완료 — 평균 이웃 수: {w_knn.mean_neighbors:.1f}")

# ─────────────────────────────────────────
# 4. 전역 Moran's I 검정
# ─────────────────────────────────────────
print("\n▶ 전역 Moran's I 검정...")

TARGET_VARS = {
    "공시가격(만원)":   "notice_amt_man",
    "실거래가(만원)":   "median_price",
    "평가비율":         "assess_ratio",
}

moran_results = {}

for label, col in TARGET_VARS.items():
    y = df[col].values
    mi = Moran(y, w_knn, permutations=999)
    moran_results[label] = {
        "I": mi.I,
        "E[I]": mi.EI,
        "p_sim": mi.p_sim,
        "z_norm": mi.z_norm,
        "significant": mi.p_sim < 0.05
    }
    sig = "★ 유의" if mi.p_sim < 0.05 else "  비유의"
    print(f"  {sig} [{label}]  I={mi.I:.4f}  E[I]={mi.EI:.4f}  "
          f"p={mi.p_sim:.4f}  z={mi.z_norm:.2f}")

# ─────────────────────────────────────────
# 5. 최종 모형 선택 판단
# ─────────────────────────────────────────
print("\n" + "="*55)
print(" 모형 선택 결과")
print("="*55)

assess_result = moran_results["평가비율"]
if assess_result["significant"] and assess_result["I"] > 0:
    model_choice = "SDM"
    reason = (f"평가비율의 Moran's I = {assess_result['I']:.4f}, "
              f"p = {assess_result['p_sim']:.4f} → 양의 공간 자기상관 확인")
else:
    model_choice = "Hedonic (OLS)"
    reason = (f"평가비율의 Moran's I = {assess_result['I']:.4f}, "
              f"p = {assess_result['p_sim']:.4f} → 공간 자기상관 미확인")

print(f"  채택 모형: {model_choice}")
print(f"  근거: {reason}")
print("="*55)

# ─────────────────────────────────────────
# 6. 국지적 Moran's I (LISA) — 공시가격 기준
# ─────────────────────────────────────────
print("\n▶ 국지적 Moran's I (LISA) 계산 중...")
y_assess = df["assess_ratio"].values
lisa = Moran_Local(y_assess, w_knn, permutations=999)

# LISA 클러스터 유형
# 1=HH(고-고), 2=LH(저-고), 3=LL(저-저), 4=HL(고-저)
df["lisa_q"]   = lisa.q          # 사분면
df["lisa_p"]   = lisa.p_sim      # p-value
df["lisa_sig"] = lisa.p_sim < 0.05

cluster_map = {1: "HH(과대집중)", 2: "LH(주변과대)", 3: "LL(과소집중)", 4: "HL(주변과소)"}
df["lisa_cluster"] = df.apply(
    lambda r: cluster_map.get(int(r["lisa_q"]), "비유의") if r["lisa_sig"] else "비유의",
    axis=1
)

cluster_counts = df["lisa_cluster"].value_counts()
print("  LISA 클러스터 분포:")
for k, v in cluster_counts.items():
    print(f"    {k}: {v:,}건")

# ─────────────────────────────────────────
# 7. 시각화
# ─────────────────────────────────────────
print("\n▶ 시각화 저장 중...")
os.makedirs("data/figures", exist_ok=True) if False else None
import os
os.makedirs("data/figures", exist_ok=True)

# (a) Moran 산점도
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("전역 Moran's I 산점도", fontsize=14)

for ax, (label, col) in zip(axes, TARGET_VARS.items()):
    y  = df[col].values
    y_std = (y - y.mean()) / y.std()
    wy = w_knn.sparse.dot(y_std)   # 공간 시차
    mi_val = moran_results[label]["I"]
    p_val  = moran_results[label]["p_sim"]

    ax.scatter(y_std, wy, alpha=0.15, s=3, color="#378ADD")
    m, b = np.polyfit(y_std, wy, 1)
    x_line = np.linspace(y_std.min(), y_std.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#E24B4A", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"{label}\nI={mi_val:.4f}  p={p_val:.4f}", fontsize=11)
    ax.set_xlabel("표준화 값")
    ax.set_ylabel("공간 시차")

plt.tight_layout()
plt.savefig("data/figures/moran_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  data/figures/moran_scatter.png 저장 완료")

# (b) LISA 클러스터 지도 (산점도 형태)
fig, ax = plt.subplots(figsize=(8, 8))
color_map = {
    "HH(과대집중)": "#E24B4A",
    "LH(주변과대)": "#85B7EB",
    "LL(과소집중)": "#378ADD",
    "HL(주변과소)": "#F09595",
    "비유의":       "#D3D1C7",
}
for cluster, color in color_map.items():
    mask = df["lisa_cluster"] == cluster
    ax.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"],
               c=color, label=f"{cluster} ({mask.sum():,})",
               alpha=0.6, s=4)
ax.set_title("LISA 클러스터 — 평가비율 (공시가격/실거래가)", fontsize=12)
ax.set_xlabel("경도")
ax.set_ylabel("위도")
ax.legend(loc="upper left", fontsize=9, markerscale=3)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("data/figures/lisa_cluster_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("  data/figures/lisa_cluster_map.png 저장 완료")

# ─────────────────────────────────────────
# 8. 결과 저장
# ─────────────────────────────────────────
df.to_csv("data/Int_Data_geo.csv", index=False, encoding="utf-8-sig")
print("\n▶ 결과 저장 완료: data/Int_Data_geo.csv (lisa_cluster 컬럼 추가)")

# 모형 선택 결과 텍스트 저장
with open("data/model_choice.txt", "w", encoding="utf-8") as f:
    f.write(f"채택 모형: {model_choice}\n")
    f.write(f"근거: {reason}\n")
    for label, res in moran_results.items():
        f.write(f"\n[{label}]\n")
        f.write(f"  Moran's I  = {res['I']:.6f}\n")
        f.write(f"  E[I]       = {res['E[I]']:.6f}\n")
        f.write(f"  p-value    = {res['p_sim']:.6f}\n")
        f.write(f"  z-score    = {res['z_norm']:.4f}\n")
        f.write(f"  유의여부   = {'유의(p<0.05)' if res['significant'] else '비유의'}\n")
print("▶ 모형 선택 결과 저장: data/model_choice.txt")

print(f"\n✅ Step 2 완료 → Step 3에서 [{model_choice}] 모형을 적용합니다.")

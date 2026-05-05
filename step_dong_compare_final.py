# -*- coding: utf-8 -*-
"""
동별 특성 비교 분석
- 동별 평균 실제/예측 비율
- 동별 평균 공시/예측 비율
- 동별 주요 변수 평균
- 중앙값, 표준편차, 표본수 포함
- 결과 저장
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------------
# 1. 데이터 불러오기
# -----------------------------------
df = pd.read_csv("data/result_full.csv", encoding="utf-8-sig")

dff = pd.read_csv("data/data_imputed.csv", encoding="utf-8-sig")

# -----------------------------------
# 2. 동별 요약 분석
# -----------------------------------
dong_summary = df.groupby("bj_dong_name").agg(
    # 핵심 비교지표
    real_pred_ratio_mean   = ("real_ratio", "mean"),
    real_pred_ratio_median = ("real_ratio", "median"),
    act_ratio_mean = ("actual_ratio", "mean"),
    act_ratio_median = ("actual_ratio", "median"),
    calc_ratio_mean = ("calc_ratio", "mean"),
    calc_ratio_median = ("calc_ratio", "median"),

    # 가격 수준
    slm_pred_price_mean    = ("slm_pred", "mean"),
    median_price_mean      = ("median_price", "mean"),

    # 표본수
    count                  = ("real_ratio", "count"),

).reset_index()

# 정렬
dong_summary = dong_summary.sort_values("real_pred_ratio_mean", ascending=False)

# 저장
dong_summary.to_csv("data/dong_feature_summary.csv",
                    index=False, encoding="utf-8-sig")

print("\n=== 동별 요약표 ===")
print(dong_summary)

dong_sum= dff.groupby("bj_dong_name").agg(
    # 변수 평균
    dist_subway_mean       = ("dist_subway", "mean"),
    dist_cbd_mean          = ("dist_cbd", "mean"),
    dist_park_mean         = ("dist_park", "mean"),
    slope_mean             = ("slope", "mean"),
    age_mean               = ("age", "mean"),
    priv_area_mean         = ("priv_area", "mean")
).reset_index()


# 저장
dong_sum.to_csv("data/dong_feature_summary2.csv",
                    index=False, encoding="utf-8-sig")


print("\n저장 완료")
print("data/dong_feature_summary.csv")
print("data/dong_feature_summary2.csv")

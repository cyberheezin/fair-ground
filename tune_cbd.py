"""
업무지구 변수 튜닝 — 3버전 비교
================================

버전 1: 기존 (여의도·강남·광화문)
버전 2: 기존 + 관악 (여의도·강남·광화문·서울대·서울대입구역·신림역)
버전 3: 관악 단독 (서울대·서울대입구역·신림역)

실행: python3 tune_cbd.py
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import statsmodels.api as sm
from libpysal.weights import KNN
from spreg import GM_Lag
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. 업무지구 좌표 정의
# ─────────────────────────────────────────

CBD_ORIGINAL = {
    '여의도':  (37.5219, 126.9245),
    '강남':    (37.4979, 127.0276),
    '광화문':  (37.5720, 126.9769),
}

CBD_GWANAK = {
    '서울대학교정문':      (37.4601, 126.9523),
    '서울대입구역(관악구청)': (37.4813, 126.9527),
    '신림역상권':          (37.4843, 126.9295),
}

CBD_COMBINED = {**CBD_ORIGINAL, **CBD_GWANAK}

CBD_VERSIONS = {
    'V1_기존':       CBD_ORIGINAL,
    'V2_기존+관악':  CBD_COMBINED,
    'V3_관악단독':   CBD_GWANAK,
}

# ─────────────────────────────────────────
# 2. Haversine 거리 계산 함수
# ─────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def min_dist_to_cbds(df, cbd_dict):
    """각 세대에서 업무지구들까지의 최단거리(km) 계산"""
    dist_cols = []
    for name, (clat, clon) in cbd_dict.items():
        col = f'_dist_{name}'
        df[col] = df.apply(
            lambda r: haversine(r['lat'], r['lon'], clat, clon), axis=1
        )
        dist_cols.append(col)
    result = df[dist_cols].min(axis=1)
    df.drop(columns=dist_cols, inplace=True)
    return result

# ─────────────────────────────────────────
# 3. 데이터 로드 및 기본 전처리
# ─────────────────────────────────────────
print("=" * 65)
print("  업무지구 변수 튜닝 — 3버전 비교")
print("  V1: 기존(여의도·강남·광화문)")
print("  V2: 기존+관악(6곳)")
print("  V3: 관악단독(서울대·서울대입구역·신림역)")
print("=" * 65)

print("\n[1/4] 데이터 로드...")
df = pd.read_csv('data/data_imputed.csv', low_memory=False)
df = df.dropna(subset=['median_price', 'land_price', 'lat', 'lon',
                        'dist_subway', 'dist_park', 'slope',
                        'age', 'floor_clean', 'notice_amt',
                        'priv_area'])
df = df[df['median_price'] > 0].copy().reset_index(drop=True)
print(f"  분석 세대수: {len(df):,}")

# 로그 변환
df['log_land_price']   = np.log(df['land_price'].clip(lower=1))
df['log_priv_area']    = np.log(df['priv_area'].clip(lower=1))
df['log_median_price'] = np.log(df['median_price'].clip(lower=1))

# MinMax 스케일링 (dist, age, slope)
from sklearn.preprocessing import MinMaxScaler
scale_cols = ['age', 'dist_subway', 'dist_park', 'slope']
scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# 공간 가중치 행렬 (공통)
print("\n[2/4] 공간 가중치 행렬 구성 (KNN k=8)...")
coords = list(zip(df['lon'], df['lat']))
w = KNN.from_array(coords, k=8)
w.transform = 'r'
print(f"  완료: {len(df):,}개 세대")

# ─────────────────────────────────────────
# 4. 3버전 비교 실행
# ─────────────────────────────────────────
print("\n[3/4] 3버전 모형 추정...")

INDEP_BASE = ['log_land_price', 'age', 'log_priv_area', 'floor_clean',
              'dist_subway', 'dist_park', 'slope']

results = {}

for version, cbd_dict in CBD_VERSIONS.items():
    print(f"\n  === {version} ===")
    print(f"  업무지구: {list(cbd_dict.keys())}")

    # 업무지구 거리 계산
    df['dist_cbd_v'] = min_dist_to_cbds(df.copy(), cbd_dict)

    # MinMax 스케일링 (dist_cbd)
    cbd_scaler = MinMaxScaler()
    df['dist_cbd_scaled'] = cbd_scaler.fit_transform(df[['dist_cbd_v']])

    INDEP = INDEP_BASE + ['dist_cbd_scaled']

    # OLS
    X_ols = sm.add_constant(df[INDEP])
    y = df['log_median_price']
    ols = sm.OLS(y, X_ols).fit()

    y_pred_ols_log = ols.predict(X_ols)
    rmse_ols = np.sqrt(np.mean((y - y_pred_ols_log)**2))
    mae_ols = np.mean(np.abs(df['median_price'] - np.exp(y_pred_ols_log)))
    ols_resid_moran_I = None

    # OLS 잔차 Moran's I
    from esda.moran import Moran
    mi = Moran(ols.resid.values, w)
    ols_resid_moran_I = mi.I

    print(f"  OLS: R²={ols.rsquared:.4f} | RMSE={rmse_ols:.4f} | "
          f"잔차 Moran's I={ols_resid_moran_I:.4f}")
    print(f"  OLS dist_cbd 계수: {ols.params['dist_cbd_scaled']:.4f} "
          f"(p={ols.pvalues['dist_cbd_scaled']:.3f})")

    # SLM
    X_slm = df[INDEP].values
    y_slm = df['log_median_price'].values.reshape(-1, 1)
    slm = GM_Lag(y_slm, X_slm, w=w, robust='white',
                 name_y='log_price', name_x=INDEP)

    rho = float(np.array(slm.rho).flatten()[0])
    betas = slm.betas[:-1]
    X_const = np.column_stack([np.ones(len(X_slm)), X_slm])
    y_pred_slm_log = (X_const @ betas + rho * (w.sparse @ y_slm)).flatten()
    rmse_slm = np.sqrt(np.mean((y_slm.flatten() - y_pred_slm_log)**2))
    mae_slm = np.mean(np.abs(df['median_price'].values - np.exp(y_pred_slm_log)))
    r2_slm = 1 - np.sum((y_slm.flatten() - y_pred_slm_log)**2) / \
                 np.sum((y_slm.flatten() - y_slm.mean())**2)

    # SLM dist_cbd 계수 및 p값
    cbd_idx = INDEP.index('dist_cbd_scaled')
    cbd_coef = slm.betas[cbd_idx + 1][0]
    cbd_std  = slm.std_err[cbd_idx + 1]
    cbd_z    = cbd_coef / cbd_std
    cbd_p    = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(cbd_z)))

    print(f"  SLM: R²={r2_slm:.4f} | RMSE={rmse_slm:.4f} | ρ={rho:.4f}")
    print(f"  SLM dist_cbd 계수: {cbd_coef:.4f} (z={cbd_z:.2f}, p={cbd_p:.3f})")

    results[version] = {
        'version': version,
        'cbd_points': list(cbd_dict.keys()),
        'ols_r2': ols.rsquared,
        'ols_rmse': rmse_ols,
        'ols_mae': mae_ols,
        'ols_cbd_coef': ols.params['dist_cbd_scaled'],
        'ols_cbd_pval': ols.pvalues['dist_cbd_scaled'],
        'ols_resid_moran': ols_resid_moran_I,
        'slm_r2': r2_slm,
        'slm_rmse': rmse_slm,
        'slm_mae': mae_slm,
        'slm_rho': rho,
        'slm_cbd_coef': cbd_coef,
        'slm_cbd_z': cbd_z,
        'slm_cbd_pval': cbd_p,
    }

# ─────────────────────────────────────────
# 5. 결과 비교표 출력
# ─────────────────────────────────────────
print("\n[4/4] 버전별 비교 결과")
print("=" * 65)

res_df = pd.DataFrame(results).T

print("\n[OLS 비교]")
print(f"{'':20s} {'V1_기존':>12s} {'V2_기존+관악':>12s} {'V3_관악단독':>12s}")
print(f"{'R²':20s} "
      f"{results['V1_기존']['ols_r2']:>12.4f} "
      f"{results['V2_기존+관악']['ols_r2']:>12.4f} "
      f"{results['V3_관악단독']['ols_r2']:>12.4f}")
print(f"{'RMSE(log)':20s} "
      f"{results['V1_기존']['ols_rmse']:>12.4f} "
      f"{results['V2_기존+관악']['ols_rmse']:>12.4f} "
      f"{results['V3_관악단독']['ols_rmse']:>12.4f}")
print(f"{'MAE(만원)':20s} "
      f"{results['V1_기존']['ols_mae']:>12.0f} "
      f"{results['V2_기존+관악']['ols_mae']:>12.0f} "
      f"{results['V3_관악단독']['ols_mae']:>12.0f}")
print(f"{'dist_cbd 계수':20s} "
      f"{results['V1_기존']['ols_cbd_coef']:>12.4f} "
      f"{results['V2_기존+관악']['ols_cbd_coef']:>12.4f} "
      f"{results['V3_관악단독']['ols_cbd_coef']:>12.4f}")
print(f"{'dist_cbd p값':20s} "
      f"{results['V1_기존']['ols_cbd_pval']:>12.3f} "
      f"{results['V2_기존+관악']['ols_cbd_pval']:>12.3f} "
      f"{results['V3_관악단독']['ols_cbd_pval']:>12.3f}")
print(f"{'잔차 Moran I':20s} "
      f"{results['V1_기존']['ols_resid_moran']:>12.4f} "
      f"{results['V2_기존+관악']['ols_resid_moran']:>12.4f} "
      f"{results['V3_관악단독']['ols_resid_moran']:>12.4f}")

print("\n[SLM 비교]")
print(f"{'':20s} {'V1_기존':>12s} {'V2_기존+관악':>12s} {'V3_관악단독':>12s}")
print(f"{'pseudo-R²':20s} "
      f"{results['V1_기존']['slm_r2']:>12.4f} "
      f"{results['V2_기존+관악']['slm_r2']:>12.4f} "
      f"{results['V3_관악단독']['slm_r2']:>12.4f}")
print(f"{'RMSE(log)':20s} "
      f"{results['V1_기존']['slm_rmse']:>12.4f} "
      f"{results['V2_기존+관악']['slm_rmse']:>12.4f} "
      f"{results['V3_관악단독']['slm_rmse']:>12.4f}")
print(f"{'MAE(만원)':20s} "
      f"{results['V1_기존']['slm_mae']:>12.0f} "
      f"{results['V2_기존+관악']['slm_mae']:>12.0f} "
      f"{results['V3_관악단독']['slm_mae']:>12.0f}")
print(f"{'ρ (공간자기회귀)':20s} "
      f"{results['V1_기존']['slm_rho']:>12.4f} "
      f"{results['V2_기존+관악']['slm_rho']:>12.4f} "
      f"{results['V3_관악단독']['slm_rho']:>12.4f}")
print(f"{'dist_cbd 계수':20s} "
      f"{results['V1_기존']['slm_cbd_coef']:>12.4f} "
      f"{results['V2_기존+관악']['slm_cbd_coef']:>12.4f} "
      f"{results['V3_관악단독']['slm_cbd_coef']:>12.4f}")
print(f"{'dist_cbd z값':20s} "
      f"{results['V1_기존']['slm_cbd_z']:>12.2f} "
      f"{results['V2_기존+관악']['slm_cbd_z']:>12.2f} "
      f"{results['V3_관악단독']['slm_cbd_z']:>12.2f}")
print(f"{'dist_cbd p값':20s} "
      f"{results['V1_기존']['slm_cbd_pval']:>12.3f} "
      f"{results['V2_기존+관악']['slm_cbd_pval']:>12.3f} "
      f"{results['V3_관악단독']['slm_cbd_pval']:>12.3f}")

print("\n" + "=" * 65)
print("  채택 기준 가이드:")
print("  1) SLM RMSE가 가장 낮은 버전")
print("  2) dist_cbd 계수가 유의(p<0.05)하고 부호가 음수(-)")
print("     → 업무지구 가까울수록 가격 상승 (이론 부합)")
print("  3) OLS 잔차 Moran's I가 가장 낮은 버전")
print("     → 공간 자기상관이 덜 남아있을수록 좋음")
print("=" * 65)

# 결과 CSV 저장
res_df.to_csv('data/tune_cbd_results.csv',
              encoding='utf-8-sig', index=True)
print("\n  저장: data/tune_cbd_results.csv")

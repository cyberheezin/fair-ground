"""
Step 1c: 전세 실거래가 수집 — 아파트 + 오피스텔 + 연립다세대
================================================================
엔드포인트: apis.data.go.kr (2024년부터 변경된 새 주소)
전세 필터: deposit > 0 AND monthlyRent == 0
매칭 기준: 건물명(aptNm) + 전용면적(excluUseAr) 반올림
126% 기준: 전세보증금(만원) > 공시가격(원) ÷ 10000 × 1.26
"""

import os
import time
import requests
import xmltodict
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import date
from dateutil.relativedelta import relativedelta

load_dotenv()
SERVICE_KEY  = os.getenv("LAND_PRICE_API_KEY")
RESULT_CSV   = "data/Int_Data_result.csv"
CACHE_CSV    = "data/jeonse_cache.csv"
LAWD_CD      = "11620"
# 최근 12개월 자동 계산
from datetime import date
from dateutil.relativedelta import relativedelta

today = date.today()
YEAR_MONTHS = []
for i in range(60, 0, -1):
    d = today - relativedelta(months=i)
    YEAR_MONTHS.append(f"{d.year}{str(d.month).zfill(2)}")
print(f"  수집 기간: {YEAR_MONTHS[0]} ~ {YEAR_MONTHS[-1]}")

# ─────────────────────────────────────────
# API 설정 — 새 엔드포인트 기준
# ─────────────────────────────────────────
API_CONFIGS = {
    "아파트": {
        "url": "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent",
        "name_field":    "aptNm",
        "area_field":    "excluUseAr",
        "deposit_field": "deposit",
        "monthly_field": "monthlyRent",
    },
    "오피스텔": {
        "url": "https://apis.data.go.kr/1613000/RTMSDataSvcOffiRent/getRTMSDataSvcOffiRent",
        "name_field":    "offiNm",
        "area_field":    "excluUseAr",
        "deposit_field": "deposit",
        "monthly_field": "monthlyRent",
    },
    "연립다세대": {
        "url": "https://apis.data.go.kr/1613000/RTMSDataSvcRHRent/getRTMSDataSvcRHRent",
        "name_field":    "mhouseNm",
        "area_field":    "excluUseAr",
        "deposit_field": "deposit",
        "monthly_field": "monthlyRent",
    },
}

# ─────────────────────────────────────────
# 1. API 호출
# ─────────────────────────────────────────
def fetch_one(url: str, lawd_cd: str, deal_ymd: str, service_key: str) -> list:
    params = {
        "serviceKey": service_key,
        "LAWD_CD":    lawd_cd,
        "DEAL_YMD":   deal_ymd,
        "numOfRows":  "1000",
        "pageNo":     "1",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = xmltodict.parse(resp.text)
        body  = data.get("response", {}).get("body", {})
        items = body.get("items", {})
        if not items:
            return []
        item = items.get("item", [])
        return [item] if isinstance(item, dict) else item
    except Exception as e:
        print(f"    [오류] {deal_ymd}: {e}")
        return []


def to_num(val, default=0.0):
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return default


# ─────────────────────────────────────────
# 2. 전체 수집
# ─────────────────────────────────────────
def collect_all(year_months: list, service_key: str) -> pd.DataFrame:
    if os.path.exists(CACHE_CSV):
        print(f"  캐시 로드: {CACHE_CSV}")
        df = pd.read_csv(CACHE_CSV)
        print(f"  캐시 건수: {len(df):,}건")
        return df

    all_records = []

    for prop_type, cfg in API_CONFIGS.items():
        print(f"\n  [{prop_type}] 수집 중...")
        type_cnt = 0
        for ym in tqdm(year_months, desc=f"    {prop_type}"):
            items = fetch_one(cfg["url"], LAWD_CD, ym, service_key)
            for r in items:
                deposit = to_num(r.get(cfg["deposit_field"], 0))
                monthly = to_num(r.get(cfg["monthly_field"], 0))

                # 전세 필터: 보증금 있고 월세 0
                if deposit <= 0 or monthly > 0:
                    continue

                name = str(r.get(cfg["name_field"], "")).strip()
                area = to_num(r.get(cfg["area_field"], 0))

                if not name or area <= 0:
                    continue

                all_records.append({
                    "prop_type": prop_type,
                    "apt_name":  name,
                    "priv_area": area,
                    "deposit":   int(deposit),  # 만원
                    "deal_ym":   ym,
                    "roadnm":    str(r.get("roadnm", "")).strip(),   # 도로명
                    "umdNm":     str(r.get("umdNm", "")).strip(),    # 법정동명
                    "jibun":     str(r.get("jibun", "")).strip(),    # 지번
                })
                type_cnt += 1
            time.sleep(0.05)

        print(f"    → {type_cnt:,}건 수집")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"\n  전세 수집 완료: {len(df):,}건")
    print(f"  유형별 건수:")
    for pt, cnt in df["prop_type"].value_counts().items():
        print(f"    {pt}: {cnt:,}건")
    print(f"  보증금 범위: {df['deposit'].min():,} ~ {df['deposit'].max():,} 만원")
    print(f"  보증금 중앙값: {df['deposit'].median():,.0f} 만원")

    df.to_csv(CACHE_CSV, index=False, encoding="utf-8-sig")
    print(f"  캐시 저장: {CACHE_CSV}")
    return df


# ─────────────────────────────────────────
# 3. 단지별·면적별 전세 중앙값 산출
# ─────────────────────────────────────────
def calc_median(df_jeonse: pd.DataFrame) -> pd.DataFrame:
    df = df_jeonse.copy()
    df["area_round"] = df["priv_area"].round(0)
    # deal_ym 기준 내림차순 정렬 후 그룹별 가장 최근 거래가 채택
    df = df.sort_values("deal_ym", ascending=False)
    latest_df = (
        df.groupby(["apt_name", "area_round"])
        .agg(
            median_deposit=("deposit", "first"),   # 가장 최근 거래
            latest_ym=("deal_ym", "first"),        # 최근 거래 시점
            trade_cnt=("deposit", "count"),        # 거래 건수
        )
        .reset_index()
    )
    print(f"  단지·면적 그룹: {len(latest_df):,}개")
    print(f"  최근 거래 시점 범위: {latest_df['latest_ym'].min()} ~ {latest_df['latest_ym'].max()}")
    return latest_df


# ─────────────────────────────────────────
# 4. 메인
# ─────────────────────────────────────────
print("▶ 전세 실거래가 수집 시작")
print(f"  대상: 아파트 + 오피스텔 + 연립다세대")
print(f"  기간: {YEAR_MONTHS[0]} ~ {YEAR_MONTHS[-1]}")
print(f"  지역: 관악구 ({LAWD_CD})")

df_jeonse = collect_all(YEAR_MONTHS, SERVICE_KEY)

if df_jeonse is None or len(df_jeonse) == 0:
    print("\n⚠ 수집된 전세 데이터가 없습니다. API 키를 확인해주세요.")
    exit()

# 중앙값 산출
print("\n▶ 단지별·면적별 전세 중앙값 산출 중...")
median_df = calc_median(df_jeonse)

# ─────────────────────────────────────────
# 5. 분석 결과와 매칭
# ─────────────────────────────────────────
print("\n▶ Int_Data_result.csv와 매칭 중...")
df_result = pd.read_csv(RESULT_CSV, low_memory=False)
# 기존 전세 관련 컬럼 초기화 (재실행 시 중복 방지)
for col in ["median_deposit", "over_126pct", "match_type"]:
    if col in df_result.columns:
        df_result = df_result.drop(columns=[col])
print(f"  분석 데이터: {len(df_result):,}세대")

df_result["area_round"] = df_result["priv_area"].round(0)

# ── 1차 매칭: 단지명 + 면적
print("  1차 매칭: 단지명 + 면적...")
df_result["apt_name_clean"] = df_result["apt_name"].astype(str).str.strip()
median_df["apt_name_clean"] = median_df["apt_name"].astype(str).str.strip()

df_merged = df_result.merge(
    median_df[["apt_name_clean", "area_round", "median_deposit"]],
    on=["apt_name_clean", "area_round"],
    how="left"
)
match1 = df_merged["median_deposit"].notna().sum()
print(f"  1차 매칭 성공: {match1:,}건 ({match1/len(df_merged)*100:.1f}%)")

# ── 2차 매칭: 도로명 키워드 + 면적 (1차 미매칭 대상)
print("  2차 매칭: 도로명 키워드 + 면적...")

# 전세 데이터에 도로명 기반 키 생성 (도로명에서 번호 제거, 핵심 도로명만 추출)
df_jeonse2 = df_jeonse.copy()
df_jeonse2["area_round"] = df_jeonse2["priv_area"].round(0)
# 도로명에서 숫자 제거해서 핵심 도로명 추출 (예: "관악로30길 27" → "관악로30길")
df_jeonse2["road_key"] = df_jeonse2["roadnm"].str.extract(r"^([^\d]+\d*[^\s\d]*)")[0].str.strip()

road_median = (
    df_jeonse2.groupby(["road_key", "area_round"])["deposit"]
    .median()
    .reset_index()
    .rename(columns={"deposit": "median_deposit_road"})
)

# result에도 도로명 키 생성
df_merged["road_key"] = df_merged["road_name"].str.extract(r"([^\s]+길|[^\s]+로|[^\s]+대로)")[0].str.strip()

df_merged = df_merged.merge(
    road_median,
    on=["road_key", "area_round"],
    how="left"
)

# 1차 미매칭인 경우 2차 결과로 보완
mask_fill = df_merged["median_deposit"].isna() & df_merged["median_deposit_road"].notna()
df_merged.loc[mask_fill, "median_deposit"] = df_merged.loc[mask_fill, "median_deposit_road"]
df_merged.loc[mask_fill, "match_type"] = "도로명매칭"
df_merged.loc[df_merged["match_type"].isna() & df_merged["median_deposit"].notna(), "match_type"] = "단지명매칭"

df_merged = df_merged.drop(columns=["median_deposit_road", "road_key"], errors="ignore")

match2 = df_merged["median_deposit"].notna().sum()
print(f"  2차 매칭 후 성공: {match2:,}건 ({match2/len(df_merged)*100:.1f}%)")
print(f"  2차 매칭으로 추가: {match2-match1:,}건")
match_cnt = match2

# ─────────────────────────────────────────
# 6. 126% 초과 여부 계산
# ─────────────────────────────────────────
print("\n▶ 전세대출 가능 여부 계산 중...")
notice_man = df_merged["notice_amt"] / 10000

df_merged["over_126pct"] = np.where(
    df_merged["median_deposit"].notna(),
    df_merged["median_deposit"] > notice_man * 1.26,
    None
)

over_cnt    = (df_merged["over_126pct"] == True).sum()
ok_cnt      = (df_merged["over_126pct"] == False).sum()
unknown_cnt = df_merged["over_126pct"].isna().sum()

print(f"\n  결과:")
print(f"    전세대출 불가 (보증금 > 공시가격 × 126%): {over_cnt:,}건 ({over_cnt/len(df_merged)*100:.1f}%)")
print(f"    전세대출 가능 (보증금 ≤ 공시가격 × 126%): {ok_cnt:,}건 ({ok_cnt/len(df_merged)*100:.1f}%)")
print(f"    전세 데이터 없음 (미확인):                 {unknown_cnt:,}건 ({unknown_cnt/len(df_merged)*100:.1f}%)")

# ─────────────────────────────────────────
# 7. 저장
# ─────────────────────────────────────────
df_merged = df_merged.drop(columns=["area_round","apt_name_clean"], errors="ignore")
df_merged.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

print(f"\n▶ 저장 완료: {RESULT_CSV}")
print(f"   추가 컬럼: median_deposit(전세 중앙값 만원), over_126pct(대출가능여부)")
print(f"\n✅ Step 1c 완료")

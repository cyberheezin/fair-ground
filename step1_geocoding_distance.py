"""
Step 1: 지오코딩 + 거리 변수 계산
=====================================
목적:
  - 카카오 로컬 API로 도로명주소 → 위경도 변환
  - 공공 GIS 데이터(지하철역, 업무지구, 근린공원)와 공간 조인
  - 8개 독립변수 중 거리 변수 3개(dist_subway, dist_cbd, dist_park) 산출
  - 건축경과연수(age) 추가
  - 결과를 Int_Data_geo.csv로 저장

사전 준비:
  pip install pandas geopandas requests shapely pyproj tqdm

카카오 API 키 발급:
  https://developers.kakao.com → 앱 생성 → REST API 키 복사
"""

import os
import time
import json
import math
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# ─────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
INPUT_CSV     = "data/Int_Data.csv"
OUTPUT_CSV    = "data/Int_Data_geo.csv"
GEOCODE_CACHE = "data/geocode_cache.json"

BASE_YEAR = 2026  # 건축경과연수 기준 연도

# 업무지구 중심점 (직접 정의 — 관악구 접근성 기준)
# 여의도, 강남, 광화문 세 곳의 위경도
CBD_POINTS = [
    {"name": "여의도",  "lat": 37.5219, "lon": 126.9245},
    {"name": "강남",    "lat": 37.4979, "lon": 127.0276},
    {"name": "광화문",  "lat": 37.5759, "lon": 126.9769},
]

# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
print("▶ 데이터 로드 중...")
df = pd.read_csv(INPUT_CSV, encoding="cp949")
print(f"  전체 세대: {len(df):,}행 / 고유 주소: {df['road_name'].nunique():,}개")


# ─────────────────────────────────────────
# 2. 카카오 지오코딩
# ─────────────────────────────────────────
def kakao_geocode(address: str, api_key: str) -> dict | None:
    """
    카카오 로컬 API - 주소 검색
    반환: {"lat": float, "lon": float} 또는 None
    """
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address, "analyze_type": "similar"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        if docs:
            return {"lat": float(docs[0]["y"]), "lon": float(docs[0]["x"])}
        # 주소 검색 실패 시 키워드 검색으로 재시도
        url2 = "https://dapi.kakao.com/v2/local/search/keyword.json"
        resp2 = requests.get(url2, headers=headers,
                             params={"query": address}, timeout=5)
        docs2 = resp2.json().get("documents", [])
        if docs2:
            return {"lat": float(docs2[0]["y"]), "lon": float(docs2[0]["x"])}
    except Exception as e:
        print(f"  [오류] {address}: {e}")
    return None


def run_geocoding(df: pd.DataFrame, api_key: str, cache_path: str) -> dict:
    """
    고유 주소에 대해 지오코딩 수행. 캐시 파일 활용.
    반환: {주소: {"lat": float, "lon": float}, ...}
    """
    # 캐시 로드
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  캐시 로드: {len(cache):,}건")

    unique_addrs = df["road_name"].dropna().unique().tolist()
    todo = [a for a in unique_addrs if a not in cache]
    print(f"  신규 지오코딩 대상: {len(todo):,}건")

    for addr in tqdm(todo, desc="  지오코딩"):
        result = kakao_geocode(addr, api_key)
        cache[addr] = result
        time.sleep(0.05)  # API rate limit 방지 (20 req/s 이하 유지)

        # 100건마다 캐시 저장 (중단 시 손실 최소화)
        if len(cache) % 100 == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)

    # 최종 캐시 저장
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    success = sum(1 for v in cache.values() if v is not None)
    print(f"  지오코딩 성공: {success:,}/{len(cache):,}건 "
          f"({success/len(cache)*100:.1f}%)")
    return cache


print("\n▶ 지오코딩 시작...")
geocode_cache = run_geocoding(df, KAKAO_API_KEY, GEOCODE_CACHE)

# 위경도 컬럼 추가
df["lat"] = df["road_name"].map(
    lambda a: geocode_cache.get(a, {}).get("lat") if geocode_cache.get(a) else None
)
df["lon"] = df["road_name"].map(
    lambda a: geocode_cache.get(a, {}).get("lon") if geocode_cache.get(a) else None
)
print(f"  위경도 결측: {df['lat'].isna().sum():,}건")


# ─────────────────────────────────────────
# 3. 거리 계산 유틸
# ─────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """두 위경도 간 직선거리 (km), Haversine 공식"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ─────────────────────────────────────────
# 4. 업무지구까지 거리 (dist_cbd)
# ─────────────────────────────────────────
print("\n▶ 업무지구까지 거리 계산...")

def min_dist_to_cbd(lat, lon):
    """세 업무지구 중 가장 가까운 곳까지의 거리 (km)"""
    if pd.isna(lat) or pd.isna(lon):
        return None
    return min(
        haversine_km(lat, lon, p["lat"], p["lon"])
        for p in CBD_POINTS
    )

df["dist_cbd"] = df.apply(
    lambda r: min_dist_to_cbd(r["lat"], r["lon"]), axis=1
)
print(f"  dist_cbd 완료 (평균: {df['dist_cbd'].mean():.2f} km)")


# ─────────────────────────────────────────
# 5. 지하철역까지 거리 (dist_subway)
# ─────────────────────────────────────────
print("\n▶ 지하철역 GIS 데이터 로드 및 거리 계산...")
"""
서울 지하철역 좌표 데이터 출처 (무료 공개):
  서울 열린데이터광장 → '서울시 역사마스터 정보'
  https://data.seoul.go.kr/dataList/OA-121/S/1/datasetView.do
  또는 공공데이터포털 → '전국 도시철도 역사 정보'

아래는 관악구 인근 주요 역 위경도를 직접 정의한 fallback 목록.
실제 운영 시 위 공공데이터를 pandas로 불러와서 대체하세요:
  subway_df = pd.read_csv("seoul_subway_stations.csv", encoding="utf-8")
  subway_points = list(zip(subway_df["위도"], subway_df["경도"]))
"""

# 관악구 인근 주요 역 (2호선, 4호선, 7호선, 경전철 등)
SUBWAY_STATIONS = [
    # 2호선
    (37.4812, 126.9527, "서울대입구"),
    (37.4847, 126.9296, "낙성대"),
    (37.4843, 126.9014, "사당"),
    (37.4965, 126.9255, "방배"),
    (37.5047, 126.9244, "서초"),
    # 4호선
    (37.4767, 126.9814, "남태령"),
    (37.4749, 126.9683, "선바위"),
    # 7호선
    (37.4825, 126.9007, "남성"),
    (37.4791, 126.9066, "숭실대입구"),
    (37.4968, 126.9234, "이수"),
    (37.5136, 126.9007, "장승배기"),
    # 경전철(신림선)
    (37.4840, 126.9295, "서원"),
    (37.4760, 126.9314, "신림"),
    (37.4715, 126.9299, "당곡"),
    (37.4740, 126.9159, "서울대벤처타운"),
    (37.4769, 126.9027, "관악산"),
]

def min_dist_to_subway(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None
    return min(haversine_km(lat, lon, s[0], s[1]) for s in SUBWAY_STATIONS)

df["dist_subway"] = df.apply(
    lambda r: min_dist_to_subway(r["lat"], r["lon"]), axis=1
)
print(f"  dist_subway 완료 (평균: {df['dist_subway'].mean():.3f} km)")


# ─────────────────────────────────────────
# 6. 근린공원까지 거리 (dist_park)
# ─────────────────────────────────────────
print("\n▶ 근린공원 GIS 데이터 로드 및 거리 계산...")
"""
근린공원 데이터 출처:
  국토교통부 공원 정보 (공공데이터포털)
  https://www.data.go.kr → '전국 공원정보 표준 데이터'
  → 시도=서울, 시군구=관악구 필터링

GeoJSON/CSV 파일을 받은 경우:
  parks_df = pd.read_csv("gwanak_parks.csv", encoding="utf-8")
  park_points = list(zip(parks_df["위도"], parks_df["경도"]))

아래는 관악구 주요 근린공원 fallback 목록 (직접 정의)
"""

PARKS = [
    (37.4870, 126.9530, "관악산근린공원"),
    (37.4994, 126.9193, "보라매공원"),
    (37.4964, 126.9338, "봉천근린공원"),
    (37.4888, 126.9401, "삼성산시민공원"),
    (37.5049, 126.9491, "난곡근린공원"),
    (37.4833, 126.9200, "신림근린공원"),
    (37.4905, 126.9260, "행운근린공원"),
    (37.4851, 126.9607, "청룡근린공원"),
    (37.5030, 126.9320, "은천근린공원"),
    (37.4780, 126.9450, "낙성대근린공원"),
]

def min_dist_to_park(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None
    return min(haversine_km(lat, lon, p[0], p[1]) for p in PARKS)

df["dist_park"] = df.apply(
    lambda r: min_dist_to_park(r["lat"], r["lon"]), axis=1
)
print(f"  dist_park 완료 (평균: {df['dist_park'].mean():.3f} km)")


# ─────────────────────────────────────────
# 7. 나머지 독립변수 정리
# ─────────────────────────────────────────
print("\n▶ 파생 변수 생성...")

# 건축경과연수
df["age"] = BASE_YEAR - df["built_year"]

# 층수: 지하층(-1 등)은 0으로 클리핑
df["floor_clean"] = df["floor"].clip(lower=0)

# 고저 더미: 현재 데이터에 지형 정보 없음 → 추후 DEM 데이터로 보완 가능
# 임시로 NaN 처리 (Step 2 전까지 외부 데이터 조인 필요)
if "elevation_dummy" not in df.columns:
    df["elevation_dummy"] = None
    print("  ⚠ 고저 더미변수 없음 — DEM 데이터 조인 필요 (아래 안내 참조)")

# 공시지가: 현재 데이터에 별도 없음 → notice_amt를 priv_area로 나눠 단위면적당 값 사용
# 실제 개별공시지가는 국토부 '부동산 공시가격 알리미' API에서 별도 수집 필요
if "land_price" not in df.columns:
    df["land_price"] = None
    print("  ⚠ 공시지가 없음 — 국토부 공시지가 API 조인 필요 (아래 안내 참조)")

# 단위 통일: median_price (만원) → 원 단위로 변환
df["median_price_won"] = df["median_price"] * 10000

print("\n▶ 독립변수 현황:")
for col in ["land_price", "age", "priv_area", "floor_clean",
            "dist_cbd", "dist_subway", "dist_park", "elevation_dummy"]:
    null_cnt = df[col].isna().sum()
    status = "✓" if null_cnt == 0 else f"✗ 결측 {null_cnt:,}건"
    print(f"  {col:20s}: {status}")


# ─────────────────────────────────────────
# 8. 저장
# ─────────────────────────────────────────
print(f"\n▶ 저장 중: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"  완료. {len(df):,}행 × {len(df.columns)}컬럼")
print(f"\n  주요 컬럼 목록:")
print(f"  {df.columns.tolist()}")


# ─────────────────────────────────────────
# 9. 보완 필요 데이터 안내
# ─────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 추가 수집 필요한 외부 데이터
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 공시지가 (land_price)
   → 출처: 국토교통부 '개별공시지가 정보' API
     https://www.data.go.kr/data/15056947/openapi.do
   → 도로명주소 또는 PNU 코드(bjd_code + 번지)로 조회
   → 현재 데이터의 bjd_code + bunji 컬럼 활용 가능

2. 고저 더미 (elevation_dummy: 0=경사지, 1=평지)
   → 출처: 국토지리정보원 수치표고모델(DEM) 1m 해상도
     https://map.ngii.go.kr/ms/pblictn/RDNM0082.do
   → 각 주택 좌표의 고도값 추출 후 임계값(예: 경사도 5% 미만=평지) 기준으로 분류

3. 지하철역 전체 목록 (더 정확한 거리 계산용)
   → 출처: 서울 열린데이터광장 '서울시 역사마스터 정보'
     https://data.seoul.go.kr/dataList/OA-121/S/1/datasetView.do

4. 근린공원 전체 목록
   → 출처: 공공데이터포털 '전국 공원정보 표준 데이터'
     https://www.data.go.kr/data/15012690/fileData.do
   → 관악구 필터링 후 위경도 컬럼 활용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

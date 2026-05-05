"""
공시지가 수집 스크립트 (국토부 공개 API)
============================================
API: 국토교통부 개별공시지가 정보
등록: https://www.data.go.kr/data/15056947/openapi.do
     → 활용 신청(무료) → 서비스키 발급 (보통 1~2일 소요)

기준: bjd_code(법정동코드 10자리) + bunji(번지) 조합으로 조회
"""

import requests
import pandas as pd
import time
import json
import os
from tqdm import tqdm

import os
from dotenv import load_dotenv
load_dotenv()
SERVICE_KEY = os.getenv("95f179f238434eeff7ff1717be9e557f1c9c2dbbc0eeeb5e00e1f529a8763737")
INPUT_CSV   = "data/Int_Data_geo.csv"
CACHE_PATH  = "data/land_price_cache.json"
OUTPUT_CSV  = "data/Int_Data_geo.csv"

BASE_YEAR = 2025   # 공시기준연도 (데이터의 base_year와 일치)

def get_land_price(bjd_code: str, bunji: str, ho: str, year: int, svc_key: str) -> int | None:
    """
    PNU = 법정동코드(10자리) + 산/대지구분(1) + 본번(4자리) + 부번(4자리)
    bunji=본번, ho=부번
    """
    try:
        bcode  = str(bjd_code).strip().zfill(10)
        main   = str(int(float(bunji))).zfill(4)   # 본번
        sub    = str(int(float(ho))).zfill(4)       # 부번
        pnu    = f"{bcode}1{main}{sub}"             # 대지=1

        url = "http://apis.data.go.kr/1613000/IndvdLandPriceService/getIndvdLandPriceInfo"
        params = {
            "serviceKey": svc_key,
            "pnu":        pnu,
            "stdrYear":   str(year),
            "numOfRows":  "1",
            "pageNo":     "1",
            "_type":      "json",
        }
        r = requests.get(url, params=params, timeout=10)
        body = r.json().get("response", {}).get("body", {})
        items = body.get("items", "")
        if items and items != "":
            item = items["item"]
            if isinstance(item, list):
                item = item[0]
            return int(item.get("pblntfPclnd", 0))
    except Exception:
        pass
    return None


def run_land_price_collection(df: pd.DataFrame, svc_key: str) -> pd.DataFrame:
    # 캐시 로드
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)

    unique_keys = df[["bjd_code", "bunji", "ho"]].drop_duplicates()
    todo = [
        (str(row.bjd_code), str(row.bunji), str(row.ho))
        for _, row in unique_keys.iterrows()
        if f"{row.bjd_code}_{row.bunji}_{row.ho}" not in cache
    ]
    print(f"  공시지가 조회 대상: {len(todo):,}건")

    for bjd, bunji, ho in tqdm(todo, desc="  공시지가 수집"):
        key = f"{bjd}_{bunji}_{ho}"
        result = get_land_price(bjd, bunji, ho, BASE_YEAR, svc_key)
        cache[key] = result
        time.sleep(0.1)

        if len(cache) % 100 == 0:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f)

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    # 데이터프레임에 매핑
    df["land_price"] = df.apply(
        lambda r: cache.get(f"{r['bjd_code']}_{r['bunji']}_{r['ho']}"), axis=1
    )
    success = df["land_price"].notna().sum()
    print(f"  수집 성공: {success:,}/{len(df):,}건 ({success/len(df)*100:.1f}%)")
    return df


print("▶ 데이터 로드...")
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

print("▶ 공시지가 수집 시작...")
df = run_land_price_collection(df, SERVICE_KEY)

print(f"▶ 저장: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("완료!")

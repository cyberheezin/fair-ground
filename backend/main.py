"""
관악구 공시가격 형평성 진단 플랫폼 — FastAPI 백엔드
실행: uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import os

app = FastAPI(title="관악구 공시가격 형평성 진단 플랫폼")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "final_result.csv")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

df = pd.read_csv(DATA_PATH, low_memory=False)
df = df[df['lat'].notna() & df['lon'].notna()].copy()
print(f"데이터 로드 완료: {len(df):,}세대")

def safe(val):
    if val is None: return None
    try:
        if isinstance(val, float) and np.isnan(val): return None
    except: pass
    return val

def row_to_dict(r):
    price_src = safe(r.get('price_source'))
    over      = safe(r.get('over_126'))

    def sf(key):
        v = r.get(key)
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)): return None
            return round(float(v), 4)
        except: return None

    return {
        "rid":          safe(r.get("rid")),
        "apt_name":     safe(r.get("apt_name")),
        "road_name":    str(safe(r.get("road_name")) or "").replace("서울특별시 관악구 ", ""),
        "dong":         safe(r.get("bj_dong_name")),
        "purpose":      safe(r.get("purpose")),
        "floor":        safe(r.get("floor")),
        "priv_area":    safe(r.get("priv_area")),
        "built_year":   safe(r.get("built_year")),
        "lat":          safe(r.get("lat")),
        "lon":          safe(r.get("lon")),
        "notice_amt":   sf("notice_man"),       # 공시가격 (만원)
        "median_price": sf("median_price"),     # 최근 매매가 (만원)
        "price_source": price_src,
        "jeonse_price": sf("jeonse_price"),     # 전세보증금 (만원)
        "actual_ratio": sf("actual_ratio"),     # 실제 현실화율
        "fair_price":   sf("fair_price_man"),   # 적정 공시가격 (만원)
        "tier_ratio":   sf("tier_ratio"),       # 적용 현실화율
        "equity_ratio": sf("equity_ratio"),     # 형평성 비율
        "equity_grade": safe(r.get("equity_grade")),  # 형평성 등급
        "over_126":     over if price_src == "전세가 역산" else None,
    }

@app.get("/api/search")
def search(q: str = Query(..., min_length=1)):
    q = q.strip()
    mask = (df['apt_name'].str.contains(q, na=False) |
            df['road_name'].str.contains(q, na=False))
    results = df[mask].head(20)
    return {"count": len(results),
            "items": [row_to_dict(r) for _, r in results.iterrows()]}

@app.get("/api/nearby")
def nearby(lat: float = Query(...), lon: float = Query(...),
           radius: float = Query(default=0.2)):
    dlat = df['lat'] - lat
    dlon = df['lon'] - lon
    dist = np.sqrt((dlat * 111.0)**2 +
                   (dlon * 111.0 * np.cos(np.radians(lat)))**2)
    mask = dist <= radius
    results = df[mask].copy()
    results['_dist'] = dist[mask]
    results = results.nsmallest(50, '_dist')
    return {"count": len(results),
            "items": [row_to_dict(r) for _, r in results.iterrows()]}

@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# requirements.txt 파일에 아래 두 줄을 꼭 적어주세요:
# fastapi
# uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import math

app = FastAPI()

# 프론트엔드(깃허브 페이지)에서 접근할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "통합사회 빅데이터 분석 서버가 정상 작동 중입니다."}

# ==========================================
# 📡 [기능 1] 위성 데이터 분석 (NDVI 및 도시화 지수)
# ==========================================
@app.get("/analyze-satellite")
def analyze_satellite(region: str, year: int):
    # 가상의 알고리즘: 연도가 최신일수록 식생지수(NDVI)는 감소, 도시 열섬 강도는 증가
    base_year = 1990
    years_passed = max(0, year - base_year)
    
    # 식생지수 (1.0에 가까울수록 숲이 울창함)
    ndvi = max(0.2, 0.85 - (years_passed * 0.015))
    
    # 도시 열섬 강도 (도심과 외곽의 온도차)
    heat_island = min(5.5, 0.5 + (years_passed * 0.12))
    
    # 지역별 특성 가중치
    if "서울" in region or "seoul" in region.lower():
        heat_island += 1.5
        ndvi -= 0.1
    elif "제주" in region or "강원" in region:
        heat_island -= 1.0
        ndvi += 0.2

    insight = f"{year}년 {region}의 위성 데이터 분석 결과, "
    if ndvi < 0.4:
        insight += "심각한 식생 파괴와 급격한 도시화가 관찰됩니다. 열섬 현상에 대비해야 합니다."
    else:
        insight += "비교적 양호한 식생(녹지)이 보존되어 있습니다."

    return {
        "region": region,
        "year": year,
        "ndvi": round(ndvi, 2),
        "heat_island": round(heat_island, 1),
        "insight": insight
    }

# ==========================================
# 👨‍👩‍👧‍👦 [기능 2] 인구 예측 AI 모델 (로지스틱 성장 곡선)
# ==========================================
@app.get("/predict-population")
def predict_population(city: str, current_pop: int, target_year: int):
    current_year = 2026
    years_diff = target_year - current_year
    
    if years_diff < 0:
        return {"error": "목표 연도는 현재(2026년)보다 미래여야 합니다."}
    
    # 가상의 알고리즘: 지역 이름에 따른 성장률(r) 및 수용 환경(K) 차등 적용
    # 실제로는 통계청 API에서 데이터를 끌어와서 ARIMA나 선형회귀 모델을 돌리는 자리입니다.
    if city in ["서울", "부산", "대구", "대전", "광주"]:
        growth_rate = -0.008  # 대도시는 저출산/고령화로 인구 감소 추세 적용
        carrying_capacity = current_pop * 1.1 
    elif city in ["화성", "평택", "세종", "용인"]:
        growth_rate = 0.035   # 신도시/산업단지는 인구 급증 적용
        carrying_capacity = current_pop * 2.5
    else:
        growth_rate = -0.015  # 지방 중소도시는 인구 소멸 추세 적용 (가파른 감소)
        carrying_capacity = current_pop * 1.0

    # 자연 지수 모델 (P = P0 * e^(rt))
    predicted_pop = current_pop * math.exp(growth_rate * years_diff)
    
    # 증감률 계산
    percent_change = ((predicted_pop - current_pop) / current_pop) * 100
    
    insight = f"{city}의 {target_year}년 예상 인구는 약 {int(predicted_pop):,}명입니다. "
    if percent_change > 0:
        insight += f"현재 대비 인구가 {percent_change:.1f}% 증가하며 도시 인프라 확충이 시급합니다."
    else:
        insight += f"현재 대비 인구가 {abs(percent_change):.1f}% 감소하며 '지방 소멸' 현상에 대한 대비책이 필요합니다."

    return {
        "city": city,
        "current_pop": current_pop,
        "target_year": target_year,
        "predicted_pop": int(predicted_pop),
        "percent_change": round(percent_change, 1),
        "insight": insight
    }

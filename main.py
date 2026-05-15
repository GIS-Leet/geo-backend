from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 📊 [데이터베이스 세팅] pandas DataFrame 준비
# ==========================================
# [선생님 전용 팁] 실제 서버 운영 시에는 아래처럼 CSV 파일을 불러옵니다.
# pop_df = pd.read_csv("population_data.csv")
# sat_df = pd.read_csv("satellite_data.csv")

# 현재는 테스트를 위해 코드 내부에서 pandas 데이터프레임을 직접 생성합니다.
# 1. 인구 데이터 (과거~현재)
pop_data = {
    'year': [2000, 2005, 2010, 2015, 2020, 2025],
    '서울': [10370000, 10160000, 10050000, 9900000, 9660000, 9400000], # 감소 추세
    '화성': [210000, 280000, 500000, 600000, 850000, 1000000],        # 급증 추세
    '부산': [3800000, 3650000, 3560000, 3440000, 3390000, 3280000]    # 감소 추세
}
pop_df = pd.DataFrame(pop_data)

# 2. 위성(식생/도시화) 데이터
sat_data = {
    'region': ['서울', '서울', '서울', '화성', '화성', '화성', '제주', '제주', '제주'],
    'year': [2000, 2010, 2020, 2000, 2010, 2020, 2000, 2010, 2020],
    'ndvi': [0.55, 0.48, 0.42, 0.78, 0.65, 0.51, 0.85, 0.83, 0.80], # 식생지수 (높을수록 숲)
    'lst': [24.5, 25.8, 27.2, 22.1, 23.5, 25.4, 21.0, 21.5, 22.0]   # 지표면 온도 (열섬)
}
sat_df = pd.DataFrame(sat_data)


@app.get("/")
def home():
    return {"message": "데이터 사이언스 백엔드 서버가 가동 중입니다."}

# ==========================================
# 🛰️ [기능 1] pandas를 활용한 위성 데이터 검색 및 추세 분석
# ==========================================
@app.get("/analyze-satellite")
def analyze_satellite(region: str, year: int):
    # 1. 입력된 지역의 데이터만 pandas 필터링
    region_df = sat_df[sat_df['region'] == region]
    
    if region_df.empty:
        return {"insight": f"'{region}'의 위성 데이터가 DB에 없습니다. (현재: 서울, 화성, 제주 지원)", "ndvi": 0, "heat_island": 0}
    
    # 2. scikit-learn 선형회귀를 이용해 원하는 연도의 NDVI, LST 예측
    # (과거 데이터를 학습하여 입력된 year의 값을 추론)
    X = region_df[['year']]
    
    model_ndvi = LinearRegression().fit(X, region_df['ndvi'])
    pred_ndvi = model_ndvi.predict([[year]])[0]
    
    model_lst = LinearRegression().fit(X, region_df['lst'])
    pred_lst = model_lst.predict([[year]])[0]

    # 3. 인사이트 도출
    insight = f"{year}년 {region}의 머신러닝 추론 결과, "
    if pred_ndvi < 0.4:
        insight += "식생 파괴가 심각하며 "
    else:
        insight += "식생이 양호하며 "
        
    if pred_lst > 26.0:
        insight += "강력한 열섬 현상이 경고됩니다."
    else:
        insight += "표면 온도가 안정적입니다."

    return {
        "region": region,
        "year": year,
        "ndvi": round(float(pred_ndvi), 2),
        "heat_island": round(float(pred_lst), 1),
        "insight": insight
    }


# ==========================================
# 🤖 [기능 2] scikit-learn을 활용한 머신러닝 인구 예측
# ==========================================
@app.get("/predict-population")
def predict_population(city: str, current_pop: int, target_year: int):
    if city not in pop_df.columns:
        return {"error": f"'{city}'의 역사적 인구 데이터가 DB에 없습니다. (현재: 서울, 화성, 부산 지원)"}
    
    # 1. 머신러닝 모델 학습 준비
    X_train = pop_df[['year']]  # 독립 변수 (연도)
    y_train = pop_df[city]      # 종속 변수 (해당 도시의 과거 인구)
    
    # 2. scikit-learn 선형 회귀(Linear Regression) 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 3. AI 모델에 목표 연도(target_year)를 넣어 미래 인구 예측
    predicted_pop = model.predict([[target_year]])[0]
    
    # 예측값이 음수가 나오는 오류 방지 (인구는 0 이하가 될 수 없음)
    predicted_pop = max(0, int(predicted_pop))
    
    # 4. 증감률 계산
    percent_change = ((predicted_pop - current_pop) / current_pop) * 100
    
    insight = f"통계청 과거 데이터를 학습한 AI 예측 결과, {target_year}년 {city}의 인구는 약 {predicted_pop:,}명으로 예상됩니다. "
    if percent_change > 0:
        insight += "지속적인 인구 유입에 대비한 인프라 확충이 필요합니다."
    else:
        insight += "인구 감소 추세가 확인되며, 지역 균형 발전 정책이 요구됩니다."

    return {
        "city": city,
        "current_pop": current_pop,
        "target_year": target_year,
        "predicted_pop": predicted_pop,
        "percent_change": round(percent_change, 1),
        "insight": insight
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

# 프론트엔드(깃허브 홈페이지)에서 이 파이썬 서버에 접근할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 📊 [데이터베이스 세팅] pandas DataFrame 준비
# ==========================================
# (나중에 진짜 통계청/기상청 데이터를 구하시면 pd.read_csv("파일.csv")로 바꾸시면 됩니다.)

# 1. 인구 데이터 (과거~현재)
pop_data = {
    'year': [2000, 2005, 2010, 2015, 2020, 2025],
    '서울': [10370000, 10160000, 10050000, 9900000, 9660000, 9400000],
    '화성': [210000, 280000, 500000, 600000, 850000, 1000000],
    '부산': [3800000, 3650000, 3560000, 3440000, 3390000, 3280000],
    '제주': [540000, 560000, 570000, 620000, 670000, 680000]
}
pop_df = pd.DataFrame(pop_data)

# 2. 위성(식생/도시화) 데이터
sat_data = {
    'region': ['서울', '서울', '서울', '화성', '화성', '화성', '제주', '제주', '제주', '부산', '부산', '부산'],
    'year': [2000, 2010, 2020, 2000, 2010, 2020, 2000, 2010, 2020, 2000, 2010, 2020],
    'ndvi': [0.55, 0.48, 0.42, 0.78, 0.65, 0.51, 0.85, 0.83, 0.80, 0.60, 0.55, 0.50], # 식생지수
    'lst': [24.5, 25.8, 27.2, 22.1, 23.5, 25.4, 21.0, 21.5, 22.0, 23.5, 24.5, 25.5]   # 지표면 온도
}
sat_df = pd.DataFrame(sat_data)

# 3. 지도 표시용 지역별 위경도 좌표 데이터 (Leaflet.js 연동용)
region_coords = {
    "서울": [37.5665, 126.9780],
    "화성": [37.1995, 126.8311],
    "제주": [33.4996, 126.5312],
    "부산": [35.1796, 129.0756]
}


@app.get("/")
def home():
    return {"message": "통합사회 AI 빅데이터 백엔드 서버가 정상 작동 중입니다."}


# ==========================================
# 🛰️ [기능 1] 위성 데이터 분석 (지도 연동)
# ==========================================
@app.get("/analyze-satellite")
def analyze_satellite(region: str, year: int):
    # 1. 지역 좌표 가져오기 (없으면 한반도 중앙값)
    coords = region_coords.get(region, [36.5, 127.5])
    
    # 2. 입력된 지역의 데이터 필터링
    region_df = sat_df[sat_df['region'] == region]
    
    if region_df.empty:
        # DB에 없는 지역일 경우 가상의 수식으로 데이터 생성 방어
        fake_ndvi = max(0.2, (0.85 - ((year - 1990) * 0.015)))
        fake_heat = min(5.5, (0.5 + ((year - 1990) * 0.12)))
        return {
            "region": region,
            "year": year,
            "coords": coords,
            "ndvi": round(fake_ndvi, 2),
            "heat_island": round(fake_heat, 1),
            "insight": f"DB에 '{region}' 데이터가 없어 임의의 시뮬레이션 결과를 출력합니다."
        }
    
    # 3. 선형회귀를 이용해 입력된 year의 NDVI, LST 예측 추론
    X = region_df[['year']]
    
    model_ndvi = LinearRegression().fit(X, region_df['ndvi'])
    pred_ndvi = model_ndvi.predict([[year]])[0]
    
    model_lst = LinearRegression().fit(X, region_df['lst'])
    pred_lst = model_lst.predict([[year]])[0]

    # 4. 인사이트 생성
    insight = f"{year}년 {region}의 위성 데이터 AI 분석 결과, "
    if pred_ndvi < 0.4:
        insight += "식생(녹지) 파괴가 상당 부분 진행되었으며 "
    else:
        insight += "비교적 양호한 식생이 보존되어 있으며 "
        
    if pred_lst > 26.0:
        insight += "도심의 열섬 현상 심화가 우려됩니다."
    else:
        insight += "지표면 온도가 안정적인 수준입니다."

    return {
        "region": region,
        "year": year,
        "coords": coords,
        "ndvi": round(float(pred_ndvi), 2),
        "heat_island": round(float(pred_lst), 1),
        "insight": insight
    }


# ==========================================
# 👨‍👩‍👧‍👦 [기능 2] 인구 예측 AI 모델 (차트 연동)
# ==========================================
@app.get("/predict-population")
def predict_population(city: str, current_pop: int, target_year: int):
    if city not in pop_df.columns:
        return {"error": f"'{city}'의 과거 인구 데이터가 DB에 없습니다. (현재: 서울, 화성, 부산, 제주 지원)"}
    
    # 1. AI 머신러닝 학습 (과거 연도와 인구 데이터 핏팅)
    X_train = pop_df[['year']]
    y_train = pop_df[city]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 2. 목표 연도 단일 예측 (음수 방지 로직 포함)
    predicted_pop = model.predict([[target_year]])[0]
    predicted_pop = max(0, int(predicted_pop))
    
    # 3. Chart.js 그래프용 시계열 데이터(배열) 생성
    # (2000년부터 사용자가 입력한 목표 연도까지 5년 단위로 X, Y축 데이터를 모두 만들어 줌)
    years_list = list(range(2000, max(2030, target_year + 5), 5)) 
    if target_year not in years_list:
        years_list.append(target_year)
        years_list.sort()
        
    # 위에서 뽑은 모든 년도에 대한 예측값 리스트 생성
    X_predict = np.array(years_list).reshape(-1, 1)
    pop_predictions = model.predict(X_predict)
    pop_predictions = [max(0, int(p)) for p in pop_predictions]
    
    # 4. 증감률 계산 및 인사이트 도출
    percent_change = ((predicted_pop - current_pop) / current_pop) * 100
    
    insight = f"과거 데이터를 학습한 AI 모델 분석 결과, {target_year}년 {city} 인구는 {predicted_pop:,}명으로 예측됩니다. "
    if percent_change > 0:
        insight += "상승 곡선을 그리고 있으므로 주택 및 교통 인프라 확충이 필요합니다."
    else:
        insight += "하락 곡선을 띠고 있어 고령화 및 인구 감소 대책 수립이 필요합니다."

    return {
        "city": city,
        "current_pop": current_pop,
        "target_year": target_year,
        "predicted_pop": predicted_pop,
        "percent_change": round(percent_change, 1),
        "years": years_list,           # 👉 프론트엔드 차트의 가로축 (X축 데이터)
        "population": pop_predictions, # 👉 프론트엔드 차트의 세로축 (Y축 데이터)
        "insight": insight
    }

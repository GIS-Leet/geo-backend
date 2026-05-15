from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ⚠️ 중요: 깃허브 페이지(프론트엔드)에서 내 서버에 접속할 수 있도록 허용하는 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 실제 배포 시에는 선생님의 깃허브 주소만 넣는 것이 안전합니다.
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "지리 데이터 분석 서버가 가동 중입니다!"}

@app.get("/predict-climate")
def predict_climate(temp: float, humidity: float):
    # 여기에 선생님만의 복잡한 파이썬 로직(GIS 분석, AI 모델 등)을 넣을 수 있습니다.
    # 예시: 간단한 불쾌지수 및 기후 위험도 계산 로직
    discomfort_index = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) + 46.3
    risk_level = "높음" if discomfort_index > 80 else "보통"
    
    return {
        "score": round(discomfort_index, 2),
        "risk": risk_level,
        "advice": "적도 수렴대의 영향으로 강수량이 증가할 수 있습니다."
    }
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. 앱 제목 설정
st.title("동물 판별기")
st.write("이미지를 업로드하면 티처블 머신 모델이 어떤 동물인지 분석합니다.")

# 2. 모델과 라벨 로드
@st.cache_resource
def load_my_model():
    # 왼쪽에 파일이 있으므로 이름만 똑같이 맞춰줍니다
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

try:
    model, class_names = load_my_model()
except Exception as e:
    st.error(f"모델 또는 라벨 파일을 로드하는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 이미지 업로드 창
uploaded_file = st.file_uploader("동물 사진을 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)
    
    st.write("🔄 모델이 열심히 분석 중입니다...")
    
    # 4. 이미지 전처리 (티처블 머신 기준)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # 5. 예측 및 결과 출력
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    st.success(f"결과 예측: **{class_name}**")
    st.metric(label="분석 신뢰도", value=f"{confidence_score * 100:.2f}%")

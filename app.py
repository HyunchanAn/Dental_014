import os
import streamlit as st
from PIL import Image
import pandas as pd

import sys
sys.path.append('src')

try:
    from dental_014.inference import OsteoporosisInferencer
except ImportError:
    OsteoporosisInferencer = None

st.set_page_config(page_title="Dental_014: Osteoporosis Screening", layout="wide")

st.title("🦷 Dental_014: Osteoporosis Risk Screening")
st.markdown("하악 피질골/골소주(Mandibular Cortical Bone) 패치 이미지를 분석하여 골다공증 위험도를 예측합니다.")

weight_path = "weights/best.pt"

@st.cache_resource
def load_inferencer(path):
    if not os.path.exists(path):
        return None
    return OsteoporosisInferencer(path)

if OsteoporosisInferencer is None:
    st.error("`dental_014` 모듈을 로드할 수 없습니다. `src` 경로 및 환경을 확인해주세요.")
else:
    inferencer = load_inferencer(weight_path)
    
    if inferencer is None:
        st.warning(f"모델 가중치 파일(`{weight_path}`)을 찾을 수 없습니다. 학습을 먼저 진행해주세요.")
    else:
        st.info("✅ 모델 로딩 완료.")
        
        uploaded_file = st.file_uploader("하악골 피질골 크롭 이미지(Patch) 업로드", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
            with col2:
                with st.spinner("분석 중..."):
                    pred_class, probs = inferencer.predict(image)
                    
                st.subheader(f"예측 결과: **{pred_class}**")
                
                # Bar chart
                df = pd.DataFrame({
                    "Risk Level": list(probs.keys()),
                    "Probability (%)": [v * 100 for v in probs.values()]
                })
                st.bar_chart(df.set_index("Risk Level"))

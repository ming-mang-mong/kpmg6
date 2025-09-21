import streamlit as st
import pandas as pd
import numpy as np

st.title("테스트 앱")

try:
    from utils import load_data
    st.success("✅ utils.py import 성공")
    
    df = load_data()
    st.write(f"데이터 로드 성공: {len(df)}행")
    st.dataframe(df.head())
    
except Exception as e:
    st.error(f"❌ 오류 발생: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

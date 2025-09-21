import streamlit as st
import pandas as pd
import numpy as np

st.title("데이터 디버깅")

try:
    from utils import load_data
    st.success("✅ utils.py import 성공")
    
    df = load_data()
    st.write(f"데이터 로드 성공: {len(df)}행, {len(df.columns)}열")
    
    if not df.empty:
        st.write("컬럼 목록:", df.columns.tolist())
        st.write("Segment 컬럼 존재:", 'Segment' in df.columns)
        if 'Segment' in df.columns:
            st.write("Segment 값들:", df['Segment'].unique())
            st.write("Segment 분포:", df['Segment'].value_counts())
        
        st.dataframe(df.head())
    else:
        st.error("데이터가 비어있습니다!")
        
except Exception as e:
    st.error(f"❌ 오류 발생: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

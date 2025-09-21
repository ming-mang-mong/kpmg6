import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data, SEGMENT_ORDER, SEGMENT_COLORS

st.title("시각화 테스트")

try:
    # 데이터 로드
    df = load_data()
    st.success(f"✅ 데이터 로드 성공: {df.shape}")
    
    # 기본 정보 표시
    st.write(f"컬럼 수: {len(df.columns)}")
    st.write(f"Segment 컬럼 존재: {'Segment' in df.columns}")
    
    if 'Segment' in df.columns:
        st.write(f"Segment 값: {df['Segment'].unique()}")
        st.write(f"Segment 분포:")
        st.write(df['Segment'].value_counts())
    
    # 간단한 차트 테스트
    if 'Segment' in df.columns and '총이용금액_B0M' in df.columns:
        st.subheader("세그먼트별 총이용금액")
        segment_sum = df.groupby('Segment', observed=True)['총이용금액_B0M'].sum()
        st.bar_chart(segment_sum)
        
        st.subheader("세그먼트별 고객 수")
        segment_count = df['Segment'].value_counts()
        st.bar_chart(segment_count)
    
    # 데이터 샘플
    st.subheader("데이터 샘플")
    st.dataframe(df.head())
    
except Exception as e:
    st.error(f"오류 발생: {e}")
    import traceback
    st.code(traceback.format_exc())

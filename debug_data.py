import streamlit as st
from utils import load_data

try:
    df = load_data()
    print(f"데이터 크기: {df.shape}")
    print(f"컬럼 수: {len(df.columns)}")
    print(f"첫 5개 컬럼: {list(df.columns[:5])}")
    print(f"Segment 컬럼 존재: {'Segment' in df.columns}")
    
    if 'Segment' in df.columns:
        print(f"Segment 값: {df['Segment'].unique()}")
        print(f"Segment 타입: {df['Segment'].dtype}")
    else:
        print("Segment 컬럼이 없습니다!")
    
    # 필요한 컬럼들 확인
    required_cols = ['총이용금액_B0M', '총이용건수_B0M', '연체여부', '카드이용한도액']
    for col in required_cols:
        exists = col in df.columns
        print(f"{col}: {'존재' if exists else '없음'}")
    
    # 데이터 샘플 확인
    print(f"\n데이터 샘플 (첫 3행):")
    print(df.head(3))
    
except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()

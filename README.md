# 신용카드 세그먼트 분석 대시보드

Streamlit을 사용한 신용카드 고객 세그먼트 분석 대시보드입니다.

## 기능

- 📊 세그먼트별 KPI 분석
- 🔍 세그먼트별 세부특성 분석
- 📈 트렌드 분석 (시계열)
- ⚠️ 리스크 분석
- 🎯 행동마케팅 분석

## Streamlit Cloud 배포

1. GitHub 저장소에 코드 업로드
2. [Streamlit Cloud](https://share.streamlit.io/)에서 새 앱 생성
3. 저장소 연결 및 `app.py` 파일 선택
4. 자동 배포 완료

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 요구사항

- Python 3.8+
- Streamlit 1.28.0+
- Pandas, NumPy, Plotly
- PyTorch (GPU 가속용, 선택사항)

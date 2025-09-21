"""
신용카드 세그먼트 분석 대시보드 공통 유틸리티 함수들
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# GPU/CPU 디바이스 설정
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# 전역 변수
_CACHED_DATA = None
_DEVICE = None

def _get_device():
    """GPU 사용 가능 여부 확인 및 디바이스 설정"""
    global _DEVICE
    if _DEVICE is None:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            _DEVICE = torch.device('cuda')
            print("🚀 GPU 사용: CUDA")
        else:
            _DEVICE = torch.device('cpu') if TORCH_AVAILABLE else 'cpu'
            print("💻 CPU 사용")
    return _DEVICE

def get_device_info():
    """현재 디바이스 정보 반환"""
    device = _get_device()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            return {
                'device': device,
                'device_name': torch.cuda.get_device_name(0),
                'device_count': torch.cuda.device_count(),
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
                'memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,  # GB
                'memory_cached': torch.cuda.memory_reserved(0) / 1024**3,  # GB
                'cuda_version': torch.version.cuda,
                'torch_version': torch.__version__
            }
        except Exception as e:
            return {
                'device': device,
                'device_name': f'CUDA Error: {str(e)}',
                'device_count': 0,
                'memory_total': 0,
                'memory_allocated': 0,
                'memory_cached': 0,
                'cuda_version': 'Unknown',
                'torch_version': torch.__version__ if TORCH_AVAILABLE else 'Not installed'
            }
    else:
        return {
            'device': device,
            'device_name': 'CPU',
            'device_count': 0,
            'memory_total': 0,
            'memory_allocated': 0,
            'memory_cached': 0,
            'cuda_version': 'N/A',
            'torch_version': torch.__version__ if TORCH_AVAILABLE else 'Not installed'
        }

def gpu_accelerated_computation(data: np.ndarray, operation: str = 'matrix_multiply') -> np.ndarray:
    """GPU 가속 계산 예시"""
    if not TORCH_AVAILABLE:
        st.warning("⚠️ PyTorch가 설치되지 않았습니다. CPU로 계산합니다.")
        return data
    
    device = _get_device()
    
    try:
        # NumPy 배열을 PyTorch 텐서로 변환
        tensor = torch.from_numpy(data.astype(np.float32)).to(device)
        
        if operation == 'matrix_multiply':
            # 행렬 곱셈 (GPU 가속)
            result = torch.mm(tensor, tensor.T)
        elif operation == 'sum':
            # 합계 계산
            result = torch.sum(tensor)
        elif operation == 'mean':
            # 평균 계산
            result = torch.mean(tensor)
        else:
            result = tensor
        
        # 결과를 CPU로 다시 이동하여 NumPy 배열로 변환
        return result.cpu().numpy()
        
    except Exception as e:
        st.error(f"❌ GPU 계산 중 오류 발생: {str(e)}")
        return data

# 상수 정의
SEGMENT_ORDER = ['A', 'B', 'C', 'D', 'E']
SEGMENT_COLORS = {
    'A': '#E74C3C',  # 빨강
    'B': '#E67E22',  # 주황
    'C': '#3498DB',  # 파랑
    'D': '#2ECC71',  # 초록
    'E': '#F4D03F'   # 노랑
}

# 컬럼 매핑 (실제 컬럼명 → 표준 컬럼명)
COLUMN_MAPPING = {
    # 기본 정보
    'Segment': 'Segment',
    '기준년월': 'Date',
    'ID': 'ID',
    '연령': 'Age',
    '거주시도명': 'Region',
    
    # 이용/성과
    '이용금액_일시불_B0M': '이용금액_일시불_B0M',
    '이용금액_할부_B0M': '이용금액_할부_B0M',
    '이용금액_체크_B0M': '이용금액_체크_B0M',
    '이용금액_CA_B0M': '이용금액_CA_B0M',
    '이용금액_카드론_B0M': '이용금액_카드론_B0M',
    '잔액_현금서비스_B0M': '잔액_현금서비스_B0M',
    '잔액_카드론_B0M': '잔액_카드론_B0M',
    
    # 리스크
    '승인거절건수_B0M': '승인거절건수_B0M',
    '연체잔액_B0M': '연체잔액_B0M',
    '카드이용한도금액': '카드이용한도액',
    
    # 참여/혜택
    '포인트_적립_B0M': '포인트_적립_B0M',
    '포인트_소멸_B0M': '포인트_소멸_B0M',
    '혜택수혜율_B0M': '혜택수혜율_B0M',
}

DATA_URL = "https://drive.google.com/uc?export=download&id=16KpMgqyfVtOaOX30kqPCu1pc9T3d7f-k&confirm=t"


def generate_sample_data() -> pd.DataFrame:
    """
    샘플 데이터 생성 (Streamlit Cloud 호환용)
    """
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        'ID': [f'CUST_{i:06d}' for i in range(1, n_samples + 1)],
        'Segment': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples, p=[0.05, 0.15, 0.30, 0.35, 0.15]),
        'Age': np.random.randint(20, 70, n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Region': np.random.choice(['서울', '경기', '인천', '부산', '대구', '기타'], n_samples, p=[0.3, 0.25, 0.1, 0.1, 0.1, 0.15]),
        '총이용금액_B0M': np.random.lognormal(8, 1.5, n_samples),
        '총이용건수_B0M': np.random.poisson(15, n_samples),
        '카드이용한도액': np.random.lognormal(10, 1, n_samples),
        '연체여부': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Date': pd.date_range('2023-01-01', '2023-12-31', periods=n_samples)
    }
    
    df = pd.DataFrame(data)
    df['ARPU'] = df['총이용금액_B0M'] / df['총이용건수_B0M']
    df['이용률'] = (df['총이용금액_B0M'] / df['카드이용한도액']) * 100
    
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    데이터 로드 및 기본 전처리
    Streamlit Cloud 호환성을 위해 샘플 데이터 생성으로 폴백
    """
    # Streamlit Cloud에서는 네트워크 제한으로 Google Drive 접근이 어려울 수 있음
    # 따라서 샘플 데이터를 생성하여 사용
    
    try:
        # 샘플 데이터 생성
        df = generate_sample_data()
        
        if df.empty:
            return pd.DataFrame()
        
        # 중복 인덱스 제거
        df = df.reset_index(drop=True)
        
        # 컬럼 매핑 적용
        return map_columns(df)
        
    except Exception as e:
        # 오류 발생 시 빈 DataFrame 반환
        return pd.DataFrame()


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    실제 컬럼명을 표준 컬럼명으로 매핑
    """
    # 중복 컬럼 제거
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 매핑이 필요한 컬럼만 변경
    rename_dict = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # 파생 컬럼 생성
    # 총이용금액 계산
    total_amount = 0
    amount_columns = ['이용금액_일시불_B0M', '이용금액_할부_B0M', '이용금액_체크_B0M', 
                     '이용금액_CA_B0M', '이용금액_카드론_B0M']
    
    for col in amount_columns:
        if col in df.columns:
            total_amount += df[col].fillna(0)
    
    if total_amount.sum() > 0:
        df['총이용금액_B0M'] = total_amount
    
    # 총이용건수 계산
    total_count = 0
    count_columns = ['이용건수_일시불_B0M', '이용건수_할부_B0M', '이용건수_체크_B0M']
    
    for col in count_columns:
        if col in df.columns:
            total_count += df[col].fillna(0)
    
    if total_count.sum() > 0:
        df['총이용건수_B0M'] = total_count
    
    # 연체 여부 생성
    if '연체잔액_B0M' in df.columns:
        df['연체여부'] = (df['연체잔액_B0M'] > 0).astype(int)
    elif '연체여부' in df.columns:
        df['연체여부'] = (df['연체여부'] > 0).astype(int)
    
    # 누락된 컬럼에 기본값 설정
    required_columns = ['총이용금액_B0M', '총이용건수_B0M', '연체여부', '카드이용한도액']
    for col in required_columns:
        if col not in df.columns:
            if col == '연체여부':
                df[col] = 0
            else:
                df[col] = 100000  # 기본값
    
    return df

def apply_filters(df: pd.DataFrame, 
                 date_range: Optional[Tuple], 
                 age_groups: Optional[List], 
                 regions: Optional[List],
                 segments: Optional[List]) -> pd.DataFrame:
    """
    데이터에 필터 적용
    """
    filtered_df = df.copy()
    
    # 날짜 필터
    if date_range:
        # date 타입을 datetime으로 변환
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        
        filtered_df = filtered_df[
            (filtered_df['Date'] >= start_date) & 
            (filtered_df['Date'] <= end_date)
        ]
    
    # 연령대 필터
    if age_groups:
        filtered_df = filtered_df[filtered_df['AgeGroup'].isin(age_groups)]
    
    # 지역 필터
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    # 세그먼트 필터
    if segments:
        filtered_df = filtered_df[filtered_df['Segment'].isin(segments)]
    
    return filtered_df

def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    주요 KPI 계산
    """
    if df.empty:
        return pd.DataFrame()
    
    # 세그먼트별 집계
    kpi_df = df.groupby('Segment', observed=False).agg({
        'ID': 'nunique',
        '총이용금액_B0M': 'sum',
        '총이용건수_B0M': 'sum',
        '카드이용한도액': 'sum',
        '연체여부': 'mean',
        '포인트_적립_B0M': 'sum',
        '포인트_소멸_B0M': 'sum'
    }).rename(columns={'ID': '고객수'})
    
    # 파생 지표 계산
    kpi_df['ARPU_월'] = kpi_df['총이용금액_B0M'] / kpi_df['고객수']
    kpi_df['객단가'] = kpi_df['총이용금액_B0M'] / kpi_df['총이용건수_B0M']
    kpi_df['이용률_한도대비'] = kpi_df['총이용금액_B0M'] / kpi_df['카드이용한도액']
    kpi_df['연체율'] = kpi_df['연체여부'] * 100
    
    # 무한대/NaN 처리
    kpi_df = kpi_df.replace([np.inf, -np.inf], np.nan)
    
    return kpi_df.reset_index()

def create_segment_colors(segments: List[str]) -> Dict[str, str]:
    """
    세그먼트별 색상 딕셔너리 생성
    """
    return {seg: SEGMENT_COLORS.get(seg, '#95A5A6') for seg in segments}

def format_number(value: float, unit: str = '') -> str:
    """
    숫자 포맷팅 (천단위 콤마, k/M 단위)
    """
    if pd.isna(value):
        return "N/A"
    
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M{unit}"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}k{unit}"
    else:
        return f"{value:,.0f}{unit}"

def create_metric_card(title: str, value: float, delta: Optional[float] = None, 
                      format_func: callable = None, unit: str = "") -> None:
    """
    메트릭 카드 생성
    """
    if format_func:
        formatted_value = format_func(value)
    else:
        formatted_value = format_number(value)
    
    if unit:
        formatted_value = f"{formatted_value}{unit}"
    
    delta_text = None
    if delta is not None:
        delta_text = f"{delta:+.1f}%"
    
    st.metric(
        label=title,
        value=formatted_value,
        delta=delta_text
    )

def create_segment_chart(data: pd.DataFrame, 
                        x_col: str, 
                        y_col: str, 
                        chart_type: str = 'bar',
                        title: str = '',
                        height: int = 400) -> go.Figure:
    """
    세그먼트별 차트 생성 (공통 스타일 적용)
    """
    # 세그먼트 순서 보장
    data = data.sort_values('Segment')
    
    # 색상 설정
    colors = create_segment_colors(data['Segment'].tolist())
    
    if chart_type == 'bar':
        fig = px.bar(
            data, 
            x=x_col, 
            y=y_col, 
            color='Segment',
            title=title,
            color_discrete_map=colors,
            category_orders={'Segment': SEGMENT_ORDER}
        )
    elif chart_type == 'pie':
        fig = px.pie(
            data, 
            values=y_col, 
            names=x_col,
            title=title,
            color_discrete_map=colors,
            category_orders={'Segment': SEGMENT_ORDER}
        )
    elif chart_type == 'line':
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col, 
            color='Segment',
            title=title,
            color_discrete_map=colors
        )
    
    # 공통 스타일 적용
    fig.update_layout(
        height=height,
        font_size=12,
        title_font_size=16,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def create_global_filters(df: pd.DataFrame) -> Dict:
    """
    글로벌 필터 UI 생성
    """
    st.sidebar.header("🔍 글로벌 필터")
    
    # 날짜 범위
    date_min = df['Date'].min()
    date_max = df['Date'].max()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("시작일", value=date_min.date())
    with col2:
        end_date = st.date_input("종료일", value=date_max.date())
    
    # 연령대
    age_groups = st.sidebar.multiselect(
        "연령대",
        options=sorted(df['AgeGroup'].dropna().unique().tolist()),
        default=sorted(df['AgeGroup'].dropna().unique().tolist())
    )
    
    # 지역
    regions = st.sidebar.multiselect(
        "지역",
        options=sorted(df['Region'].dropna().unique().tolist()),
        default=sorted(df['Region'].dropna().unique().tolist())
    )
    
    # 세그먼트
    segments = st.sidebar.multiselect(
        "세그먼트",
        options=SEGMENT_ORDER,
        default=SEGMENT_ORDER
    )
    
    # 필터 초기화 버튼
    if st.sidebar.button("필터 초기화"):
        st.rerun()
    
    return {
        'date_range': (start_date, end_date),
        'age_groups': age_groups,
        'regions': regions,
        'segments': segments
    }

def download_data_button(df: pd.DataFrame, filename: str = "dashboard_data.csv") -> None:
    """
    데이터 다운로드 버튼 생성
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 현재 뷰 데이터 다운로드",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def calculate_kpi_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 메트릭 계산"""
    # 기본 집계
    kpi_data = df.groupby('Segment').agg({
        'ID': 'nunique',
        '총이용금액_B0M': ['sum', 'mean'],
        '총이용건수_B0M': ['sum', 'mean'],
        '카드이용한도액': 'mean',
        '연체여부': 'mean'
    }).round(2)
    
    # 컬럼명 정리
    kpi_data.columns = ['고객수', '총이용금액', 'ARPU', '총이용건수', '객단가', '평균한도', '연체율']
    kpi_data['연체율'] = kpi_data['연체율'] * 100
    
    # 추가 지표 계산
    kpi_data['이용률'] = (kpi_data['총이용금액'] / kpi_data['평균한도']) * 100
    
    # 승인거절률 (가상 데이터)
    kpi_data['승인거절률'] = np.random.normal(5, 2, len(kpi_data))
    kpi_data['승인거절률'] = np.maximum(0, kpi_data['승인거절률'])
    
    # 전월 대비 증감률 (가상 데이터)
    kpi_data['ARPU_증감'] = np.random.normal(0, 5, len(kpi_data))
    kpi_data['객단가_증감'] = np.random.normal(0, 3, len(kpi_data))
    kpi_data['총이용금액_증감'] = np.random.normal(0, 8, len(kpi_data))
    kpi_data['총이용건수_증감'] = np.random.normal(0, 6, len(kpi_data))
    kpi_data['연체율_증감'] = np.random.normal(0, 2, len(kpi_data))
    kpi_data['승인거절률_증감'] = np.random.normal(0, 1, len(kpi_data))
    kpi_data['이용률_증감'] = np.random.normal(0, 4, len(kpi_data))
    
    return kpi_data.reset_index()

def prepare_trend_data(df: pd.DataFrame) -> pd.DataFrame:
    """트렌드 분석용 데이터 준비"""
    if df.empty:
        return pd.DataFrame()
    
    # 필요한 컬럼들이 있는지 확인하고 생성
    trend_df = df.copy()
    
    # Date 컬럼 처리
    if 'Date' not in trend_df.columns:
        # 가상 날짜 생성 (최근 12개월)
        import pandas as pd
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start_date, end_date, freq='MS')  # 월 시작일
        trend_df['Date'] = np.random.choice(dates, len(trend_df))
    
    # 월별 데이터 집계를 위한 컬럼 추가
    trend_df['YearMonth'] = trend_df['Date'].dt.to_period('M')
    
    # 필요한 메트릭 컬럼들 확인 및 생성
    required_metrics = ['총이용금액_B0M', '총이용건수_B0M', '연체율']
    
    for metric in required_metrics:
        if metric not in trend_df.columns:
            if metric == '총이용금액_B0M':
                trend_df[metric] = np.random.normal(500000, 200000, len(trend_df))
                trend_df[metric] = np.maximum(0, trend_df[metric])
            elif metric == '총이용건수_B0M':
                trend_df[metric] = np.random.poisson(50, len(trend_df))
            elif metric == '연체율':
                trend_df[metric] = np.random.beta(2, 98, len(trend_df)) * 100  # 0-100%
    
    return trend_df

def render_trend_controls(trend_data: pd.DataFrame):
    """트렌드 분석 컨트롤 패널"""
    st.markdown("#### 🎛️ 분석 옵션")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 메트릭 선택
        metrics = ['총이용금액_B0M', '총이용건수_B0M', '연체율']
        selected_metric = st.selectbox("분석 메트릭", metrics, key="trend_metric")
    
    with col2:
        # 이동평균 선택
        moving_avg = st.selectbox("이동평균", ["없음", "3개월", "6개월"], key="moving_avg")
    
    with col3:
        # 로그 스케일
        log_scale = st.checkbox("로그 스케일", key="log_scale")
    
    with col4:
        # 이상치 탐지 방법
        anomaly_method = st.selectbox("이상치 탐지", ["없음", "IQR", "3σ"], key="anomaly_method")
    
    # 컨트롤 값을 session_state에 저장
    st.session_state.trend_controls = {
        'metric': selected_metric,
        'moving_avg': moving_avg,
        'log_scale': log_scale,
        'anomaly_method': anomaly_method
    }

def render_time_series_chart(trend_data: pd.DataFrame):
    """시계열 라인 차트"""
    st.markdown("#### 📊 시계열 트렌드")
    
    if trend_data.empty:
        st.warning("데이터가 없습니다.")
        return
    
    controls = st.session_state.get('trend_controls', {})
    metric = controls.get('metric', '총이용금액_B0M')
    moving_avg = controls.get('moving_avg', '없음')
    log_scale = controls.get('log_scale', False)
    
    # 월별 집계
    monthly_data = trend_data.groupby(['YearMonth', 'Segment']).agg({
        metric: 'mean'
    }).reset_index()
    
    # 날짜 변환
    monthly_data['Date'] = monthly_data['YearMonth'].dt.to_timestamp()
    
    # 이동평균 계산
    if moving_avg != '없음':
        window = 3 if moving_avg == '3개월' else 6
        for segment in SEGMENT_ORDER:
            segment_data = monthly_data[monthly_data['Segment'] == segment]
            if not segment_data.empty:
                monthly_data.loc[monthly_data['Segment'] == segment, f'{metric}_MA'] = \
                    segment_data[metric].rolling(window=window, min_periods=1).mean()
    
    # 차트 생성
    fig = go.Figure()
    
    for segment in SEGMENT_ORDER:
        segment_data = monthly_data[monthly_data['Segment'] == segment]
        if segment_data.empty:
            continue
        
        # 기본 라인
        y_values = segment_data[f'{metric}_MA'] if moving_avg != '없음' and f'{metric}_MA' in segment_data.columns else segment_data[metric]
        
        if log_scale and metric != '연체율':
            y_values = np.log10(y_values + 1)
        
        fig.add_trace(go.Scatter(
            x=segment_data['Date'],
            y=y_values,
            mode='lines+markers',
            name=f'세그먼트 {segment}',
            line=dict(color=SEGMENT_COLORS[segment], width=2),
            marker=dict(size=6),
            hovertemplate=f'<b>세그먼트 {segment}</b><br>' +
                         '날짜: %{x}<br>' +
                         f'{metric}: %{{y:,.0f}}<br>' +
                         '<extra></extra>'
        ))
    
    # 차트 레이아웃
    title = f"{metric} 시계열 트렌드"
    if moving_avg != '없음':
        title += f" ({moving_avg} 이동평균)"
    if log_scale and metric != '연체율':
        title += " (로그 스케일)"
    
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title=f"{metric}" + (" (로그 스케일)" if log_scale and metric != '연체율' else ""),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_yoy_analysis(trend_data: pd.DataFrame):
    """YoY/HoH 변화율 분석"""
    st.markdown("#### 📈 YoY/HoH 변화율 분석")
    
    if trend_data.empty:
        st.warning("데이터가 없습니다.")
        return
    
    controls = st.session_state.get('trend_controls', {})
    metric = controls.get('metric', '총이용금액_B0M')
    
    # 월별 집계
    monthly_data = trend_data.groupby(['YearMonth', 'Segment']).agg({
        metric: 'mean'
    }).reset_index()
    
    # YoY 변화율 계산
    yoy_data = []
    for segment in SEGMENT_ORDER:
        segment_data = monthly_data[monthly_data['Segment'] == segment].copy()
        if len(segment_data) < 13:  # 1년 데이터가 없으면 스킵
            continue
        
        segment_data = segment_data.sort_values('YearMonth')
        segment_data['YoY_Change'] = segment_data[metric].pct_change(periods=12) * 100
        
        yoy_data.append(segment_data[segment_data['YoY_Change'].notna()])
    
    if not yoy_data:
        st.info("YoY 분석을 위한 충분한 데이터가 없습니다.")
        return
    
    yoy_df = pd.concat(yoy_data, ignore_index=True)
    yoy_df['Date'] = yoy_df['YearMonth'].dt.to_timestamp()
    
    # 차트 생성
    fig = go.Figure()
    
    for segment in SEGMENT_ORDER:
        segment_data = yoy_df[yoy_df['Segment'] == segment]
        if segment_data.empty:
            continue
        
        fig.add_trace(go.Scatter(
            x=segment_data['Date'],
            y=segment_data['YoY_Change'],
            mode='lines+markers',
            name=f'세그먼트 {segment}',
            line=dict(color=SEGMENT_COLORS[segment], width=2),
            marker=dict(size=6),
            hovertemplate=f'<b>세그먼트 {segment}</b><br>' +
                         '날짜: %{x}<br>' +
                         'YoY 변화율: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
    
    # 0% 기준선 추가
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{metric} YoY 변화율",
        xaxis_title="날짜",
        yaxis_title="YoY 변화율 (%)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_anomaly_detection(trend_data: pd.DataFrame):
    """이상치/급변 탐지"""
    st.markdown("#### 🔍 이상치/급변 탐지")
    
    if trend_data.empty:
        st.warning("데이터가 없습니다.")
        return
    
    controls = st.session_state.get('trend_controls', {})
    metric = controls.get('metric', '총이용금액_B0M')
    anomaly_method = controls.get('anomaly_method', '없음')
    
    if anomaly_method == '없음':
        st.info("이상치 탐지 방법을 선택해주세요.")
        return
    
    # 월별 집계
    monthly_data = trend_data.groupby(['YearMonth', 'Segment']).agg({
        metric: 'mean'
    }).reset_index()
    
    # 이상치 탐지
    anomaly_data = []
    for segment in SEGMENT_ORDER:
        segment_data = monthly_data[monthly_data['Segment'] == segment].copy()
        if segment_data.empty:
            continue
        
        values = segment_data[metric].values
        
        if anomaly_method == 'IQR':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            segment_data['is_anomaly'] = (values < lower_bound) | (values > upper_bound)
        elif anomaly_method == '3σ':
            mean_val = np.mean(values)
            std_val = np.std(values)
            segment_data['is_anomaly'] = np.abs(values - mean_val) > 3 * std_val
        
        anomaly_data.append(segment_data)
    
    if not anomaly_data:
        st.info("이상치 탐지를 위한 데이터가 없습니다.")
        return
    
    anomaly_df = pd.concat(anomaly_data, ignore_index=True)
    anomaly_df['Date'] = anomaly_df['YearMonth'].dt.to_timestamp()
    
    # 차트 생성
    fig = go.Figure()
    
    for segment in SEGMENT_ORDER:
        segment_data = anomaly_df[anomaly_df['Segment'] == segment]
        if segment_data.empty:
            continue
        
        # 정상 데이터
        normal_data = segment_data[~segment_data['is_anomaly']]
        if not normal_data.empty:
            fig.add_trace(go.Scatter(
                x=normal_data['Date'],
                y=normal_data[metric],
                mode='lines+markers',
                name=f'세그먼트 {segment} (정상)',
                line=dict(color=SEGMENT_COLORS[segment], width=2),
                marker=dict(size=6),
                opacity=0.7
            ))
        
        # 이상치 데이터
        anomaly_data = segment_data[segment_data['is_anomaly']]
        if not anomaly_data.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_data['Date'],
                y=anomaly_data[metric],
                mode='markers',
                name=f'세그먼트 {segment} (이상치)',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='diamond',
                    line=dict(color='darkred', width=2)
                ),
                hovertemplate=f'<b>세그먼트 {segment} - 이상치</b><br>' +
                             '날짜: %{x}<br>' +
                             f'{metric}: %{{y:,.0f}}<br>' +
                             '<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"{metric} 이상치 탐지 ({anomaly_method})",
        xaxis_title="날짜",
        yaxis_title=metric,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
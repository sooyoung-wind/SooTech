import matplotlib.pyplot as plt
import numpy as np

def plot_performance_metrics(metrics: list, based_value: float) -> None:
    """
    성능 지표를 시각화하는 함수입니다.

    이 함수는 주어진 성능 지표를 기반으로 레이더 차트를 생성하여 
    다양한 성능 지표(ME, MAE, RMSE, IOA, R)의 상대적인 성능을 
    시각적으로 비교할 수 있도록 합니다.

    Parameters:
    -----------
    metrics : list
        성능 지표의 리스트로, 각 지표는 다음과 같습니다:
        - ME (Mean Error)
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Square Error)
        - IOA (Index of Agreement)
        - R (Correlation Coefficient)

    Returns:
    --------
    None
        이 함수는 차트를 표시하며, 반환값은 없습니다.

    Example:
    -------
    metrics = [987.71, 1297.44, 2119.20, 0.86, 0.73]
    plot_performance_metrics(metrics)
    """
    # 데이터와 라벨
    labels = ['ME', 'MAE', 'RMSE', 'IOA', 'R']
    values = metrics
    values[0] = (1 - values[0] / based_value) * 100
    values[1] = (1 - values[1] / based_value) * 100
    values[2] = (1 - values[2] / based_value) * 100
    values[3] = values[3] * 100
    values[4] = values[4] * 100
    values = np.round(values, 2)

    # 라벨을 각도 계산에 맞추기
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # 각 값에 대한 닫기 처리를 values와 angles에 추가
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]

    # 차트 생성
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.3)  # 내부 영역
    ax.plot(angles, values, color='blue', linewidth=2)   # 외곽선

    # 각 지점에 라벨 붙이기
    ax.set_yticklabels([])  # 중간 축 라벨 제거
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='black', fontsize=14, fontweight='bold')

    # 축 스타일 설정
    ax.spines['polar'].set_visible(False)  # 외곽선 제거
    ax.grid(color='gray', linestyle='--', linewidth=0.5)  # 격자 스타일 조정
    ax.set_ylim(0, 100)  # 값 범위 설정

    # 각도별 축을 위한 색상 추가
    for i, angle in enumerate(angles[:-1]):
        ax.plot([angle, angle], [0, 100], color="lightgray", linestyle="--", linewidth=0.5)

    # 각 데이터 값에 다크 레드 레이블 추가, 값에 따라 위치 조정, % 표시 추가
    for i in range(len(values) - 1):
        offset = -5 if values[i] > 70 else 5  # 값이 높을 경우 아래로, 낮을 경우 위로 오프셋 조정
        ax.text(angles[i], values[i] + offset, f"{values[i]:.0f}%", ha='center', color='darkred', fontsize=16, fontweight='bold')

    plt.title("Performance Metrics", fontsize=16, fontweight='bold', color='navy', pad=20)
    plt.show()

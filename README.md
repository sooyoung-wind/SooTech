# SooTech

데이터 분석 및 머신러닝을 위한 유틸리티 함수 모음입니다. 이 라이브러리는 데이터 처리, 통계 계산, 시각화 및 모델링을 위한 다양한 기능을 제공합니다.

## 설치 방법

다음 명령어를 사용하여 SooTech를 설치할 수 있습니다:

```bash
pip install git+https://github.com/sooyoung-wind/SooTech.git
```
또는
```bash
poetry add git+https://github.com/sooyoung-wind/SooTech.git
```

## 사용 예제

SooTech를 사용하여 데이터프레임의 요약 정보를 출력하고, 랜덤 시드를 고정하는 방법은 다음과 같습니다:

### OOP API

```python
import SooTech as soo
import pandas as pd

# 예시 데이터프레임 생성
df = pd.DataFrame({
    'A': [1, 2, 3, None],
    'B': [4, 5, 6, 7]
})

# SooTech 클래스 인스턴스 생성
sootech = soo.SooTech(raw_data=df)

# DataFrame 요약 정보 출력
summary = sootech.resumetable(sootech.raw_data)
print(summary)

# 시드 값 고정
sootech.seed_everything(42)
```
### Functaion API

```python
import SooTech as soo
import pandas as pd

# 예시 데이터프레임 생성
df = pd.DataFrame({
    'A': [1, 2, 3, None],
    'B': [4, 5, 6, 7]
})

# DataFrame 요약 정보 출력
summary = soo.resumetable(df)
print(summary)

# 시드 값 고정
soo.seed_everything(42)
```

## 주요 기능

- **데이터 요약**: `resumetable(df)` 함수를 사용하여 데이터프레임의 요약 정보를 확인할 수 있습니다.
- **랜덤 시드 설정**: `seed_everything(seed)` 함수를 사용하여 실험의 재현성을 확보할 수 있습니다.
- **통계 계산**: 다양한 통계 지표를 계산하는 함수들이 포함되어 있습니다.
- **시각화**: 성능 지표를 시각화하는 함수도 제공됩니다.
```
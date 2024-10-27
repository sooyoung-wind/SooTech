# SooTech

데이터 분석 및 머신러닝을 위한 유틸리티 함수 모음입니다.

## 설치 방법

```bash
pip install git+https://github.com/sooyoung-wind/SooTech.git
```
or
```bash
poetry add git+https://github.com/sooyoung-wind/SooTech.git
```

## 사용 예제

```python
import SooTech as soo

# DataFrame 요약 정보 출력
soo.resumetable(df)

# 시드 값 고정
soo.seed_everything(42)
```

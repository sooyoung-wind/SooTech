from dataclasses import dataclass
import pandas as pd

@dataclass
class WeatherData:
    ws: pd.Series  # 풍속
    wd: pd.Series  # 풍향

@dataclass
class StatisticsResult:
    mean: float
    median: float
    mode: float
    rmse: float
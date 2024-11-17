import sys
import types
import pandas as pd
from . import soo_functions, soo_statistic, soo_plots

class SooTech:
    """
    SooTech 클래스는 데이터 처리 및 분석을 위한 기능을 제공합니다.

    Attributes:
    raw_data (pd.DataFrame): 초기 데이터로 사용할 pandas DataFrame입니다.

    Examples:
    >>> soo = SooTech(pd.DataFrame({'A': [1, 2, 3]}))
    >>> soo.raw_data
    A
    0  1
    1  2
    2  3
    """
    def __init__(self, raw_data=None):
        """
        SooTech 클래스의 초기화 메서드입니다.

        Parameters:
        raw_data (pd.DataFrame or str): 초기 데이터로 사용할 pandas DataFrame 또는 CSV 파일 경로입니다.

        Returns:
        None

        Examples:
        >>> soo = SooTech(pd.DataFrame({'A': [1, 2, 3]}))
        >>> soo.raw_data
        A
        0  1
        1  2
        2  3
        """
        self.raw_data = self._load_data(raw_data)
        self._add_module_functions(soo_functions)
        self._add_module_functions(soo_statistic)
        self._add_module_functions(soo_plots)

    def _load_data(self, raw_data):
        """
        raw_data를 적절한 형식으로 로드합니다.

        Parameters:
        raw_data (pd.DataFrame or str): 로드할 데이터입니다. pandas DataFrame 또는 CSV 파일 경로여야 합니다.

        Returns:
        pd.DataFrame: 로드된 데이터입니다.

        Examples:
        >>> soo = SooTech('data.csv')
        >>> soo.raw_data
        A  B
        0  1  4
        1  2  5
        2  3  6
        """
        if isinstance(raw_data, pd.DataFrame):
            return raw_data
        elif isinstance(raw_data, str):
            try:
                return pd.read_csv(raw_data)
            except FileNotFoundError:
                raise ValueError(f"파일을 찾을 수 없습니다: {raw_data}")
        else:
            raise ValueError("raw_data는 DataFrame 또는 파일 경로여야 합니다.")

    def _add_module_functions(self, module):
        """
        모듈에 정의된 함수를 현재 클래스에 추가합니다.

        Parameters:
        module (module): 함수를 포함하는 모듈입니다.
        startswitch를 사용해서 비공개 함수들은 제외합니다.

        Returns:
        None
        """
        for name, func in vars(module).items():
            if isinstance(func, types.FunctionType) and not name.startswith('_'):
                setattr(self, name, func)

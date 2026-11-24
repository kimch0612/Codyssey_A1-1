"""쇼핑몰 단골 찾기 분석 파이프라인.

원본 데이터를 읽어 전처리, 피처 엔지니어링, 요약 통계, RFM 분석을 수행하는
`DataAnalyzer` 클래스를 제공한다. 이 클래스는 데이터 처리와 계산만 맡고,
차트는 노트북(notebooks/analysis_report.ipynb)에서 그린다.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


class DataAnalyzer:
    """H&M 부분집합 데이터(거래·고객·상품)를 다루는 분석 파이프라인.

    사용 예:
        analyzer = DataAnalyzer().load_data()
        analyzer.transactions  # 거래 DataFrame
    """

    # 데이터 폴더 안의 파일 이름. 이 데이터셋의 약속이므로 상수로 둔다.
    TRANSACTIONS_FILE = "transactions.csv"
    CUSTOMERS_FILE = "customers.csv"
    ARTICLES_FILE = "articles.csv"

    # 숫자처럼 보여도 문자열로 읽어야 하는 ID 컬럼.
    # 숫자로 읽으면 article_id "0706016001"의 앞자리 0이 사라져
    # 이미지 경로나 다른 테이블과의 연결이 끊긴다.
    TRANSACTIONS_DTYPES = {"customer_id": str, "article_id": str}
    CUSTOMERS_DTYPES = {"customer_id": str}
    ARTICLES_DTYPES = {"article_id": str, "product_code": str}

    # 날짜형으로 읽을 컬럼.
    TRANSACTIONS_DATE_COLUMNS = ["order_date"]
    ARTICLES_DATE_COLUMNS = ["first_sold_date"]

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """분석기를 만든다. 데이터는 `load_data()`를 호출할 때 읽는다.

        Args:
            data_dir: CSV 파일이 있는 폴더. None이면 이 파일 위치를 기준으로
                프로젝트 루트의 `data/` 폴더를 쓴다. 실행 위치(작업 폴더)와
                상관없이 같은 폴더를 가리킨다.
        """
        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[1] / "data"
        self.data_dir = Path(data_dir)

        # load_data() 전에는 비어 있다.
        self.transactions = None  # 거래: 1행 = 상품 1개 구매
        self.customers = None  # 고객: 1행 = 고객 1명
        self.articles = None  # 상품: 1행 = 상품 1개

    def load_data(self) -> "DataAnalyzer":
        """거래·고객·상품 CSV를 읽어 속성에 저장하고 `self`를 반환한다.

        - ID 컬럼은 문자열로 읽어 앞자리 0을 유지한다.
        - 날짜 컬럼(`order_date`, `first_sold_date`)은 날짜형으로 읽는다.
        - 다시 호출하면 파일에서 새로 읽으므로, 이후 단계에서 테이블을
          바꿨더라도 처음 상태로 되돌릴 수 있다.

        Returns:
            self. `DataAnalyzer().load_data()`처럼 이어서 쓸 수 있다.

        Raises:
            FileNotFoundError: 필요한 CSV 파일이 data_dir에 없을 때.
        """
        transactions_path = self.data_dir / self.TRANSACTIONS_FILE
        customers_path = self.data_dir / self.CUSTOMERS_FILE
        articles_path = self.data_dir / self.ARTICLES_FILE

        missing = [
            str(path)
            for path in (transactions_path, customers_path, articles_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "데이터 파일을 찾을 수 없다: "
                + ", ".join(missing)
                + " (scripts/make_subset.py로 data/ 폴더를 만들 수 있다)"
            )

        self.transactions = pd.read_csv(
            transactions_path,
            dtype=self.TRANSACTIONS_DTYPES,
            parse_dates=self.TRANSACTIONS_DATE_COLUMNS,
        )
        self.customers = pd.read_csv(
            customers_path,
            dtype=self.CUSTOMERS_DTYPES,
        )
        self.articles = pd.read_csv(
            articles_path,
            dtype=self.ARTICLES_DTYPES,
            parse_dates=self.ARTICLES_DATE_COLUMNS,
        )
        return self

    def add_text_features(self, text_col: str = "prod_name") -> "DataAnalyzer":
        """텍스트 컬럼에서 단어 수와 문자열 길이를 계산해 상품 테이블에 새 컬럼으로 추가한다.

        추가되는 컬럼(원본 컬럼은 바꾸지 않는다):
            - `{text_col}_word_count`: 단어 수. 공백(연속된 공백은 하나로 본다)으로
              나눈 조각의 개수이며, 구두점은 정리하지 않는다. 예: "Tilly (1)" -> 2
            - `{text_col}_length`: 문자열 길이. 공백을 포함한 글자 수다.

        값이 비어 있는(결측) 행은 두 컬럼도 결측으로 남긴다(0으로 채우지 않는다).
        pandas 문자열 메서드로 컬럼 전체를 한 번에 계산하며 Python 반복문을 쓰지 않는다.

        Args:
            text_col: 계산 대상 텍스트 컬럼 이름. 기본값은 상품명 `prod_name`.

        Returns:
            self. 같은 컬럼을 다시 계산하면 덮어쓴다.

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
        """
        if self.articles is None:
            raise RuntimeError("load_data()를 먼저 호출해야 한다.")

        text = self.articles[text_col]
        # str.split()은 인자 없이 쓰면 공백 연속을 하나의 구분자로 본다.
        self.articles[f"{text_col}_word_count"] = text.str.split().str.len()
        self.articles[f"{text_col}_length"] = text.str.len()
        return self

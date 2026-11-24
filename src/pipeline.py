"""쇼핑몰 단골 찾기 분석 파이프라인.

원본 데이터를 읽어 전처리, 피처 엔지니어링, 요약 통계, RFM 분석을 수행하는
`DataAnalyzer` 클래스를 제공한다. 이 클래스는 데이터 처리와 계산만 맡고,
차트는 노트북(notebooks/analysis_report.ipynb)에서 그린다.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.image import imread


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

    def add_image_features(
        self, image_col: str = "image_path", step: int = 1
    ) -> "DataAnalyzer":
        """상품 이미지에서 평균(Mean)과 표준편차(Std)를 계산해 새 컬럼으로 추가한다.

        추가되는 컬럼(원본 컬럼은 바꾸지 않는다):
            - `img_mean`: 이미지의 모든 값(세로·가로·채널 전부)의 평균. 밝기에 해당한다.
            - `img_std`: 이미지의 모든 값의 표준편차(모집단, ddof=0). 밝고 어두운
              부분의 차이(대비)에 해당한다.

        이미지는 `matplotlib.image.imread()`로 읽어 NumPy 배열로 만든다. 파일을 여는
        반복은 파일마다 한 번씩 필요하지만, 이미지 한 장 안의 픽셀 계산(평균·표준편차)은
        반복문 없이 NumPy 배열 연산으로 한 번에 한다. 이미지는 크기가 서로 달라도
        한 장씩 계산하므로 크기를 맞추지 않으며, 한꺼번에 쌓지도 않는다.
        이미지 경로가 비어 있는 상품은 두 컬럼도 결측으로 남긴다.

        Args:
            image_col: 이미지 파일 경로 컬럼. 경로는 `data_dir` 기준 상대 경로다.
            step: 다운샘플링 간격. 1이면 전체 해상도를 쓰고, 1보다 크면
                `arr[::step, ::step]` 슬라이싱으로 픽셀을 건너뛰어 계산한다.

        Returns:
            self. 다시 호출하면 두 컬럼을 덮어쓴다.

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
            ValueError: `step`이 1 이상의 정수가 아닐 때.
            FileNotFoundError: 경로에 해당하는 이미지 파일이 없을 때.
        """
        if self.articles is None:
            raise RuntimeError("load_data()를 먼저 호출해야 한다.")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError(f"step은 1 이상의 정수여야 한다: {step!r}")

        has_image = self.articles[image_col].notna()
        # 파일을 여는 반복은 파일 수만큼 필요하다.
        stats = [
            self._image_mean_std(self.data_dir / path, step)
            for path in self.articles.loc[has_image, image_col]
        ]

        self.articles["img_mean"] = np.nan
        self.articles["img_std"] = np.nan
        if stats:
            self.articles.loc[has_image, ["img_mean", "img_std"]] = np.array(stats)
        return self

    @staticmethod
    def _image_mean_std(path: Path, step: int) -> Tuple[float, float]:
        """이미지 한 장을 읽어 (평균, 표준편차)를 반환한다."""
        image = imread(str(path))  # 모양 (세로, 가로, 채널), 자료형 uint8, 값 0~255
        pixels = image[::step, ::step]  # step=1이면 전체
        return pixels.mean(), pixels.std()  # 모든 값에 대한 배열 연산(반복문 없음)

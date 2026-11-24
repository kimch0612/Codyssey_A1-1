"""쇼핑몰 단골 찾기 분석 파이프라인.

원본 데이터를 읽어 전처리, 피처 엔지니어링, 요약 통계, RFM 분석을 수행하는
`DataAnalyzer` 클래스를 제공한다. 이 클래스는 데이터 처리와 계산만 맡고,
차트는 노트북(notebooks/analysis_report.ipynb)에서 그린다.
"""

import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

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

    # 테이블 이름과, 결측 대치에 쓸 수 있는 통계.
    TABLE_NAMES = ("transactions", "customers", "articles")
    MISSING_STRATEGIES = ("median", "mean")

    # 상관계수 계산 방법: 피어슨(값 자체), 스피어만(값의 순위)
    CORRELATION_METHODS = ("pearson", "spearman")

    # RFM 세그먼트 이름(보고서와 차트에 쓰는 순서). 분류 규칙은 `segment_customers()`에 있다.
    SEGMENT_ORDER = ("VIP", "Loyal", "New", "At Risk", "Cooling", "Churned")

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

    def missing_report(self) -> pd.DataFrame:
        """세 테이블에서 결측이 있는 컬럼의 결측 개수와 비율(%)을 표로 반환한다.

        Returns:
            컬럼 `table`, `column`, `missing`(결측 개수), `missing_pct`(비율, %)로 이루어진
            DataFrame. 결측이 없는 컬럼은 포함하지 않는다.

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
        """
        frames = []
        for name in self.TABLE_NAMES:
            df = self._get_table(name)
            missing = df.isna().sum()
            missing = missing[missing > 0]
            frames.append(
                pd.DataFrame(
                    {
                        "table": name,
                        "column": missing.index,
                        "missing": missing.to_numpy(),
                        "missing_pct": (missing.to_numpy() / len(df) * 100).round(2),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def handle_missing_values(
        self,
        table: str,
        target_col: str,
        group_col: Optional[str] = None,
        strategy: str = "median",
    ) -> "DataAnalyzer":
        """숫자 컬럼의 결측치를 그룹별(또는 전체) 통계로 대치한다.

        - `group_col`을 주면 같은 그룹 안의 값으로 계산한 통계로 채운다(GroupBy).
        - 그룹 값 자체가 결측이거나 그 그룹에 값이 하나도 없어 통계를 구할 수 없으면
          컬럼 전체의 통계로 채운다.
        - `group_col`이 None이면 컬럼 전체의 통계로 채운다.
        - 컬럼 전체가 결측이라 통계를 구할 수 없으면 결측으로 남긴다.
        - 원래 값이 있던 행은 바꾸지 않는다. 같은 컬럼을 그 자리에서 덮어쓰므로
          대치 전 값이 필요하면 호출 전에 복사해 둔다.

        Args:
            table: 대상 테이블 이름. "transactions", "customers", "articles" 중 하나.
            target_col: 결측을 채울 숫자 컬럼.
            group_col: 그룹을 나누는 컬럼. None이면 그룹 없이 전체 통계를 쓴다.
            strategy: 채울 통계. "median"(중앙값) 또는 "mean"(평균).

        Returns:
            self.

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
            ValueError: 알 수 없는 table 또는 strategy일 때.
            TypeError: target_col이 숫자형이 아닐 때.
        """
        df = self._get_table(table)
        if strategy not in self.MISSING_STRATEGIES:
            raise ValueError(
                f"strategy는 {self.MISSING_STRATEGIES} 중 하나여야 한다: {strategy!r}"
            )
        column = df[target_col]
        if not pd.api.types.is_numeric_dtype(column):
            raise TypeError(f"숫자형 컬럼만 대치할 수 있다: {target_col!r} ({column.dtype})")
        if not column.isna().any():
            return self  # 채울 값이 없다.

        overall = column.agg(strategy)  # 결측을 뺀 전체 통계(전부 결측이면 NaN)
        if group_col is None:
            fill_values = overall
        else:
            # 그룹별 통계를 각 행에 붙인다. 그룹 값이 결측이거나 그룹에 값이 없으면
            # NaN이 되므로 전체 통계로 대신한다.
            fill_values = df.groupby(group_col)[target_col].transform(strategy)
            fill_values = fill_values.fillna(overall)

        df[target_col] = column.fillna(fill_values)
        return self

    def iqr_bounds(
        self, table: str, column: str, threshold: float = 1.5
    ) -> Tuple[float, float]:
        """IQR 방식의 이상치 경계를 계산한다.

        데이터를 작은 값부터 줄 세웠을 때 1/4 지점의 값이 Q1, 3/4 지점의 값이 Q3이고
        IQR = Q3 - Q1이다. 경계는 다음과 같다(결측은 계산에서 제외한다).
            - 아래 경계 = Q1 - threshold * IQR
            - 위 경계   = Q3 + threshold * IQR

        Args:
            table: 대상 테이블 이름. "transactions", "customers", "articles" 중 하나.
            column: 숫자 컬럼.
            threshold: IQR에 곱하는 임계값. 0 이상의 수. 클수록 경계가 넓어진다.

        Returns:
            (아래 경계, 위 경계). 컬럼 전체가 결측이면 둘 다 NaN이다.

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
            ValueError: 알 수 없는 table이거나 threshold가 0 이상의 유한한 수가 아닐 때.
            TypeError: column이 숫자형이 아닐 때.
        """
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError(f"threshold는 0 이상의 유한한 수여야 한다: {threshold!r}")
        values = self._get_table(table)[column]
        if not pd.api.types.is_numeric_dtype(values):
            raise TypeError(f"숫자형 컬럼만 계산할 수 있다: {column!r} ({values.dtype})")

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        return float(q1 - threshold * iqr), float(q3 + threshold * iqr)

    def detect_outliers(
        self, table: str, column: str, threshold: float = 1.5
    ) -> pd.DataFrame:
        """IQR 경계 밖에 있는 값을 가진 행을 찾아 반환한다.

        `값 < 아래 경계` 또는 `값 > 위 경계`인 행만 이상치다. 경계와 같은 값은 이상치가
        아니다. 결측은 이상치로 보지 않는다. 테이블은 바꾸지 않는다.

        Args:
            table: 대상 테이블 이름.
            column: 숫자 컬럼.
            threshold: IQR에 곱하는 임계값(기본 1.5).

        Returns:
            이상치 행들의 DataFrame(원래 행 번호를 유지한다). 이상치가 없으면 빈 표다.

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
            ValueError: 알 수 없는 table이거나 threshold가 0 이상의 유한한 수가 아닐 때.
            TypeError: column이 숫자형이 아닐 때.
        """
        lower, upper = self.iqr_bounds(table, column, threshold)
        df = self._get_table(table)
        values = df[column]
        return df[(values < lower) | (values > upper)]

    def treat_outliers(
        self, table: str, column: str, threshold: float = 1.5
    ) -> "DataAnalyzer":
        """이상치를 경계값으로 대체한 값을 새 컬럼 `{column}_clipped`에 저장한다.

        경계 밖의 값은 가까운 쪽 경계값으로 바꾸고, 경계 안의 값은 그대로 둔다. 행은
        지우지 않는다. 원본 컬럼은 바꾸지 않으므로 처리 전후를 비교할 수 있다.

        Args:
            table: 대상 테이블 이름.
            column: 숫자 컬럼.
            threshold: IQR에 곱하는 임계값(기본 1.5).

        Returns:
            self. 다시 호출하면 `{column}_clipped`를 덮어쓴다.

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
            ValueError: 알 수 없는 table이거나 threshold가 0 이상의 유한한 수가 아닐 때.
            TypeError: column이 숫자형이 아닐 때.
        """
        lower, upper = self.iqr_bounds(table, column, threshold)
        df = self._get_table(table)
        df[f"{column}_clipped"] = df[column].clip(lower=lower, upper=upper)
        return self

    def summarize(self, table: str, columns: Sequence[str]) -> pd.DataFrame:
        """숫자 컬럼들의 요약 통계(평균·중앙값·표준편차·사분위수 등)를 표로 반환한다.

        결측은 계산에서 제외한다. 표준편차는 표본 표준편차(pandas 기본값, ddof=1)다.
        번호 컬럼처럼 숫자로 저장돼 있지만 통계 의미가 없는 컬럼이 섞이지 않도록
        `columns`를 반드시 직접 지정해야 한다.

        Args:
            table: 대상 테이블 이름. "transactions", "customers", "articles" 중 하나.
            columns: 요약할 숫자 컬럼 이름들.

        Returns:
            컬럼 이름을 행으로 하고 `count`, `mean`, `median`, `std`, `min`, `q1`(25%),
            `q3`(75%), `max`를 열로 하는 DataFrame.

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
            ValueError: 알 수 없는 table이거나 columns가 비어 있을 때.
            TypeError: 숫자형이 아닌 컬럼이 있을 때.
        """
        data = self._numeric_data(table, columns, minimum=1)
        return pd.DataFrame(
            {
                "count": data.count(),
                "mean": data.mean(),
                "median": data.median(),
                "std": data.std(),
                "min": data.min(),
                "q1": data.quantile(0.25),
                "q3": data.quantile(0.75),
                "max": data.max(),
            }
        )

    def correlation(
        self, table: str, columns: Sequence[str], method: str = "pearson"
    ) -> pd.DataFrame:
        """숫자 컬럼들 사이의 상관행렬을 계산한다.

        상관계수는 -1~1이며, 1에 가까우면 함께 커지고 -1에 가까우면 한쪽이 커질 때
        다른 쪽이 작아지는 경향이다. 0에 가까우면 그런 경향이 약하다. 상관이 있다고
        해서 한쪽이 다른 쪽의 원인이라는 뜻은 아니다. 결측이 있는 행은 해당 두 컬럼의
        계산에서만 제외한다. 값이 모두 같은 컬럼은 결과가 결측이다.

        Args:
            table: 대상 테이블 이름.
            columns: 숫자 컬럼 이름들(2개 이상).
            method: "pearson"(값 자체의 직선 관계) 또는 "spearman"(값의 순위 사이 관계).
                스피어만은 극단값의 영향을 덜 받는다.

        Returns:
            행과 열이 모두 `columns`인 대칭 DataFrame. 대각선은 1이다.

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
            ValueError: 알 수 없는 table·method이거나 columns가 2개 미만일 때.
            TypeError: 숫자형이 아닌 컬럼이 있을 때.
        """
        if method not in self.CORRELATION_METHODS:
            raise ValueError(
                f"method는 {self.CORRELATION_METHODS} 중 하나여야 한다: {method!r}"
            )
        data = self._numeric_data(table, columns, minimum=2)
        return data.corr(method=method)

    def monthly_sales(
        self,
        table: str = "transactions",
        date_col: str = "order_date",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """거래를 월별로 집계해 건수와 매출 합계, 하루 평균 매출을 표로 반환한다.

        데이터가 한 달의 일부 기간만 있으면 합계가 작게 보이므로, 데이터가 있는 날 수와
        그 달의 일수를 함께 돌려주고 일부 기간만 있는 달을 표시한다.

        Args:
            table: 거래 테이블 이름.
            date_col: 날짜형 컬럼.
            amount_col: 금액 숫자 컬럼.

        Returns:
            월별 한 행의 DataFrame. 컬럼은 `month`(그 달 1일), `orders`(거래 행 수),
            `total_amount`(금액 합계), `days`(데이터가 있는 날 수), `days_in_month`(그 달의
            일수), `is_partial`(days < days_in_month), `daily_avg`(합계 / days).

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
            ValueError: 알 수 없는 table일 때.
            TypeError: date_col이 날짜형이 아니거나 amount_col이 숫자형이 아닐 때.
        """
        df = self._get_table(table)
        dates, amounts = df[date_col], df[amount_col]
        if not pd.api.types.is_datetime64_any_dtype(dates):
            raise TypeError(f"날짜형 컬럼이어야 한다: {date_col!r} ({dates.dtype})")
        if not pd.api.types.is_numeric_dtype(amounts):
            raise TypeError(f"숫자형 컬럼이어야 한다: {amount_col!r} ({amounts.dtype})")

        month = dates.dt.to_period("M")  # 날짜를 "2019-09" 같은 월 단위로 바꾼다.
        result = pd.DataFrame(
            {
                "orders": amounts.groupby(month).size(),
                "total_amount": amounts.groupby(month).sum(),
                "days": dates.dt.normalize().groupby(month).nunique(),
            }
        )
        result["month"] = result.index.to_timestamp()
        result["days_in_month"] = result.index.days_in_month
        result["is_partial"] = result["days"] < result["days_in_month"]
        result["daily_avg"] = result["total_amount"] / result["days"]
        columns = [
            "month", "orders", "total_amount", "days",
            "days_in_month", "is_partial", "daily_avg",
        ]
        return result.reset_index(drop=True)[columns]

    def calculate_rfm(
        self,
        table: str = "transactions",
        customer_col: str = "customer_id",
        date_col: str = "order_date",
        amount_col: str = "amount",
        reference_date: Optional[Union[str, datetime.date, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """고객별 Recency(최근성), Frequency(빈도), Monetary(금액)를 계산한다.

        - Recency: 기준일에서 그 고객의 마지막 구매일까지 지난 날수(정수). 작을수록 최근에
          구매한 것이다. 기준일에 구매한 고객은 0이다.
        - Frequency: 서로 다른 구매일의 수. 같은 날 여러 상품을 사도 한 번으로 센다.
        - Monetary: 금액의 합계.

        고객 ID나 날짜가 비어 있는 행은 계산에서 제외한다. 금액이 비어 있는 행은 구매일로는
        세고 금액 합계에는 더하지 않는다(0). 계산은 새 표를 만들어 반환하며 테이블을
        바꾸지 않는다.

        Args:
            table: 거래 테이블 이름.
            customer_col: 고객 ID 컬럼.
            date_col: 날짜형 구매일 컬럼.
            amount_col: 숫자형 금액 컬럼.
            reference_date: Recency의 기준일. 문자열("2020-09-22"), 날짜, Timestamp를
                받는다. None이면 테이블의 마지막 구매일이다.

        Returns:
            고객당 한 행의 DataFrame(고객 ID 순). 컬럼은 `customer_col`의 이름,
            `recency`, `frequency`, `monetary`다.

        Raises:
            RuntimeError: `load_data()`를 호출하기 전일 때.
            ValueError: 알 수 없는 table이거나, 기준일을 날짜로 해석할 수 없거나, 기준일이
                어느 고객의 마지막 구매일보다 이를 때.
            TypeError: 날짜 컬럼이 날짜형이 아니거나 금액 컬럼이 숫자형이 아니거나,
                기준일이 문자열·날짜·Timestamp가 아닐 때.
        """
        df = self._get_table(table)
        customers, dates, amounts = df[customer_col], df[date_col], df[amount_col]
        if not pd.api.types.is_datetime64_any_dtype(dates):
            raise TypeError(f"날짜형 컬럼이어야 한다: {date_col!r} ({dates.dtype})")
        if not pd.api.types.is_numeric_dtype(amounts):
            raise TypeError(f"숫자형 컬럼이어야 한다: {amount_col!r} ({amounts.dtype})")

        valid = customers.notna() & dates.notna()
        orders = pd.DataFrame(
            {
                "customer": customers[valid],
                "day": dates[valid].dt.normalize(),  # 시각을 버리고 날짜만 남긴다.
                "amount": amounts[valid],
            }
        )
        grouped = orders.groupby("customer")
        last_day = grouped["day"].max()  # 고객별 마지막 구매일

        if reference_date is None:
            reference = orders["day"].max()
        else:
            if not isinstance(reference_date, (str, datetime.date, pd.Timestamp)):
                raise TypeError(
                    f"기준일은 문자열·날짜·Timestamp여야 한다: {reference_date!r}"
                )
            try:
                reference = pd.Timestamp(reference_date)
            except ValueError as error:
                raise ValueError(f"기준일을 날짜로 해석할 수 없다: {reference_date!r}") from error
            if pd.isna(reference):  # pd.NaT도 날짜 객체로 취급되므로 따로 막는다.
                raise ValueError(f"기준일을 날짜로 해석할 수 없다: {reference_date!r}")
            reference = reference.normalize()  # 시각을 버리고 날짜만 남긴다.
            if len(last_day) > 0 and reference < last_day.max():
                raise ValueError(
                    f"기준일({reference.date()})이 마지막 구매일({last_day.max().date()})보다 "
                    "이르다. 기준일은 모든 고객의 마지막 구매일 이후여야 한다."
                )

        return pd.DataFrame(
            {
                customer_col: last_day.index,
                "recency": (reference - last_day).dt.days.to_numpy(),
                "frequency": grouped["day"].nunique().to_numpy(),
                "monetary": grouped["amount"].sum().to_numpy(),
            }
        )

    def score_rfm(
        self,
        rfm: pd.DataFrame,
        n_bins: int = 5,
        frequency_edges: Sequence[float] = (1, 2, 4, 9),
    ) -> pd.DataFrame:
        """RFM 값을 1~`n_bins`점으로 바꾼다. 점수가 높을수록 좋은 고객이다.

        - Recency, Monetary: 값을 같은 인원씩 `n_bins`구간으로 나눈(분위수) 점수.
          Recency는 작을수록 좋으므로 가장 최근 구간이 최고점이다.
        - Frequency: 같은 값이 많아(구매일 1일뿐인 고객이 많다) 분위수 경계가 겹치므로
          `frequency_edges`로 직접 나눈다. 경계 이하의 값이 해당 구간이다. 기본값
          (1, 2, 4, 9)는 구매일 수 1 / 2 / 3~4 / 5~9 / 10 이상을 1~5점으로 한다.

        Args:
            rfm: `calculate_rfm()`의 결과(`recency`, `frequency`, `monetary` 컬럼).
            n_bins: 점수 단계 수(2 이상). 분위수 구간 수이기도 하다.
            frequency_edges: Frequency 구간 경계. 오름차순이어야 하고 길이가
                `n_bins - 1`이어야 한다.

        Returns:
            입력에 `r_score`, `f_score`, `m_score`(정수) 컬럼을 추가한 새 표. 입력은 바뀌지 않는다.

        Raises:
            TypeError: rfm이 DataFrame이 아닐 때.
            KeyError: 필요한 컬럼이 없을 때.
            ValueError: n_bins·frequency_edges가 올바르지 않거나, 값이 비어 있거나,
                분위수 경계가 겹쳐 구간을 만들 수 없을 때.
        """
        self._check_rfm_table(rfm, ("recency", "frequency", "monetary"))
        if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError(f"n_bins는 2 이상의 정수여야 한다: {n_bins!r}")
        edges = list(frequency_edges)
        if len(edges) != n_bins - 1:
            raise ValueError(
                f"frequency_edges는 {n_bins - 1}개여야 한다(n_bins={n_bins}): {edges!r}"
            )
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError(f"frequency_edges는 오름차순이어야 한다: {edges!r}")

        labels = list(range(1, n_bins + 1))
        try:
            r_bins = pd.qcut(rfm["recency"], n_bins, labels=labels[::-1])
            m_bins = pd.qcut(rfm["monetary"], n_bins, labels=labels)
        except ValueError as error:
            raise ValueError(f"분위수 경계가 겹쳐 점수 구간을 만들 수 없다: {error}") from error
        f_bins = pd.cut(rfm["frequency"], bins=[-np.inf] + edges + [np.inf], labels=labels)

        scored = rfm.copy()
        scored["r_score"] = r_bins.astype(int)
        scored["f_score"] = f_bins.astype(int)
        scored["m_score"] = m_bins.astype(int)
        return scored

    def segment_customers(self, scored: pd.DataFrame) -> pd.DataFrame:
        """R·F 점수(1~5)로 고객을 6개 세그먼트로 나눈다. 모든 고객은 정확히 한 곳에 속한다.

        규칙은 위에서부터 차례로 적용한다.
            1. VIP     : R >= 4 그리고 F >= 4  (최근에 오고 자주 오는 고객)
            2. Loyal   : R >= 3 그리고 F >= 3  (꾸준히 오는 고객)
            3. New     : R >= 4 그리고 F <= 2  (최근에 왔지만 방문이 적은 고객)
            4. At Risk : R <= 2 그리고 F >= 3  (한때 자주 왔지만 최근 오지 않는 고객)
            5. Cooling : R == 3                 (위에 안 걸린 고객: 최근성 중간, 방문 적음)
            6. Churned : 나머지(R <= 2 그리고 F <= 2) (오래 오지 않고 방문도 적은 고객)

        M 점수는 규칙에 쓰지 않는다(F와 정보가 겹친다). 세그먼트별 금액 차이를 확인하는 데 쓴다.

        Args:
            scored: `score_rfm()`의 결과(`r_score`, `f_score` 컬럼, 값은 1~5).

        Returns:
            입력에 `segment` 컬럼을 추가한 새 표. 입력은 바뀌지 않는다.

        Raises:
            TypeError: scored가 DataFrame이 아닐 때.
            KeyError: 필요한 컬럼이 없을 때.
            ValueError: 점수가 비어 있거나 1~5의 정수가 아닐 때.
        """
        self._check_rfm_table(scored, ("r_score", "f_score"))
        r, f = scored["r_score"], scored["f_score"]
        for name, score in (("r_score", r), ("f_score", f)):
            if not (score.between(1, 5).all() and (score % 1 == 0).all()):
                raise ValueError(f"{name}는 1~5의 정수여야 한다.")

        conditions = [
            (r >= 4) & (f >= 4),
            (r >= 3) & (f >= 3),
            (r >= 4) & (f <= 2),
            (r <= 2) & (f >= 3),
            r == 3,
        ]
        names = ["VIP", "Loyal", "New", "At Risk", "Cooling"]
        result = scored.copy()
        result["segment"] = np.select(conditions, names, default="Churned")
        return result

    def segment_summary(self, segmented: pd.DataFrame) -> pd.DataFrame:
        """세그먼트별 고객 수, 평균 R·F·M, 금액 합계와 비중을 표로 반환한다.

        고객이 없는 세그먼트도 `SEGMENT_ORDER` 순서대로 항상 포함한다(고객 수 0).

        Args:
            segmented: `segment_customers()`의 결과(`segment`, `recency`, `frequency`,
                `monetary` 컬럼).

        Returns:
            컬럼 `segment`, `customers`(고객 수), `avg_recency`, `avg_frequency`,
            `avg_monetary`, `total_monetary`(금액 합계), `customer_pct`(고객 비중 %),
            `monetary_pct`(금액 비중 %)의 DataFrame.

        Raises:
            TypeError: segmented가 DataFrame이 아닐 때.
            KeyError: 필요한 컬럼이 없을 때.
            ValueError: 알 수 없는 세그먼트 이름이 있을 때.
        """
        self._check_rfm_table(segmented, ("segment", "recency", "frequency", "monetary"))
        unknown = set(segmented["segment"].unique()) - set(self.SEGMENT_ORDER)
        if unknown:
            raise ValueError(f"알 수 없는 세그먼트: {sorted(unknown)}")

        grouped = segmented.groupby("segment")
        summary = pd.DataFrame(
            {
                "customers": grouped.size(),
                "avg_recency": grouped["recency"].mean(),
                "avg_frequency": grouped["frequency"].mean(),
                "avg_monetary": grouped["monetary"].mean(),
                "total_monetary": grouped["monetary"].sum(),
            }
        ).reindex(list(self.SEGMENT_ORDER))
        summary["customers"] = summary["customers"].fillna(0).astype(int)
        summary["total_monetary"] = summary["total_monetary"].fillna(0.0)
        summary["customer_pct"] = summary["customers"] / len(segmented) * 100
        summary["monetary_pct"] = summary["total_monetary"] / segmented["monetary"].sum() * 100
        summary.index.name = "segment"
        return summary.reset_index()

    @staticmethod
    def _check_rfm_table(table: pd.DataFrame, columns: Sequence[str]) -> None:
        """RFM 표가 DataFrame이고 필요한 컬럼이 있으며 그 값이 비어 있지 않은지 확인한다."""
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"DataFrame이어야 한다: {type(table).__name__}")
        missing = [name for name in columns if name not in table.columns]
        if missing:
            raise KeyError(f"필요한 컬럼이 없다: {missing}")
        if table[list(columns)].isna().any().any():
            raise ValueError(f"{list(columns)} 컬럼에 결측이 있다.")

    def _numeric_data(
        self, table: str, columns: Sequence[str], minimum: int
    ) -> pd.DataFrame:
        """테이블에서 숫자 컬럼들만 골라 반환한다. 컬럼이 부족하거나 숫자형이 아니면 오류를 낸다."""
        df = self._get_table(table)
        names = [columns] if isinstance(columns, str) else list(columns)
        if len(names) < minimum:
            raise ValueError(f"컬럼을 {minimum}개 이상 지정해야 한다: {names!r}")
        for name in names:
            if not pd.api.types.is_numeric_dtype(df[name]):
                raise TypeError(f"숫자형 컬럼만 계산할 수 있다: {name!r} ({df[name].dtype})")
        return df[names]

    def _get_table(self, name: str) -> pd.DataFrame:
        """테이블 이름으로 DataFrame을 돌려준다. 읽기 전이거나 이름이 틀리면 오류를 낸다."""
        if name not in self.TABLE_NAMES:
            raise ValueError(f"table은 {self.TABLE_NAMES} 중 하나여야 한다: {name!r}")
        df = getattr(self, name)
        if df is None:
            raise RuntimeError("load_data()를 먼저 호출해야 한다.")
        return df

"""H&M 원본 데이터에서 과제용 부분집합을 만든다.

과제 제출물이 아니라 데이터 준비 도구다. 원본(data/raw/)은 3.5GB가 넘어
그대로 분석하기 어렵기 때문에, 과제 요구사항을 채우는 크기로 잘라 data/에 저장한다.

출처: H&M Personalized Fashion Recommendations (Kaggle)
https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

만드는 파일
- data/transactions.csv : 분석 기간에 1회 이상 구매한 고객 중 N_CUSTOMERS명을 무작위로 뽑고,
                          그 고객들의 기간 내 거래를 전부 남긴 것 (RFM, 월별 추이용)
                          원본 t_dat → order_date, price → amount 로 이름만 바꾼다.
- data/customers.csv    : 위 고객들의 고객 정보 (원본 그대로, 결측치 포함)
- data/articles.csv     : 분석 기간 전체에서 판매 건수 상위 N_ARTICLES개 상품의 정보 (원본 그대로)
                          + 원본 거래 전체에서 집계한 sales_count, avg_price, first_sold_date

결측치·이상치는 과제에서 직접 다루도록 일부러 손대지 않는다.

실행: 과제 폴더에서 `python scripts/make_subset.py`
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data"

WINDOW_MONTHS = 12      # 분석 기간: 최종 거래일로부터 12개월
N_CUSTOMERS = 2000      # 거래 테이블에 남길 고객 수
N_ARTICLES = 1500       # 상품 테이블에 남길 상품 수
SEED = 42               # 무작위 표본을 매번 같게 뽑기 위한 시드
CHUNK_SIZE = 4_000_000  # 원본 거래를 한 번에 읽을 행 수

# 앞자리 0이 사라지지 않도록 ID는 문자열로 읽는다.
ID_DTYPES = {"article_id": str, "customer_id": str}


def read_transactions():
    """원본 거래를 나눠 읽으며 상품별 최초 판매일을 구하고, 분석 기간 후보 행만 남긴다."""
    first_sold = None
    recent = []
    for chunk in pd.read_csv(RAW_DIR / "transactions_train.csv",
                             chunksize=CHUNK_SIZE, dtype=ID_DTYPES):
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
        chunk_first = chunk.groupby("article_id")["t_dat"].min()
        first_sold = chunk_first if first_sold is None else \
            pd.concat([first_sold, chunk_first]).groupby(level=0).min()
        # 기간 시작일은 아직 모르므로, 원본 최종일(2020-09-22) 기준 13개월 전부터 넉넉히 보관한다.
        recent.append(chunk[chunk["t_dat"] >= "2019-08-01"])
        print(f"  읽은 행: {chunk.index[-1] + 1:,}", flush=True)
    return pd.concat(recent, ignore_index=True), first_sold


def main():
    print("원본 거래 읽는 중...")
    tx, first_sold = read_transactions()

    end = tx["t_dat"].max()
    start = end - pd.DateOffset(months=WINDOW_MONTHS) + pd.Timedelta(days=1)
    window = tx[tx["t_dat"] >= start]
    print(f"분석 기간: {start.date()} ~ {end.date()} / 거래 {len(window):,}건")

    # 거래 테이블: 고객 표본의 기간 내 거래 전부
    customers_in_window = np.sort(window["customer_id"].unique())
    rng = np.random.default_rng(SEED)
    picked = rng.choice(customers_in_window, size=N_CUSTOMERS, replace=False)
    transactions = window[window["customer_id"].isin(picked)] \
        .sort_values(["t_dat", "customer_id"]).reset_index(drop=True)
    transactions["t_dat"] = transactions["t_dat"].dt.strftime("%Y-%m-%d")
    # Subject가 요구하는 RFM 컬럼명(customer_id, order_date, amount)에 맞춘다.
    # H&M 거래는 상품 1개 = 1행(수량 컬럼 없음)이라 행의 price가 곧 그 거래의 금액이다.
    transactions = transactions.rename(columns={"t_dat": "order_date", "price": "amount"})

    # 고객 테이블
    customers = pd.read_csv(RAW_DIR / "customers.csv", dtype=ID_DTYPES)
    customers = customers[customers["customer_id"].isin(picked)].reset_index(drop=True)

    # 상품 테이블: 기간 전체 판매 상위 상품 + 판매 집계
    stats = window.groupby("article_id").agg(sales_count=("price", "size"),
                                             avg_price=("price", "mean"))
    top = stats.sort_values("sales_count", ascending=False).head(N_ARTICLES)
    top["first_sold_date"] = first_sold.reindex(top.index).dt.strftime("%Y-%m-%d")

    articles = pd.read_csv(RAW_DIR / "articles.csv", dtype={"article_id": str, "product_code": str})
    articles = articles.merge(top, left_on="article_id", right_index=True, how="inner") \
        .sort_values("sales_count", ascending=False).reset_index(drop=True)

    transactions.to_csv(OUT_DIR / "transactions.csv", index=False)
    customers.to_csv(OUT_DIR / "customers.csv", index=False)
    articles.to_csv(OUT_DIR / "articles.csv", index=False)

    print(f"transactions.csv: {transactions.shape} / 고객 {transactions['customer_id'].nunique():,}명")
    print(f"customers.csv   : {customers.shape}")
    print(f"articles.csv    : {articles.shape}")


if __name__ == "__main__":
    main()

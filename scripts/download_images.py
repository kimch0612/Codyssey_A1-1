"""data/articles.csv에 있는 상품의 이미지만 Kaggle에서 내려받는다.

과제 제출물이 아니라 데이터 준비 도구다. make_subset.py를 먼저 실행해야 한다.
원본 이미지 폴더 전체는 수십 GB라서, 과제에 쓰는 상품의 이미지만 한 장씩 받는다.

준비 (한 번만)
1. Kaggle 대회 규칙에 동의한다.
2. `pip install kaggle` 후 `kaggle auth login`으로 로그인한다.

결과
- data/images/<article_id 앞 3자리>/<article_id>.jpg
- 모든 상품을 확인하고 나면 data/articles.csv에 image_path 컬럼을 추가한다.
  값은 data/ 기준 상대 경로이고, Kaggle에 이미지가 없는 상품은 빈 값(결측)으로 둔다.

Kaggle은 짧은 시간에 요청이 많으면 429(요청 과다)로 한동안 막는다.
막히면 받은 데까지 두고 종료 코드 2로 끝나므로, 시간이 지난 뒤 다시 실행하면 이어서 받는다.

실행: 과제 폴더에서 `python scripts/download_images.py`
"""

import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTICLES_CSV = DATA_DIR / "articles.csv"
COMPETITION = "h-and-m-personalized-fashion-recommendations"
REQUEST_DELAY = 1.0   # 요청 사이 간격(초). 한 장씩 천천히 받는다.
NOT_FOUND_TRIES = 2   # 404를 이만큼 받으면 Kaggle에 이미지가 없는 상품으로 본다.


class RateLimited(Exception):
    """Kaggle이 429로 요청을 막았다."""


_api = None


def get_api():
    """Kaggle API 객체를 한 번 만들어 재사용한다."""
    global _api
    if _api is None:
        from kaggle.api.kaggle_api_extended import KaggleApi
        _api = KaggleApi()
        _api.authenticate()
    return _api


def refresh_login():
    """로그인 토큰(약 1시간짜리)을 새로 발급받는다.

    kaggle 2.2.4는 만료 후 30분이 더 지나야 스스로 갱신하므로, 401을 받으면 직접 갱신한다.
    """
    global _api
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk import KaggleCredentials
    with KaggleApi().build_kaggle_client() as client:
        KaggleCredentials.load(client=client).refresh_access_token()
    _api = None


def relative_path(article_id):
    """Kaggle과 같은 폴더 구조: images/<앞 3자리>/<article_id>.jpg"""
    return f"images/{article_id[:3]}/{article_id}.jpg"


def fetch(rel, tmp):
    """파일 하나를 tmp에 받는다. 받으면 True, Kaggle에 없으면 False."""
    not_found = 0
    while True:
        try:
            get_api().competition_download_file(COMPETITION, rel, path=tmp, quiet=True)
            return True
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429:
                raise RateLimited() from e
            if status == 401:
                refresh_login()
            elif status == 404:
                not_found += 1
                if not_found >= NOT_FOUND_TRIES:
                    return False
                time.sleep(5)
            else:
                raise


def download_one(article_id):
    """이미지 한 장을 받아 제자리에 둔다. 성공하면 상대 경로, Kaggle에 없으면 None."""
    rel = relative_path(article_id)
    target = DATA_DIR / rel
    if target.exists():
        return rel
    time.sleep(REQUEST_DELAY)
    with tempfile.TemporaryDirectory() as tmp:
        if not fetch(rel, tmp):
            print(f"  Kaggle에 이미지 없음: {article_id}", flush=True)
            return None
        got = next(Path(tmp).iterdir())
        target.parent.mkdir(parents=True, exist_ok=True)
        if got.suffix == ".zip":  # 서버가 압축해서 보내는 경우
            with zipfile.ZipFile(got) as z:
                with z.open(z.namelist()[0]) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            shutil.move(str(got), target)
    return rel


def main():
    articles = pd.read_csv(ARTICLES_CSV, dtype={"article_id": str, "product_code": str})
    ids = articles["article_id"].tolist()
    print(f"이미지 {len(ids)}장 확인·다운로드 시작", flush=True)

    paths = []
    for i, article_id in enumerate(ids, start=1):
        try:
            paths.append(download_one(article_id))
        except RateLimited:
            done = sum((DATA_DIR / relative_path(a)).exists() for a in ids)
            print(f"요청 한도 도달: {done}장까지 받음. 잠시 뒤 다시 실행하면 이어서 받는다.", flush=True)
            sys.exit(2)
        if i % 100 == 0:
            print(f"  {i}/{len(ids)}", flush=True)

    articles["image_path"] = paths
    articles.to_csv(ARTICLES_CSV, index=False)
    missing = articles["image_path"].isna().sum()
    print(f"완료: 이미지 있음 {len(ids) - missing}장 / 없음 {missing}장")


if __name__ == "__main__":
    main()

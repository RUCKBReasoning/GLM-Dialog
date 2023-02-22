from xkdai.logger import logger
from xkdai.search_engine.base import SearchEngine


class BaiduSearcher(SearchEngine):
    def __init__(self, timeout=5, max_tries=5, mongo_db=None):
        raise NotImplementedError

    def search(self, query: str) -> list:
        raise NotImplementedError

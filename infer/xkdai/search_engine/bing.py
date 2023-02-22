import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from xkdai.logger import logger
from xkdai.search_engine.base import SearchEngine


class BingSearcher(SearchEngine):
    def __init__(self, timeout=5, max_tries=5, mongo_db=None):
        self.url = "https://www.bing.com/search?q={}"
        self.timeout = timeout
        self.max_tries = max_tries
        self.mongo_db = mongo_db
        logger.info(
            f"BingSearcher timeout:{timeout}, max_tries:{max_tries} mongo_db:{mongo_db}")
        if self.mongo_db:
            self.cache_collection = self.mongo_db['search_cache']
            self.cache_collection.create_index('query')

    def _search_single(self, query):
        response = requests.get(
            self.url.format(query), timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []
        for result in soup.find_all('li', class_='b_algo'):
            link = result.find('a').get('href')
            title = result.find('a').text
            snippet = result.find('div', class_='b_caption').text
            results.append(
                {'link': link, 'title': title, 'snippet': snippet})
        return results

    def search(self, query):
        if self.mongo_db:
            cached_result = self.cache_collection.find_one({'query': query})
            if cached_result:
                return cached_result['results']

        for i in range(self.max_tries):
            try:
                results = self._search_single(query)
                if not results:
                    logger.warning(
                        f"Empty search results from bing for query:[{query}]")
                if self.mongo_db and results:
                    self.cache_collection.replace_one(
                        {'query': query}, {'query': query, 'results': results}, upsert=True)
                return results
            except:
                if i == self.max_tries - 1:
                    logger.warning("BING API FAIL")
                    raise IOError("BING API FAIL")

from xkdai.search_engine import SearchEngine
from xkdai.logger import logger
from pymongo.database import Database
from xkdai.generation_model import GenerationModel


class Chatbot:
    def __init__(self, model: GenerationModel, searcher: SearchEngine, mongo_db: Database, query_limit: int = 10, response_limit: int = 100):
        self.model = model
        self.mongo_db = mongo_db
        self.searcher = searcher
        self.query_limit = query_limit
        self.response_limit = response_limit

    def chat(self, utters) -> str:
        query = self.query_generation(utters)
        knowledge = self.get_knowledge_by_search_engine(query)
        response = self.response_generation(knowledge, utters)
        logger.info({"call": "chatbot", "utters": utters, "query": query,
                    "knowledge": knowledge, "response": response})
        self.mongo_db["conversations"].insert_one(
            {"utters": utters, "query": query, "knowledge": knowledge, "response": response})
        return response

    def query_generation(self, utters) -> str:
        dialog_line = self._linearize_utters(utters)
        prompt = f'生成查询:对话:{dialog_line} 此时应该查询:[sMASK]'
        return self.model.generate(prompt, self.query_limit)

    def response_generation(self, knowledge, utters) -> str:
        dialog_line = self._linearize_utters(utters)
        prompt = f'背景:{knowledge} 对话:{dialog_line} B:[sMASK]'
        return self.model.generate(prompt, self.response_limit)

    def get_knowledge_by_search_engine(self, query: str) -> str:
        try:
            results = self.searcher.search(query)
        except IOError:
            logger.warn(f'No Result from searcher {self.searcher}')
            results = []
        if results:
            pickup = results[0]  # 目前只考虑第一个结果
            knowledge = f"{pickup['title']}:{pickup['snippet']}"
            return knowledge
        return ""

    def _linearize_utters(self, utters):
        a, b = 'A: ', 'B: '
        dialog_line = ''
        for idx, utter in enumerate(utters):
            if idx % 2 == 0:
                part_sentence = a + utter
            else:
                part_sentence = b + utter
            dialog_line += part_sentence
            dialog_line += '\t'
        return dialog_line

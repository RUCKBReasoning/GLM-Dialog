import abc


class SearchEngine(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def search(self, query: str) -> list:
        pass

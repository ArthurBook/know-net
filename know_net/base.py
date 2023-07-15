import abc
from typing import Iterator
import networkx as nx


class ContentRetriever(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[str]:
        ...


class GraphBuilder(abc.ABC):
    @abc.abstractmethod
    def add_content(self, content: str) -> None:
        ...

    @property
    @abc.abstractmethod
    def graph(self) -> nx.Graph:
        ...


class Retriever(abc.ABC):
    @abc.abstractmethod
    def get_nodes(self, question: str) -> str:
        ...


class LanguageModel:
    @abc.abstractmethod
    def answer(self, question) -> str:
        ...

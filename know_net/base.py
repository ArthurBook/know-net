import abc
from typing import Iterator
import networkx as nx


class WebScraper(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[str]:
        ...


class GraphBuilder(abc.ABC):
    @abc.abstractmethod
    def add_content(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def graph(self) -> nx.Graph:
        ...


class LanguageModel:
    @abc.abstractmethod
    def answer(self, question) -> str:
        ...

import abc
from typing import Iterable, Iterator, NamedTuple
import networkx as nx


class Content(NamedTuple):
    text: str
    url: str


class ContentRetriever(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[Content]:
        ...


class GraphBuilder(abc.ABC):
    def add_content_batch(self, contents: Iterable[str]) -> None:
        [self.add_content(c) for c in contents]

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

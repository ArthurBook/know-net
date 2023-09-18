import abc
import asyncio
from collections import UserString
from typing import (
    Annotated,
    AsyncGenerator,
    Generator,
    Iterable,
    Iterator,
    List,
    NamedTuple,
)

import networkx as nx
import numpy as np

from know_net import mixins

###
# Content Retrieval (raw text to be processed by models)
###

Source = Annotated[str, "reference to where the content came from"]


class Content(UserString):
    source: Source

    def __init__(self, text: str, source: Source) -> None:
        super().__init__(text)
        self.source = source


class ContentRetriever(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[Content]:
        ...


class AsynchronousContentRetriever(ContentRetriever, abc.ABC):
    def __iter__(self) -> Iterator[Content]:
        yield from mixins.iterate_through(self.__aiter__())

    @abc.abstractmethod
    async def __aiter__(self) -> AsyncGenerator[Content, None]:
        ...


###
# Extracting graph triplets
###


class KGTriple(NamedTuple):
    source: Source
    subject: str
    predicate: str
    object_: str


class TripletExtractor(abc.ABC):
    @abc.abstractmethod
    def extract_triplets(self, content: Content) -> Iterator[KGTriple]:
        ...


class AsynchronousTripletExtractor(TripletExtractor, abc.ABC):
    def extract_triplets(self, content: Content) -> Iterator[KGTriple]:
        yield from mixins.iterate_through(self.extract_triplets_async(content))

    @abc.abstractmethod
    def extract_triplets_async(
        self, content: Content
    ) -> AsyncGenerator[KGTriple, None]:
        ...


###
# Embedder
###


class Embedder:
    @abc.abstractmethod
    def embed(self, strings: str) -> np.ndarray:
        ...


class AsyncEmbedder(Embedder):
    def embed(self, strings: str) -> np.ndarray:
        return asyncio.run(self.embed_async(strings))

    @abc.abstractmethod
    async def embed_async(self, strings: str) -> np.ndarray:
        ...


###
# Building the graph
###


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


class Test:
    def __iter__(self) -> Generator[str, str, None]:
        string = ""
        while True:
            string = yield string

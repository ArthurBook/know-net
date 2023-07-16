import asyncio
import functools
import itertools
from typing import (
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    cast,
)

import diskcache
import networkx as nx
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.graphs.networkx_graph import NetworkxEntityGraph
from langchain.indexes import GraphIndexCreator
from langchain.llms import HuggingFaceTextGenInference
from langchain.vectorstores import Chroma, FAISS
from loguru import logger
from pydantic import BaseModel

from know_net.base import GraphBuilder


MAX_CONCURRENCY = 100
logger = logger.opt(ansi=True)


class Entity(BaseModel):
    name: str
    is_a: Optional["Entity"] = None

    def __hash__(self) -> int:
        return hash(self.name)


class KGTriple(NamedTuple):
    subject: Entity
    predicate: str
    object_: Entity


RootEntity = Entity(name="root", is_a=None)


class LLMGraphBuilder(GraphBuilder):
    _semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    def __init__(self) -> None:
        super().__init__()
        # llm = HuggingFaceTextGenInference(
        #     inference_server_url="http://100.79.46.78:8081",
        #     verbose=True,
        #     max_new_tokens=512,
        #     top_k=10,
        #     top_p=0.95,
        #     typical_p=0.95,
        #     temperature=0.01,
        # )  # type: ignore
        llm = ChatOpenAI(temperature=0)
        self.index_creator = GraphIndexCreator(llm=llm)
        self.triples: List[KGTriple] = []
        self.embeddings = OpenAIEmbeddings()  # type: ignore
        self.vectorstore = FAISS.from_texts(["root"], self.embeddings)
        self.match_threshold = 0.95
        self.doc_to_entity: Dict[str, Entity] = {}
        self.llm_cache = diskcache.Cache(".triples_cache")
        logger.info("Initialized LLMGraphBuilder")

    def add_content_batch(self, contents: Iterable[str]) -> None:
        contents = list(contents)  # in case of generator
        asyncio.run(self.add_missing_contents_to_cache(contents))
        graphs = [cast(NetworkxEntityGraph, self.llm_cache[t]) for t in contents]
        normalized_graph_triples = map(self.normalize_graph_triples, graphs)
        self.triples.extend(itertools.chain.from_iterable(normalized_graph_triples))

    async def add_missing_contents_to_cache(self, contents: Iterable[str]) -> None:
        missing_from_cache = itertools.filterfalse(self.is_in_cache, contents)
        tasks = [self.add_to_cache_async(t) for t in missing_from_cache]
        await asyncio.gather(*tasks)

    async def add_to_cache_async(self, text: str) -> None:
        async with self._semaphore:
            try:
                entity_graph = await self.index_creator.afrom_text(text)
                self.llm_cache[text] = entity_graph
            except:
                entity_graph = await self.index_creator.afrom_text(text[: 4000 * 3])
                self.llm_cache[text] = entity_graph

    def add_content(self, content: str) -> None:
        if self.is_in_cache(content):
            graph = cast(NetworkxEntityGraph, self.llm_cache[content])
        else:
            self.llm_cache[content] = graph = self.index_creator.from_text(content)
        self.triples.extend(self.normalize_graph_triples(graph))

    def is_in_cache(self, text: str) -> bool:
        hit = text in self.llm_cache
        if hit:
            logger.debug("cache <green>hit</green> for content: {}", text[:18])
        else:
            logger.debug("cache <red>miss</red> for content: {}", text[:18])
        return hit

    def normalize_graph_triples(self, graph: NetworkxEntityGraph) -> Iterator[KGTriple]:
        return itertools.starmap(self._normalize_triple, graph.get_triples())

    def _normalize_triple(self, subject: str, object_: str, predicate: str) -> KGTriple:
        s = self.vectorstore.similarity_search_with_relevance_scores(subject, k=1)
        o = self.vectorstore.similarity_search_with_relevance_scores(object_, k=1)
        if len(s) and s[0][1] > self.match_threshold:
            logger.debug("match for subject: {}", subject)
            # if it already exists, then take existing entity
            s_entity = self.doc_to_entity[s[0][0].page_content]
        else:
            logger.debug("miss for subject: {}", subject)
            # if doesn't exist, add the entity to the vectorebase
            s_entity = Entity(name=subject)
            s_doc = Document(page_content=subject)
            self.vectorstore.add_documents([s_doc])
            self.doc_to_entity[s_doc.page_content] = s_entity
        if len(o) and o[0][1] > self.match_threshold:
            logger.debug("match for object: {}", object_)
            o_entity = self.doc_to_entity[o[0][0].page_content]
        else:
            logger.debug("miss for object: {}", object_)
            o_entity = Entity(name=object_)
            o_doc = Document(page_content=object_)
            self.vectorstore.add_documents([o_doc])
            self.doc_to_entity[o_doc.page_content] = o_entity
        return KGTriple(s_entity, predicate, o_entity)

    @property
    def graph(self) -> nx.Graph:
        def convert_to_graph(triples: List[KGTriple]) -> nx.Graph:
            graph = nx.Graph()
            for triple in triples:
                subject = triple.subject
                predicate = triple.predicate
                object_ = triple.object_
                graph.add_node(subject)
                graph.add_node(object_)
                graph.add_edge(subject, object_, label=predicate)
            return graph

        return convert_to_graph(self.triples)

    def search(self, q: str):
        return self.vectorstore.similarity_search(q)


from langchain.indexes import GraphIndexCreator

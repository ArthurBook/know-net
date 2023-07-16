import asyncio
import functools
import itertools
from typing import (
    Coroutine,
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
from langchain.embeddings import base as embeddings_base, openai as openai_embeddings
from langchain.docstore.document import Document
from langchain.graphs.networkx_graph import NetworkxEntityGraph
from langchain.indexes import GraphIndexCreator
from langchain.llms import base as llm_base
from langchain.llms import huggingface_text_gen_inference, openai
from langchain.vectorstores import Chroma
from loguru import logger
from openai import InvalidRequestError
from pydantic import BaseModel

from know_net.base import GraphBuilder

logger = logger.opt(ansi=True)

MAX_LLM_CONCURRENCY = 100
MAX_EMBEDDER_CONCURRENCY = 100
DEFAULT_MATCH_THRESHOLD = 0.95
TRIPLES_CACHE_PATH = ".triples_cache/%s"
CHROMA_PERSISTENT_DISK_DIR = ".chroma_cache/%s"

DEFAULT_LLM = openai.OpenAIChat()  # type: ignore
DEFAULT_EMBEDDER = openai_embeddings.OpenAIEmbeddings()  # type: ignore


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
    _llm_semaphore = asyncio.Semaphore(MAX_LLM_CONCURRENCY)
    _embedder_semaphore = asyncio.Semaphore(MAX_EMBEDDER_CONCURRENCY)

    def __init__(
        self,
        llm: Optional[llm_base.BaseLLM] = None,
        embedding_model: Optional[embeddings_base.Embeddings] = None,
        match_treshold: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.llm = llm or DEFAULT_LLM
        self.embeddings = embedding_model or DEFAULT_EMBEDDER
        self.match_threshold = match_treshold or DEFAULT_MATCH_THRESHOLD

        self.index_creator = GraphIndexCreator(llm=self.llm)

        self.vectorstore = get_chroma_vectorstore(self.embeddings)
        self.llm_cache = get_cache_triples_cache(self.llm)

        self.doc_to_entity: Dict[str, Entity] = {}
        self.triples: List[KGTriple] = []

        logger.info("Initialized LLMGraphBuilder")

    def add_content_batch(self, contents: Iterable[str]) -> None:
        contents = list(contents)  # in case of generator
        asyncio.run(self.update_llm_cache_async(contents))
        graphs = [cast(NetworkxEntityGraph, self.llm_cache[t]) for t in contents]
        normalized_graph_triples = map(self.normalize_graph_triples, graphs)
        self.triples.extend(itertools.chain.from_iterable(normalized_graph_triples))

    def add_content(self, content: str) -> None:
        if self.is_in_llm_cache(content):
            graph = cast(NetworkxEntityGraph, self.llm_cache[content])
        else:
            self.llm_cache[content] = graph = self.index_creator.from_text(content)
        self.triples.extend(self.normalize_graph_triples(graph))

    ## Updating LLM cache
    async def update_llm_cache_async(self, contents: Iterable[str]) -> None:
        missing_from_cache = itertools.filterfalse(self.is_in_llm_cache, contents)
        tasks = [self.add_to_item_cache_async(t) for t in missing_from_cache]
        await asyncio.gather(*tasks)

    async def add_to_item_cache_async(self, text: str) -> None:
        async with self._llm_semaphore:
            try:
                entity_graph = await self.index_creator.afrom_text(text)
            except InvalidRequestError as e:  # TODO pretty hacky
                logger.exception(e)
                entity_graph = await self.index_creator.afrom_text(text[:12000])

            self.llm_cache[text] = entity_graph

    ## Normalizing triples
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

    def is_in_llm_cache(self, text: str) -> bool:
        hit = text in self.llm_cache
        if hit:
            logger.debug("cache <green>hit</green> for content: {}", text[:18])
        else:
            logger.debug("cache <red>miss</red> for content: {}", text[:18])
        return hit


def get_chroma_vectorstore(embedder: embeddings_base.Embeddings) -> Chroma:
    if isinstance(embedder, openai_embeddings.OpenAIEmbeddings):
        name = embedder.model
    else:
        name = embedder.__class__.__name__
        logger.warning("Unknown llm type: {}", name)
    return Chroma(
        embedding_function=embedder,
        persist_directory=CHROMA_PERSISTENT_DISK_DIR % name,
    )


def get_cache_triples_cache(
    llm: llm_base.BaseLLM,
) -> diskcache.Cache:
    if isinstance(llm, openai.OpenAIChat):
        name = llm.model_name
    elif isinstance(llm, huggingface_text_gen_inference.HuggingFaceTextGenInference):
        name = "huggingface_TGI_server"
    else:
        name = llm.__class__.__name__
        logger.warning("Unknown llm type: {}", name)
    return diskcache.Cache(TRIPLES_CACHE_PATH % name)

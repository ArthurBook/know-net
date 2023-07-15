from typing import List, Tuple, NamedTuple, Optional, Dict
import networkx as nx
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceTextGenInference
from langchain.indexes import GraphIndexCreator
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from know_net.base import GraphBuilder

from loguru import logger


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
    def __init__(self) -> None:
        super().__init__()
        llm = HuggingFaceTextGenInference(
            inference_server_url="http://100.79.46.78:8081",
            verbose=True,
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
        )  # type: ignore
        self.index_creator = GraphIndexCreator(llm=llm)
        self.triples: List[KGTriple] = []
        self.embeddings = OpenAIEmbeddings()  # type: ignore
        self.vectorstore = Chroma.from_documents([], self.embeddings)
        self.match_threshold = 0.95
        self.doc_to_entity: Dict[str, Entity] = {}
        logger.info("Initialized LLMGraphBuilder")

    def add_content(self, content: str) -> None:
        graph = self.index_creator.from_text(content)
        unnormalized_triples: List[Tuple[str, str, str]] = graph.get_triples()
        normalized_triples: List[KGTriple] = [
            self._normalize_triple(s, p, o) for s, o, p in unnormalized_triples
        ]
        self.triples.extend(normalized_triples)

    def _normalize_triple(self, subject: str, predicate: str, object_: str) -> KGTriple:
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

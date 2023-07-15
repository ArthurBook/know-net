from typing import List, Tuple, NamedTuple, Optional, Dict
import networkx as nx
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceTextGenInference
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
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
        )
        self.index_creator = GraphIndexCreator(llm=llm)
        self.triples: List[KGTriple] = []
        self.embeddings = HuggingFaceEmbeddings(model_kwargs={"device": "cuda"})
        self.vectorstore = Chroma.from_documents([], self.embeddings)
        self.match_threshold = 0.95
        self.doc_to_entity: Dict[Document, Entity] = {}
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


T = """"New Breakthrough in Renewable Energy: Solar-Powered Artificial Leaves Set to Revolutionize Clean Energy Production"

Date: July 15, 2023

In a groundbreaking scientific discovery, researchers at a prominent university have developed a revolutionary solar-powered artificial leaf that promises to transform the landscape of renewable energy. This cutting-edge technology aims to provide a sustainable and efficient solution for clean energy production, paving the way towards a greener and more sustainable future.

The research team, led by Dr. Amanda Reynolds, has successfully engineered an artificial leaf capable of mimicking the natural process of photosynthesis. By harnessing sunlight, water, and carbon dioxide, the artificial leaf can generate renewable energy in the form of hydrogen fuel. This breakthrough has the potential to revolutionize the way we produce and utilize clean energy.

Unlike traditional solar panels, which convert sunlight directly into electricity, this new technology directly produces hydrogen gas. The artificial leaf consists of a thin, flexible silicon-based material embedded with specialized catalysts that facilitate the splitting of water molecules into hydrogen and oxygen. The generated hydrogen can then be stored and used as a clean and efficient source of energy.

The key advantage of this solar-powered artificial leaf lies in its efficiency. The device can convert sunlight into hydrogen fuel with an impressive efficiency rate of 15%, surpassing the efficiency of most commercially available solar panels. This advancement brings us significantly closer to achieving a cost-effective and sustainable method of producing clean energy on a large scale.

Dr. Reynolds and her team are also working on integrating the artificial leaf into existing renewable energy infrastructure. They envision a future where solar-powered artificial leaves can be easily deployed in large-scale arrays, transforming sunlight and water into hydrogen fuel, which can be utilized for various applications such as powering vehicles, heating buildings, and generating electricity.

The implications of this breakthrough are far-reaching. With the demand for renewable energy on the rise, solar-powered artificial leaves have the potential to meet a substantial portion of global energy needs while reducing carbon emissions. Additionally, this technology holds promise for remote areas and developing countries, providing them with a decentralized and sustainable source of power.

Industry experts and environmentalists alike are hailing this breakthrough as a game-changer in the field of renewable energy. If successfully implemented on a large scale, solar-powered artificial leaves could play a significant role in mitigating climate change and ushering in a cleaner, greener future for generations to come.

As researchers continue to refine and optimize the technology, the prospects of solar-powered artificial leaves are becoming increasingly promising. With continued investment and support, this innovation could soon be a common sight, bringing us closer to achieving a more sustainable and environmentally friendly world."""

if __name__ == "__main__":
    from langchain.indexes import GraphIndexCreator
    from langchain.chat_models import ChatOpenAI

    # from langchain.document_loaders import TextLoader
    # from langchain.chains import GraphQAChain

    builder = LLMGraphBuilder()
    builder.add_content(T)
    print(builder.graph)
    print(builder.triples)
    print(builder.search("hypoxaemia"))

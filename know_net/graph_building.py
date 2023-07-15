from typing import List, Tuple, NamedTuple
import networkx as nx
from know_net.base import GraphBuilder


class KGTriple(NamedTuple):
    subject: str
    predicate: str
    object_: str


class LLMGraphBuilder(GraphBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.index_creator = GraphIndexCreator(llm=ChatOpenAI(temperature=0))
        self.triples: List[KGTriple] = []

    def add_content(self, content: str) -> None:
        graph = self.index_creator.from_text(content)
        triples = [KGTriple(s, p, o) for s, o, p in graph.get_triples()]
        self.triples.extend(triples)

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


T = """Epidemiology of hypoxaemia in children with acute lower respiratory infection. To determine the prevalence of hypoxaemia in children aged under 5 years suffering acute lower respiratory infections (ALRI), the risk factors for hypoxaemia in children under 5 years of age with ALRI, and the association of hypoxaemia with an increased risk of dying in children of the same age. Systematic review of the published literature. Out-patient clinics, emergency departments and hospitalisation wards in 23 health centres from 10 countries. Cohort studies reporting the frequency of hypoxaemia in children under 5 years of age with ALRI, and the association between hypoxaemia and the risk of dying. Prevalence of hypoxaemia measured in children with ARI and relative risks for the association between the severity of illness and the frequency of hypoxaemia, and between hypoxaemia and the risk of dying. Seventeen published studies were found that included 4,021 children under 5 with acute respiratory infections (ARI) and reported the prevalence of hypoxaemia. Out-patient children and those with a clinical diagnosis of upper ARI had a low risk of hypoxaemia (pooled estimate of 6% to 9%). The prevalence increased to 31% and to 43% in patients in emergency departments and in cases with clinical pneumonia, respectively, and it was even higher among hospitalised children (47%) and in those with radiographically confirmed pneumonia (72%). The cumulated data also suggest that hypoxaemia is more frequent in children living at high altitude. Three papers reported an association between hypoxaemia and death, with relative risks varying between 1.4 and 4.6. Papers describing predictors of hypoxaemia have focused on clinical signs for detecting hypoxaemia rather than on identifying risk factors for developing this complication. Hypoxaemia is a common and potentially lethal complication of ALRI in children under 5, particularly among those with severe disease and those living at high altitude. Given the observed high prevalence of hypoxaemia and its likely association with increased mortality, efforts should be made to improve the detection of hypoxaemia and to provide oxygen earlier to more children with severe ALRI."""

if __name__ == "__main__":
    from langchain.indexes import GraphIndexCreator
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import TextLoader
    from langchain.chains import GraphQAChain

    builder = LLMGraphBuilder()
    builder.add_content(T)
    print(builder.graph)
    print(builder.triples)

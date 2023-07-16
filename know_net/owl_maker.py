from langchain import BasePromptTemplate
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceTextGenInference

_PROMPT = """Given the following knowledge graph triples:
   triples: {triples}.
  Please extend the following Turtle OWL ontology.
  ontology: {ontology}.

 Output the result in JSON:
   {{"turtle": "value"}} with no other text please.
"""
prompt = PromptTemplate(
    input_variables=["triples", "ontology"],
    template=_PROMPT,
)
llm = ChatOpenAI(temperature=0, model="gpt-4")
chain = LLMChain(llm=llm, prompt=prompt)
TRIPLES = """['(Cybertruck, is on its way to, consumers)', '(Cybertruck, rolled off, Giga Texas assembly line)', '(Cybertruck, is a, production intent model)', '(Cybertruck, is on track to meet, timeline)', '(Cybertruck, takes time to get, manufacturing line going)', '(Cybertruck, is a, radical product)', '(Cybertruck, is not made in the way that, other cars are made)', '(Cybertruck, is, first production)', '(Cybertruck, is, electric pickup)', '(Cybertruck, has rolled off, assembly line)', '(assembly line, is, off)', '(Cybertruck, is, off the assembly line)', '(Cybertruck, is, two years behind the original schedule)', '(Cybertruck, looks, nothing like a traditional pickup)', '(Cybertruck, built, Tesla)', '(Tesla, designed, the new vehicle)', '(Tesla, would hold, Cybertruck delivery event in the third quarter of 2023)', '(Tesla, has encountered, repeated bottlenecks involving next-generation 4680 battery)', '(Tesla, said, vehicle would start at $39,900 for single-motor variant)', '(Tesla, says, its first production Cybertruck electric pickup has rolled off the assembly line)', '(Tesla, had said, production would start in late 2021)', '(Tesla, originally said, it would make three versions of the truck)', '(Tesla, is scheduled to report, second-quarter financial results)', '(Cybertruck, introduced by, Elon Musk)', '(Elon Musk, introduced, pickup truck)']"""
ONT = """@prefix : <http://www.semanticweb.org/ontologies/technology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

: a owl:Ontology .

:Technology a owl:Class .

:Hardware a owl:Class ;
    rdfs:subClassOf :Technology .

:Software a owl:Class ;
    rdfs:subClassOf :Technology .

:AI a owl:Class ;
    rdfs:subClassOf :Software .

:MachineLearning a owl:Class ;
    rdfs:subClassOf :AI .

:DeepLearning a owl:Class ;
    rdfs:subClassOf :MachineLearning .

:Robotics a owl:Class ;
    rdfs:subClassOf :Hardware .

:Drone a owl:Class ;
    rdfs:subClassOf :Robotics .

:Computer a owl:Class ;
    rdfs:subClassOf :Hardware .

:Smartphone a owl:Class ;
    rdfs:subClassOf :Hardware .

:OperatingSystem a owl:Class ;
    rdfs:subClassOf :Software .

:Linux a owl:Class ;
    rdfs:subClassOf :OperatingSystem .

:Windows a owl:Class ;
    rdfs:subClassOf :OperatingSystem .

:MacOS a owl:Class ;
    rdfs:subClassOf :OperatingSystem .

:Android a owl:Class ;
    rdfs:subClassOf :OperatingSystem .

:iOS a owl:Class ;
    rdfs:subClassOf :OperatingSystem .

:Person a owl:Class .

:Developer a owl:Class ;
    rdfs:subClassOf :Person .

:Researcher a owl:Class ;
    rdfs:subClassOf :Person .

:CEO a owl:Class ;
    rdfs:subClassOf :Person .

:Company a owl:Class .

:Startup a owl:Class ;
    rdfs:subClassOf :Company .

:Multinational a owl:Class ;
    rdfs:subClassOf :Company .

:Innovation a owl:Class .

:Patent a owl:Class ;
    rdfs:subClassOf :Innovation .

:ResearchPaper a owl:Class ;
    rdfs:subClassOf :Innovation .

:News a owl:Class .

:BlogPost a owl:Class ;
    rdfs:subClassOf :News .

:PressRelease a owl:Class ;
    rdfs:subClassOf :News .

:Conference a owl:Class .

:Webinar a owl:Class ;
    rdfs:subClassOf :Conference .

:Seminar a owl:Class ;
    rdfs:subClassOf :Conference .

:Product a owl:Class .

:SoftwareProduct a owl:Class ;
    rdfs:subClassOf :Product .

:HardwareProduct a owl:Class ;
    rdfs:subClassOf :Product .

:Service a owl:Class .

:CloudService a owl:Class ;
    rdfs:subClassOf :Service .

:ConsultingService a owl:Class ;
    rdfs:subClassOf :Service .

:Investment a owl:Class .

:VentureCapital a owl:Class ;
    rdfs:subClassOf :Investment .

:Acquisition a owl:Class ;
    rdfs:subClassOf :Investment .

:Regulation a owl:Class .

:PrivacyPolicy a owl:Class ;
    rdfs:subClassOf :Regulation .

:DataProtection a owl:Class ;
    rdfs:subClassOf :Regulation .

:worksFor a owl:ObjectProperty ;
    rdfs:domain :Person ;
    rdfs:range :Company .

:develops a owl:ObjectProperty ;
    rdfs:domain :Developer ;
    rdfs:range :Product .

:investsIn a owl:ObjectProperty ;
    rdfs:domain :VentureCapital ;
    rdfs:range :Startup .

:publishes a owl:ObjectProperty ;
    rdfs:domain :Researcher ;
    rdfs:range :ResearchPaper .

:attends a owl:ObjectProperty ;
    rdfs:domain :Person ;
    rdfs:range :Conference .

:owns a owl:ObjectProperty ;
    rdfs:domain :Company ;
    rdfs:range :Product .

:regulates a owl:ObjectProperty ;
    rdfs:domain :Regulation ;
    rdfs:range :Company .
"""
res = chain.predict(triples=TRIPLES, ontology=ONT)
print(res)

from typing import List
import os

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


def main() -> None:
    files = os.listdir("data")
    turtles: List[str] = [ONT]
    for file in files:
        with open(f"data/{file}", "r") as f:
            turtle = f.read()
        turtles.append(turtle)
    with open("ontology.owl", "w") as f:
        f.write("\n".join(turtles))


if __name__ == "__main__":
    main()

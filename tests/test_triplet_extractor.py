import asyncio

from know_net import base

text = """
Tesla CEO and part-time Twitter nuisance Elon Musk showed the world the Cybertruck back in November 2019 which, wow, feels like an eon ago. Ridiculous as it was – and still is – the big, spaceship-looking Cybertruck continues to be a hot commodity, with some sources estimating as many as 1.8 million pre-orders for Tesla’s bonkers brutalist beast.
"""

if __name__ == "__main__":
    from know_net.triplet_extraction import langchain_graphindexcreator

    from langchain.chat_models import openai
    from langchain.indexes import graph

    llm = openai.ChatOpenAI(model="gpt-4")  # type: ignore
    model = graph.GraphIndexCreator(llm=llm)

    extractor = langchain_graphindexcreator.LangChainGraphIndexCreator(model)

    triplets = extractor.extract_triplets(base.Content(text, base.Source("test")))
    for triplet in triplets:
        print(triplet)

from know_net.graph_building import LLMGraphBuilder

test_doc1 = """"New Breakthrough in Renewable Energy: Solar-Powered Artificial Leaves Set to Revolutionize Clean Energy Production"

Date: July 15, 2023

In a groundbreaking scientific discovery, researchers at a prominent university have developed a revolutionary solar-powered artificial leaf that promises to transform the landscape of renewable energy. This cutting-edge technology aims to provide a sustainable and efficient solution for clean energy production, paving the way towards a greener and more sustainable future.

The research team, led by Dr. Amanda Reynolds, has successfully engineered an artificial leaf capable of mimicking the natural process of photosynthesis. By harnessing sunlight, water, and carbon dioxide, the artificial leaf can generate renewable energy in the form of hydrogen fuel. This breakthrough has the potential to revolutionize the way we produce and utilize clean energy.

Unlike traditional solar panels, which convert sunlight directly into electricity, this new technology directly produces hydrogen gas. The artificial leaf consists of a thin, flexible silicon-based material embedded with specialized catalysts that facilitate the splitting of water molecules into hydrogen and oxygen. The generated hydrogen can then be stored and used as a clean and efficient source of energy.
"""
test_doc2 = """nThe new iPhone 12 is expected to be released in September 2020. It will have a 5G network, improved camera, and a new design."""

if __name__ == "__main__":
    builder = LLMGraphBuilder()
    builder.add_content(test_doc1)
    builder.add_content(test_doc2)
    print(builder.graph)
    print(builder.triples)
    print(builder.search("iphone"))

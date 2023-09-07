import asyncio
from typing import AsyncGenerator, Optional
from langchain.indexes import graph
from langchain.llms import huggingface_text_gen_inference, openai
from langchain.schema import language_model

from know_net import base, mixins

DEFAULT_MAX_CONCURRENCY = 20


class LangChainGraphIndexCreator(
    base.AsynchronousTripletExtractor,
    mixins.WithDiskCache,
):
    def __init__(
        self,
        graph_index_creator: graph.GraphIndexCreator,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        super().__init__()
        self.graph_index_creator = graph_index_creator
        self.semaphore = semaphore or asyncio.Semaphore(DEFAULT_MAX_CONCURRENCY)

    async def extract_triplets_async(
        self, content: base.Content
    ) -> AsyncGenerator[base.KGTriple, None]:
        nx_graph = await self.graph_index_creator.afrom_text(content.data)
        for triplet in nx_graph.get_triples():
            subject, object_, predicate = triplet
            yield base.KGTriple(
                source=content.source,
                subject=subject,
                predicate=predicate,
                object_=object_,
            )

    @property
    def cache_name(self) -> str:
        assert (llm := self.graph_index_creator.llm) is not None
        model_name = get_model_name(llm)
        return f"graph_index_creator_{model_name}"


def get_model_name(llm: language_model.BaseLanguageModel) -> str:
    if isinstance(llm, openai.OpenAIChat):
        name = llm.model_name  # gpt 3 or gpt 4 etc
    elif isinstance(llm, huggingface_text_gen_inference.HuggingFaceTextGenInference):
        name = "huggingface_TGI_server"
    else:
        name = llm.__class__.__name__
    return name

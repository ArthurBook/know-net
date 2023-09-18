import asyncio
import threading
import uuid
from typing import Dict, List, Tuple, cast

import loguru
import numpy as np
import torch
import transformers
from transformers import modeling_outputs

from know_net import base, mixins

DEFAULT_BATCH_SIZE = 32

logger = loguru.logger


class BertEmbedder(base.AsyncEmbedder, mixins.WithDiskCache):
    """
    Embeds strings using a BERT model asynchronously by collecting requests
    and passing them through the model in batches.

    Example:
    >>> embedder = BertEmbedder.from_pretrained("bert-base-uncased")
    >>> embedder.start()
    >>> embeddings = asyncio.run(embedder.embed_async("Hello, my dog is cute"))
    >>> print(embeddings.shape)
    """

    def __init__(
        self,
        model: transformers.BertModel,
        tokenizer: transformers.BertTokenizer,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.request_queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop()
        self.results: Dict[uuid.UUID, np.ndarray] = {}
        self._closed = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def embed_async(self, string: str) -> np.ndarray:
        """
        Embed a string asynchronously, returning the embedding as a numpy array
        """
        if self._closed:
            raise RuntimeError("Cannot embed after closing")
        if self.is_in_cache(string):
            return cast(np.ndarray, self.from_cache(string))

        ## Handle the request
        request_id = uuid.uuid4()
        await self.request_queue.put((request_id, string))
        while request_id not in self.results:
            await asyncio.sleep(0.1)  # yield control to other tasks
        result = self.results.pop(request_id)

        self.set_in_cache(string, result)
        return result

    ## Embedding
    async def _embedding_daemon(self) -> None:
        while True:
            batch = await self._collect_batch()
            await self._process_batch(batch)
            await asyncio.sleep(0)  # yield control to other tasks

    async def _collect_batch(self) -> List[Tuple[uuid.UUID, str]]:
        batch = []
        while len(batch) < self.batch_size and not self.request_queue.empty():
            request_id, string = await self.request_queue.get()
            batch.append((request_id, string))
        return batch

    async def _process_batch(self, batch: List[Tuple[uuid.UUID, str]]) -> None:
        if not batch:
            return
        logger.info("passing batch of size {} through encoder", len(batch))
        tokenized = self._tokenize_batch(batch)
        outputs = cast(modeling_outputs.BaseModelOutput, self.model(**tokenized))
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
        for (request_id, _), embedding in zip(batch, embeddings):
            self.results[request_id] = embedding

    def _tokenize_batch(
        self, batch: List[Tuple[uuid.UUID, str]]
    ) -> transformers.BatchEncoding:
        tokenized = self.tokenizer.batch_encode_plus(
            [s for _, s in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return tokenized

    ## Async stuff
    def start(self):
        self._daemon_task = self.loop.create_task(self._embedding_daemon())
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.start()
        self._closed = False

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def close(self):
        self._closed = True
        if self.request_queue.empty():
            self._daemon_task.cancel()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()

    ## Caching
    @property
    def cache_name(self) -> str:
        return "bert_embedder"

    ## Initialization
    @classmethod
    def from_pretrained(cls, name: str) -> "BertEmbedder":
        model = transformers.BertModel.from_pretrained(name)
        assert isinstance(model, transformers.BertModel)
        tokenizer = transformers.BertTokenizer.from_pretrained(name)
        return cls(model, tokenizer)

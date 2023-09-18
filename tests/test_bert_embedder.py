from typing import List
import asyncio
import loguru
import numpy as np
from know_net.embedders import bert_embedder

logger = loguru.logger


def test_bert_embedder():
    embedder = bert_embedder.BertEmbedder.from_pretrained("bert-base-uncased")
    embedder.clear_cache()
    embedder.start()

    strings_to_embed = [
        "Hello, my dog is cute",
        "Hello, my cat is cute",
    ] * bert_embedder.DEFAULT_BATCH_SIZE + ["rax"]

    async def test_embeddings() -> List[np.ndarray]:
        tasks = []
        for string in strings_to_embed:
            task = embedder.embed_async(string)
            tasks.append(task)
        return await asyncio.gather(*tasks)

    embeddings = asyncio.run(test_embeddings())
    embedder.close()
    logger.success("done with embeddings of shape: {}", np.stack(embeddings).shape)


if __name__ == "__main__":
    test_bert_embedder()

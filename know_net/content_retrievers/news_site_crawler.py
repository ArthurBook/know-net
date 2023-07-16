import asyncio
import loguru
from typing import AsyncGenerator, Iterator, List, Optional, Set

import aiohttp
from typing_extensions import Annotated

from know_net import base
from know_net.content_retrievers import content_filters, content_parsers

DEFAULT_CONTENT_FILTER = content_filters.SimpleHeuristicFilter(3, 30)
DEFAULT_CONTENT_PARSER = content_parsers.YahooFinanceParser()
MAX_CONCURRENCY = 20
MAX_RETRIES = 3

logger = loguru.logger.opt(colors=True)


class NewsCrawler(base.ContentRetriever):
    _semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    def __init__(
        self,
        url: Annotated[str, "Link the news frontpage that you want to crawl"],
        link_finder: Optional[content_filters.HyperLinkFinder] = None,
        content_parser: Optional[content_parsers.ContentParser] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.url = url
        self.hyperlink_finder = link_finder or DEFAULT_CONTENT_FILTER
        self.content_parser = content_parser or DEFAULT_CONTENT_PARSER
        self.max_retries = max_retries or MAX_RETRIES
        self.queue: asyncio.Queue[base.Content] = asyncio.Queue()
        logger.info("initialized {}", self.__class__.__name__)

    def __iter__(self) -> Iterator[base.Content]:
        """
        Synchronous iterator over loaded articles
        """
        loaded_articles: List[base.Content] = list()

        async def load_sync():
            nonlocal loaded_articles
            await self.load()
            while not self.queue.empty():
                article = await self.queue.get()
                loaded_articles.append(article)

        asyncio.run(load_sync())
        yield from loaded_articles

    def __aiter__(self) -> "NewsCrawler":
        asyncio.create_task(self.load())
        return self

    async def __anext__(self) -> base.Content:
        while True:
            try:
                content_piece = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.1)
                continue
            return content_piece

    async def load(self) -> None:
        async for content_piece in self._load_frontpage_articles(self.url):
            if content_piece is not None:
                await self.queue.put(content_piece)

    async def _load_frontpage_articles(
        self, url: str
    ) -> AsyncGenerator[Optional[base.Content], None]:
        async with self._semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    text = await response.text()

                    links = self.hyperlink_finder.get_links_from_html(text, url)
                    tasks = [self._load_article(link) for link in links]

                    for result in asyncio.as_completed(tasks):
                        yield await result

    async def _load_article(self, url: str) -> Optional[base.Content]:
        async with self._semaphore:
            for attempt in range(1, self.max_retries + 1):
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(url) as response:
                            response.raise_for_status()
                            text = await response.text()
                            parsed_text = self.content_parser.parse(text)
                            logger.debug(f"Retrieved {url}")
                            return base.Content(text=parsed_text, url=url)
                    except aiohttp.ClientResponseError:
                        logger.debug(f"Request to {url} failed on attempt {attempt}")
                        await asyncio.sleep(2**attempt)  # exponential backoff
            logger.warning(f"All attempts to retrieve the URL failed: {url}")
            return None  # or some appropriate default value

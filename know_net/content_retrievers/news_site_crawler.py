import asyncio
from typing import AsyncGenerator, Optional, Union

import aiohttp
import loguru
from typing_extensions import Annotated

from know_net import base
from know_net.content_retrievers import content_parsers, hyperlink_finders

DEFAULT_CONTENT_FILTER = hyperlink_finders.SimpleHeuristicFilter(3, 30)
DEFAULT_CONTENT_PARSER = content_parsers.ParagraphParser()
DEFAULT_MAX_CONCURRENCY = 20
DEFAULT_MAX_RETRIES = 3

logger = loguru.logger


class NewsCrawler(base.AsynchronousContentRetriever):
    """
    A class that crawls a news website and returns the articles mentioned on the frontpage
    """

    _SENTINEL = object()

    def __init__(
        self,
        url: Annotated[str, "Link the news frontpage that you want to crawl"],
        link_finder: Optional[hyperlink_finders.HyperLinkFinder] = None,
        content_parser: Optional[content_parsers.ContentParser] = None,
        max_retries: Optional[int] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        self.url = url
        self.hyperlink_finder = link_finder or DEFAULT_CONTENT_FILTER
        self.content_parser = content_parser or DEFAULT_CONTENT_PARSER
        self.max_retries = max_retries or DEFAULT_MAX_RETRIES
        self.queue: asyncio.Queue[Union[base.Content, object]] = asyncio.Queue()
        self.semaphore = semaphore or asyncio.Semaphore(DEFAULT_MAX_CONCURRENCY)
        logger.info("initialized {}", self.__class__.__name__)

    async def __aiter__(self) -> AsyncGenerator[base.Content, None]:
        """
        Asynchronous iterator over loaded articles
        """

        async with aiohttp.ClientSession() as session:
            ## Load all articles from the frontpage
            asyncio.create_task(self.load_articles_from_frontpage(self.url, session))

            ## Yield all articles from the queue as they are loaded
            while True:
                article = await self.queue.get()
                if article is self._SENTINEL:
                    break
                if article is None:
                    continue
                if isinstance(article, base.Content):
                    yield article

    async def load_articles_from_frontpage(
        self,
        url: str,
        session: aiohttp.ClientSession,
    ) -> None:
        """
        Loads all articles from the frontpage and puts them in the queue
        """
        frontpage_text = await self.load_frontpage(session)
        links = self.hyperlink_finder.get_links_from_html(frontpage_text, url)
        tasks = [self._load_article(link, session) for link in links]
        await asyncio.gather(*tasks)
        await self.queue.put(self._SENTINEL)

    async def load_frontpage(self, session: aiohttp.ClientSession) -> str:
        """
        Loads the frontpage of the news website
        """
        async with self.semaphore:
            async with session.get(self.url) as response:
                response.raise_for_status()
                text = await response.text()
        return text

    async def _load_article(
        self, url: str, session: aiohttp.ClientSession
    ) -> Optional[base.Content]:
        """
        Loads an article from a URL and puts it in the queue
        """
        for attempt in range(1, self.max_retries + 1):
            async with self.semaphore:
                async with session.get(url) as response:
                    try:
                        response.raise_for_status()
                    except aiohttp.ClientResponseError:
                        logger.debug(f"Request to {url} failed ({attempt=})")
                        await asyncio.sleep(2**attempt)  # exponential backoff
                    else:
                        text = await response.text()
                        parsed_text = self.content_parser.parse(text)
                        result = base.Content(text=parsed_text, source=url)
                        logger.debug(f"Successfully retrieved {url}")
                        await self.queue.put(result)
                        break
        else:
            logger.warning(f"All attempts to retrieve the URL failed: {url}")

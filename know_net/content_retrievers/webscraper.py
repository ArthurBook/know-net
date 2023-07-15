from typing import Iterator, Optional
from know_net import base
import bs4

DEFAULT_DEPTH = 1


class WebScraper(base.ContentRetriever):
    def __init__(self, url: str, depth: Optional[int] = None) -> None:
        self.url = url
        self.depth = depth or DEFAULT_DEPTH
        self.content: str = ""

    def __iter__(self) -> Iterator[str]:
        ...

    async def load(self) -> None:
        ...

if __name__ 
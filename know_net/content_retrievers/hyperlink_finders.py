import abc
from typing import Iterator, Optional
from urllib import parse

import bs4

DEFAULT_MAX_LINKS = None  # if None, then no cap


class HyperLinkFinder(abc.ABC):
    """
    Responsible for finding hyperlinks worth clicking in a html
    """

    def __init__(self, max_links: Optional[int] = None) -> None:
        self.max_links = max_links if max_links is not None else DEFAULT_MAX_LINKS

    def get_links_from_html(self, html: str, base_url: str) -> Iterator[str]:
        soup = bs4.BeautifulSoup(html, "html.parser")
        links = soup.find_all("a", href=True)
        for i, tag in enumerate(filter(self, links)):
            if self.max_links and i > self.max_links:
                break
            link: str = tag["href"]
            if not link.startswith("http"):
                link = parse.urljoin(base_url, link)
            yield link

    @abc.abstractmethod
    def __call__(self, tag: bs4.Tag) -> bool:
        ...


class SimpleHeuristicFilter(HyperLinkFinder):
    """
    TODO write intelligent class that uses a language model to filter out
    """

    def __init__(
        self,
        min_words: int,
        max_words: int,
        max_links: Optional[int] = None,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        super().__init__(max_links)

    def __call__(self, tag: bs4.Tag) -> bool:
        words = tag.text.split()
        if len(words) < self.min_words:
            return False
        if len(words) > 20:
            return False
        if all(word[0].isupper() for word in words):
            return False
        return True

import abc
from typing_extensions import Annotated

import bs4


class ContentParser(abc.ABC):
    """
    Responsible for parsing a website html into a string
    """

    @abc.abstractmethod
    def parse(self, html: Annotated[str, "html string"]) -> str:
        ...


class YahooFinanceParser(ContentParser):
    def parse(self, html: str) -> str:
        soup = bs4.BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        article_content = "\n".join([p.get_text() for p in paragraphs])
        return article_content

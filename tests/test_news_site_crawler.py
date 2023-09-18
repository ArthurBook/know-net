import asyncio

from know_net import base


def crawl(scraper: base.ContentRetriever) -> None:
    for content_piece in scraper:
        print_contentpiece(content_piece)


def crawl_asynchronously(scraper: base.AsynchronousContentRetriever) -> None:
    async def crawl_async():
        async for content_piece in scraper:
            print_contentpiece(content_piece)

    asyncio.run(crawl_async())


def print_contentpiece(content_piece: base.Content) -> None:
    print(content_piece)
    print("-----")


if __name__ == "__main__":
    from know_net.content_retrievers import news_site_crawler

    test_url = "https://news.yahoo.com/"

    scraper = news_site_crawler.NewsCrawler(test_url)
    crawl_asynchronously(scraper)

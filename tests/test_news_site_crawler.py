from know_net.content_retrievers import news_site_crawler


test_url = "https://news.yahoo.com/"

scraper = news_site_crawler.NewsCrawler(test_url)

for content_piece in scraper:
    print(content_piece)
    print("-----")

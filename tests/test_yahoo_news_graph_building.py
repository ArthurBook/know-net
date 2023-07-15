if __name__ == "__main__":
    from know_net.content_retrievers import news_site_crawler
    from know_net import graph_building

    test_url = "https://news.yahoo.com/"

    builder = graph_building.LLMGraphBuilder()
    scraper = news_site_crawler.NewsCrawler(test_url)

    builder.add_content_batch(scraper)

    print(builder.graph)
    print(builder.triples)
    print(builder.search("hypoxaemia"))

    import pickle
    with open("builder.pkl", 'wb') as f:
        pickle.dump(builder, f)

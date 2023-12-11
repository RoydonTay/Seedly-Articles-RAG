from pathlib import Path
import os
import scrapy
from bs4 import BeautifulSoup
import random


class ArticleSpider(scrapy.Spider):
    name = "articles"

    def start_requests(self):
        urls = [
            "https://blog.seedly.sg/bto-balloting-system-does-it-need-to-be-revamped/"
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f"seedly-{page}.txt"

        # Extract content from the specified <div class="sc-4f49d391-6 cPhffo">
        extracted_content = response.css('div.sc-4f49d391-6.cPhffo ::text').getall()
        extracted_content = ' '.join(extracted_content)
        extracted_content += '\n' + str(response.url)

        with open(os.path.join("/home/cowboygarage/seedly_scrape/Scraped", filename), 'w', encoding='utf-8') as file:
            file.write(extracted_content)

        rand_int = random.randint(1, 4)
    
        next_page = response.xpath(f'/html/body/div/div/div[2]/div[2]/div[2]/div[1]/div[1]/article/section/div[2]/ul[last()]/li[{rand_int}]').get()
        soup = BeautifulSoup(next_page, 'html.parser')
        a_tag = soup.find('a')
        href = a_tag['href']
        if href is not None:
            yield response.follow(href, callback=self.parse, dont_filter=True)
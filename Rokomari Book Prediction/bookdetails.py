import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager

def fetch_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return None

def scrape_page(url):
    html_content = fetch_html(url)
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        books = soup.find_all('div', class_='book-list-wrapper')
        data = []
        for book in books:
            title = book.find('h4', class_='book-title').text.strip() if book.find('h4', class_='book-title') else 'N/A'
            author = book.find('p', class_='book-author').text.strip() if book.find('p', class_='book-author') else 'N/A'
            rating_section = book.find('div', class_='rating-section text-center')
            text_secondary = rating_section.find('span', class_='text-secondary').text.strip() if rating_section and rating_section.find('span', 'text-secondary') else 'N/A'
            price = book.find('p', class_='book-price').text.strip() if book.find('p', class_='book-price') else 'N/A'
            book_detail_link = book.find('a')['href'] if book.find('a') else 'N/A'
            full_detail_link = f"https://www.rokomari.com{book_detail_link}" if book_detail_link != 'N/A' else 'N/A'
            data.append({
                'Title': title,
                'Author': author,
                'Text Secondary': text_secondary,
                'Price': price,
                'Detail Link': full_detail_link
            })
        return data
    else:
        print(f"Failed to fetch HTML content from {url}.")
        return None

def scrape_all_pages(base_url, num_pages):
    all_data = []
    for page in range(1, num_pages + 1):
        url = f"{base_url}&page={page}"
        print(f"Scraping page {page}...")
        page_data = scrape_page(url)
        if page_data:
            all_data.extend(page_data)
        time.sleep(1)
    return all_data

def scrape_book_details(book_url, driver):
    if book_url == 'N/A':
        return 'N/A', 'N/A'
    driver.get(book_url)
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    category_tag = soup.find('div', class_='details-book-info__content-category')
    category = category_tag.find('a').text.strip() if category_tag and category_tag.find('a') else 'N/A'
    review_tag = soup.find('div', class_='details-book-info__content-rating')
    reviews = review_tag.find('a').text.strip().split(' ')[0] if review_tag and review_tag.find('a') else 'N/A'
    return category, reviews

# Set up the base URL and number of pages to scrape
base_url = 'https://www.rokomari.com/book/publisher/2/anannya?ref=mm_p28'
num_pages = 1

# Scrape all book listings from the publisher's pages
scraped_data = scrape_all_pages(base_url, num_pages)

# Set up the WebDriver for scraping detailed book information
driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()))

# Scrape detailed information for each book
for book in scraped_data:
    print(f"Scraping details for book: {book['Title']}")
    category, reviews = scrape_book_details(book['Detail Link'], driver)
    book['Category'] = category
    book['Number of Reviews'] = reviews

# Close the WebDriver
driver.quit()

# Save the scraped data to a CSV file if data scraping was successful
if scraped_data:
    df = pd.DataFrame(scraped_data)
    df.to_csv('anannya_books_data.csv', index=False)
    print(f"Scraped {len(df)} books and saved to books_data.csv")
else:
    print("Failed to scrape data from the webpage.")

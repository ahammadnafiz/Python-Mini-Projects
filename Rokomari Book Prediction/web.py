import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return None

def get_book_details(url):
    if url == 'N/A':
        return 'N/A', 'N/A'
    html_content = fetch_html(url)
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the category
        category_elem = soup.select_one('.details-book-info__content.category a.ml-2')
        category = category_elem.text.strip() if category_elem else 'N/A'
        
        # Find the number of reviews
        reviews_elem = soup.select_one('.details-book-info__content.rating span.ml-2 a')
        reviews = reviews_elem.text.strip() if reviews_elem else 'N/A'
        
        print(f"Extracted - Category: {category}, Reviews: {reviews}")  # Debugging line
        return category, reviews
    return 'N/A', 'N/A'

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
            text_secondary = rating_section.find('span', class_='text-secondary').text.strip() if rating_section and rating_section.find('span', class_='text-secondary') else 'N/A'
            
            price = book.find('p', class_='book-price').text.strip() if book.find('p', class_='book-price') else 'N/A'
            
            # Extract book URL
            book_url_elem = book.find('a', class_='book-title-link')
            book_url = 'https://www.rokomari.com' + book_url_elem['href'] if book_url_elem and 'href' in book_url_elem.attrs else 'N/A'
            
            print(f"Scraping details for book: {title}")  # Debugging line
            
            # Get category and reviews from individual book page
            category, reviews = get_book_details(book_url)
            
            data.append({
                'Title': title,
                'Author': author,
                'Text Secondary': text_secondary,
                'Price': price,
                'URL': book_url,
                'Category': category,
                'Reviews': reviews
            })
            
            # Add a short delay to be polite to the server
            time.sleep(2)
        
        return data
    else:
        print(f"Failed to fetch HTML content from {url}.")
        return None

def scrape_all_pages(base_url, num_pages):
    all_data = []
    
    for page in range(1, num_pages + 1):
        url = f"{base_url}?page={page}"
        print(f"Scraping page {page}...")
        
        page_data = scrape_page(url)
        
        if page_data:
            all_data.extend(page_data)
        
        # Add a short delay to be polite to the server
        time.sleep(3)
    
    return all_data

# Base URL of the webpage you want to scrape (without page number)
base_url = 'https://www.rokomari.com/book/publisher/1369/anyaprokash'
# Number of pages to scrape
num_pages = 5  # Reduced to 5 pages for testing

# Scrape all pages
scraped_data = scrape_all_pages(base_url, num_pages)

# If data is scraped successfully, convert to DataFrame and save to CSV
if scraped_data:
    df = pd.DataFrame(scraped_data)
    df.to_csv('books_data_extended.csv', index=False, encoding='utf-8-sig')
    print(f"Scraped {len(df)} books and saved to books_data_extended.csv")
else:
    print("Failed to scrape data from the webpage.")
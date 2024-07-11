import requests
from bs4 import BeautifulSoup
import pandas as pd
import time  # for adding delays between requests

def fetch_html(url):
    # Send a GET request to the webpage
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.content  # Return the HTML content
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return None

def scrape_page(url):
    # Fetch HTML content
    html_content = fetch_html(url)
    
    if html_content:
        # Parse the webpage content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all book containers on the page
        books = soup.find_all('div', class_='book-list-wrapper')
        
        # List to hold the extracted data
        data = []
        
        # Iterate through each book container
        for book in books:
            # Find title and author
            title = book.find('h4', class_='book-title').text.strip() if book.find('h4', class_='book-title') else 'N/A'
            author = book.find('p', class_='book-author').text.strip() if book.find('p', class_='book-author') else 'N/A'
            
            # Find text secondary (assuming it's in rating section)
            rating_section = book.find('div', class_='rating-section text-center')
            text_secondary = rating_section.find('span', class_='text-secondary').text.strip() if rating_section and rating_section.find('span', class_='text-secondary') else 'N/A'
            
            # Find price
            price = book.find('p', class_='book-price').text.strip() if book.find('p', class_='book-price') else 'N/A'
            
            # Append the data to the list
            data.append({
                'Title': title,
                'Author': author,
                'Text Secondary': text_secondary,
                'Price': price
            })
        
        return data
    else:
        print(f"Failed to fetch HTML content from {url}.")
        return None

def scrape_all_pages(base_url, num_pages):
    # List to hold all scraped data
    all_data = []
    
    # Iterate through each page
    for page in range(1, num_pages + 1):
        url = f"{base_url}?page={page}"
        print(f"Scraping page {page}...")
        
        # Scrape the current page
        page_data = scrape_page(url)
        
        if page_data:
            all_data.extend(page_data)
        
        # Add a short delay to be polite to the server
        time.sleep(1)
    
    return all_data

# Base URL of the webpage you want to scrape (without page number)
base_url = 'https://www.rokomari.com/book/publisher/1/anyaprokash'
# Number of pages to scrape
num_pages = 30  # Adjust this based on how many pages you want to scrape

# Scrape all pages
scraped_data = scrape_all_pages(base_url, num_pages)

# If data is scraped successfully, convert to DataFrame and save to CSV
if scraped_data:
    df = pd.DataFrame(scraped_data)
    df.to_csv('books_data.csv', index=False)
    print(f"Scraped {len(df)} books and saved to books_data.csv")
else:
    print("Failed to scrape data from the webpage.")
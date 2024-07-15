import pandas as pd
import re

def clean_price(price_str):
    if pd.isna(price_str):
        return pd.NA
    # Remove 'TK.' and any commas, then convert to float
    return float(price_str.replace('TK.', '').replace(',', '').strip())

def extract_ratings(ratings_str):
    # Extract the number from parentheses
    match = re.search(r'\((\d+)\)', str(ratings_str))
    return int(match.group(1)) if match else pd.NA

def process_prices(price_str):
    if pd.isna(price_str):
        return pd.NA, pd.NA
    # Split the string by 'TK.'
    prices = price_str.split('TK.')
    # Clean and return the prices
    original = clean_price(prices[1]) if len(prices) > 1 else pd.NA
    discounted = clean_price(prices[2]) if len(prices) > 2 else pd.NA
    return original, discounted

def process_data(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Rename columns
    df.columns = ['Title', 'Author', 'Text Secondary', 'Price', 'Detail Link', 'Category', 'Number of Reviews']

    # Extract total ratings
    df['Total Ratings'] = df['Text Secondary'].apply(extract_ratings)

    # Process prices
    df[['Original Price', 'Discounted Price']] = df['Price'].apply(process_prices).tolist()

    # Select and reorder columns
    result_df = df[['Title', 'Author', 'Total Ratings', 'Original Price', 'Discounted Price', 'Category', 'Number of Reviews']]

    # Write to CSV
    result_df.to_csv(output_file, index=False, encoding='utf-8')

# Usage
input_file = 'anannya_books_data.csv'
output_file = 'anannya_books_data_clean.csv'
process_data(input_file, output_file)
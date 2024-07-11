import pandas as pd

# Load the CSV file
df = pd.read_csv('books_data.csv')

# Function to parse price and extract original and discounted prices
def parse_price(price_str):
    # Split the price string into parts based on 'TK.'
    parts = price_str.split('TK.')
    
    # Extract original price and discounted price
    if len(parts) == 3:  # If both original and discounted prices are present
        original_price = parts[1].strip()
        discounted_price = parts[2].strip()
    else:
        original_price = 'N/A'
        discounted_price = parts[1].strip() if len(parts) == 2 else 'N/A'
    
    return original_price, discounted_price

# Apply the parse_price function to Price column
df['Original Price'], df['Discounted Price'] = zip(*df['Price'].apply(parse_price))

# Split Text Secondary to extract the number in parenthesis
df['Text Secondary'] = df['Text Secondary'].str.extract(r'\((\d+)\)', expand=False)

# Reorder columns as per the requested structure
df = df[['Title', 'Author', 'Text Secondary', 'Original Price', 'Discounted Price']]

# Save the updated DataFrame back to CSV
df.to_csv('books_data_updated.csv', index=False)

print("Updated CSV file 'books_data_updated.csv' created successfully.")

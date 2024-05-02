import requests
from tkinter import *
from PIL import ImageTk, Image
import webbrowser
import io

class NewsApp:
    def __init__(self):
        # Fetch data from the API
        api_key = 'Your API Key'
        country_code = 'bd'
        api_url = f'https://newsdata.io/api/1/news?country={country_code}&apikey={api_key}'
        self.data = self.fetch_data(api_url)

        # Initialize GUI
        self.root = Tk()
        self.root.geometry('600x800')
        self.root.title('News App')
        self.root.configure(bg='#070707')  # Set background color

        # Load the first news item if data is successfully fetched
        if self.data:
            self.load_news_item(0)

        self.root.mainloop()

    def fetch_data(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad response status
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def load_news_item(self, index):
        # Clear previous content
        for widget in self.root.winfo_children():
            widget.destroy()

        # Check if 'results' key exists and is a list
        if 'results' in self.data and isinstance(self.data['results'], list):
            results = self.data['results']
            if index < len(results):
                news_item = results[index]
                title = news_item.get('title', 'Title not available')
                description = news_item.get('description', 'Description not available')
                image_url = news_item.get('image_url', '')
                link = news_item.get('link', '')

                # Display news item title
                title_label = Label(self.root, text=title, font=('Hind Siliguri', 24, 'bold'), wraplength=580, padx=10, pady=10, fg='white', bg='#070707')
                title_label.pack()

                # Display news item image
                if image_url:
                    try:
                        response = requests.get(image_url)
                        response.raise_for_status()  # Raise HTTPError for bad response status
                        image_data = response.content
                        image = Image.open(io.BytesIO(image_data))
                        image = image.resize((560, 360), Image.LANCZOS)  # Resize image with antialiasing
                        photo = ImageTk.PhotoImage(image)
                        image_label = Label(self.root, image=photo, bg='#070707')
                        image_label.image = photo  # Keep a reference to prevent garbage collection
                        image_label.pack()

                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching image: {e}")

                # Display news item description
                description_label = Label(self.root, text=description, font=('Hind Siliguri', 14), wraplength=580, padx=10, pady=10, fg='white', bg='#070707')
                description_label.pack()

                # Open link in web browser when 'Read more' button is clicked
                def open_link():
                    webbrowser.open_new(link)

                read_more_button = Button(self.root, text='Read more', font=('Helvetica', 16), command=open_link, fg='white', bg='#070707', padx=20, pady=10, width=20)
                read_more_button.pack(pady=20)

                # Navigation buttons (Prev and Next)
                frame = Frame(self.root, bg='#070707')
                frame.pack(pady=10)

                if index > 0:
                    prev_button = Button(frame, text='Previous', font=('Helvetica', 16), command=lambda: self.load_news_item(index - 1), fg='white', bg='#070707', padx=20, pady=10, width=10)
                    prev_button.pack(side=LEFT, padx=10)

                if index < len(results) - 1:
                    next_button = Button(frame, text='Next', font=('Helvetica', 16), command=lambda: self.load_news_item(index + 1), fg='white', bg='#070707', padx=20, pady=10, width=10)
                    next_button.pack(side=RIGHT, padx=10)

        else:
            # Handle case where 'results' key or its value is missing or invalid
            error_label = Label(self.root, text='Error: Invalid data format', font=('Helvetica', 16, 'bold'), padx=10, pady=10, fg='white', bg='#070707')
            error_label.pack()

if __name__ == '__main__':
    app = NewsApp()


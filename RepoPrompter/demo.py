from repoprompter.repoprompter.main import main
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Replace with your repository URL
repo_url = "ahammadnafiz/FizTorch"  # Correct repository URL format
access_token = os.getenv('GITHUB_ACCESS_TOKEN')
output_file = "output_prompt.txt"

try:
    # Generate the prompt and save it to a file
    main(repo_url, access_token, output_file)
except Exception as e:
    print(f"Error: {str(e)}")
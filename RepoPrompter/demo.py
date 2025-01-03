from repoprompter.repoprompter.main import main
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

def get_user_input():
    # Prompt the user to enter the repository URL
    repo_url = input("Enter the GitHub repository URL (e.g., ahammadnafiz/FizTorch): ")

    # Get the GitHub access token from environment variables
    access_token = os.getenv('GITHUB_ACCESS_TOKEN')
    if not access_token:
        # If the access token is not found in the .env file, prompt the user to enter it
        access_token = input("Enter your GitHub access token: ")

    # Prompt the user to enter the output file name
    output_file = input("Enter the output file name (e.g., output_prompt.txt): ")

    return repo_url, access_token, output_file

def main_interactive():
    # Get user inputs
    repo_url, access_token, output_file = get_user_input()

    try:
        # Generate the prompt using the main function and save it to the output file
        main(repo_url, access_token, output_file)
        print(f"Prompt successfully generated and saved to {output_file}")
    except Exception as e:
        # Print an error message if something goes wrong
        print(f"Error: {str(e)}")
        print("Please check the repository URL, access token, and try again.")

# Run the interactive main function
if __name__ == "__main__":
    main_interactive()
from repoprompter.repoprompter.main import main

# Replace with your repository URL and access_token
repo_url = "ahammadnafiz/FizTorch"  # Correct repository URL format
access_token = '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'
output_file = "output_prompt.txt"

try:
    # Generate the prompt and save it to a file
    main(repo_url, access_token, output_file)
except Exception as e:
    print(f"Error: {str(e)}")
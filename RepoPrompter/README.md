# RepoPrompter

RepoPrompter is a Python package designed to convert the contents of a GitHub repository into a structured text format suitable for prompting large language models (LLMs). This tool is particularly useful for developers and researchers who need to analyze or understand the contents of a repository programmatically.

## Features

- **Fetch Repository Content**: Easily fetch the contents of any public or private GitHub repository.
- **Convert to Text**: Convert the repository contents into a structured text format.
- **Generate Prompts**: Create prompts that can be understood by LLMs.
- **Handle Large Repositories**: Efficiently process large repositories with nested directories and files.
- **Error Handling**: Robust error handling for non-UTF-8 content and other encoding issues.
- **Environment Variables**: Securely manage GitHub access tokens using environment variables.

## Installation

To install RepoPrompter, you can use pip:

```bash
pip install repoprompter
```

## Usage

### Basic Usage

Here is a basic example of how to use RepoPrompter to convert a GitHub repository into a prompted text format:

```python
from repoprompter.repoprompter.main import main
import os

# Replace with your repository URL
repo_url = "stanford-oval/storm"  # Correct repository URL format

# Set your GitHub access token as an environment variable
os.environ['GITHUB_ACCESS_TOKEN'] = "your_access_token"

output_file = "output_prompt.txt"

try:
    # Generate the prompt and save it to a file
    prompt = main(repo_url, os.getenv('GITHUB_ACCESS_TOKEN'), output_file)
    print(prompt)
except Exception as e:
    print(f"Error: {str(e)}")
```

### Detailed Steps

1. **Set Up Environment Variable**:
   Ensure your GitHub access token is set as an environment variable to avoid hardcoding sensitive information.

   ```bash
   export GITHUB_ACCESS_TOKEN="your_access_token"
   ```

2. **Run the Script**:
   Execute your Python script to fetch the repository content, convert it to text, and generate the prompt.

   ```bash
   python demo.py
   ```

### Handling Non-UTF-8 Content

RepoPrompter includes robust error handling to manage files that cannot be decoded as UTF-8. These files are marked as binary or non-UTF-8 content in the output.

### Saving the Prompt to a File

The generated prompt can be saved to a text file for further analysis or processing.

```python
output_file = "output_prompt.txt"
prompt = main(repo_url, os.getenv('GITHUB_ACCESS_TOKEN'), output_file)
```

## Example

Here is a complete example script (`demo.py`) that demonstrates how to use RepoPrompter:

```python
# demo.py
import os
from repoprompter.repoprompter.main import main

# Replace with your repository URL
repo_url = "stanford-oval/storm"  # Correct repository URL format

# Set your GitHub access token as an environment variable
os.environ['GITHUB_ACCESS_TOKEN'] = "your_access_token"

output_file = "output_prompt.txt"

try:
    # Generate the prompt and save it to a file
    prompt = main(repo_url, os.getenv('GITHUB_ACCESS_TOKEN'), output_file)
    print(prompt)
except Exception as e:
    print(f"Error: {str(e)}")
```

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

### Forking the Repo

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgements

- Thanks to the developers of PyGithub for providing a great library to interact with the GitHub API.
- Inspiration and code snippets from various open-source projects.
# repoprompter/repoprompter/main.py
from .fetcher import fetch_repo_content
from .converter import convert_to_text
from .structurer import structure_text_for_llm
from .prompter import create_prompt

def main(repo_url, access_token, output_file=None):
    try:
        repo, contents = fetch_repo_content(repo_url, access_token)
        text_content = convert_to_text(contents, repo)
        structured_text = structure_text_for_llm(text_content)
        prompt = create_prompt(structured_text)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(prompt)

        return prompt
    except Exception as e:
        return f"Error: {str(e)}"
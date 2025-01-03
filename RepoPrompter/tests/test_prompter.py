# tests/test_prompter.py
import pytest
from repoprompter.repoprompter.prompter import create_prompt

def test_create_prompt():
    structured_text = "File: example.py\nprint('Hello, World!')\n"
    prompt = create_prompt(structured_text)
    assert isinstance(prompt, str)

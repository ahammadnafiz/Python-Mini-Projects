# tests/test_main.py
import pytest
from repoprompter.repoprompter.main import main

def test_main():
    repo_url = "owner/repo"
    access_token = "your_access_token"
    prompt = main(repo_url, access_token)
    assert isinstance(prompt, str)

# tests/test_fetcher.py
import pytest
from repoprompter.repoprompter.fetcher import fetch_repo_content

def test_fetch_repo_content():
    repo_url = "owner/repo"
    access_token = "your_access_token"
    contents = fetch_repo_content(repo_url, access_token)
    assert contents is not None

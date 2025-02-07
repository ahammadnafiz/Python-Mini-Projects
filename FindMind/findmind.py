import os
from dotenv import load_dotenv
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq  # For LLM tag/category enrichment
from sentence_transformers import SentenceTransformer, util

######################################
# 1. Initialize Clients and Models   #
######################################

# Groq client for LLM-powered enrichment
# Load the Groq API key from the environment.
load_dotenv(".env")
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key=GROQ_API_KEY)

# Embedding model (open-source) for semantic search
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

######################################
# 2. Bookmark Parsing Utility        #
######################################

def parse_bookmarks(file_obj):
    """
    Parse the uploaded bookmarks HTML file and return a list of bookmarks.
    Each bookmark is a dict with a title, URL, and placeholders for tags,
    category, priority, and embedding.
    """
    content = file_obj.read().decode("utf-8")
    soup = BeautifulSoup(content, "html.parser")
    bookmarks = []

    # Standard bookmark exports usually use <a> tags.
    for a in soup.find_all("a"):
        title = a.get_text(strip=True)
        url = a.get("href")
        bookmarks.append({
            "title": title,
            "url": url,
            "tags": [],
            "category": None,
            "priority": None,
            "embedding": None,  # Will be computed later
        })
    return bookmarks

######################################
# 3. Enrich Bookmark Metadata via LLM #
######################################

def enrich_bookmark(bookmark):
    """
    Uses the Groq API (powered by an open-source LLM) to enrich a bookmark.
    The prompt instructs the model to generate:
      - Tags (a comma-separated list)
      - A Category (e.g., 'tech', 'news', etc.)
      - A Priority level (High, Medium, or Low)
    
    Expected response format:
      Tags: tag1, tag2, tag3
      Category: category_name
      Priority: High/Medium/Low
    """
    prompt = (
        f"You are an intelligent assistant that organizes bookmarks. "
        f"For the bookmark with title '{bookmark['title']}' and URL {bookmark['url']}, "
        "generate a comma-separated list of relevant tags, assign a category (e.g., 'tech', 'news', 'tutorial', etc.), "
        "and decide its priority level (High, Medium, or Low) based on its importance. "
        "Format the response exactly as follows (with no extra text):\n\n"
        "Tags: tag1, tag2, tag3\n"
        "Category: category_name\n"
        "Priority: High/Medium/Low"
    )
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",
        )
        response_text = chat_completion.choices[0].message.content.strip()
        # Expecting three lines: one for tags, one for category, one for priority.
        lines = response_text.splitlines()
        if len(lines) < 3:
            raise ValueError("Unexpected response format.")

        tags_line = lines[0]
        category_line = lines[1]
        priority_line = lines[2]

        tags = [tag.strip() for tag in tags_line.replace("Tags:", "").split(",") if tag.strip()]
        category = category_line.replace("Category:", "").strip()
        priority = priority_line.replace("Priority:", "").strip()

        bookmark["tags"] = tags
        bookmark["category"] = category
        bookmark["priority"] = priority

    except Exception as e:
        st.error(f"Error enriching bookmark '{bookmark['title']}': {e}")
    return bookmark

def enrich_all_bookmarks(bookmarks):
    """
    Enrich each bookmark with tags, category, and priority using the LLM.
    """
    for i, bm in enumerate(bookmarks):
        with st.spinner(f"Enriching bookmark {i+1}/{len(bookmarks)}: {bm['title']}"):
            enrich_bookmark(bm)
    return bookmarks

######################################
# 4. Compute Embeddings for Bookmarks #
######################################

def embed_all_bookmarks(bookmarks):
    """
    Compute and store an embedding for each bookmark. The text used
    for embedding is a combination of the bookmark title, tags, and category.
    """
    for bm in bookmarks:
        # Use title as the primary text; include tags/category if available.
        text = bm["title"]
        if bm.get("tags"):
            text += " " + " ".join(bm["tags"])
        if bm.get("category"):
            text += " " + bm["category"]
        bm["embedding"] = embedding_model.encode(text, convert_to_tensor=True)
    return bookmarks

######################################
# 5. Grouping and Search Utilities     #
######################################

def group_bookmarks_by_category(bookmarks):
    """
    Group bookmarks by their 'category' field.
    """
    groups = {}
    for bm in bookmarks:
        cat = bm.get("category") or "Uncategorized"
        groups.setdefault(cat, []).append(bm)
    return groups

def textual_search_bookmarks(bookmarks, query):
    """
    Perform a basic textual search over bookmarks based on title, URL, tags,
    category, or priority.
    """
    results = []
    query_lower = query.lower()
    for bm in bookmarks:
        if (query_lower in bm["title"].lower() or
            query_lower in bm["url"].lower() or
            any(query_lower in tag.lower() for tag in bm["tags"]) or
            (bm.get("category") and query_lower in bm["category"].lower()) or
            (bm.get("priority") and query_lower in bm["priority"].lower())):
            results.append(bm)
    return results

def semantic_search_bookmarks(bookmarks, query, threshold=0.2):
    """
    Perform semantic search using the pre-computed embeddings.
    Returns bookmarks sorted by cosine similarity to the query.
    """
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    results = []
    for bm in bookmarks:
        if bm.get("embedding") is not None:
            sim = util.cos_sim(query_embedding, bm["embedding"]).item()
        else:
            sim = 0.0
        results.append((bm, sim))
    # Sort results in descending order of similarity.
    results = sorted(results, key=lambda x: x[1], reverse=True)
    # Optionally, filter out low-similarity items.
    return [bm for bm, score in results if score >= threshold]

######################################
# 6. Streamlit App Interface           #
######################################

def main():
    st.title("Tagmind")
    st.markdown(
        "Welcome to Tagmind—your second brain for bookmarks! "
        "Upload your bookmarks HTML file to automatically enrich, embed, and organize your links. "
        "Never lose track of an important link again."
    )

    # File uploader for bookmarks.
    uploaded_file = st.file_uploader("Choose your bookmarks HTML file", type=["html", "htm"])
    bookmarks = None

    if uploaded_file is not None:
        with st.spinner("Parsing bookmarks..."):
            bookmarks = parse_bookmarks(uploaded_file)
        st.success(f"Parsed {len(bookmarks)} bookmarks!")

        # Option: Enrich bookmarks with LLM-powered metadata.
        if st.button("Enrich Bookmarks"):
            bookmarks = enrich_all_bookmarks(bookmarks)
            st.success("Bookmarks enriched successfully!")
            st.write("Sample enriched bookmark:")
            st.json(bookmarks[0] if bookmarks else {})

        # Option: Compute embeddings for semantic search.
        if bookmarks and st.button("Compute Embeddings"):
            with st.spinner("Computing embeddings for all bookmarks..."):
                bookmarks = embed_all_bookmarks(bookmarks)
            st.success("Embeddings computed!")

        # Dashboard: Group bookmarks by category.
        if bookmarks and st.checkbox("Show Dashboard"):
            groups = group_bookmarks_by_category(bookmarks)
            st.subheader("Bookmarks by Category")
            for category, bm_list in groups.items():
                st.markdown(f"**{category}** ({len(bm_list)} bookmarks)")
                for bm in bm_list:
                    st.markdown(
                        f"- **{bm['title']}**  \n"
                        f"  URL: [Link]({bm['url']})  \n"
                        f"  Tags: {', '.join(bm['tags']) if bm['tags'] else 'None'}  \n"
                        f"  Priority: {bm.get('priority', 'Unknown')}"
                    )
            st.markdown("---")

        # Search interface: Choose search method and enter a query.
        if bookmarks:
            search_method = st.radio("Search Method", options=["Textual Search", "Semantic Search"])
            query = st.text_input("Search bookmarks (by title, URL, tag, category, or priority):")
            if query:
                if search_method == "Textual Search":
                    results = textual_search_bookmarks(bookmarks, query)
                else:
                    # Ensure embeddings have been computed.
                    results = semantic_search_bookmarks(bookmarks, query)
                st.markdown(f"### Found {len(results)} matching bookmarks:")
                for bm in results:
                    st.markdown(
                        f"**{bm['title']}**  \n"
                        f"[Visit Link]({bm['url']})  \n"
                        f"Tags: {', '.join(bm['tags']) if bm['tags'] else 'None'}  \n"
                        f"Category: {bm.get('category', 'Uncategorized')}  \n"
                        f"Priority: {bm.get('priority', 'Unknown')}\n---"
                    )
            else:
                st.info("Enter a search query above to filter your bookmarks.")

if __name__ == "__main__":
    main()

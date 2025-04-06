import streamlit as st
import requests
import os
import logging

# Import the MovieRecommender class
# Make sure the movie_recommender.py file is in the same directory
from movie_recommender import MovieRecommender

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TMDB API for movie posters
TMDB_API_KEY = "d5a5901be0f6afbf7d05419e1cb2f804"  # Replace with your actual API key
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Function to get movie poster URL using TMDB API
def get_movie_poster(movie_title, release_year=None):
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        if release_year:
            search_url += f"&year={release_year}"
        
        response = requests.get(search_url)
        data = response.json()
        
        if 'results' in data and data['results']:
            poster_path = data['results'][0]['poster_path']
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            
        # Return a placeholder if no poster found
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"
    except Exception as e:
        logger.error(f"Error fetching poster for {movie_title}: {e}")
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"

# Function to create a movie card with poster
def create_movie_card(movie_info, show_scores=True):
    # Extract movie details
    title = movie_info.get("title", "Unknown Title")
    year = movie_info.get("year", "Unknown Year")
    genre = movie_info.get("genre", "Unknown Genre")
    director = movie_info.get("director", "Unknown Director")
    stars = movie_info.get("stars", "Unknown Cast")
    rating = movie_info.get("imdb_rating", 0.0)
    
    # Get poster URL
    poster_url = get_movie_poster(title, year)
    
    # Create card HTML with dark modern theme
    card_html = f"""
    <div style="display: flex; margin-bottom: 20px; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); background-color: #1f2937; color: #e5e7eb;">
        <div style="flex: 0 0 150px; margin-right: 15px;">
            <img src="{poster_url}" style="width: 100%; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);" alt="{title} poster">
        </div>
        <div style="flex: 1;">
            <h3 style="margin-top: 0; color: #f3f4f6;">{title} ({year})</h3>
            <p><strong style="color: #9ca3af;">Genre:</strong> {genre}</p>
            <p><strong style="color: #9ca3af;">Director:</strong> {director}</p>
            <p><strong style="color: #9ca3af;">Stars:</strong> {stars}</p>
            <p><strong style="color: #9ca3af;">IMDb Rating:</strong> ⭐ {rating}/10</p>
    """
    
    # Add similarity scores if available and requested
    if show_scores:
        if "similarity_score" in movie_info:
            similarity = movie_info["similarity_score"] * 100
            card_html += f"<p><strong>Similarity:</strong> {similarity:.1f}%</p>"
        
        if "combined_score" in movie_info:
            combined = movie_info["combined_score"] * 100
            card_html += f"<p><strong>Relevance:</strong> {combined:.1f}%</p>"
            
        if "aspect_scores" in movie_info:
            card_html += "<details><summary><strong>Detailed Scores</strong></summary><ul>"
            for aspect, score in movie_info["aspect_scores"].items():
                card_html += f"<li>{aspect.title()}: {score*100:.1f}%</li>"
            card_html += "</ul></details>"
            
    card_html += """
        </div>
    </div>
    """
    
    return card_html

# Initialize the recommender
@st.cache_resource
def get_recommender():
    # Default paths - modify as needed
    data_path = "Data/imdb_top_1000.csv"  # Path to your IMDb dataset
    db_path = "chroma_db_movies"  # Path to store ChromaDB
    cache_file = "movie_embeddings_cache.npz"  # Embedding cache file
    
    # Check if the data file exists, otherwise show a file uploader
    if not os.path.exists(data_path):
        st.warning("IMDb dataset not found at the default location.")
    
    # Initialize recommender
    try:
        recommender = MovieRecommender(
            data_path=data_path,
            db_path=db_path,
            use_cached_embeddings=True,
            cache_file=cache_file
        )
        return recommender
    except Exception as e:
        st.error(f"Error initializing recommender: {e}")
        return None

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="CinemaSync: AI Movie Recommender",
        page_icon="🎬",
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 42px;
            font-weight: 800;
            color: #FAFAFA;
            text-align: center;
            margin-bottom: 30px;
            # background: linear-gradient(90deg, #3B82F6, #6366F1);
            # -webkit-background-clip: text;
            # -webkit-text-fill-color: transparent;
        }
        .sub-header {
            font-size: 24px;
            font-weight: 600;
            color: #4B5563;
            margin-top: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #3B82F6;
            padding-left: 10px;
        }
        .stButton > button {
            background-color: #3B82F6;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #2563EB;
            box-shadow: 0 6px 10px rgba(59, 130, 246, 0.3);
            transform: translateY(-2px);
        }
        .stSelectbox, .stSlider {
            border-radius: 8px;
        }

        </style>
        <div class="main-header">🎬 CinemaSync: AI-Powered Movie Recommendations</div>
    """, unsafe_allow_html=True)
    
    # Initialize recommender
    recommender = get_recommender()
    
    if recommender is None:
        st.error("Failed to initialize the recommender system. Please check logs for details.")
        return
    
    # Get list of all movies
    all_movies = recommender.df["Series_Title"].tolist()
    
    # Sidebar for user options
    st.sidebar.markdown("## 🔍 Recommendation Options")
    
    # Get all unique genres
    all_genres = set()
    for genres in recommender.df["Genre"].str.split(", "):
        all_genres.update(genres)
    all_genres = sorted(list(all_genres))
    
    # Create tabs for different recommendation methods
    tab1, tab2, tab3, tab4 = st.tabs([
        "Similar Movies", 
        "Genre Mix", 
        "Text Search", 
        "Personalized"
    ])
    
    # Tab 1: Similar Movies
    with tab1:
        st.markdown('<div class="sub-header">Find Movies Similar to Your Favorites</div>', unsafe_allow_html=True)
        
        # Movie selection
        selected_movie = st.selectbox(
            "Select a movie you enjoyed:",
            options=all_movies,
            index=0,
            key="similar_movie_select"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            num_recommendations = st.slider(
                "Number of recommendations:",
                min_value=1,
                max_value=10,
                value=5,
                key="similar_count_slider"
            )
        
        with col2:
            advanced_search = st.checkbox("Use advanced hybrid search", value=True)
        
        # Display aspect weights if advanced search is selected
        if advanced_search:
            st.markdown("#### Aspect Importance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                overall_weight = st.slider("Overall", 0.0, 1.0, 0.4, 0.1, key="weight_overall")
            
            with col2:
                genre_weight = st.slider("Genre", 0.0, 1.0, 0.3, 0.1, key="weight_genre")
            
            with col3:
                plot_weight = st.slider("Plot/Theme", 0.0, 1.0, 0.2, 0.1, key="weight_plot")
            
            with col4:
                cast_weight = st.slider("Cast", 0.0, 1.0, 0.1, 0.1, key="weight_cast")
            
            # Normalize weights
            total = overall_weight + genre_weight + plot_weight + cast_weight
            aspect_weights = {
                "overall": overall_weight / total,
                "genre": genre_weight / total,
                "plot": plot_weight / total,
                "cast": cast_weight / total
            }
        
        get_recommendations = st.button("Get Similar Movies", key="btn_similar")
        
        if get_recommendations:
            with st.spinner("Finding similar movies..."):
                if advanced_search:
                    recommendations = recommender.hybrid_content_based_search(
                        selected_movie,
                        k=num_recommendations,
                        aspect_weights=aspect_weights
                    )
                else:
                    recommendations = recommender.get_similar_movies(
                        selected_movie,
                        k=num_recommendations
                    )
                
                if recommendations:
                    st.markdown(f"### Movies Similar to '{selected_movie}'")
                    
                    for movie in recommendations:
                        st.markdown(create_movie_card(movie), unsafe_allow_html=True)
                else:
                    st.warning("No similar movies found. Try another movie or adjust your search criteria.")
    
    # Tab 2: Genre Mix
    with tab2:
        st.markdown('<div class="sub-header">Find Movies by Genre Combination</div>', unsafe_allow_html=True)
        
        selected_genres = st.multiselect(
            "Select genres you're interested in:",
            options=all_genres,
            default=["Action", "Adventure"] if "Action" in all_genres and "Adventure" in all_genres else None,
            key="genre_multiselect"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            genre_count = st.slider(
                "Number of recommendations:",
                min_value=1,
                max_value=10,
                value=5,
                key="genre_count_slider"
            )
        
        with col2:
            min_rating = st.slider(
                "Minimum IMDb rating:",
                min_value=5.0,
                max_value=9.0,
                value=7.0,
                step=0.1,
                key="min_rating_slider"
            )
        
        get_genre_recommendations = st.button("Find Movies", key="btn_genre")
        
        if get_genre_recommendations:
            if not selected_genres:
                st.warning("Please select at least one genre.")
            else:
                with st.spinner("Finding movies with your selected genres..."):
                    recommendations = recommender.recommend_by_genre_mix(
                        selected_genres,
                        k=genre_count,
                        min_rating=min_rating
                    )
                    
                    if recommendations:
                        st.markdown(f"### Recommended Movies with {', '.join(selected_genres)} Genres")
                        
                        for movie in recommendations:
                            st.markdown(create_movie_card(movie), unsafe_allow_html=True)
                    else:
                        st.warning("No movies found matching your criteria. Try different genres or lower the minimum rating.")
    
    # Tab 3: Text Search
    with tab3:
        st.markdown('<div class="sub-header">Natural Language Movie Search</div>', unsafe_allow_html=True)
        
        query = st.text_input(
            "Describe what you're looking for:",
            placeholder="E.g., 'Sci-fi movies with time travel themes' or 'Heartwarming drama with great acting'",
            key="text_query_input"
        )
        
        text_count = st.slider(
            "Number of results:",
            min_value=1,
            max_value=10,
            value=5,
            key="text_count_slider"
        )
        
        search_movies = st.button("Search Movies", key="btn_text_search")
        
        if search_movies:
            if not query.strip():
                st.warning("Please enter a search query.")
            else:
                with st.spinner(f"Searching for: '{query}'"):
                    recommendations = recommender.get_recommendations_by_text_query(
                        query,
                        k=text_count
                    )
                    
                    if recommendations:
                        st.markdown(f"### Movies Matching Your Search")
                        
                        for movie in recommendations:
                            st.markdown(create_movie_card(movie), unsafe_allow_html=True)
                    else:
                        st.warning("No movies found matching your search. Try a different query.")
    
    # Tab 4: Personalized Recommendations
    with tab4:
        st.markdown('<div class="sub-header">Personalized Recommendations</div>', unsafe_allow_html=True)
        st.markdown("Tell us about your preferences to get personalized movie recommendations")
        
        # Movie watch history
        watched_movies = st.multiselect(
            "Movies you've watched and enjoyed:",
            options=all_movies,
            key="watched_movies_multiselect"
        )
        
        # Genre preferences
        liked_genres = st.multiselect(
            "Your favorite genres:",
            options=all_genres,
            key="liked_genres_multiselect"
        )
        
        # Cast preferences
        all_actors = list(set(recommender.df["Star1"].tolist() + recommender.df["Star2"].tolist() + 
                             recommender.df["Star3"].tolist() + recommender.df["Star4"].tolist()))
        all_actors = [actor for actor in all_actors if actor != "Unknown"]
        all_actors.sort()
        
        # Use a text input for actors instead of a huge dropdown
        favorite_actors_input = st.text_input(
            "Your favorite actors/actresses (comma separated):",
            placeholder="E.g., Tom Hanks, Meryl Streep, Leonardo DiCaprio",
            key="favorite_actors_input"
        )
        favorite_actors = [actor.strip() for actor in favorite_actors_input.split(",")] if favorite_actors_input else []
        
        # Decade preferences
        available_decades = sorted(list(set((recommender.df["Released_Year"].astype('int') // 10 * 10).astype(int).tolist())))
        preferred_decades = st.multiselect(
            "Your preferred decades:",
            options=available_decades,
            key="preferred_decades_multiselect"
        )
        
        # Recommendation count
        personal_count = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=10,
            value=5,
            key="personal_count_slider"
        )
        
        # Diversity factor
        diversity_factor = st.slider(
            "Recommendation diversity (higher = more diverse):",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            key="diversity_slider"
        )
        
        get_personal_recommendations = st.button("Get Personal Recommendations", key="btn_personal")
        
        if get_personal_recommendations:
            if not watched_movies and not liked_genres and not favorite_actors:
                st.warning("Please provide at least some of your preferences to get personalized recommendations.")
            else:
                with st.spinner("Generating personalized recommendations..."):
                    # Create user profile
                    user_profile = {
                        "user_id": "user_1",  # Default user id
                        "recent_watches": watched_movies,
                        "liked_genres": liked_genres,
                        "favorite_actors": favorite_actors,
                        "preferred_decades": preferred_decades,
                        "min_rating": 7.0
                    }
                    
                    recommendations = recommender.get_personalized_recommendations(
                        user_profile,
                        k=personal_count,
                        diversity_factor=diversity_factor
                    )
                    
                    if recommendations:
                        st.markdown("### Your Personalized Movie Recommendations")
                        
                        for movie in recommendations:
                            st.markdown(create_movie_card(movie), unsafe_allow_html=True)
                    else:
                        st.warning("Couldn't generate personalized recommendations. Try adding more preferences or watched movies.")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #1f2937; border-radius: 10px; color: #e5e7eb; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <p>Powered by advanced NLP and semantic search technology</p>
        <p style="font-size: 12px; color: #9ca3af;">Movie data from IMDb. Posters provided by TMDB.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
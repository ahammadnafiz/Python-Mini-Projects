import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import logging
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self, data_path: str = "Data/imdb_top_1000.csv", 
                 db_path: str = "chroma_db_movies",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 use_cached_embeddings: bool = True,
                 cache_file: str = "movie_embeddings_cache.npz"):
        """
        Initialize the movie recommender system with advanced vector embeddings.
        
        Args:
            data_path: Path to the IMDB dataset
            db_path: Path to store ChromaDB
            embedding_model: Model to use for generating embeddings
            use_cached_embeddings: Whether to use cached embeddings if available
            cache_file: File to store embedding cache
        """
        self.data_path = data_path
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.use_cached_embeddings = use_cached_embeddings
        self.cache_file = cache_file
        self.df = None
        self.embedding_model = None
        self.chroma_client = None
        self.movie_db = None
        
        # Create directory for ChromaDB if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize the system
        self._load_data()
        self._initialize_embedding_model()
        self._initialize_vector_db()
        
    # In the _load_data method of MovieRecommender class:
    def _load_data(self) -> None:
        """Load and preprocess the IMDB dataset."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load IMDB dataset
        self.df = pd.read_csv(self.data_path)
        
        # Select relevant features
        self.df = self.df[["Series_Title", "Genre", "IMDB_Rating", "Overview", "Director",
                        "Star1", "Star2", "Star3", "Star4", "No_of_Votes", "Gross", "Runtime", "Released_Year"]]
        
        # Convert Released_Year to numeric first
        self.df["Released_Year"] = pd.to_numeric(self.df["Released_Year"], errors='coerce')
        
        # Handle missing values more robustly
        for col in self.df.columns:
            if col == "Released_Year":
                # Fill missing years with median year
                median_year = self.df[col].median()
                self.df[col].fillna(median_year, inplace=True)
                self.df[col] = self.df[col].astype(int)
            elif self.df[col].dtype == object:
                self.df[col].fillna("Unknown", inplace=True)
            else:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Create a rich movie description for better semantic understanding
        self._generate_movie_descriptions()
        
        logger.info(f"Loaded {len(self.df)} movies")
        
    def _generate_movie_descriptions(self) -> None:
        """Generate comprehensive textual representations of movies."""
        logger.info("Generating rich movie descriptions")
        
        self.df["movie_description"] = self.df.apply(
            lambda row: f"""Title: {row['Series_Title']}
            Year: {row['Released_Year']}
            Genres: {row['Genre']}
            Director: {row['Director']}
            Stars: {row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}
            Runtime: {row['Runtime']}
            IMDB Rating: {row['IMDB_Rating']} based on {row['No_of_Votes']} votes
            Overview: {row['Overview']}""".replace('\n            ', ' ').strip(),
            axis=1
        )
        
        # Create separate embeddings for each aspect to enable more nuanced recommendations
        self.df["title_director"] = self.df.apply(
            lambda row: f"Title: {row['Series_Title']} Director: {row['Director']}", axis=1
        )
        
        self.df["genre_info"] = self.df.apply(
            lambda row: f"Genres: {row['Genre']}", axis=1
        )
        
        self.df["cast_info"] = self.df.apply(
            lambda row: f"Cast: {row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}", axis=1
        )
        
        self.df["plot_info"] = self.df.apply(
            lambda row: f"Plot: {row['Overview']}", axis=1
        )
        
    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model."""
        logger.info(f"Initializing embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def _generate_embeddings(self) -> None:
        """Generate vector embeddings for movies."""
        # Check if embeddings cache exists and should be used
        if self.use_cached_embeddings and os.path.exists(self.cache_file):
            logger.info(f"Loading cached embeddings from {self.cache_file}")
            cache_data = np.load(self.cache_file, allow_pickle=True)
            self.df["embeddings"] = list(cache_data["embeddings"])
            self.df["title_director_embeddings"] = list(cache_data["title_dir_emb"])
            self.df["genre_embeddings"] = list(cache_data["genre_emb"])
            self.df["cast_embeddings"] = list(cache_data["cast_emb"])
            self.df["plot_embeddings"] = list(cache_data["plot_emb"])
            return
        
        logger.info("Generating embeddings for movies")
        
        # Generate embeddings using batch processing for efficiency
        # Main description embeddings
        descriptions = self.df["movie_description"].tolist()
        
        # Use batched encoding for efficiency
        batch_size = 64
        embeddings = []
        
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Generating main embeddings"):
            batch = descriptions[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            embeddings.extend(batch_embeddings)
        
        self.df["embeddings"] = embeddings
        
        # Generate aspect-specific embeddings
        logger.info("Generating aspect-specific embeddings")
        
        # Title and director embeddings
        title_dir_texts = self.df["title_director"].tolist()
        title_dir_emb = []
        for i in tqdm(range(0, len(title_dir_texts), batch_size), desc="Title/Director embeddings"):
            batch = title_dir_texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            title_dir_emb.extend(batch_embeddings)
        self.df["title_director_embeddings"] = title_dir_emb
        
        # Genre embeddings
        genre_texts = self.df["genre_info"].tolist()
        genre_emb = []
        for i in tqdm(range(0, len(genre_texts), batch_size), desc="Genre embeddings"):
            batch = genre_texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            genre_emb.extend(batch_embeddings)
        self.df["genre_embeddings"] = genre_emb
        
        # Cast embeddings
        cast_texts = self.df["cast_info"].tolist()
        cast_emb = []
        for i in tqdm(range(0, len(cast_texts), batch_size), desc="Cast embeddings"):
            batch = cast_texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            cast_emb.extend(batch_embeddings)
        self.df["cast_embeddings"] = cast_emb
        
        # Plot embeddings
        plot_texts = self.df["plot_info"].tolist()
        plot_emb = []
        for i in tqdm(range(0, len(plot_texts), batch_size), desc="Plot embeddings"):
            batch = plot_texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            plot_emb.extend(batch_embeddings)
        self.df["plot_embeddings"] = plot_emb
        
        # Cache embeddings for future use
        logger.info(f"Saving embeddings cache to {self.cache_file}")
        np.savez(
            self.cache_file,
            embeddings=np.array(self.df["embeddings"].tolist()),
            title_dir_emb=np.array(self.df["title_director_embeddings"].tolist()),
            genre_emb=np.array(self.df["genre_embeddings"].tolist()),
            cast_emb=np.array(self.df["cast_embeddings"].tolist()),
            plot_emb=np.array(self.df["plot_embeddings"].tolist())
        )
    
    def _initialize_vector_db(self) -> None:
        """Initialize and populate the vector database."""
        logger.info(f"Initializing ChromaDB at {self.db_path}")
        
        # Generate embeddings if not already done
        if "embeddings" not in self.df.columns:
            self._generate_embeddings()
        
        # Initialize ChromaDB
        self.chroma_client = PersistentClient(path=self.db_path)
        
        # Define embedding function using the same model
        hf_embeddings = SentenceTransformerEmbeddingFunction(self.embedding_model_name)
        
        # Check if collection exists and recreate if needed
        try:
            self.movie_db = self.chroma_client.get_collection(name="movies")
            logger.info("Using existing ChromaDB collection")
        except Exception:
            logger.info("Creating new ChromaDB collection")
            self.movie_db = self.chroma_client.create_collection(
                name="movies", 
                embedding_function=hf_embeddings,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity for semantic search
            )
            
            # Add movies to ChromaDB
            self._populate_vector_db()
            
            self.index_aspects()  # Add this line
    
            logger.info("All collections initialized")
    
    def _populate_vector_db(self) -> None:
        """Populate the vector database with movie data and embeddings."""
        logger.info("Populating ChromaDB with movie data and embeddings")
        
        # Add movies in batches for better performance
        batch_size = 50
        total_movies = len(self.df)
        
        for start_idx in tqdm(range(0, total_movies, batch_size), desc="Adding movies to database"):
            end_idx = min(start_idx + batch_size, total_movies)
            batch_df = self.df.iloc[start_idx:end_idx]
            
            batch_ids = [str(i) for i in batch_df.index.tolist()]
            batch_embeddings = batch_df["embeddings"].tolist()
            
            batch_metadatas = []
            for _, row in batch_df.iterrows():
                metadata = {
                    "title": row["Series_Title"],
                    "year": str(row["Released_Year"]),
                    "genre": row["Genre"],
                    "director": row["Director"],
                    "stars": ", ".join([row["Star1"], row["Star2"], row["Star3"], row["Star4"]]),
                    "imdb_rating": float(row["IMDB_Rating"]),
                    "votes": int(row["No_of_Votes"]),
                    "overview": row["Overview"],
                    "gross": str(row["Gross"]),
                    "runtime": str(row["Runtime"])
                }
                batch_metadatas.append(metadata)
            
            self.movie_db.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        
        logger.info(f"Added {total_movies} movies to ChromaDB")
        
    def index_aspects(self) -> None:
        """Create separate collections for different movie aspects."""
        logger.info("Creating aspect-specific collections")
        
        aspects = {
            "genre": ("genre_info", "genre_embeddings"),
            "cast": ("cast_info", "cast_embeddings"),
            "plot": ("plot_info", "plot_embeddings")
        }
        
        for aspect_name, (text_col, emb_col) in aspects.items():
            try:
                # Get or create collection
                aspect_collection = self.chroma_client.get_or_create_collection(
                    name=f"movies_{aspect_name}",
                    embedding_function=SentenceTransformerEmbeddingFunction(self.embedding_model_name),
                    metadata={"hnsw:space": "cosine"}
                )
                
                # Check if collection is empty
                if aspect_collection.count() == 0:
                    logger.info(f"Populating {aspect_name} collection")
                    
                    # Add data in batches
                    batch_size = 50
                    total_movies = len(self.df)
                    
                    for start_idx in tqdm(range(0, total_movies, batch_size), 
                                        desc=f"Adding {aspect_name} data"):
                        end_idx = min(start_idx + batch_size, total_movies)
                        batch_df = self.df.iloc[start_idx:end_idx]
                        
                        batch_ids = [str(i) for i in batch_df.index.tolist()]
                        batch_embeddings = batch_df[emb_col].tolist()
                        
                        batch_metadatas = []
                        for _, row in batch_df.iterrows():
                            metadata = {
                                "title": row["Series_Title"],
                                "genre": row["Genre"],
                                "director": row["Director"],
                                "stars": ", ".join(filter(None, [row["Star1"], row["Star2"], 
                                                            row["Star3"], row["Star4"]]))
                            }
                            batch_metadatas.append(metadata)
                        
                        aspect_collection.add(
                            ids=batch_ids,
                            embeddings=batch_embeddings,
                            metadatas=batch_metadatas
                        )
                else:
                    logger.info(f"{aspect_name.capitalize()} collection already populated")
                    
            except Exception as e:
                logger.error(f"Error creating {aspect_name} collection: {e}")
                raise
    
    def get_similar_movies(self, movie_name: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find movies similar to the given movie using semantic search.
        
        Args:
            movie_name: Name of the movie to find similar movies to
            k: Number of similar movies to return
            
        Returns:
            List of dictionaries containing similar movie information
        """
        try:
            # Find the movie in the dataframe
            movie_info = self.df[self.df["Series_Title"] == movie_name]
            
            if movie_info.empty:
                logger.warning(f"Movie '{movie_name}' not found in the database")
                return []
            
            movie_info = movie_info.iloc[0]
            query_embedding = movie_info["embeddings"]
            
            # Retrieve similar movies with metadata
            results = self.movie_db.query(
                query_embeddings=[query_embedding],
                n_results=k+1,  # +1 because the movie itself will be in results
                include=["metadatas", "distances"]
            )
            
            # Filter out the query movie
            similar_movies = []
            for i, (metadata, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
                if metadata["title"] != movie_name:
                    metadata["similarity_score"] = 1 - distance  # Convert distance to similarity score
                    similar_movies.append(metadata)
            
            # Return top k results
            return similar_movies[:k]
            
        except Exception as e:
            logger.error(f"Error finding similar movies: {e}")
            return []
    
    def hybrid_content_based_search(self, 
                                   movie_name: str, 
                                   k: int = 5,
                                   aspect_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Advanced hybrid search using multiple aspects of movies with weighted scores.
        
        Args:
            movie_name: Name of the movie to find similar movies to
            k: Number of similar movies to return
            aspect_weights: Dictionary of weights for different aspects (plot, genre, cast, etc.)
            
        Returns:
            List of dictionaries containing similar movie information
        """
        # Default weights if not provided
        if aspect_weights is None:
            aspect_weights = {
                "overall": 0.4,
                "genre": 0.3,
                "plot": 0.2,
                "cast": 0.1
            }
        
        try:
            # Find the movie in the dataframe
            movie_info = self.df[self.df["Series_Title"] == movie_name]
            
            if movie_info.empty:
                logger.warning(f"Movie '{movie_name}' not found in the database")
                return []
            
            movie_info = movie_info.iloc[0]
            
            # Get movie index
            movie_idx = movie_info.name
            
            # Calculate similarities for each aspect
            results = {}
            scores = {}
            
            # Overall similarity (main embeddings)
            main_emb = movie_info["embeddings"]
            # MAIN QUERY - remove 'ids' from include list
            main_results = self.movie_db.query(
                query_embeddings=[main_emb],
                n_results=50,
                include=["metadatas", "distances"]  # REMOVED 'ids' FROM HERE
            )
            
            # Map distances to ids for each aspect
            for i, movie_id in enumerate(main_results["ids"][0]):
                movie_id = int(movie_id)
                if movie_id != movie_idx:  # Skip the query movie
                    distance = main_results["distances"][0][i]
                    similarity = 1 - distance
                    scores[movie_id] = {"overall": similarity * aspect_weights["overall"]}
            
            # Get aspect-specific similarities if collections exist
            aspects = {
                "genre": "genre_embeddings",
                "plot": "plot_embeddings",
                "cast": "cast_embeddings"
            }
            
            for aspect, emb_col in aspects.items():
                try:
                    aspect_collection = self.chroma_client.get_collection(name=f"movies_{aspect}")
                    aspect_emb = movie_info[emb_col]
                    
                    aspect_results = aspect_collection.query(
                        query_embeddings=[aspect_emb],
                        n_results=50,
                        include=["distances"]
                    )
                    
                    for i, movie_id in enumerate(aspect_results["ids"][0]):
                        movie_id = int(movie_id)
                        if movie_id != movie_idx:  # Skip the query movie
                            distance = aspect_results["distances"][0][i]
                            similarity = 1 - distance
                            
                            if movie_id not in scores:
                                scores[movie_id] = {"overall": 0.0}
                            
                            scores[movie_id][aspect] = similarity * aspect_weights[aspect]
                except Exception:
                    logger.warning(f"Aspect collection for {aspect} not found, skipping")
            
            # Calculate total scores
            total_scores = []
            for movie_id, aspect_scores in scores.items():
                total_score = sum(aspect_scores.values())
                total_scores.append((movie_id, total_score))
            
            # Sort by total score
            total_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top k results with detailed metadata
            top_movies = []
            for movie_id, score in total_scores[:k]:
                movie_data = self.df.iloc[movie_id]
                movie_info = {
                    "title": movie_data["Series_Title"],
                    "year": str(movie_data["Released_Year"]),
                    "genre": movie_data["Genre"],
                    "director": movie_data["Director"],
                    "stars": ", ".join([movie_data["Star1"], movie_data["Star2"], movie_data["Star3"], movie_data["Star4"]]),
                    "imdb_rating": float(movie_data["IMDB_Rating"]),
                    "similarity_score": score,
                    "aspect_scores": scores[movie_id]
                }
                top_movies.append(movie_info)
            
            return top_movies
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def get_personalized_recommendations(self, 
                                        user_profile: Dict[str, Any], 
                                        k: int = 5,
                                        diversity_factor: float = 0.3) -> List[Dict[str, Any]]:
        """
        Generate personalized movie recommendations based on user profile.
        
        Args:
            user_profile: Dictionary containing user preferences
            k: Number of recommendations to return
            diversity_factor: Factor to control recommendation diversity (0-1)
            
        Returns:
            List of dictionaries containing recommended movie information
        """
        try:
            # Extract user preferences
            liked_genres = set(user_profile.get("liked_genres", []))
            favorite_actors = set(user_profile.get("favorite_actors", []))
            recent_watches = user_profile.get("recent_watches", [])
            preferred_decades = set(user_profile.get("preferred_decades", []))
            min_rating = user_profile.get("min_rating", 6.0)
            
            # Get candidate recommendations based on recently watched movies
            candidate_movies = {}
            
            for movie in recent_watches:
                # Use hybrid search for better recommendations
                similar_movies = self.hybrid_content_based_search(
                    movie, 
                    k=10,
                    aspect_weights={
                        "overall": 0.3,
                        "genre": 0.4,
                        "plot": 0.2,
                        "cast": 0.1
                    }
                )
                
                # Add to candidates with similarity score
                for movie_info in similar_movies:
                    title = movie_info["title"]
                    if title not in recent_watches and title not in candidate_movies:
                        # Calculate a personalization score
                        personalization_score = 0.0
                        
                        # Genre matching
                        movie_genres = set(movie_info["genre"].split(", "))
                        genre_overlap = len(movie_genres.intersection(liked_genres))
                        if genre_overlap > 0:
                            personalization_score += 0.3 * (genre_overlap / len(movie_genres))
                        
                        # Actor matching
                        movie_actors = set(movie_info["stars"].split(", "))
                        actor_overlap = len(movie_actors.intersection(favorite_actors))
                        if actor_overlap > 0:
                            personalization_score += 0.2 * (actor_overlap / len(movie_actors))
                        
                        # Rating threshold
                        if movie_info["imdb_rating"] >= min_rating:
                            personalization_score += 0.1 * (movie_info["imdb_rating"] / 10.0)
                        
                        # Decade preference (if available)
                        if "year" in movie_info:
                            decade = (int(movie_info["year"]) // 10) * 10
                            if decade in preferred_decades:
                                personalization_score += 0.1
                        
                        # Combine similarity and personalization
                        combined_score = (movie_info["similarity_score"] * (1 - diversity_factor) + 
                                         personalization_score * diversity_factor)
                        
                        candidate_movies[title] = {
                            **movie_info,
                            "combined_score": combined_score
                        }
            
            # Sort by combined score
            sorted_candidates = sorted(
                candidate_movies.values(), 
                key=lambda x: x["combined_score"], 
                reverse=True
            )
            
            # Apply diversity filtering to ensure variety in recommendations
            diverse_recommendations = []
            genres_added = set()
            
            for movie in sorted_candidates:
                movie_genres = set(movie["genre"].split(", "))
                
                # Add movie if it introduces new genres or if we need more recommendations
                if len(diverse_recommendations) < k * 0.6 or len(genres_added.intersection(movie_genres)) < 2:
                    diverse_recommendations.append(movie)
                    genres_added.update(movie_genres)
                
                if len(diverse_recommendations) >= k:
                    break
            
            return diverse_recommendations
            
        except Exception as e:
            logger.error(f"Error generating personalized recommendations: {e}")
            return []
    
    def get_recommendations_by_text_query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on natural language query.
        
        Args:
            query: Natural language query text (e.g., "action movies with car chases")
            k: Number of recommendations to return
            
        Returns:
            List of dictionaries containing recommended movie information
        """
        try:
            # Generate embedding for the query text
            query_embedding = self.embedding_model.encode(query)
            
            # Search in the vector database
            results = self.movie_db.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["metadatas", "distances"]
            )
            
            # Format results
            recommendations = []
            for i, (metadata, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
                metadata["relevance_score"] = 1 - distance
                recommendations.append(metadata)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in text query search: {e}")
            return []
    
    def recommend_by_genre_mix(self, genres: List[str], k: int = 5, min_rating: float = 7.0) -> List[Dict[str, Any]]:
        """
        Recommend movies based on a mix of genres.
        
        Args:
            genres: List of genres to include
            k: Number of recommendations to return
            min_rating: Minimum IMDB rating threshold
            
        Returns:
            List of dictionaries containing recommended movie information
        """
        try:
            # Create a query string from genres
            query_text = f"Movies with genres: {', '.join(genres)}"
            
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query_text)
            
            # Search in the vector database
            results = self.movie_db.query(
                query_embeddings=[query_embedding],
                n_results=50,  # Get more to filter
                include=["metadatas", "distances"]
            )
            
            # Filter and rank results
            recommendations = []
            
            for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
                # Check if movie contains at least one of the requested genres
                movie_genres = set(metadata["genre"].split(", "))
                if any(genre in movie_genres for genre in genres) and metadata["imdb_rating"] >= min_rating:
                    # Calculate genre match score
                    genre_match_count = sum(1 for genre in genres if genre in movie_genres)
                    genre_match_score = genre_match_count / len(genres)
                    
                    # Calculate combined score (semantic similarity + genre match + rating boost)
                    semantic_score = 1 - distance
                    rating_boost = (metadata["imdb_rating"] - min_rating) / (10 - min_rating)
                    
                    combined_score = (0.4 * semantic_score + 
                                     0.4 * genre_match_score + 
                                     0.2 * rating_boost)
                    
                    recommendations.append({
                        **metadata,
                        "combined_score": combined_score,
                        "genre_match": genre_match_score
                    })
            
            # Sort by combined score and return top k
            recommendations.sort(key=lambda x: x["combined_score"], reverse=True)
            return recommendations[:k]
            
        except Exception as e:
            logger.error(f"Error in genre mix recommendation: {e}")
            return []
    
    def build_user_profile(self, 
                      user_id: str,
                      watched_movies: List[str],
                      ratings: Dict[str, float] = None,
                      explicit_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Build or update a user profile based on watch history and ratings.
        
        Args:
            user_id: Unique user identifier
            watched_movies: List of movies the user has watched
            ratings: Dictionary mapping movie titles to user ratings
            explicit_preferences: Dictionary of explicitly stated user preferences
            
        Returns:
            Dictionary containing user profile
        """
        try:
            # Initialize profile
            user_profile = {
                "user_id": user_id,
                "recent_watches": [],
                "liked_genres": [],
                "favorite_actors": [],
                "preferred_decades": [],
                "genre_preferences": {},
                "actor_preferences": {},
                "director_preferences": {},
                "min_rating": 7.0,
                "last_updated": time.time()
            }
            
            # Update with explicit preferences if provided
            if explicit_preferences:
                for key, value in explicit_preferences.items():
                    if key in user_profile:
                        user_profile[key] = value
            
            # Initialize counters
            genre_counts = {}
            actor_counts = {}
            director_counts = {}
            decade_counts = {}
            
            # Process watched movies
            valid_movies = []
            
            for movie in watched_movies:
                movie_info = self.df[self.df["Series_Title"] == movie]
                
                if not movie_info.empty:
                    movie_info = movie_info.iloc[0]
                    valid_movies.append(movie)
                    
                    # Get user rating or default to IMDB rating if not provided
                    user_rating = ratings.get(movie, movie_info["IMDB_Rating"]) if ratings else movie_info["IMDB_Rating"]
                    rating_weight = user_rating / 10.0  # Normalize to 0-1
                    
                    # Update genre preferences
                    for genre in movie_info["Genre"].split(", "):
                        if genre in genre_counts:
                            genre_counts[genre] += rating_weight
                        else:
                            genre_counts[genre] = rating_weight
                    
                    # Update actor preferences
                    for actor in [movie_info["Star1"], movie_info["Star2"], movie_info["Star3"], movie_info["Star4"]]:
                        if actor and actor != "Unknown":
                            if actor in actor_counts:
                                actor_counts[actor] += rating_weight
                            else:
                                actor_counts[actor] = rating_weight
                    
                    # Update director preferences
                    director = movie_info["Director"]
                    if director and director != "Unknown":
                        if director in director_counts:
                            director_counts[director] += rating_weight
                        else:
                            director_counts[director] = rating_weight
                    
                    # Update decade preferences
                    decade = (int(movie_info["Released_Year"]) // 10) * 10
                    if decade in decade_counts:
                        decade_counts[decade] += rating_weight
                    else:
                        decade_counts[decade] = rating_weight
            
            # Sort by counts and select top preferences
            user_profile["genre_preferences"] = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))
            user_profile["actor_preferences"] = dict(sorted(actor_counts.items(), key=lambda x: x[1], reverse=True))
            user_profile["director_preferences"] = dict(sorted(director_counts.items(), key=lambda x: x[1], reverse=True))
            
            # Select top-rated items for simplified lists
            user_profile["liked_genres"] = list(user_profile["genre_preferences"].keys())[:5]
            user_profile["favorite_actors"] = list(user_profile["actor_preferences"].keys())[:5]
            user_profile["preferred_decades"] = [decade for decade, count in 
                                                sorted(decade_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
            
            # Add recent watches (most recent first, limited to 10)
            user_profile["recent_watches"] = valid_movies[-10:][::-1]
            
            return user_profile
            
        except Exception as e:
            logger.error(f"Error building user profile: {e}")
            return {
                "user_id": user_id,
                "error": str(e)
            }
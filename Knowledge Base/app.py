import streamlit as st
import json
import nltk
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network

# Download NLTK resources (only once)
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    # Add punkt_tab resource download to fix tokenization error
    try:
        nltk.download('punkt_tab')
    except Exception as e:
        st.warning(f"Note: punkt_tab download failed, but application may still work with punkt: {str(e)}")
    return True

download_nltk_resources()
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Custom CSS for modern layout
st.markdown("""
    <style>
        .main {
            max-width: 2500px;
            padding: 2rem;
        }
        .header {
            background: linear-gradient(45deg, #6366f1, #8b5cf6);
            padding: 3rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background: #6366f1;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# Modern header
st.markdown("""
    <div class="header">
        <h1 style="font-size: 2.5rem; margin: 0;">TEXT RELATION EXPLORER</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Semantic & Syntactic Relationship Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Initialize NLP pipeline
@st.cache_resource
def load_models():
    try:
        # Load BERT model for embeddings
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        
        # Load named entity recognition pipeline as alternative to spaCy's token processing
        ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        
        # Load English stopwords
        stop_words = set(stopwords.words('english'))
        
        return tokenizer, model, ner_pipeline, stop_words
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

tokenizer, model, ner_pipeline, stop_words = load_models()

def preprocess_text(text_content):
    """Split text into sentences using NLTK instead of spaCy"""
    try:
        sentences = sent_tokenize(text_content)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 0]
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return []

def get_token_offsets(text):
    """Get word offsets in a text using NLTK tokenizer"""
    offsets = []
    tokens = []
    current_pos = 0
    
    for token in word_tokenize(text):
        token_start = text.find(token, current_pos)
        if token_start != -1:
            token_end = token_start + len(token)
            offsets.append((token_start, token_end))
            tokens.append(token)
            current_pos = token_end
    
    return tokens, offsets

# Improved embedding extraction with alignment checks
def extract_embeddings(sentences):
    word_embeddings = defaultdict(list)
    case_mapping = defaultdict(set)
    
    for sent in sentences:
        try:
            # Get BERT tokenization
            inputs = tokenizer(sent, return_tensors="pt", return_offsets_mapping=True, truncation=True)
            offset_mapping = inputs.pop('offset_mapping').squeeze(0).tolist()
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
            
            hidden_states = outputs.hidden_states[-4:]
            embeddings = torch.mean(torch.stack(hidden_states), dim=0).squeeze(0)
            
            # Get word tokens and their positions using NLTK
            tokens, token_offsets = get_token_offsets(sent)
            
            for token_idx, (token_start, token_end) in enumerate(token_offsets):
                token = tokens[token_idx]
                
                # Skip short tokens, punctuation, and stopwords
                if len(token) < 2 or token.lower() in stop_words or not any(c.isalpha() for c in token):
                    continue
                
                token_vectors = []
                
                # Find all BERT subword tokens that overlap with this word
                for bert_idx, (bert_start, bert_end) in enumerate(offset_mapping):
                    if bert_end <= token_start:
                        continue
                    if bert_start >= token_end:
                        break
                    # Handle subword tokens
                    overlap = min(bert_end, token_end) - max(bert_start, token_start)
                    if overlap > 0:
                        token_vectors.append(embeddings[bert_idx].numpy())
                
                if token_vectors:
                    lower_text = token.lower()
                    avg_vector = np.mean(token_vectors, axis=0)
                    word_embeddings[lower_text].append(avg_vector)
                    case_mapping[lower_text].add(token)
        except Exception as e:
            st.warning(f"Error processing sentence: {sent[:50]}... ({str(e)})")
            continue
    
    final_embeddings = {}
    for word, vectors in word_embeddings.items():
        final_embeddings[word] = {
            "embeddings": np.mean(vectors, axis=0),
            "variants": list(case_mapping[word]),
            "count": len(vectors)
        }
    return final_embeddings

# Enhanced PMI calculation with smoothing
def build_co_occurrence(sentences, window_size=3):
    co_occur = defaultdict(lambda: defaultdict(int))
    word_counts = defaultdict(int)
    total_pairs = 0
    
    for sent in sentences:
        try:
            # Tokenize the sentence and filter out stopwords/punctuation
            words = [token.lower() for token in word_tokenize(sent) 
                    if token.lower() not in stop_words and token.isalpha() and len(token) > 1]
            
            for i, word in enumerate(words):
                word_counts[word] += 1
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j and word != words[j]:
                        co_occur[word][words[j]] += 1
                        total_pairs += 1
        except Exception as e:
            st.warning(f"Error analyzing sentence: {sent[:50]}... ({str(e)})")
            continue

    pmi_matrix = defaultdict(lambda: defaultdict(float))
    for w1, neighbors in co_occur.items():
        for w2, count in neighbors.items():
            try:
                p_xy = (count + 1e-8) / (total_pairs + 1e-8)
                p_x = (word_counts[w1] + 1e-8) / (total_pairs + 1e-8)
                p_y = (word_counts[w2] + 1e-8) / (total_pairs + 1e-8)
                pmi = np.log(p_xy / (p_x * p_y))
                pmi_matrix[w1][w2] = max(pmi, 0)
            except:
                pmi_matrix[w1][w2] = 0
        
    return pmi_matrix, word_counts

# Weighted knowledge base creation
def create_knowledge_base(embeddings, co_occur, semantic_weight=0.7, similarity_threshold=0.7):
    knowledge_base = defaultdict(lambda: defaultdict(list))
    words = list(embeddings.keys())
    
    if words:
        try:
            embeddings_matrix = np.array([embeddings[w]["embeddings"] for w in words])
            sim_matrix = cosine_similarity(embeddings_matrix)
            
            for i, w1 in enumerate(words):
                for j, w2 in enumerate(words):
                    if i != j and sim_matrix[i][j] > similarity_threshold:
                        for variant in embeddings[w1]["variants"]:
                            knowledge_base[variant][w2].append({
                                "type": "semantic",
                                "score": float(sim_matrix[i][j]),
                                "weight": semantic_weight
                            })
        except Exception as e:
            st.error(f"Semantic analysis error: {str(e)}")
    
    syntactic_weight = 1 - semantic_weight
    for w1, neighbors in co_occur.items():
        if w1 in embeddings:
            try:
                for variant in embeddings[w1]["variants"]:
                    for w2, pmi_score in neighbors.items():
                        knowledge_base[variant][w2].append({
                            "type": "syntactic",
                            "score": float(pmi_score),
                            "weight": syntactic_weight
                        })
            except Exception as e:
                st.warning(f"Relation processing error: {str(e)}")
                continue
    
    return knowledge_base

# Interactive visualization with configurable physics
def visualize_with_pyvis(knowledge_base, target_word):
    target_lower = target_word.lower()
    matches = [word for word in knowledge_base.keys() if word.lower() == target_lower]
    
    if not matches:
        return None
    
    G = nx.MultiDiGraph()
    edge_colors = {
        "semantic": "#6366f1",
        "syntactic": "#10b981"
    }
    
    try:
        for neighbor, relations in knowledge_base[matches[0]].items():
            for rel in relations:
                G.add_edge(
                    matches[0], neighbor,
                    title=f"{rel['type'].title()} ({rel['score']:.2f})",
                    color=edge_colors[rel["type"]],
                    weight=rel["score"] * 0.1,
                    type=rel["type"]
                )
    except Exception as e:
        st.error(f"Graph construction error: {str(e)}")
        return None
    
    net = Network(height="800px", width="100%", notebook=True, cdn_resources="remote")
    net.from_nx(G)
    
    physics_options = {
        "forceAtlas2Based": {
            "gravitationalConstant": -100,
            "springLength": 100,
            "springConstant": 0.01
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
    }
    
    net.set_options(json.dumps({
        "nodes": {
            "font": {"size": 18},
            "shape": "dot",
            "size": 25
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "scaling": {"min": 1, "max": 5}
        },
        "physics": physics_options
    }))
    
    return net

# Modern Streamlit layout
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📤 Upload Text File", type=["txt"])
    with col2:
        target_word = st.text_input("🎯 Target Word", key='target_word')

process_btn = st.button("Analyze Text →", type="primary")

# Enhanced sidebar with weight controls
with st.sidebar:
    st.markdown("## Configuration Panel")
    with st.expander("🧠 Semantic Settings", expanded=True):
        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.0, 1.0, 0.7, 0.05,
            help="Minimum cosine similarity for semantic relations"
        )
        semantic_weight = st.slider(
            "Semantic Weight",
            0.0, 1.0, 0.7, 0.1,
            help="Relative importance of semantic vs syntactic relations"
        )
    
    with st.expander("🔗 Syntactic Settings"):
        window_size = st.slider(
            "Context Window Size",
            1, 5, 3,
            help="Number of words to consider around each word for co-occurrence"
        )
    
    st.markdown("---")
    st.markdown("### Model Information")
    st.info("""
    This application uses:
    - NLTK for tokenization and sentence splitting
    - BERT embeddings for semantic similarity
    - Transformer-based NER for entity recognition
    - PMI-enhanced co-occurrence analysis
    - Interactive network visualizations
    """)

# Processing pipeline with progress indicators
if uploaded_file and process_btn:
    current_content = uploaded_file.read().decode()
    uploaded_file.seek(0)

    if 'file_content' not in st.session_state or st.session_state.file_content != current_content:
        with st.spinner("🔍 Analyzing text relationships..."):
            try:
                sentences = preprocess_text(current_content)
                if not sentences:
                    st.error("No valid sentences found in the uploaded file")
                    st.stop()
                
                progress_bar = st.progress(0)
                
                progress_bar.text("Extracting embeddings...")
                embeddings = extract_embeddings(sentences)
                if not embeddings:
                    st.error("Failed to extract word embeddings")
                    st.stop()
                progress_bar.progress(33)
                
                progress_bar.text("Analyzing co-occurrence...")
                co_occur, word_counts = build_co_occurrence(sentences, window_size)
                progress_bar.progress(66)
                
                progress_bar.text("Building knowledge base...")
                knowledge_base = create_knowledge_base(
                    embeddings, co_occur,
                    semantic_weight=semantic_weight,
                    similarity_threshold=similarity_threshold
                )
                progress_bar.progress(100)
                
                st.session_state.knowledge_base = knowledge_base
                st.session_state.file_content = current_content
                st.session_state.sentences = sentences
                st.session_state.word_counts = word_counts
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

# Results visualization and export
if 'knowledge_base' in st.session_state and st.session_state.target_word.strip():
    st.markdown("---")
    st.markdown("## Visualization Results")
    
    with st.container():
        try:
            net = visualize_with_pyvis(st.session_state.knowledge_base, st.session_state.target_word)
            if net:
                html_file = "graph.html"
                net.save_graph(html_file)
                st.components.v1.html(open(html_file).read(), height=800)
                
                # Show word statistics
                target_lower = st.session_state.target_word.lower()
                matches = [w for w in st.session_state.knowledge_base if w.lower() == target_lower]
                
                if matches:
                    # Find matching word variants
                    found_variants = []
                    for match in matches:
                        for key, value in st.session_state.knowledge_base[match].items():
                            if "variants" in value:
                                found_variants.extend(value["variants"])
                    
                    if found_variants:
                        st.info(f"Word variants found: {', '.join(found_variants)}")
                    
                    # Show frequency information
                    if hasattr(st.session_state, 'word_counts') and target_lower in st.session_state.word_counts:
                        st.success(f"Frequency: {st.session_state.word_counts[target_lower]} occurrences")
            else:
                st.error("Target word not found in analysis results")
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")

    # Enhanced export section with statistics
    st.markdown("---")
    with st.expander("📤 Export Results", expanded=True):
        st.markdown("### Download Analysis Data")
        if st.button("Generate JSON Report"):
            target_lower = st.session_state.target_word.lower()
            matches = [w for w in st.session_state.knowledge_base if w.lower() == target_lower]
            
            if matches:
                try:
                    export_data = {
                        "target": matches[0],
                        "statistics": {
                            "total_sentences": len(st.session_state.sentences),
                            "total_relations": len(st.session_state.knowledge_base[matches[0]]),
                            "frequency": st.session_state.word_counts.get(target_lower, 0)
                        },
                        "relations": st.session_state.knowledge_base[matches[0]]
                    }
                    json_str = json.dumps(export_data, indent=2)
                    
                    st.download_button(
                        "⬇️ Download JSON",
                        json_str,
                        file_name=f"{target_lower}_relations.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
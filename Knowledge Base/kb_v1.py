import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(file_path):
    """Split text into sentences using NLTK"""
    with open(file_path, "r") as f:
        text = f.read()
    return sent_tokenize(text)

def extract_embeddings(sentences):
    """Case-sensitive embedding extraction with case-insensitive index"""
    word_embeddings = defaultdict(list)
    case_mapping = defaultdict(set)
    
    for sent in sentences:
        try:
            inputs = tokenizer(sent, return_tensors="pt", return_offsets_mapping=True, truncation=True)
            offset_mapping = inputs.pop('offset_mapping').squeeze(0).tolist()
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            hidden_states = outputs.hidden_states[-4:]
            embeddings = torch.mean(torch.stack(hidden_states), dim=0).squeeze(0)
            
            words = word_tokenize(sent)
            for word in words:
                if not word.isalnum() or word.lower() in stop_words or len(word) < 2:
                    continue
                
                token_vectors = []
                word_positions = []
                start_idx = 0
                
                # Find word positions in the sentence
                while True:
                    pos = sent.find(word, start_idx)
                    if pos == -1:
                        break
                    word_positions.append((pos, pos + len(word)))
                    start_idx = pos + 1
                
                for word_start, word_end in word_positions:
                    for idx, (start, end) in enumerate(offset_mapping):
                        if end <= word_start:
                            continue
                        if start >= word_end:
                            break
                        # Check for overlap
                        if min(end, word_end) > max(start, word_start):
                            token_vectors.append(embeddings[idx].numpy())
                
                if token_vectors:
                    lower_text = word.lower()
                    word_embeddings[lower_text].append(np.mean(token_vectors, axis=0))
                    case_mapping[lower_text].add(word)
        except Exception as e:
            print(f"Error processing sentence: {sent[:30]}... ({str(e)})")
            continue
    
    final_embeddings = {}
    for word, vectors in word_embeddings.items():
        if vectors:
            final_embeddings[word] = {
                "embeddings": np.mean(vectors, axis=0),
                "variants": list(case_mapping[word])
            }
    
    return final_embeddings, case_mapping

def build_co_occurrence(sentences, window_size=3):
    """Case-sensitive co-occurrence tracking with normalization"""
    co_occur = defaultdict(lambda: defaultdict(int))
    word_counts = defaultdict(int)
    case_mapping = defaultdict(set)
    total_words = 0
    
    for sent in sentences:
        words = word_tokenize(sent)
        filtered_words = []
        
        for word in words:
            if not word.isalnum() or word.lower() in stop_words or len(word) < 2:
                continue
            filtered_words.append(word)
            word_counts[word.lower()] += 1
            case_mapping[word.lower()].add(word)
            total_words += 1
        
        for i, word in enumerate(filtered_words):
            start = max(0, i - window_size)
            end = min(len(filtered_words), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    w1_lower = word.lower()
                    w2_lower = filtered_words[j].lower()
                    co_occur[w1_lower][w2_lower] += 1
    
    # Calculate PMI
    pmi_scores = defaultdict(dict)
    for w1, neighbors in co_occur.items():
        for w2, count in neighbors.items():
            if count > 0:
                p_xy = count / total_words
                p_x = word_counts[w1] / total_words
                p_y = word_counts[w2] / total_words
                pmi = max(0, np.log2(p_xy / (p_x * p_y)))
                pmi_scores[w1][w2] = pmi
    
    return pmi_scores, case_mapping

def create_knowledge_base(embeddings, co_occur, case_mapping, similarity_threshold=0.7):
    """Knowledge base with preserved casing and case-insensitive lookup"""
    knowledge_base = defaultdict(dict)
    words = list(embeddings.keys())
    
    # Get similarity matrix
    embeddings_matrix = np.array([embeddings[w]["embeddings"] for w in words])
    sim_matrix = cosine_similarity(embeddings_matrix)
    
    # Add semantic relationships
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if i != j and sim_matrix[i][j] > similarity_threshold:
                for variant in case_mapping[w1]:
                    if variant not in knowledge_base:
                        knowledge_base[variant]["variants"] = list(case_mapping[w1])
                    if "semantic" not in knowledge_base[variant]:
                        knowledge_base[variant]["semantic"] = {}
                    for w2_variant in case_mapping[w2]:
                        knowledge_base[variant]["semantic"][w2_variant] = float(sim_matrix[i][j])
    
    # Add syntactic relationships
    for w1, neighbors in co_occur.items():
        for w2, score in neighbors.items():
            if score > 0.1:  # Filter low-scoring relationships
                for variant in case_mapping[w1]:
                    if variant not in knowledge_base:
                        knowledge_base[variant]["variants"] = list(case_mapping[w1])
                    if "syntactic" not in knowledge_base[variant]:
                        knowledge_base[variant]["syntactic"] = {}
                    for w2_variant in case_mapping[w2]:
                        knowledge_base[variant]["syntactic"][w2_variant] = float(score)
    
    return knowledge_base

def visualize_word_graph(knowledge_base, target_word):
    """Visualization with original casing"""
    G = nx.DiGraph()
    
    if target_word not in knowledge_base:
        target_lower = target_word.lower()
        matches = [w for w in knowledge_base.keys() if w.lower() == target_lower]
        if matches:
            target_word = matches[0]
        else:
            return None
    
    G.add_node(target_word, size=25)
    
    for rel_type in ["semantic", "syntactic"]:
        if rel_type in knowledge_base[target_word]:
            for neighbor, score in knowledge_base[target_word][rel_type].items():
                G.add_node(neighbor, size=15)
                G.add_edge(target_word, neighbor, weight=score, type=rel_type)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 10))
    
    # Draw edges with different colors
    semantic_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "semantic"]
    syntactic_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "syntactic"]
    
    nx.draw_networkx_edges(G, pos, edgelist=semantic_edges, alpha=0.6, edge_color='blue', width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=syntactic_edges, alpha=0.6, edge_color='green', width=1.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[target_word], node_color='red', node_size=2000, alpha=0.8)
    neighbors = list(G.neighbors(target_word))
    nx.draw_networkx_nodes(G, pos, nodelist=neighbors, node_color='lightblue', node_size=1200, alpha=0.8)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    plt.axis('off')
    plt.title(f"Word Relationships for '{target_word}'")
    plt.tight_layout()
    
    return plt

def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

if __name__ == "__main__":
    target_word = input("Enter a word to analyze: ").strip()
    if not target_word:
        print("Please enter a valid word")
        exit()
    
    sentences = preprocess_text("test.txt")
    
    embeddings, case_mapping = extract_embeddings(sentences)
    co_occur, case_mapping = build_co_occurrence(sentences)
    knowledge_base = create_knowledge_base(embeddings, co_occur, case_mapping)
    
    plt = visualize_word_graph(knowledge_base, target_word)
    if plt:
        plt.show()
    
    # Save target word's knowledge base entries to JSON
    target_lower = target_word.lower()
    matches = [word for word in knowledge_base if word.lower() == target_lower]
    
    if not matches:
        print(f"Target word '{target_word}' not found in knowledge base. JSON file not created.")
    else:
        # Collect all entries matching the target word (case-insensitive)
        target_data = {word: knowledge_base[word] for word in matches}
        # Convert defaultdicts to regular dicts
        target_data = default_to_regular(target_data)
        
        filename = f"{target_lower}_knowledge_base.json"
    
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(target_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved knowledge base entries for '{target_word}' to {filename}")
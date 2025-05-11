import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK resources (first-time only)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

sentences = [
    "DNA replication requires DNA polymerase.",
    "This enzyme synthesizes new DNA strands using existing strands as templates."
]

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# Process the first sentence
stop_words = set(stopwords.words('english'))
words = [word for word in word_tokenize(sentences[0]) 
         if word.isalnum() and word.lower() not in stop_words and len(word) > 1]
print(words)


def extract_embeddings(sentences):
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
                "variants": list(case_mapping[word]),
                "count": len(vectors)
            }
    
    return final_embeddings

embeddings = extract_embeddings(sentences)
print(embeddings)

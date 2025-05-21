# %%
# Step 1: Load and Preprocess Dataset
import json
import re
import pandas as pd

# Load the JSON dataset
with open("Corona2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Preprocess function
def preprocess_dataset(data):
    examples = data.get("examples", [])
    contents = []

    for example in examples:
        content = example.get("content", "")
        # Clean text: remove newlines, excessive whitespace, etc.
        clean_content = re.sub(r'\s+', ' ', content).strip()
        contents.append({
            "id": example.get("id"),
            "content": clean_content
        })

    df = pd.DataFrame(contents)
    return df

# Run preprocessing
preprocessed_df = preprocess_dataset(data)

# View sample
print(preprocessed_df.head())


# %%
# Step 2: Embed the Text
# Step 2.1 
# Option A: Using sentence-transformers
from sentence_transformers import SentenceTransformer

# Load model
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Small and fast

# Generate embeddings
embeddings = embedder.encode(preprocessed_df['content'].tolist(), show_progress_bar=True)

# Store in DataFrame
preprocessed_df['embedding'] = embeddings.tolist()

 # Step 2.2
 # Option B: Using HuggingFace Transformers
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load model and tokenizer (BioBERT as an example)
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Apply to dataset
preprocessed_df['embedding'] = preprocessed_df['content'].apply(get_embedding)



# %%
# Step 3: Store in a Vector Database (FAISS)
import faiss
import numpy as np

# Convert embeddings to numpy array
embedding_matrix = np.vstack(preprocessed_df['embedding'].values)

# Initialize FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)


# %%
# Step 4: Build a Retrieval Pipeline
def retrieve(query, k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    results = preprocessed_df.iloc[indices[0]]
    return results[['id', 'content']]


# %%
# Step 5: Feed Retrieved Text to the LLaMA Model
# Example with llama-cpp-python (assuming local model):
from llama_cpp import Llama

llm = Llama(model_path="path/to/llama/model.bin", n_ctx=2048)

def generate_response(query, retrieved_docs):
    context = " ".join(retrieved_docs['content'].tolist())
    prompt = f"Use the following context to answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    output = llm(prompt, max_tokens=256)
    return output['choices'][0]['text'].strip()

# %%
# Step 6: Evaluate Output Quality
query = "What are treatments for diarrhea?"
retrieved = retrieve(query)
rag_response = generate_response(query, retrieved)
print("RAG Response:\n", rag_response)


# %%
# Without RAG (Baseline using BioBERT for classification/QA)
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="dmis-lab/biobert-base-cased-v1.1", tokenizer="dmis-lab/biobert-base-cased-v1.1")

# Use one full document as input context
context = preprocessed_df['content'].iloc[0]
bert_response = qa_pipeline(question=query, context=context)

print("Direct BERT Response:\n", bert_response['answer'])

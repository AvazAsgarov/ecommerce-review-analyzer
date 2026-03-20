import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Setup & Initialization
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

# Define our specific OpenAI client for custom embedding generation later
client = OpenAI(api_key=api_key)

def load_data(filepath):
    """Loads and cleans the dataset, extracting non-null text reviews."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Review Text']).reset_index(drop=True)
    print(f"Loaded {len(df)} textual reviews.")
    return df

def setup_chroma_db(reviews_list):
    """Initializes ChromaDB, creates the collection, and stores/retrieves embeddings."""
    print("Connecting to ChromaDB and initializing collection...")
    
    # Set up a persistent Chroma vector database locally
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Define the embedding function utilizing OpenAI's model natively in Chroma
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    
    # Get or create the collection
    collection = chroma_client.get_or_create_collection(
        name="clothing_reviews", 
        embedding_function=openai_ef
    )
    
    if collection.count() == 0:
        print("Database is empty. Generating and storing embeddings in ChromaDB...\n(This might take a moment based on the amount of documents).")
        batch_size = 500
        for i in range(0, len(reviews_list), batch_size):
            batch_docs = reviews_list[i:i+batch_size]
            batch_ids = [str(j) for j in range(i, i+len(batch_docs))]
            
            collection.add(
                documents=batch_docs,
                ids=batch_ids
            )
            print(f"Stored {i+len(batch_docs)} / {len(reviews_list)} reviews.")
            
        print("Finished storing all review embeddings in ChromaDB.")
    else:
        print("Embeddings previously generated and found in local ChromaDB cache. Loading...")

    # Fetch all stored embeddings to utilize them for PCA and the categorization logic
    db_data = collection.get(include=['embeddings', 'documents'])
    sorted_pairs = sorted(zip(db_data['ids'], db_data['embeddings']), key=lambda x: int(x[0]))
    
    embeddings = [pair[1] for pair in sorted_pairs]
    
    return collection, embeddings

def visualize_with_pca(embeddings, output_file="reviews_2d_plot.png"):
    """Reduces the embeddings to 2D using PCA and plots them in a scatter plot."""
    print("Reducing dimensionality to 2D using PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(np.array(embeddings))

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, color='dodgerblue')
    plt.title("2D Visualization of Customer Reviews (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    plt.savefig(output_file)
    print(f"Plot saved successfully as '{output_file}'")
    plt.show()

def categorize_feedback(df, review_embeddings):
    """Categorizes reviews into specific themes based on semantic similarity."""
    print("\nCategorizing feedback via Semantic Similarity...")
    categories = ['quality', 'fit', 'style', 'comfort']
    
    # Embed the specific category words to evaluate against the database
    cat_response = client.embeddings.create(input=categories, model="text-embedding-3-small")
    category_embeddings = [data.embedding for data in cat_response.data]

    # Calculate distance and link each snippet to a category
    similarity_matrix = cosine_similarity(review_embeddings, category_embeddings)
    df['Category'] = [categories[np.argmax(sim)] for sim in similarity_matrix]

    for cat in categories:
        print(f"\n--- Examples of reviews discussing '{cat.upper()}' ---")
        examples = df[df['Category'] == cat]['Review Text'].head(2).tolist()
        for ex in examples:
            print(f" - {ex}")

def find_similar_reviews_chroma(collection, target_review, top_k=3):
    """Perform a similarity search efficiently utilizing ChromaDB's native functions."""
    print(f"\nSearching ChromaDB for top {top_k} reviews similar to: '{target_review}'")
    
    results = collection.query(
        query_texts=[target_review],
        n_results=top_k + 1 # +1 in case the target is exactly inside our dataset
    )
    
    closest_reviews = []
    # results['documents'][0] targets the first batch result strings directly
    for doc in results['documents'][0]:
        if doc != target_review:
            closest_reviews.append(doc)
        if len(closest_reviews) == top_k:
            break
            
    return closest_reviews

def main():
    # 1. Load Data
    data_path = 'data/womens_clothing_e-commerce_reviews.csv'
    df = load_data(data_path)
    reviews_list = df['Review Text'].tolist()
    
    # 2. Setup Vector Database (ChromaDB)
    collection, embeddings = setup_chroma_db(reviews_list)
    
    # 3. PCA Dimensionality Reduction
    visualize_with_pca(embeddings)
    
    # 4. Feedback Categorization
    categorize_feedback(df, embeddings)
    
    # 5. ChromaDB Native Similarity Search
    target_review_text = "Absolutely wonderful - silky and sexy and comfortable"
    most_similar_reviews = find_similar_reviews_chroma(collection, target_review_text, top_k=3)
    
    print("\nMost Similar Reviews Found:")
    for i, review in enumerate(most_similar_reviews, 1):
        print(f"{i}. {review}")

if __name__ == "__main__":
    main()
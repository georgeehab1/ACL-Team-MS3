import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import os
import numpy as np

# --- 1. Configuration & Setup ---

def get_config(config_file="config.txt"):
    config = {}
    try:
        with open(config_file, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key] = value.strip()
        return config
    except FileNotFoundError:
        print("Config file not found. Using defaults.")
        return {}

config = get_config()
NEO4J_URI = config.get("URI", "neo4j://localhost:7687")
NEO4J_USER = config.get("USERNAME", "neo4j")
NEO4J_PASSWORD = config.get("PASSWORD", "your_password")

# --- 2. Data Preparation ---

def load_and_prepare_data():
    print("Loading datasets...")
    
    # Define paths (checking common locations)
    base_paths = ["Dataset", "Milestone2/Dataset", "."]
    hotels_path = None
    reviews_path = None
    
    for bp in base_paths:
        h_p = os.path.join(bp, "hotels.csv")
        r_p = os.path.join(bp, "reviews.csv")
        if os.path.exists(h_p) and os.path.exists(r_p):
            hotels_path = h_p
            reviews_path = r_p
            break
            
    if not hotels_path or not reviews_path:
        print("Error: Dataset files (hotels.csv, reviews.csv) not found.")
        exit(1)

    # Read CSVs
    df_hotels = pd.read_csv(hotels_path)
    df_reviews = pd.read_csv(reviews_path)

    # Merge to get Hotel Name linked to the Review
    # We keep review details and add hotel metadata
    print("Merging data...")
    df_merged = pd.merge(df_reviews, df_hotels[['hotel_id', 'hotel_name', 'city', 'country']], on='hotel_id', how='left')

    # Create the "Combined Feature Vector" as text
    # Structure: Hotel Name + Location + Scores + Review Text
    print("Creating combined feature text...")
    
    def create_feature_text(row):
        # Handle potential missing values
        name = str(row['hotel_name'])
        city = str(row['city'])
        country = str(row['country'])
        text = str(row['review_text'])
        
        # Format scores as a textual description of amenities/quality
        scores = (
            f"Overall Score: {row['score_overall']}, "
            f"Cleanliness: {row['score_cleanliness']}, "
            f"Comfort: {row['score_comfort']}, "
            f"Facilities: {row['score_facilities']}, "
            f"Location Rating: {row['score_location']}, "
            f"Staff: {row['score_staff']}, "
            f"Value: {row['score_value_for_money']}"
        )
        
        # Combine into one semantic block
        return f"Hotel: {name}. Located in: {city}, {country}. Ratings: {scores}. Review: {text}"

    df_merged['combined_features'] = df_merged.apply(create_feature_text, axis=1)
    
    print(f"Prepared {len(df_merged)} records for embedding.")
    return df_merged

# --- 3. Embedding Generation ---

def generate_embeddings(df):
    # [cite_start]Model 1: all-MiniLM-L6-v2 (Fast, Good for general tasks) [cite: 319]
    print("\nLoading Model 1: sentence-transformers/all-MiniLM-L6-v2 ...")
    model_1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("Generating Embeddings (Model 1)...")
    # Encode in batches is handled automatically by encode, but we can do it explicitly if needed
    embeddings_1 = model_1.encode(df['combined_features'].tolist(), show_progress_bar=True)
    
    # Model 2: all-mpnet-base-v2 (Higher quality, slower) - Requested comparison
    print("\nLoading Model 2: sentence-transformers/all-mpnet-base-v2 ...")
    model_2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    print("Generating Embeddings (Model 2)...")
    embeddings_2 = model_2.encode(df['combined_features'].tolist(), show_progress_bar=True)
    
    return embeddings_1, embeddings_2

# --- 4. Store in Neo4j ---

def update_neo4j(df, emb1, emb2):
    print("\nConnecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    query = """
    UNWIND $rows AS row
    MATCH (r:Review {review_id: row.review_id})
    SET r.embedding_minilm = row.emb1,
        r.embedding_mpnet = row.emb2,
        r.combined_text = row.text
    """
    
    batch_size = 500
    total = len(df)
    
    with driver.session() as session:
        # Check if Vector Indexes exist, if not create them (Optional but good practice)
        # We assume dimensions: MiniLM=384, MPNet=768
        try:
            session.run("CREATE VECTOR INDEX review_embedding_minilm IF NOT EXISTS FOR (r:Review) ON (r.embedding_minilm) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}")
            session.run("CREATE VECTOR INDEX review_embedding_mpnet IF NOT EXISTS FOR (r:Review) ON (r.embedding_mpnet) OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}")
            print("Vector indexes ensured.")
        except Exception as e:
            print(f"Note: Could not create indexes automatically (might already exist or permission issue): {e}")

        print("Writing embeddings to Neo4j...")
        
        batch_data = []
        for i, row in df.iterrows():
            batch_data.append({
                "review_id": row['review_id'],
                "text": row['combined_features'],
                "emb1": emb1[i].tolist(), # Convert numpy array to list
                "emb2": emb2[i].tolist()
            })
            
            if len(batch_data) >= batch_size:
                session.run(query, rows=batch_data)
                print(f"Updated {min(i + 1, total)}/{total} reviews...")
                batch_data = []
        
        # Final batch
        if batch_data:
            session.run(query, rows=batch_data)
            print(f"Updated {total}/{total} reviews.")

    driver.close()
    print("Done!")

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Prepare Data
    df = load_and_prepare_data()
    
    # 2. Generate Embeddings
    emb1, emb2 = generate_embeddings(df)
    
    # 3. Upload to Database
    update_neo4j(df, emb1, emb2)
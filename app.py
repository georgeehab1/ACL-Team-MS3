import streamlit as st
import pandas as pd
import os
import re
import time
from neo4j import GraphDatabase
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM
from pydantic import Field
from huggingface_hub import InferenceClient

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
        return {}

config = get_config()
NEO4J_URI = config.get("URI", "neo4j://localhost:7687")
NEO4J_USER = config.get("USERNAME", "neo4j")
NEO4J_PASSWORD = config.get("PASSWORD", "your_password")

# ⚠️ REPLACE WITH YOUR ACTUAL TOKEN
HF_TOKEN = "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" 

# --- 2. Database Connection ---

def get_driver():
    try:
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        return None

def run_cypher(query, params=None):
    driver = get_driver()
    if not driver: return []
    try:
        with driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    except Exception as e:
        return []
    finally:
        driver.close()

# --- 3. Database Initialization ---

def initialize_database():
    """Reads CSVs and populates Neo4j directly from Streamlit."""
    driver = get_driver()
    if not driver: return False, "Could not connect to Neo4j."
    
    # Define Paths
    base_paths = ["Dataset", "Milestone2/Dataset", "."]
    hotels_path = None
    for bp in base_paths:
        if os.path.exists(os.path.join(bp, "hotels.csv")):
            hotels_path = os.path.join(bp, "hotels.csv")
            reviews_path = os.path.join(bp, "reviews.csv")
            break
            
    if not hotels_path:
        return False, "Dataset files not found. Check your folder structure."

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with driver.session() as session:
            # 1. Clear DB
            status_text.text("Clearing Database...")
            session.run("MATCH (n) DETACH DELETE n")
            progress_bar.progress(10)
            
            # 2. Constraints
            status_text.text("Creating Constraints...")
            constraints = [
                "CREATE CONSTRAINT FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE",
                "CREATE CONSTRAINT FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
                "CREATE CONSTRAINT FOR (c:City) REQUIRE c.name IS UNIQUE"
            ]
            for q in constraints:
                try: session.run(q)
                except: pass
            progress_bar.progress(20)

            # 3. Load Hotels
            status_text.text("Loading Hotels...")
            df = pd.read_csv(hotels_path)
            query = """
            UNWIND $rows AS row
            MERGE (c:Country {name: row.country})
            MERGE (ci:City {name: row.city})
            MERGE (ci)-[:LOCATED_IN]->(c)
            MERGE (h:Hotel {hotel_id: row.hotel_id})
            ON CREATE SET 
                h.name = row.hotel_name,
                h.star_rating = toInteger(row.star_rating),
                h.average_reviews_score = toFloat(0.0)
            MERGE (h)-[:LOCATED_IN]->(ci)
            """
            session.run(query, rows=df.to_dict('records'))
            progress_bar.progress(50)

            # 4. Load Reviews (Calculate Average Ratings)
            status_text.text("Processing Reviews...")
            df_r = pd.read_csv(reviews_path)
            avg_ratings = df_r.groupby('hotel_id')['score_overall'].mean().reset_index()
            
            query_update = """
            UNWIND $rows AS row
            MATCH (h:Hotel {hotel_id: row.hotel_id})
            SET h.average_reviews_score = row.score_overall
            """
            session.run(query_update, rows=avg_ratings.to_dict('records'))
            progress_bar.progress(100)
            
        status_text.text("Database Initialized Successfully!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        return True, "Database populated!"
        
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        driver.close()

# --- 4. RAG Components (Baseline Only) ---

def extract_entities(user_query):
    match = re.search(r"\bin\s+([A-Za-z\s]+?)(?:\?|$|\s(?:please|show|give))", user_query, re.IGNORECASE)
    if match: return {"city": match.group(1).strip().title()}
    return {}

def get_hotels_in_city(city):
    """
    Baseline Query: Exact match search in Neo4j.
    """
    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
    WHERE toLower(c.name) CONTAINS toLower($city)
    RETURN h.name as Name, h.star_rating as Stars, h.average_reviews_score as Rating
    ORDER BY h.average_reviews_score DESC LIMIT 5
    """                                                     
    return run_cypher(query, {"city": city})

def get_db_stats():
    try:
        counts = run_cypher("MATCH (h:Hotel) RETURN count(h) as hotels")[0]['hotels']
        return counts
    except: return 0

# --- 5. LLM Wrapper ---

class GemmaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500
    @property
    def _llm_type(self) -> str: return "gemma_hf_api"
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message["content"]

if HF_TOKEN.startswith("hf_"):
    client = InferenceClient(model="google/gemma-2-2b-it", token=HF_TOKEN)
    llm = GemmaLangChainWrapper(client=client)
else:
    llm = None

def generate_response(user_query, context):
    if not llm: return "LLM not initialized."
    prompt = f"""
    You are a Hotel Recommendation Assistant.
    
    Context (from Knowledge Graph):
    {context}
    
    User Question: {user_query}
    
    Task: Answer based ONLY on the context. If context is empty, say no hotels found.
    """
    return llm.invoke(prompt)

# --- 6. Main UI ---

def main():
    st.set_page_config(layout="wide", page_title="Graph-RAG Assistant (Baseline)")
    st.title("🏨 Milestone 3: Hotel Recommender (Baseline)")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Using Strategy: **Baseline (Cypher Only)**")
        
        st.divider()
        st.header("🔧 Database")
        hotel_count = get_db_stats()
        st.metric("Hotels in Neo4j", hotel_count)
        
        if st.button("🔴 Initialize Database"):
            success, msg = initialize_database()
            if success: st.success(msg)
            else: st.error(msg)
            time.sleep(2)
            st.rerun()

    # Chat Interface
    user_input = st.text_input("Ask me:", "Show me top rated hotels in Paris")
    
    if st.button("Search"):
        entities = extract_entities(user_input)
        city = entities.get("city")
        context_str = ""
        
        if not city:
            st.warning("⚠️ Could not detect a city name. Try 'Hotels in Paris'.")
        else:
            st.info(f"🔎 Searching for hotels in **{city}** via Knowledge Graph...")
            
            # Baseline Retrieval
            cypher_res = get_hotels_in_city(city)
            if cypher_res:
                context_str += "--- Structured Graph Data ---\n"
                context_str += "\n".join([f"- {r['Name']} ({r['Stars']}⭐, Rating: {r['Rating']:.1f})" for r in cypher_res])
            
            # Generation
            if context_str:
                with st.expander("View Retrieved Context"):
                    st.text(context_str)
                with st.spinner("Generating Response..."):
                    answer = generate_response(user_input, context_str)
                    st.markdown("### 🤖 Answer:")
                    st.write(answer)
            else:
                st.error(f"No hotels found in {city} using the Knowledge Graph.")

if __name__ == "__main__":
    main()
import streamlit as st
from neo4j import GraphDatabase
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM  # Updated Import
from pydantic import Field
from huggingface_hub import InferenceClient

# --- 1. Configuration & Credentials ---
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
        st.error(f"Error: {config_file} not found.")
        return {}

config = get_config()
NEO4J_URI = config.get("URI", "neo4j://localhost:7687")
NEO4J_USER = config.get("USERNAME", "neo4j")
NEO4J_PASSWORD = config.get("PASSWORD", "your_password")

# ⚠️ REPLACE THIS WITH YOUR ACTUAL HUGGING FACE TOKEN
HF_TOKEN = "hf_..." 

# --- 2. Neo4j Connection ---
# We use the driver to execute Cypher queries directly for the Baseline Retrieval
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_cypher(query, params=None):
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]

# --- 3. Custom LLM Wrapper (Lab 8 Style) ---
class GemmaLangChainWrapper(LLM):
    """
    Custom wrapper to use Google's Gemma model via Hugging Face Inference API.
    Adapted from Lab 8 Manual.
    """
    client: Any = Field(...)
    max_tokens: int = 500
    
    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Call the Hugging Face Inference API
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2, # Low temperature for more factual answers
        )
        return response.choices[0].message["content"]

# Initialize Client and Model
if HF_TOKEN.startswith("hf_"):
    client = InferenceClient(model="google/gemma-2-2b-it", token=HF_TOKEN)
    llm = GemmaLangChainWrapper(client=client)
else:
    llm = None
    st.sidebar.warning("⚠️ Please set a valid HF_TOKEN in the code to use the LLM.")

# --- 4. RAG Components (Milestone 3 Implementation) ---

# A. Intent & Entity Extraction (Step 1: Simple Rule-Based)
def extract_entities(user_query):
    """
    Basic entity extraction for the 'Hotel Search' intent.
    Example: "Hotels in London" -> detects 'London'
    """
    # Simple logic: look for "in [City]"
    if "in " in user_query:
        # Split by "in ", take the second part, remove extra spaces/punctuation
        parts = user_query.split("in ")
        if len(parts) > 1:
            possible_city = parts[1].strip().rstrip("?.!")
            return {"city": possible_city.title()}
    return {}

# B. Retrieval Layer (Baseline Query #1)
def get_hotels_in_city(city):
    """
    Query 1: Find top 5 hotels in a specific city with their details.
    """
    cypher_query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
    WHERE toLower(c.name) CONTAINS toLower($city)
    RETURN h.name as Name, h.star_rating as Stars, h.average_reviews_score as Rating
    ORDER BY h.average_reviews_score DESC
    LIMIT 5
    """
    # Using CONTAINS and toLower for more robust matching
    return run_cypher(cypher_query, {"city": city})

# C. Generation Layer (Structured Prompt)
def generate_response(user_query, context):
    if not llm:
        return "Error: LLM not initialized. Please check your HF Token."
        
    prompt = f"""
    You are a helpful Travel Assistant specialized in Hotel Recommendations.
    
    Context (Retrieved from Knowledge Graph):
    {context}
    
    User Question: {user_query}
    
    Task: Answer the user's question using ONLY the provided context. 
    If the context is empty, politely say you couldn't find any hotels in that city in your database.
    Format your answer nicely.
    """
    return llm.invoke(prompt)

# --- 5. Main Application Interface (Streamlit) ---
def main():
    st.set_page_config(page_title="Graph-RAG Travel Assistant", layout="wide")
    st.title("🏨 Milestone 3: Hotel Recommender System")
    st.markdown("This system uses a **Neo4j Knowledge Graph** and **Gemma LLM** to recommend hotels.")
    
    # Sidebar for debug info
    with st.sidebar:
        st.header("System Status")
        if driver:
            st.success("Neo4j Connected")
        else:
            st.error("Neo4j Disconnected")
        if llm:
            st.success("LLM Initialized")
    
    # User Input
    user_input = st.text_input("How can I help you?", placeholder="e.g., Show me top rated hotels in London")
    
    if st.button("Submit"):
        if not user_input:
            st.warning("Please enter a question.")
            return

        # 1. Processing
        with st.spinner("Analyzing request..."):
            entities = extract_entities(user_input)
            city = entities.get("city")
        
        # 2. Logic Flow
        if city:
            st.success(f"📍 Intent Detected: Search Hotels in **{city}**")
            
            # 3. Retrieval (Baseline)
            with st.spinner("Fetching data from Knowledge Graph..."):
                results = get_hotels_in_city(city)
            
            # 4. Augmentation & Generation
            if results:
                # Format context for the LLM
                context_str = ""
                for r in results:
                    context_str += f"- Hotel: {r['Name']} | Stars: {r['Stars']} | Rating: {r['Rating']}/10\n"
                
                # Show Retrieved Context (Transparent AI)
                with st.expander("View Retrieved Context (Graph Data)"):
                    st.text(context_str)
                
                # Generate Answer
                with st.spinner("Generating recommendation..."):
                    answer = generate_response(user_input, context_str)
                    st.markdown("### 🤖 Assistant's Recommendation")
                    st.write(answer)
            else:
                st.warning(f"No hotels found in '{city}' in the Knowledge Graph.")
        else:
            st.error("Could not detect a city name. Please try phrases like 'Hotels in [City]'.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import os
import re
import time
import networkx as nx
import plotly.graph_objects as go
from neo4j import GraphDatabase
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM
from pydantic import Field
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

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

# Defined Models
LLM_MODELS = {
    "Gemma (2B)": "google/gemma-2-2b-it",
    "Mistral (7B v0.2)": "mistralai/Mistral-7B-Instruct-v0.2", 
    "Llama 3 (8B)": "meta-llama/Meta-Llama-3-8B-Instruct" 
}

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

def execute_batch(session, query, data, batch_size=1000):
    total = len(data)
    for i in range(0, total, batch_size):
        batch = data[i:i + batch_size]
        session.run(query, rows=batch)

def initialize_database():
    """Reads CSVs and populates the FULL Knowledge Graph."""
    driver = get_driver()
    if not driver: return False, "Could not connect to Neo4j."
    
    # Define Paths
    base_paths = ["Dataset", "Milestone2/Dataset", "."]
    files = {"hotels": None, "users": None, "reviews": None, "visa": None}
    
    for key in files.keys():
        for bp in base_paths:
            fpath = os.path.join(bp, f"{key}.csv")
            if os.path.exists(fpath):
                files[key] = fpath
                break
    
    if not all(files.values()):
        return False, f"Missing CSV files. Found: {files}"

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with driver.session() as session:
            # 1. Clear DB
            status_text.text("⚠️ Clearing Database...")
            session.run("MATCH (n) DETACH DELETE n")
            progress_bar.progress(5)
            
            # 2. Constraints
            status_text.text("Creating Constraints...")
            constraints = [
                "CREATE CONSTRAINT FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE",
                "CREATE CONSTRAINT FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
                "CREATE CONSTRAINT FOR (r:Review) REQUIRE r.review_id IS UNIQUE",
                "CREATE CONSTRAINT FOR (c:City) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT FOR (k:Country) REQUIRE k.name IS UNIQUE"
            ]
            for q in constraints:
                try: session.run(q)
                except: pass
            progress_bar.progress(10)

            # 3. Load Hotels
            status_text.text("Loading Hotels & Locations...")
            df_h = pd.read_csv(files["hotels"])
            df_h['city'] = df_h['city'].astype(str).str.strip()
            df_h['country'] = df_h['country'].astype(str).str.strip()
            
            query_hotels = """
            UNWIND $rows AS row
            MERGE (c:Country {name: row.country})
            MERGE (ci:City {name: row.city})
            MERGE (ci)-[:LOCATED_IN]->(c)
            MERGE (h:Hotel {hotel_id: row.hotel_id})
            SET 
                h.name = row.hotel_name,
                h.star_rating = toInteger(row.star_rating),
                h.cleanliness_base = toFloat(row.cleanliness_base),
                h.value_for_money_base = toFloat(row.value_for_money_base),
                h.average_reviews_score = toFloat(0.0)
            MERGE (h)-[:LOCATED_IN]->(ci)
            """
            execute_batch(session, query_hotels, df_h.to_dict('records'))
            progress_bar.progress(30)

            # 4. Load Travellers
            status_text.text("Loading Travellers...")
            df_u = pd.read_csv(files["users"])
            df_u['age_group'] = df_u['age_group'].astype(str).str.strip()
            df_u['traveller_type'] = df_u['traveller_type'].astype(str).str.strip()
            df_u['user_gender'] = df_u['user_gender'].astype(str).str.strip()
            
            query_users = """
            UNWIND $rows AS row
            MERGE (t:Traveller {user_id: row.user_id})
            SET 
                t.age = row.age_group,
                t.type = row.traveller_type,
                t.gender = row.user_gender
            """
            execute_batch(session, query_users, df_u.to_dict('records'))
            progress_bar.progress(50)

            # 5. Load Reviews
            status_text.text("Loading Reviews...")
            df_r = pd.read_csv(files["reviews"])
            query_reviews = """
            UNWIND $rows AS row
            MATCH (t:Traveller {user_id: row.user_id})
            MATCH (h:Hotel {hotel_id: row.hotel_id})
            MERGE (r:Review {review_id: row.review_id})
            SET 
                r.score_overall = toFloat(row.score_overall),
                r.text = row.review_text
            MERGE (t)-[:WROTE]->(r)
            MERGE (r)-[:REVIEWED]->(h)
            """
            execute_batch(session, query_reviews, df_r.to_dict('records'))
            progress_bar.progress(70)

            # 6. Load Visa Data
            status_text.text("Loading Visa Requirements...")
            df_v = pd.read_csv(files["visa"])
            df_v = df_v.rename(columns={'from': 'from_country', 'to': 'to_country'})
            
            df_v['from_country'] = df_v['from_country'].astype(str).str.strip()
            df_v['to_country'] = df_v['to_country'].astype(str).str.strip()
            
            requires = df_v[df_v['requires_visa'].isin(['Yes', '1', 'True'])].copy()
            
            query_visa = """
            UNWIND $rows AS row
            MATCH (origin:Country {name: row.from_country})
            MATCH (dest:Country {name: row.to_country})
            MERGE (origin)-[v:NEEDS_VISA]->(dest)
            SET 
                v.visa_type = row.visa_type,
                v.requires_visa = row.requires_visa
            """
            execute_batch(session, query_visa, requires.to_dict('records'))
            progress_bar.progress(90)

            # 7. Update Stats
            session.run("""
                MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)
                WITH h, avg(r.score_overall) AS avg_score
                SET h.average_reviews_score = avg_score
            """)
            progress_bar.progress(100)
            
        status_text.text("Database Fully Initialized!")
        time.sleep(2)
        status_text.empty()
        progress_bar.empty()
        return True, "Database populated!"
        
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        driver.close()

# --- 4. RAG Components ---

def extract_entities(user_query):
    entities = {}
    
    # 1. Intent
    if re.search(r"\b(clean|cleanest|cleanliness|hygiene|tidy)\b", user_query, re.IGNORECASE):
        entities["intent"] = "cleanliness_ranking"
    elif re.search(r"\b(value|money|cheap|budget|affordable|price|cost)\b", user_query, re.IGNORECASE):
        entities["intent"] = "value_ranking"
    elif re.search(r"\b(visa|passport|requirements|permit)\b", user_query, re.IGNORECASE):
        entities["intent"] = "visa_check"
        from_to = re.search(r"from\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+)", user_query, re.IGNORECASE)
        if from_to:
            entities["origin_country"] = from_to.group(1).strip().title()
            entities["dest_country"] = from_to.group(2).strip().title()
    elif re.search(r"\b(country|nation|countries|destination)\b", user_query, re.IGNORECASE):
        entities["intent"] = "country_recommendation"
    else:
        entities["intent"] = "hotel_recommendation"

    # 2. Location
    if "location" not in entities:
        loc_match = re.search(r"\bin\s+([A-Za-z\s]+?)(?:\?|$|\s(?:please|show|give|for|with))", user_query, re.IGNORECASE)
        if loc_match: 
            loc_candidate = loc_match.group(1).strip().title()
            if loc_candidate.lower() not in ['families', 'couples', 'business', 'solo', 'clean', 'cheap']:
                entities["location"] = loc_candidate
        
    # 3. Traveller Type
    type_patterns = {
        r"\b(family|families|kid|kids|child|children)\b": "Family",
        r"\b(solo|single|alone)\b": "Solo",
        r"\b(couple|couples|pair|partner|partners)\b": "Couple",
        r"\b(business|work|corporate|official)\b": "Business"
    }
    for pattern, val in type_patterns.items():
        if re.search(pattern, user_query, re.IGNORECASE):
            entities["traveller_type"] = val
            break

    # 4. Gender
    gender_patterns = {
        r"\b(male|man|men|guy|guys)\b": "Male",
        r"\b(female|woman|women|girl|girls|lady|ladies)\b": "Female"
    }
    for pattern, val in gender_patterns.items():
        if re.search(pattern, user_query, re.IGNORECASE):
            entities["gender"] = val
            break

    # 5. Age
    age_patterns = {
        r"18-24|18 to 24|young adult|student": "18-24",
        r"25-34|25 to 34": "25-34",
        r"35-44|35 to 44": "35-44",
        r"45-54|45 to 54": "45-54",
        r"55\+|55 plus|senior|elder|retiree": "55+"
    }
    for pattern, val in age_patterns.items():
        if re.search(pattern, user_query, re.IGNORECASE):
            entities["age"] = val
            break
        
    return entities

# --- Cypher Queries ---

def get_hotels_in_city(city):
    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
    WHERE toLower(c.name) CONTAINS toLower($city)
    RETURN h.name as Name, h.star_rating as Stars, h.average_reviews_score as Rating
    ORDER BY h.average_reviews_score DESC LIMIT 5
    """                                                     
    return run_cypher(query, {"city": city}), query

def get_hotels_in_country(country):
    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(ci:City)-[:LOCATED_IN]->(c:Country)
    WHERE toLower(c.name) CONTAINS toLower($country)
    RETURN h.name as Name, h.star_rating as Stars, h.average_reviews_score as Rating, ci.name as City
    ORDER BY h.average_reviews_score DESC LIMIT 5
    """
    return run_cypher(query, {"country": country}), query

def get_top_hotels_by_traveller_type(traveller_type):
    query = """
    MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
    WHERE t.type = $type
    RETURN h.name as Name, avg(r.score_overall) as Rating, count(r) as NumReviews
    ORDER BY Rating DESC LIMIT 5
    """
    return run_cypher(query, {"type": traveller_type}), query

def get_best_country_by_traveller_type(traveller_type):
    query = """
    MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(c:Country)
    WHERE t.type = $type
    RETURN c.name as Country, avg(r.score_overall) as Rating, count(r) as NumReviews
    ORDER BY Rating DESC LIMIT 5
    """
    return run_cypher(query, {"type": traveller_type}), query

def get_top_hotels_by_gender(gender):
    query = """
    MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
    WHERE toLower(t.gender) = toLower($gender)
    RETURN h.name as Name, avg(r.score_overall) as Rating, count(r) as NumReviews
    ORDER BY Rating DESC LIMIT 5
    """
    return run_cypher(query, {"gender": gender}), query

def get_best_country_by_gender(gender):
    query = """
    MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(c:Country)
    WHERE toLower(t.gender) = toLower($gender)
    RETURN c.name as Country, avg(r.score_overall) as Rating, count(r) as NumReviews
    ORDER BY Rating DESC LIMIT 5
    """
    return run_cypher(query, {"gender": gender}), query

def check_visa_requirements(origin, destination):
    query = """
    MATCH (o:Country) WHERE toLower(o.name) = toLower($origin)
    MATCH (d:Country) WHERE toLower(d.name) = toLower($dest)
    OPTIONAL MATCH (o)-[v:NEEDS_VISA]->(d)
    RETURN o.name as Origin, d.name as Destination, v.visa_type as VisaType, v.requires_visa as Required
    """
    return run_cypher(query, {"origin": origin, "dest": destination}), query

def get_best_hotels_by_cleanliness():
    query = """
    MATCH (h:Hotel)
    RETURN h.name as Name, h.cleanliness_base as Score, h.star_rating as Stars
    ORDER BY Score DESC LIMIT 5
    """
    return run_cypher(query), query

def get_best_countries_by_cleanliness():
    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(c:Country)
    RETURN c.name as Country, avg(h.cleanliness_base) as Score, count(h) as NumHotels
    ORDER BY Score DESC LIMIT 5
    """
    return run_cypher(query), query

def get_best_hotels_by_value():
    query = """
    MATCH (h:Hotel)
    RETURN h.name as Name, h.value_for_money_base as Score, h.star_rating as Stars
    ORDER BY Score DESC LIMIT 5
    """
    return run_cypher(query), query

def get_top_hotels_by_age(age):
    query = """
    MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
    WHERE t.age = $age
    RETURN h.name as Name, avg(r.score_overall) as Rating, count(r) as NumReviews
    ORDER BY Rating DESC LIMIT 5
    """
    return run_cypher(query, {"age": age}), query

def get_db_stats():
    try:
        counts = run_cypher("MATCH (n) RETURN labels(n) as label, count(*) as count")
        stats = {row['label'][0]: row['count'] for row in counts}
        return stats
    except: return {}

# --- GRAPH VISUALIZATION DATA ---

def get_graph_data(entities, intent):
    """Fetches nodes and relationships for visualization based on intent"""
    
    query = ""
    params = {}
    
    # 1. Visa: Show Countries and Link
    if intent == "visa_check" and entities.get("origin_country") and entities.get("dest_country"):
        query = """
        MATCH (o:Country)-[r:NEEDS_VISA]->(d:Country)
        WHERE toLower(o.name) = toLower($origin) AND toLower(d.name) = toLower($dest)
        RETURN o.name as source, d.name as target, "NEEDS_VISA" as type, "Country" as source_label, "Country" as target_label
        """
        params = {"origin": entities["origin_country"], "dest": entities["dest_country"]}
        
    # 2. Location: Hotel -> City -> Country
    elif entities.get("location"):
        query = """
        MATCH (h:Hotel)-[r:LOCATED_IN]->(c:City)
        WHERE toLower(c.name) CONTAINS toLower($loc) OR toLower(h.name) CONTAINS toLower($loc)
        RETURN h.name as source, c.name as target, "LOCATED_IN" as type, "Hotel" as source_label, "City" as target_label
        LIMIT 10
        UNION
        MATCH (c:City)-[r:LOCATED_IN]->(cnt:Country)
        WHERE toLower(c.name) CONTAINS toLower($loc)
        RETURN c.name as source, cnt.name as target, "LOCATED_IN" as type, "City" as source_label, "Country" as target_label
        LIMIT 10
        """
        params = {"loc": entities["location"]}
        
    # 3. Traveller Type: Traveller -> Review -> Hotel
    elif entities.get("traveller_type"):
        query = """
        MATCH (t:Traveller)-[w:WROTE]->(r:Review)-[rev:REVIEWED]->(h:Hotel)
        WHERE t.type = $type
        RETURN t.user_id as source, r.review_id as target, "WROTE" as type, "Traveller" as source_label, "Review" as target_label
        LIMIT 5
        UNION
        MATCH (t:Traveller)-[w:WROTE]->(r:Review)-[rev:REVIEWED]->(h:Hotel)
        WHERE t.type = $type
        RETURN r.review_id as source, h.name as target, "REVIEWED" as type, "Review" as source_label, "Hotel" as target_label
        LIMIT 5
        """
        params = {"type": entities["traveller_type"]}
        
    # 4. Default: Just some Hotels and Cities
    else:
        query = """
        MATCH (h:Hotel)-[r:LOCATED_IN]->(c:City)
        RETURN h.name as source, c.name as target, "LOCATED_IN" as type, "Hotel" as source_label, "City" as target_label
        LIMIT 10
        """
        
    return run_cypher(query, params)

def render_graph(data):
    """Renders the graph using NetworkX and Plotly"""
    if not data:
        st.warning("No graph data to visualize.")
        return

    G = nx.Graph()
    
    # Build Graph
    for row in data:
        G.add_node(row['source'], label=row['source_label'])
        G.add_node(row['target'], label=row['target_label'])
        G.add_edge(row['source'], row['target'], label=row['type'])
        
    # Layout
    pos = nx.spring_layout(G)
    
    # Edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    color_map = {"Hotel": "blue", "City": "green", "Country": "orange", "Traveller": "red", "Review": "purple"}
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        lbl = G.nodes[node]['label']
        node_text.append(f"{lbl}: {node}")
        node_color.append(color_map.get(lbl, "gray"))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=2),
        text=node_text)

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    st.plotly_chart(fig, use_container_width=True)


# --- VECTOR RETRIEVAL ---

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

def get_similar_reviews(user_query, model_choice, k=3):
    driver = get_driver()
    if not driver: return []
    
    if model_choice == "all-MiniLM-L6-v2":
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        index_name = "review_embedding_minilm"
    else:
        model_id = "sentence-transformers/all-mpnet-base-v2"
        index_name = "review_embedding_mpnet"

    try:
        model = load_embedding_model(model_id)
        query_vector = model.encode(user_query).tolist()
        fetch_k = k * 5
        
        query = f"""
        CALL db.index.vector.queryNodes($index_name, $fetch_k, $embedding)
        YIELD node, score
        MATCH (node)-[:REVIEWED]->(h:Hotel)-[:LOCATED_IN]->(c:City)
        RETURN h.name as Hotel, c.name as City, node.combined_text as Snippet, score
        """
        
        with driver.session() as session:
            result = session.run(query, index_name=index_name, fetch_k=fetch_k, embedding=query_vector)
            raw_results = [record.data() for record in result]
            
            unique_results = []
            seen_hotels = set()
            for r in raw_results:
                hotel_name = r['Hotel']
                if hotel_name not in seen_hotels:
                    unique_results.append(r)
                    seen_hotels.add(hotel_name)
                if len(unique_results) >= k:
                    break
            
            return unique_results
            
    except Exception as e:
        return []
    finally:
        driver.close()

# --- 5. LLM Wrapper ---

class HuggingFaceLLMWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500
    
    @property
    def _llm_type(self) -> str: return "hf_inference_api"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message["content"]

def get_llm_client(model_name_key):
    if HF_TOKEN.startswith("hf_"):
        repo_id = LLM_MODELS.get(model_name_key, "google/gemma-2-2b-it")
        client = InferenceClient(model=repo_id, token=HF_TOKEN)
        return HuggingFaceLLMWrapper(client=client)
    else:
        return None

def generate_response(llm_instance, user_query, context):
    if not llm_instance: return "LLM not initialized."
    prompt = f"""
    You are a Travel Assistant.
    
    Context (from Knowledge Graph):
    {context}
    
    User Question: {user_query}
    
    Task: Answer based ONLY on the context.
    """
    return llm_instance.invoke(prompt)

# --- 6. Main UI ---

def main():
    st.set_page_config(layout="wide", page_title="Graph-RAG Assistant")
    st.title("🏨 Milestone 3: Travel Assistant")

    with st.sidebar:
        st.header("⚙️ Settings")
        
        retrieval_mode = st.radio(
            "Retrieval Strategy:",
            ["Baseline (Cypher)", "Embeddings (Vector)", "Hybrid (Merge)"]
        )
        
        show_cypher = False
        if retrieval_mode in ["Baseline (Cypher)", "Hybrid (Merge)"]:
            show_cypher = st.checkbox("Show Cypher Query", value=True)
            
        show_graph = st.checkbox("Show Graph Visualization", value=False)
        
        embed_model_choice = "all-MiniLM-L6-v2"
        if retrieval_mode != "Baseline (Cypher)":
            embed_model_choice = st.selectbox(
                "Embedding Model (Retrieval):",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
            )
        
        st.divider()
        st.header("🤖 LLM Selection")
        llm_choice = st.selectbox("Choose Generative Model:", list(LLM_MODELS.keys()))
        
        st.divider()
        st.header("🔧 Database")
        stats = get_db_stats()
        if stats:
            st.write(f"**Hotels:** {stats.get('Hotel', 0)}")
            st.write(f"**Travellers:** {stats.get('Traveller', 0)}")
        
        if st.button("🔴 Initialize Full Database"):
            success, msg = initialize_database()
            if success: st.success(msg)
            else: st.error(msg)
            st.warning("⚠️ Clears data. Run `embedding_create.py` afterwards!")
            time.sleep(2)
            st.rerun()

    user_input = st.text_input("Ask me:", "Best hotels for families")
    
    if st.button("Search"):
        if not user_input:
            st.warning("Please enter a query.")
            return

        context_parts = []
        entities = extract_entities(user_input) # Extract entities once here
        st.info(f"🔎 Processing with **{retrieval_mode}** using **{llm_choice}**...")

        # --- A. Baseline Retrieval ---
        if retrieval_mode in ["Baseline (Cypher)", "Hybrid (Merge)"]:
            location = entities.get("location")
            traveller_type = entities.get("traveller_type")
            gender = entities.get("gender")
            age = entities.get("age")
            intent = entities.get("intent")
            origin_country = entities.get("origin_country")
            dest_country = entities.get("dest_country")
            
            cypher_res = []
            executed_query = ""
            query_type = ""

            if intent == "cleanliness_ranking":
                if re.search(r"\b(country|nation|countries)\b", user_input, re.IGNORECASE):
                    cypher_res, executed_query = get_best_countries_by_cleanliness()
                    query_type = "Best Country by Cleanliness"
                else:
                    cypher_res, executed_query = get_best_hotels_by_cleanliness()
                    query_type = "Best Cleanliness Score"
            elif intent == "value_ranking":
                cypher_res, executed_query = get_best_hotels_by_value()
                query_type = "Best Value Score"
            elif intent == "visa_check":
                if origin_country and dest_country:
                    res, executed_query = check_visa_requirements(origin_country, dest_country)
                    query_type = "Visa Check"
                    if res:
                        row = res[0]
                        if row['VisaType']:
                            context_parts.append(f"Visa Requirement: {row['VisaType']} (Required: {row['Required']})")
                        else:
                            context_parts.append(f"No visa restriction found from {row['Origin']} to {row['Destination']}.")
                    else:
                        context_parts.append("No visa data found.")
            elif age:
                cypher_res, executed_query = get_top_hotels_by_age(age)
                query_type = f"Best Hotels for Age {age}"
            elif gender:
                if intent == "country_recommendation":
                    cypher_res, executed_query = get_best_country_by_gender(gender)
                    query_type = f"Best Country for {gender}"
                else:
                    cypher_res, executed_query = get_top_hotels_by_gender(gender)
                    query_type = f"Best Hotels for {gender}"
            elif traveller_type:
                if intent == "country_recommendation":
                    cypher_res, executed_query = get_best_country_by_traveller_type(traveller_type)
                    query_type = f"Best Country for {traveller_type}"
                else:
                    cypher_res, executed_query = get_top_hotels_by_traveller_type(traveller_type)
                    query_type = f"Top Hotels for {traveller_type}"
            elif location:
                cypher_res, executed_query = get_hotels_in_city(location)
                query_type = f"Hotels in {location}"
                if not cypher_res:
                    cypher_res, executed_query = get_hotels_in_country(location)
                    query_type = f"Hotels in Country: {location}"

            if cypher_res:
                structured_txt = f"--- Structured Graph Data ({query_type}) ---\n"
                for r in cypher_res:
                    name = r.get('Name') or r.get('Country') or "Entity"
                    score = r.get('Score') or r.get('Rating') or 0
                    structured_txt += f"- {name} (Score/Rating: {score:.1f})\n"
                context_parts.append(structured_txt)
            
            if show_cypher and executed_query:
                with st.expander("📝 View Executed Cypher Query"):
                    st.code(executed_query, language='cypher')

        # --- B. Embeddings Retrieval ---
        if retrieval_mode in ["Embeddings (Vector)", "Hybrid (Merge)"]:
            vector_res = get_similar_reviews(user_input, embed_model_choice, k=3)
            if vector_res:
                vector_txt = "\n--- Semantic Review Matches ---\n"
                for r in vector_res:
                    vector_txt += f"Hotel: {r['Hotel']} ({r['City']})\nSnippet: {r['Snippet'][:200]}...\n(Similarity: {r['score']:.4f})\n\n"
                context_parts.append(vector_txt)
            elif retrieval_mode == "Embeddings (Vector)":
                st.warning("No relevant semantic matches found.")

        # --- C. Graph Visualization (New) ---
        if show_graph:
            st.markdown("### 🕸️ Knowledge Graph Subgraph")
            # Fetch graph data based on the extracted entities/intent
            graph_data = get_graph_data(entities, entities.get("intent", ""))
            render_graph(graph_data)

        # --- D. Generation ---
        full_context = "\n".join(context_parts)
        
        if full_context:
            with st.expander("View Retrieved Context"):
                st.text(full_context)
            
            with st.spinner(f"Generating Response using {llm_choice}..."):
                llm = get_llm_client(llm_choice)
                answer = generate_response(llm, user_input, full_context)
                st.markdown("### 🤖 Answer:")
                st.write(answer)
        else:
            st.error("I couldn't find any information matching your query.")

if __name__ == "__main__":
    main()
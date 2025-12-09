import pandas as pd
from neo4j import GraphDatabase, exceptions
import os

def get_db_credentials(config_file="config.txt"):
    credentials = {}
    try:
        with open(config_file, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    credentials[key] = value
        return credentials
    except FileNotFoundError:
        print(f"Error: {config_file} not found. Please ensure it exists.")
        exit(1)

config = get_db_credentials()
URI = config.get("URI", "neo4j://localhost:7687")
AUTH = (config.get("USERNAME", "neo4j"), config.get("PASSWORD", "your_password"))

driver = GraphDatabase.driver(URI, auth=AUTH)

def execute_batch(session, query, data, batch_size=1000):
    total = len(data)
    for i in range(0, total, batch_size):
        batch = data[i:i + batch_size]
        session.run(query, rows=batch)
        print(f"Processed {min(i + batch_size, total)}/{total} records...")

def create_constraints(driver):
    """
    Runs constraint creation in separate auto-commit transactions.
    If a constraint exists, it catches the error and proceeds.
    """
    print("Creating constraints...")
    queries = [
        "CREATE CONSTRAINT FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE",
        "CREATE CONSTRAINT FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
        "CREATE CONSTRAINT FOR (r:Review) REQUIRE r.review_id IS UNIQUE",
        "CREATE CONSTRAINT FOR (c:City) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT FOR (k:Country) REQUIRE k.name IS UNIQUE"
    ]

    with driver.session() as session:
        for q in queries:
            try:
                session.run(q)
            except exceptions.ClientError as e:
                if "EquivalentSchemaRuleAlreadyExists" in str(e):

                    continue 
                else:
                    raise e

def clear_database(tx):
    print("Clearing existing database (Nodes & Relationships)...")
    tx.run("MATCH (n) DETACH DELETE n")

def load_hotels(session):
    print("\n--- Loading Hotels (Nodes: Hotel, City, Country) ---")

    file_path = 'Dataset/hotels.csv' 
    if not os.path.exists(file_path): file_path = 'Milestone2/Dataset/hotels.csv'

    df = pd.read_csv(file_path)
    records = df.to_dict('records')

    query = """
    UNWIND $rows AS row
    MERGE (c:Country {name: row.country})
    MERGE (ci:City {name: row.city})
    MERGE (ci)-[:LOCATED_IN]->(c)

    MERGE (h:Hotel {hotel_id: row.hotel_id})
    ON CREATE SET 
        h.name = row.hotel_name,
        h.star_rating = toInteger(row.star_rating),
        h.cleanliness_base = toFloat(row.cleanliness_base),
        h.comfort_base = toFloat(row.comfort_base),
        h.facilities_base = toFloat(row.facilities_base),
        h.location_base = toFloat(row.location_base),
        h.staff_base = toFloat(row.staff_base),
        h.value_for_money_base = toFloat(row.value_for_money_base),
        h.average_reviews_score = toFloat(0.0)

    MERGE (h)-[:LOCATED_IN]->(ci)
    """
    execute_batch(session, query, records)

def load_travellers(session):
    print("\n--- Loading Travellers (Node: Traveller, Rel: FROM_COUNTRY) ---")
    file_path = 'Dataset/users.csv'
    if not os.path.exists(file_path): file_path = 'Milestone2/Dataset/users.csv'

    df = pd.read_csv(file_path)
    records = df.to_dict('records')

    query = """
    UNWIND $rows AS row
    MERGE (t:Traveller {user_id: row.user_id})
    ON CREATE SET 
        t.age = row.age_group,
        t.type = row.traveller_type,
        t.gender = row.user_gender

    WITH t, row
    MERGE (c:Country {name: row.country})
    MERGE (t)-[:FROM_COUNTRY]->(c)
    """
    execute_batch(session, query, records)

def load_reviews(session):
    print("\n--- Loading Reviews (Node: Review, Rels: WROTE, REVIEWED, STAYED_AT) ---")
    file_path = 'Dataset/reviews.csv'
    if not os.path.exists(file_path): file_path = 'Milestone2/Dataset/reviews.csv'

    df = pd.read_csv(file_path)
    records = df.to_dict('records')

    query = """
    UNWIND $rows AS row
    MATCH (t:Traveller {user_id: row.user_id})
    MATCH (h:Hotel {hotel_id: row.hotel_id})

    MERGE (r:Review {review_id: row.review_id})
    ON CREATE SET 
        r.text = row.review_text,
        r.date = row.review_date,
        r.score_overall = toFloat(row.score_overall),
        r.score_cleanliness = toFloat(row.score_cleanliness),
        r.score_comfort = toFloat(row.score_comfort),
        r.score_facilities = toFloat(row.score_facilities),
        r.score_location = toFloat(row.score_location),
        r.score_staff = toFloat(row.score_staff),
        r.score_value_for_money = toFloat(row.score_value_for_money)

    MERGE (t)-[:WROTE]->(r)
    MERGE (r)-[:REVIEWED]->(h)
    MERGE (t)-[:STAYED_AT]->(h)
    """
    execute_batch(session, query, records)

def load_visa(session):
    print("\n--- Loading Visa Requirements (Rel: NEEDS_VISA) ---")
    file_path = 'Dataset/visa.csv'
    if not os.path.exists(file_path): file_path = 'Milestone2/Dataset/visa.csv'

    df = pd.read_csv(file_path)
    df = df.rename(columns={'from': 'from_country', 'to': 'to_country'})
    requires = df[df['requires_visa'].isin(['Yes', '1', 'True'])].copy()
    records = requires.to_dict('records')

    query = """
    UNWIND $rows AS row
    MATCH (origin:Country {name: row.from_country})
    MATCH (dest:Country {name: row.to_country})
    MERGE (origin)-[v:NEEDS_VISA]->(dest)
    SET 
        v.visa_type = row.visa_type,
        v.requires_visa = row.requires_visa
    """
    execute_batch(session, query, records)

def main():

    create_constraints(driver)

    try:
        with driver.session() as session:
            session.execute_write(clear_database)

            load_hotels(session)
            load_travellers(session)
            load_reviews(session)
            load_visa(session)

        print("\nKnowledge Graph construction complete.")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
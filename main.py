import streamlit as st
import google.generativeai as genai
import chromadb
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="Universal IFCT 2017 RAG", layout="wide")

# 2. Secure API Validation
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.warning("⚠️ Please configure your GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()
genai.configure(api_key=api_key)

# 3. Initialize ChromaDB Vector Layer
chroma_client = chromadb.Client()
# Renamed collection to v5 to instantly force Streamlit past old broken cache frames
collection = chroma_client.get_or_create_collection(name="ifct_2017_master_v5")

# 4. INSTANT CSV INGESTION LAYER
@st.cache_resource
def load_clean_dataset():
    csv_filename = "ifct_clean.csv"
    
    if not os.path.exists(csv_filename):
        return None, False
        
    df = pd.read_csv(csv_filename)
    
    # Core vector optimization: Index rows if the database layer is blank
    if collection.count() == 0:
        documents = []
        metadatas = []
        ids = []
        
        for index, row in df.iterrows():
            # FIXED: Uses 'Food_Code' and maps directly to the master nutrient compilation string
            semantic_string = f"Food Code: {row['Food_Code']} | Food Name: {row['Food_Name']} | {row['Nutrient_Payload']}"
            documents.append(semantic_string)
            metadatas.append({"code": str(row['Food_Code']), "food": str(row['Food_Name'])})
            ids.append(f"row_{index}")
            
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
    return df, True

# Boot the app infrastructure
master_df, db_ready = load_clean_dataset()

# 5. Build UI Interface Layout
st.title("🔬 Universal IFCT 2017 Precision RAG Architecture")
st.caption("Clean Data Parsing Pipeline running over Google Gemini 2.5 Pro")

if db_ready:
    st.success(f"✅ Master Frame Online: Successfully mapped {len(master_df)} all-inclusive food asset matrices.")
else:
    st.error("📂 Database Missing: Please run your final 'parse_pdf.py' compiler script in your terminal first.")
    st.stop()

st.markdown("---")

# 6. Query Execution Block
# 6. Query Execution Block
# 6. Query Execution Block
query = st.text_input("Enter any dietary problem, nutrient deficiency, or component keyword (e.g., 'folate deficiency', 'high protein foods', 'iron'):")

if query:
    with st.spinner("Searching cross-functional vector indexes..."):
        try:
            # Query the database for the top 5 closest matched food entries
            search_results = collection.query(query_texts=[query], n_results=5)
            
            context_payload = ""
            if search_results and search_results['documents'] and search_results['documents'][0]:
                context_payload = " \n ".join(search_results['documents'][0])
                
            if not context_payload:
                st.error("No relevant nutrient data rows tracked inside the current database.")
                st.stop()
                
            # Build the structured prompt
            strict_universal_prompt = f"""
You are an expert nutritional data analyst formatting information from the ICMR-NIN IFCT 2017 data tables.
The user is searching for data regarding: '{query}'.

EXPLICIT EXTRACTED COMPONENT ROWS:
{context_payload}

Your Core Directives (Process Step-by-Step):
1. Do NOT echo or print the raw text chunks directly.
2. Extract the matching items and organize them into a clean Markdown Table with columns: Food Code, Food Name, Macronutrients, Vitamins, Minerals.
3. After creating the table, write a 'Dietary Analysis Summary' section. Look closely at the numbers in the table and logically explain which foods are objectively the highest or best suited for the user's search.
4. Keep the summary actionable by mentioning practical tips (like pairing iron with vitamin C or soaking pulses) if relevant to the matched nutrients.
5. Rely ONLY on the provided data text block. If information is missing, leave it blank in the table.
"""

            # Initialize Gemini Flash model
            model = genai.GenerativeModel("models/gemini-2.5-flash")

            # Generate response (prompt must be inside a list)
            response = model.generate_content([strict_universal_prompt])

            # Display the output
            st.markdown(response.text)

        except Exception as e:
            st.error(f"❌ Error while generating response: {e}")
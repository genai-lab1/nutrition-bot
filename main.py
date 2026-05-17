import streamlit as st
import google.generativeai as genai
import chromadb
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="Universal IFCT 2017 RAG", layout="wide")

# 2. Secure API Validation
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.warning("⚠️ Please configure your GEMINI_API_KEY in Cloud Run Environment Variables.")
    st.stop()
genai.configure(api_key=api_key)

# 3. Hybrid ChromaDB Initialization
PERSIST_DIR = "./chroma_db_data"

if os.path.exists(PERSIST_DIR):
    # Performance Mode: Load pre-built vector folder if it exists
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = chroma_client.get_collection(name="ifct_2017_master_v5")
    db_ready = True
    using_prebuilt = True
else:
    # Standard Mode: Fallback to in-memory processing if folder isn't pushed yet
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="ifct_2017_master_v5")
    db_ready = True
    using_prebuilt = False

# 4. CSV INGESTION LAYER (Runs only if standard mode is active)
@st.cache_resource
def load_and_index_dataset():
    csv_filename = "ifct_clean.csv"
    
    if not os.path.exists(csv_filename):
        return None, False
        
    df = pd.read_csv(csv_filename)
    
    # Only index row-by-row if we are in-memory and the database is empty
    if not using_prebuilt and collection.count() == 0:
        documents = []
        metadatas = []
        ids = []
        
        for index, row in df.iterrows():
            semantic_string = f"Food Code: {row['Food_Code']} | Food Name: {row['Food_Name']} | {row['Nutrient_Payload']}"
            documents.append(semantic_string)
            metadatas.append({"code": str(row['Food_Code']), "food": str(row['Food_Name'])})
            ids.append(f"row_{index}")
            
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
    return df, True

# Boot the app infrastructure
master_df, db_status = load_and_index_dataset()

# 5. Build UI Interface Layout
st.title("🔬 Universal IFCT 2017 Precision RAG Architecture")
if using_prebuilt:
    st.caption("⚡ High-Performance Mode: Running over pre-compiled vector schemas and Gemini 2.5 Flash")
else:
    st.caption("⚙️ Standard Mode: Parsing in-memory structures over Gemini 2.5 Flash")

if db_status:
    if using_prebuilt:
        st.success(f"✅ Master Vector Frame Online: Verified static assets loaded.")
    else:
        st.success(f"✅ Master Frame Online: Successfully mapped {len(master_df)} all-inclusive food asset matrices dynamically.")
else:
    st.error("📂 Dataset Missing: Please make sure 'ifct_clean.csv' is present in your root directory.")
    st.stop()

st.markdown("---")

# 6. Query Execution Block
query = st.text_input("Enter any dietary problem, nutrient deficiency, or component keyword (e.g., 'folate deficiency', 'high protein foods', 'iron'):")

if query:
    with st.spinner("Searching cross-functional vector indexes..."):
        try:
            search_results = collection.query(query_texts=[query], n_results=5)
            
            context_payload = ""
            if search_results and search_results['documents'] and search_results['documents'][0]:
                context_payload = " \n ".join(search_results['documents'][0])
                
            if not context_payload:
                st.error("No relevant nutrient data rows tracked inside the current database.")
                st.stop()
                
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

            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content([strict_universal_prompt])
            st.markdown(response.text)

        except Exception as e:
            st.error(f"❌ Error while generating response: {e}")
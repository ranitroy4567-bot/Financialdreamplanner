import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Create dummy local KB files if missing[cite: 1]
def initialize_kb():
    os.makedirs("kb", exist_ok=True)
    docs = {
        "kb/financial_guidelines.txt": "Freshers should save at least 20% of their income. Emergency funds should cover 6 months of expenses.",
        "kb/investment_categories.txt": "Short-term goals require liquid preservation. Long-term goals over 7 years benefit from equity growth categories.",
        "kb/goal_planning_rules.txt": "Goals should be evaluated separately based on distinct target timelines and priorities."
    }
    for file_path, text in docs.items():
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(text)

initialize_kb()

# Load and process vectorstore
loader_1 = TextLoader("kb/financial_guidelines.txt")
loader_2 = TextLoader("kb/investment_categories.txt")
loader_3 = TextLoader("kb/goal_planning_rules.txt")

documents = loader_1.load() + loader_2.load() + loader_3.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)

def query_rag(query: str) -> str:
    """Queries knowledge base; returns grounding fallback if context missing[cite: 1]."""
    results = vector_store.similarity_search_with_score(query, k=1)
    if not results or results[0][1] > 1.2: # Similarity threshold check
        return "The requested information is unavailable in the approved knowledge base."[cite: 1]
    return results[0][0].page_content
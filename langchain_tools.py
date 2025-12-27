from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_pdf_tool import extract_pdf_with_langchain, LangChainPdfInput
import json
import os
llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=os.getenv("GROQ_API_KEY")
)
# 1. Split
print("Splitting text...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
pdf_path = r"C:\Users\Admin\Downloads\PythonServices\my_agent\uploads\temp_6b8b6f77-0ad1-4e2c-844b-3070f43a1982.pdf"
json_output = extract_pdf_with_langchain(LangChainPdfInput(file_path=pdf_path))
pdf_text = json.loads(json_output)['text']
chunks = splitter.split_text(pdf_text)

# 2. Embed
print("Initializing Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 3. Store
print("Storing in VectorStore...")
db = InMemoryVectorStore.from_texts(chunks, embeddings)

# 4. Query
print("Querying...")
query = "create 2 quizes for me?"
docs = db.similarity_search(query, k=3)

context = "\n".join(d.page_content for d in docs)
prompt = f"""
Your teacher who can create best quizes fron students for their exam preparation.

Context:
{context}

Question:
{query}

Answer:
"""

# Example with OpenAI-like model
print("Calling LLM...")
answer = llm.invoke(prompt)
print(answer.content)



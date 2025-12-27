from pydantic import BaseModel, Field
import json
from langchain_community.document_loaders import PyPDFLoader

# -------- Input / Output Schemas --------

class LangChainPdfInput(BaseModel):
    file_path: str = Field(..., description="The absolute path to the PDF file.")

class LangChainPdfOutput(BaseModel):
    text: str
    metadata: dict

# -------- Core Logic --------

def extract_pdf_with_langchain(input: LangChainPdfInput) -> str:
    if PyPDFLoader is None:
        return json.dumps({"error": "LangChain dependencies not installed."})

    try:
        loader = PyPDFLoader(input.file_path)
        pages = loader.load()
        
        full_text = "\n\n".join([page.page_content for page in pages])
        meta = pages[0].metadata if pages else {}

        return LangChainPdfOutput(
            text=full_text,
            metadata=meta
        ).model_dump_json()
        
    except Exception as e:
        return json.dumps({"error": str(e)})

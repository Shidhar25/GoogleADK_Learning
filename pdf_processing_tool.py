import os
import shutil
import json
from pydantic import BaseModel, Field
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None

# --- Configuration ---
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Schema ---
class UserPdfInput(BaseModel):
    source_path: str = Field(..., description="The absolute path to the PDF file provided by the user.")

class UserPdfOutput(BaseModel):
    storage_path: str
    text_preview: str
    status: str

# --- Tool Function ---
def process_user_pdf(input: UserPdfInput) -> UserPdfOutput:
    """
    Takes a PDF path, stores it locally, and processes it to extract text.
    """
    if PyPDFLoader is None:
        return UserPdfOutput(storage_path="", text_preview="Error: LangChain not installed.", status="error")

    source = input.source_path.strip('"').strip("'")  # Clean quotes
    
    if not os.path.exists(source):
        return UserPdfOutput(storage_path="", text_preview=f"Error: File not found at {source}", status="error")

    # 1. Store (Copy) the file
    filename = os.path.basename(source)
    destination = os.path.join(UPLOAD_DIR, filename)
    
    try:
        shutil.copy2(source, destination)
    except Exception as e:
         return UserPdfOutput(storage_path="", text_preview=f"Error saving file: {str(e)}", status="error")

    # 2. Process (Extract)
    try:
        loader = PyPDFLoader(destination)
        pages = loader.load()
        full_text = "\n\n".join([page.page_content for page in pages])
        
        # Return result (truncated preview for brevity in tool output)
        return UserPdfOutput(
            storage_path=destination,
            text_preview=full_text[:2000], # Return first 2000 chars
            status="success"
        )
        
    except Exception as e:
        return UserPdfOutput(storage_path=destination, text_preview=f"Error reading PDF: {str(e)}", status="error")

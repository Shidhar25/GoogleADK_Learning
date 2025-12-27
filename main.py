import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import traceback
import sys
from dotenv import load_dotenv

# Load environment variables explicitly
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# Import our agents and tools
# We use relative imports assuming this is run as a module: uvicorn my_agent.main:app
from .agent import multi_agent_workflow, quiz_agent
from .pdf_processing_tool import process_user_pdf, UserPdfInput

app = FastAPI(title="Agent API")

# --- Schemas ---
class QuizRequest(BaseModel):
    text: str
    num_questions: int = 5

class PdfProcessRequest(BaseModel):
    file_path: str

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Agent API is running. Use /docs for Swagger UI."}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF file to the server and extracts its text.
    """
    temp_path = f"temp_{file.filename}"
    try:
        # Save to a temporary file first
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract text using our tool logic
        # The tool will copy from temp_path to uploads/ and process it
        tool_input = UserPdfInput(source_path=os.path.abspath(temp_path))
        tool_output = process_user_pdf(tool_input)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if tool_output.status == "error":
             raise HTTPException(status_code=400, detail=tool_output.text_preview)

        return {
            "filename": file.filename,
            "saved_path": tool_output.storage_path,
            "text_preview": tool_output.text_preview, 
            "full_text_length": len(tool_output.text_preview)
        }
    except Exception as e:
        # Ensure cleanup on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

# --- ADK Imports ---
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import uuid

# --- Agent & Runner Setup ---
# We use InMemorySessionService for simplicity in this custom API
session_service = InMemorySessionService()
runner = Runner(
    agent=multi_agent_workflow,
    app_name="my_agent",
    session_service=session_service
)

# --- Endpoints ---
# ... (read_root is fine) ...

# ... (upload_pdf is fine) ...

@app.post("/run-agent")
async def run_agent(input_text: str, session_id: str = None):
    """
    Runs the Multi-Agent Workflow using the ADK Runner.
    Auto-creates session if it doesn't exist.
    """
    try:
        app_name = "my_agent"
        user_id = "default_user"

        if not session_id:
            session_id = str(uuid.uuid4())
            
        # Check if session exists (InMemory service clears on restart)
        session = await session_service.get_session(
            app_name=app_name, 
            user_id=user_id, 
            session_id=session_id
        )
        
        if not session:
            # Create new session
            await session_service.create_session(
                app_name=app_name, 
                user_id=user_id, 
                session_id=session_id
            )
            
        events = []
        
        # Create user message content
        user_msg = types.Content(role="user", parts=[types.Part(text=input_text)])

        # Run via Runner
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_msg
        ):
            events.append(str(event))
        
        return {
            "session_id": session_id,
            "response_events": events
        }
    except Exception as e:
        traceback.print_exc() # Print to console
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz-from-pdf")
async def quiz_from_pdf(file: UploadFile = File(...), num_questions: int = 5):
    """
    1. Uploads PDF.
    2. Extracts text.
    3. Runs sequential agent to generate quiz.
    """
    temp_path = f"temp_{file.filename}"
    try:
        # 1. Save File
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Extract Text via Tool
        tool_input = UserPdfInput(source_path=os.path.abspath(temp_path))
        tool_output = process_user_pdf(tool_input)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if tool_output.status == "error":
             raise HTTPException(status_code=400, detail=tool_output.text_preview)

        # 3. Run Agent with Extracted Text
        # The sequential agent expects text input.
        input_text = f"Generate a quiz with {num_questions} questions from this content: \n\n {tool_output.text_preview}"
        
        session_id = str(uuid.uuid4())
        await session_service.create_session(app_name="my_agent", user_id="default", session_id=session_id)
        
        # Wrapped in Content
        user_msg = types.Content(role="user", parts=[types.Part(text=input_text)])
        
        events = []
        async for event in runner.run_async(
            user_id="default",
            session_id=session_id,
            new_message=user_msg
        ):
            if event.author == "user": continue
            
            # Simple parsing for this endpoint
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call and part.function_call.name == "generate_quiz":
                         events.append({"type": "quiz_generation", "args": part.function_call.args})
                    elif part.text:
                         events.append({"type": "text", "text": part.text})

        return {
            "session_id": session_id,
            "filename": file.filename,
            "results": events
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


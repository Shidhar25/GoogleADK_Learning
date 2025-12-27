from google.adk.agents.llm_agent import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from .tools import get_time, get_weather, generate_quiz, get_google_search
import os
from .langchain_pdf_tool import extract_pdf_with_langchain
from google.adk.agents.parallel_agent import ParallelAgent
# 1. Define Model First
model = LiteLlm(
    api_key=os.getenv("GROQ_API_KEY"),
    model = "groq/llama-3.3-70b-versatile"
)

# ...

# --- PDF Agent (LangChain) ---
pdf_agent = Agent(
    name="PdfReaderAgent",
    model=model,
    instruction="If the user provides a PDF path, use process_user_pdf to store and read it.",
    tools=[extract_pdf_with_langchain]
)

# --- Quiz Agent ---
quiz_agent = Agent(
    name="QuizGeneratorAgent",
    model=model,
    instruction="""
    You receive text.
    Use generate_quiz to create a quiz from it.
    Do NOT explain.
    """,
    tools=[generate_quiz]
)

# --- City/Time/Weather Agents ---
city_agent = Agent(
    name="CityExtractorAgent",
    model=model,
    instruction="Extract the city name only. Do NOT call any tools. Just output the city name as text."
)

time_agent = Agent(
    name="TimeFetcherAgent",
    model=model,
    instruction="Use get_time for the city.",
    tools=[get_time]
)

weather_agent = Agent(
    name="WeatherFetcherAgent",
    model=model,
    instruction="Use get_weather for the city.",
    tools=[get_weather]
)

# --- Search Agents ---
search_agent = Agent(
    name="SearchAgent",
    model=model,
    instruction="Search and return facts.",
    tools=[get_google_search]
)

# --- Unused Agents Removed ---

response_agent = Agent(
    name="ResponseAgent",
    model=model,
    instruction="Respond in one clear sentence."
)

# 3. Define Workflow (A2A)
# We combine them into a sequential workflow.
# This represents the "Multi AI agent A2A" system.
# 3. Define Workflow (A2A)
# We combine them into a sequential workflow.
# This represents the "Multi AI agent A2A" system.
multi_agent_workflow = SequentialAgent(
    name="MultiAgentA2AWorkflow",
    sub_agents=[
        pdf_agent,       # 1. Read PDF (if any)
        quiz_agent,      # 2. Generate Quiz from text/pdf
        city_agent,      # 3. Extract City (from extra context?)
        weather_agent,   # 4. Get Weather
        time_agent,      # 5. Get Time
        # response_agent   # 6. Final Response
    ]
)

from google.adk.agents.llm_agent import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.loop_agent import LoopAgent
from .tools import get_current_time, get_current_weather, get_google_search, get_pdf_content, generate_quiz
from google.adk.tools.google_search_tool import google_search
from google.adk.tools.exit_loop_tool import exit_loop

quiz_agent = Agent(
    name="QuizGeneratorAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You receive a paragraph.
    
    Rules:
    - Use the generate_quiz tool.
    - Do NOT explain anything.
    - Do NOT add text outside the tool result.
    """,
    tools=[generate_quiz]
)
city_agent = Agent(
    name="CityExtractorAgent",
    model="gemini-2.5-flash-lite",
    instruction="Extract the city name. Output ONLY the city."
)

time_agent = Agent(
    name="TimeFetcherAgent",
    model="gemini-2.5-flash-lite",
    instruction="Use get_current_time for the city.",
    tools=[get_current_time]
)

weather_agent = Agent(
    name="WeatherFetcherAgent",
    model="gemini-2.5-flash-lite",
    instruction="Use get_current_weather for the city.",
    tools=[get_current_weather]
)

pdf_agent = Agent(
    name="PdfReaderAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    If a PDF path is provided in the input, use get_pdf_content to read it.
    Answer user questions based on the PDF content.
    If no PDF path, pass.
    """,
    tools=[get_pdf_content]
)


search_agent = Agent(
    name="SearchAgent",
    model="gemini-2.5-flash-lite", 
    instruction="""
    Search the web for the given query and return a factual answer.
    """,
    tools=[google_search]
)

evaluator_agent = Agent(
    name="EvaluatorAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    Analyze the search results.
    If they satisfactorily answer the user's request, call exit_loop.
    If not, suggest a refined search query.
    """,
    tools=[exit_loop]
)

research_loop = LoopAgent(
    name="ResearchLoop",
    sub_agents=[search_agent, evaluator_agent],
    max_iterations=3
)

response_agent = Agent(
    name="ResponseAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You receive city, time, weather, search info, and PDF content.
    Respond politely in ONE sentence.

    Example:
    The current time in Mumbai is 10:30 AM, the weather is Sunny.
    """
)

time_workflow = SequentialAgent(
    name="TimeSequentialWorkflow",
    description="City → Weather → Time → Search → Response",
    sub_agents=[
        quiz_agent,
        city_agent,
        weather_agent,
        time_agent,
        research_loop,
        pdf_agent,
        response_agent
    ]
)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, Wiki_tool, save_tool
import re

load_dotenv()  # Load environment variables from .env file

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
            """
            You are a research assistant that will help generate a research paper.
            Use the provided tools to answer the user query. Return the response in plain text with the exact format:
            Topic: <topic>
            Summary: <summary>
            Sources: <list of sources, one per line>
            Tools Used: <list of tools used, e.g., wikipedia_search, duckduckgo_search, save_text_to_file>
            Do NOT use JSON, markdown (e.g., no ```), or any additional text. Do NOT repeat any sections (e.g., Sources or Tools Used) multiple times. Ensure the output is clean and concise.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

tools = [search_tool, Wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("Enter your research query: ")
raw_response = agent_executor.invoke({"query": query})

try:
    # Extract the output (should be plain text)
    output = raw_response.get("output", "")
    # Remove any unexpected markdown or extra newlines
    output = re.sub(r'^```(?:json)?\n|\n```$', '', output).strip()
    print(output)
except Exception as e:
    print("Error processing response:", e, "Raw response:", raw_response)
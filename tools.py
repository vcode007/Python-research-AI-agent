from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from duckduckgo_search import DDGS

def save_to_txt(data: str, filename: str = "research_output.txt"):
    print(f"Tool used: save_text_to_file (saving to {filename})")  # Debug log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"

def wikipedia_search(query: str) -> str:
    print(f"Tool used: wikipedia_search (query: {query})")  # Debug log
    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=100)
    return api_wrapper.run(query)

def duckduckgo_search(query: str) -> str:
    print(f"Tool used: duckduckgo_search (query: {query})")  # Debug log
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return "\n".join([f"{r['title']}: {r['body']}" for r in results])

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file. Takes a string input (the data to save) and an optional filename (defaults to 'research_output.txt')."
)

search_tool = Tool(
    name="duckduckgo_search",
    func=duckduckgo_search,
    description="Search the web for information on a topic using DuckDuckGo. Takes a string query as input and returns search results."
)

Wiki_tool = Tool(
    name="wikipedia_search",
    func=wikipedia_search,
    description="Search Wikipedia for information on a topic. Takes a string query as input and returns a summary."
)
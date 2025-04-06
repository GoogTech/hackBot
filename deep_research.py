from typing import Annotated, List, TypedDict, Literal, Final, Union
from pydantic import BaseModel, Field
import operator
from langgraph.types import interrupt, Command
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from IPython.display import Image, display
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import AsyncTavilyClient
import asyncio

load_dotenv()

class ModelTools():
    DEFAULT_MODEL: Final = "openai"
    MODEL_TEMPERATURE: Final = 0
    OPENAI_MODEL_NAME: Final = "gpt-4o" # or gpt-4o-mini
    OPENAI_EMBEDDING_MODEL_NAME: Final = "text-embedding-3-large"
    OPENAI_API_KEY: Final = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: Final = os.getenv("OPENAI_BASE_URL")

    @staticmethod
    def get_llm(
        model_name: str = DEFAULT_MODEL, 
        with_tools: bool = False
    ) -> Union[ChatOpenAI]:
        if model_name == "openai":
            llm = ChatOpenAI(
                model_name=ModelTools.OPENAI_MODEL_NAME,
                temperature=ModelTools.MODEL_TEMPERATURE,
                openai_api_key=ModelTools.OPENAI_API_KEY,
                base_url=ModelTools.OPENAI_BASE_URL,
            )
        else:
            raise ValueError(f"Model {model_name} not found")
        if with_tools:
            llm = llm.bind_tools(tools)
        return llm

    @staticmethod
    def get_embed(model_name: str = DEFAULT_MODEL) -> Union[OpenAIEmbeddings]:
        if model_name == "openai":
            embd = OpenAIEmbeddings(
                 model=ModelTools.OPENAI_EMBEDDING_MODEL_NAME,
                 openai_api_key=ModelTools.OPENAI_API_KEY,
                 base_url=ModelTools.OPENAI_BASE_URL,
            )
        return embd

async def execute_search(query_list: List[str]) -> List[dict]:
    """Execute the web search operation"""
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in query_list:
        search_tasks.append(
            tavily_async_client.search(
                query=query,
                max_results=3,
                include_raw_content=True,
                topic="general",
            )
        )
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

class SearchQuery(BaseModel):
    search_query: str = Field(description="Query for the web search")
    pass

class SearchQueries(BaseModel):
    search_queries: List[SearchQuery] = Field(description="List of search queries")
    pass

class AgentState(TypedDict):
    topic: str # Report topic
    search_queries: list[SearchQuery] # List of search queries
    search_results: list[dict] # List of search results
    pass

def generate_queries(state: AgentState) -> Command[Literal["human_feedback"]]:
    """Generate search queries based on the topic"""
    SYSTEM_PROMPT = """
    You are an expert technical writer crafting targeted web search queries 
    that will gather comprehensive information for writing the final summary report about topic.

    <Report topic>
    {topic}
    </Report topic>

    <Task>
    Your goal is to generate {number_of_queries} web search queries 
    that will help gather comprehensive information above the topic.

    The queries should:
    1. Be related to the topic
    2. Examine the topic from different angles
    3. Make the queries specific enough to find high-quality, relevant sources about the topic
    </Task>

    <Format>
    Call the Queries tool 
    </Format>
    """
    USER_PROMPT = """
    Generate search queries on the provided topic.
    """
    NUMBER_OF_QUERIES = 2

    topic = state["topic"]
    llm = ModelTools.get_llm()
    structured_llm = llm.with_structured_output(SearchQueries)
    SYSTEM_PROMPT = SYSTEM_PROMPT.format(topic=topic, number_of_queries=NUMBER_OF_QUERIES)
    results = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ])
    print(f"Generated search queries: {results.search_queries}")
    return Command(
        update={"search_queries": results.search_queries},
        goto="human_feedback",
    )

def human_feedback(state: AgentState) -> Command[Literal["generate_queries", "web_search"]]:
    # TODO: Implement human feedback mechanism
    return Command(
        goto="web_search",
    )

async def web_search(state: AgentState) -> Command[Literal["evaluate"]]:
    """Execute web search using the generated queries"""
    search_queries = state["search_queries"]
    query_list = [query.search_query for query in search_queries]
    search_results = await execute_search(query_list=query_list)
    return Command(
        update={"search_results": search_results},
        goto="evaluate",
    )

def evaluate(state: AgentState) -> Command[Literal["web_search", "summary"]]:
    """Evaluates the quality of the search results, 
    trigger more research if quality fails."""
    # TODO: Implement evaluation logic
    print(f"Evaluating search results: {state['search_results']}")
    return Command(
        goto="summary",
    )

def summary(state: AgentState) -> Command[Literal[END]]:
    # TODO: Implement summary generation
    DEFAULT_REPORT_STRUCTURE = """
    Use this structure to create a report on the user-provided topic:
    1. Introduction
    - Brief overview of the topic area

    2. Main Body
    - Summarize the information that focus on the user-provided topic
    
    3. Conclusion
    - Aim for 1 structural element (either a list of table) that distills the main body 
    - Provide a concise summary of the report
    """
    return Command(
        goto=END,
    )

workflow = StateGraph(state_schema=AgentState)
workflow.add_node(node="generate_queries", action=generate_queries)
workflow.add_node(node="human_feedback", action=human_feedback)
workflow.add_node(node="web_search", action=web_search)
workflow.add_node(node="evaluate", action=evaluate)
workflow.add_node(node="summary", action=summary)
workflow.add_edge(start_key=START, end_key="generate_queries")
graph = workflow.compile()

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except (ImportError, OSError) as e:
    print("Unable to display graph visualization", e)
except Exception as e:
    print("An unexpected error occurred while trying to display the graph", e)
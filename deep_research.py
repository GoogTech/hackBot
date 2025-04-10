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
import tiktoken

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

    @staticmethod
    def get_token_count(text: str, model_name: str = OPENAI_MODEL_NAME) -> int:
        encoding = tiktoken.encoding_for_model(model_name=model_name)
        tokens = encoding.encode(text=text)
        token_count = len(obj=tokens)
        return token_count

def format_search_results(
    search_results: List[dict], 
    max_tokens_per_result: int = 4000,
    include_raw_content: bool = True,
    ) -> str:
    """
    Format the search results into a readable string,
    and limits the raw_content to approximately "max_tokens_per_result" tokens.

    Args:
        search_results: List of search response dicts, each containing:
        - query: str
        - results: List of dicts with fields:
            - title: str
            - url:  str
            - content: str
            - score: float
            - raw_content: str | None
    """
    # Collect all results
    results_list = []
    for search_result in search_results:
        results_list.extend(search_result['results'])
    # Deduplicate by URL
    unique_results = {result['url']: result for result in results_list}
    # Format output
    formatted_text = ""
    for _, result in enumerate(unique_results.values(), start=1):
        formatted_text += f"{'=' * 80}\n"
        formatted_text += f"Title: {result['title']}\n"
        formatted_text += f"{'-' * 80}\n"
        formatted_text += f"URL: {result['url']}\n"
        formatted_text += f"Most relevant content: {result['content']}\n"
        if include_raw_content:
            raw_content = sources.get("raw_content", None)
            if raw_content is None:
                raw_content = ''
                print(f"Warning: Not found the raw_content in {result['url']}")
            if ModelTools.get_token_count(raw_content) > max_tokens_per_result:
                raw_content = raw_content[:max_tokens_per_result] + "...[truncated]"
            formatted_text += f"The Full result raw_content: {raw_content}\n"
        formatted_text += f"{'=' * 80}\n\n"
    return formatted_text.strip()

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
    feedback_on_search_queries: str # Feedback on the search queries
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

    <Feedback>
    Here is feedback on the search queries from human reivew,
    If the feedback value is not None, please regenerate the search queries according to the feedback.
    
    The feedback value: {feedback}
    </Feedback>

    <Format>
    Call the Queries tool 
    </Format>
    """
    USER_PROMPT = """
    Generate search queries on the provided topic.
    """
    NUMBER_OF_QUERIES = 2

    topic = state["topic"]
    feedback = state.get("feedback_on_search_queries", None)

    llm = ModelTools.get_llm()
    structured_llm = llm.with_structured_output(SearchQueries)
    SYSTEM_PROMPT = SYSTEM_PROMPT.format(topic=topic, number_of_queries=NUMBER_OF_QUERIES, feedback=feedback)
    results = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ])

    return Command(
        update={"search_queries": results.search_queries},
        goto="human_feedback",
    )

def human_feedback(state: AgentState) -> Command[Literal["generate_queries", "web_search"]]:
    """Get human feedback on the search queries, and route to next steps"""
    topic = state["topic"]
    search_queries = state["search_queries"]
    interrupt_message = f"""
    Please provide feedback on the following generated search queries:
    \n\n{search_queries}\n\n
    Dodes the report plan meet your need?\n
    Pass 'true' to approve the generated search queries.\n
    Or, Provide feedback to regenerate the search queries:
    """
    
    feedback = interrupt(value=interrupt_message)
    # If the user approves the generated search queries, kick off web search
    if isinstance(feedback, bool) and feedback is True:
        return Command(goto="web_search")
    # If the user provides feedback, regenerate the search queries
    elif isinstance(feedback, str):
        return Command(
            update={"feedback_on_search_queries": feedback},
            goto="generate_queries"
        )
    # Catch the exception
    else:
        raise TypeError(f"Interrput value of type {type(feedback)} is not supported.")

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
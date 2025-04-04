from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator
from langgraph.types import interrupt, Command
from langgraph.graph import START, END, StateGraph
from IPython.display import Image, display

class Section(BaseModel):
    description: str = Field(
        description="Brief overview of the main topics and concepts to be convered in this section"
    )
    research: bool = Field(
        description="Whether to perform web research for this section"
    )
    content: str = Field(
        description="The content of the section"
    )
    pass

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="List of sections"
    )
    pass

class SearchQueries(BaseModel):
    search_queries: List[str] = Field(
        description="List of search queries and the query used for the web search"
    )
    pass

class AgentState(BaseModel):
    topic: str # Report topic
    section: Sections # List of sections
    search_queries: SearchQueries # List of search queries
    pass

def generate_sections(state: AgentState) -> Command[Literal["generate_queries"]]:
    pass

def generate_queries(state: AgentState) -> Command[Literal["human_feedback"]]:
    """Generate search queries based on the section topic and description."""
    SYSTEM_PROMPT = """
    You are an expert technical writer crafting targeted web search queries 
    that will gather comprehensive information for writing a technical report section.

    <Report topic>
    {topic}
    </Report topic>

    <Section topic>
    {section_topic}
    </Section topic>

    <Task>
    Your goal is to generate {number_of_queries} search queries
    that will help gather comprehensive information above the section topic. 

    The queries should:
    1. Be related to the topic
    2. Examine different aspects of the topic

    Make the queries specific enough to find high-quality, relevant sources.
    </Task>

    <Format>
    Call the Queries tool 
    </Format>
    """
    USER_PROMPT = """
    Generate search queries on the provided topic.
    """
    pass

def human_feedback(state: AgentState) -> Command[Literal["generate_queries", "web_search"]]:
    pass

async def web_search(state: AgentState) -> Command[Literal["evaluate"]]:
    pass

def evaluate(state: AgentState) -> Command[Literal["web_search", "summary"]]:
    """Evaluates the quality of the search results, 
    trigger more research if quality fails."""
    pass

def summary(state: AgentState) -> Command[Literal[END]]:
    pass

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
from typing import Annotated, List, TypedDict, Literal, Final, Union
from pydantic import BaseModel, Field
import operator
from langgraph.types import interrupt, Command
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from IPython.display import Image, display
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import AsyncTavilyClient
import asyncio
import tiktoken

load_dotenv()

class Configuration:
    NUMBER_OF_QUERIES: int = 2 # Number of search queries to generate per iteration
    MAX_SEARCH_DEPTH: int = 2 # Maximum number of search iterations

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
    include_raw_content: bool = False,
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
    for _, result in enumerate(unique_results.values()):
        formatted_text += f"{'-' * 120}\n "
        formatted_text += f"TITLE: {result['title']}\n "
        formatted_text += f"URL: {result['url']}\n "
        formatted_text += f"MOST RELEVANT CONTENT: {result['content']}\n "
        if include_raw_content:
            raw_content = sources.get("raw_content", None)
            if raw_content is None:
                raw_content = ''
                print(f"Warning: Not found the raw_content in {result['url']}")
            if ModelTools.get_token_count(raw_content) > max_tokens_per_result:
                raw_content = raw_content[:max_tokens_per_result] + "...[truncated]"
            formatted_text += f"The Full result raw_content: {raw_content}\n "
        formatted_text += f"{'-' * 120}\n\n "
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

class EvaluationFeedback(BaseModel):
    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements('pass') or need revision('fail')"
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries"
    )

class AgentState(TypedDict):
    topic: str # Report topic
    search_queries: list[SearchQuery] # List of search queries
    section_titles: list[str] # Section titles of the research report
    feedback_on_search_queries: str # Feedback on the search queries
    search_results: Annotated[list[str], add_messages] # List of search results
    search_iterations: int # The number of search iterations
    sections_and_contents: str # The content of the sections is part of the research report
    full_report_content: str # The full report content
    pass

def generate_queries(state: AgentState) -> Command[Literal["human_feedback"]]:
    """Generate search queries based on the topic"""
    SYSTEM_PROMPT = """
    You are an expert technical writer crafting targeted web search queries 
    that will gather comprehensive information for writing the final report about topic.

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

    topic = state["topic"]
    feedback = state.get("feedback_on_search_queries", None)

    llm = ModelTools.get_llm()
    structured_llm = llm.with_structured_output(SearchQueries)
    SYSTEM_PROMPT = SYSTEM_PROMPT.format(
        topic=topic, 
        number_of_queries=Configuration.NUMBER_OF_QUERIES, 
        feedback=feedback
    )
    results = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ])

    return Command(
        update={
            "search_queries": results.search_queries, 
            "section_titles": results.search_queries
        },
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
        return Command(
            update={"search_iterations": 0},
            goto="web_search"
        )
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
    search_results_str = format_search_results(search_results=search_results)
    return Command(
        update={
            "search_results": search_results_str, 
            "search_iterations": state["search_iterations"] + 1},
        goto="evaluate",
    )

def evaluate(state: AgentState) -> Command[Literal["web_search", "summary"]]:
    """
    Evaluates the quality of the search results, 
    trigger more research if quality fails.
    """
    # 1.Write the sections of a research report
    # ----------------------------------------------
    # Section Title 1(SearchQuery 1)
    #   - SectionContent 1(Come from search_results)
    # Section Title 2(SearchQuery 2)
    #   - SectionContent 2(Come from search_results)
    # ----------------------------------------------
    section_titles = state["section_titles"]
    search_results = state["search_results"]
    section_writer_instructions = """
    Write the sections of a research report.

    <Section Titles>
    {section_titles}
    </Section Titles>

    <Source Material>
    {search_results}
    </Source Material>

    <Task>
    1. Get all of the section titles carefully from the Section Titles.
    2. Look at the provided Source Material, It includes many chunks of information scraped from the web, 
       and each chunk contains the keywords “TITLE”, “URL”, and “MOST RELEVANT CONTENT”.
    3. Decide which chunks you will use to write each report section.
    4. Write each report section based on the Source Material until all sections are completed.
    </Task>

    <Writing Guidelines>
    - If existing section content is not populated, write from scratch
    - If existing section content is populated, synthesize it with the Source Material
    - Strict 150-200 word limit
    - Use simple, clear language
    - Use short paragraphs (2-3 sentences max)
    - Use ## for section title (Markdown format)
    </Writing Guidelines>

    <Citation Rules>
    - Assign each unique URL a single citation number in your text
    - End with ## Sources that lists each source with corresponding numbers
    - IMPORTANT: Number sources sequentially without gaps (1,2,3...) in the final list
    - Example format:
      [1] Source Title: URL
      [2] Source Title: URL
    </Citation Rules>

    <Final Check>
    1. Verify that the content of each section is grounded in the provided Source Material.
    2. Confirm that each URL appears only once in the Source list.
    3. Verify that sources are numbered sequentially (1, 2, 3…) without any gaps.
    </Final Check>
    """
    section_writer_inputs = """
    Write the sections of a research report based on the provided Section Titles and Source Material.
    """

    llm = ModelTools.get_llm()
    section_writer_instructions_formatted = section_writer_instructions.format(
        section_titles=section_titles,
        search_results=search_results
    )
    sections_and_contents = llm.invoke([
        SystemMessage(content=section_writer_instructions_formatted),
        HumanMessage(content=section_writer_inputs),
    ])

    # 2.Grade the sections of the research report, 
    # and consider follow-up questions for missing information if necessary.
    section_grader_instructions = """
    Review the content of the section in relation to its title.

    <The titles and contents of the sections>
    {sections_and_contents}
    </The titles and contents of the sections>

    <task>
    Evaluate whether the content of section adequately addressses its title.
    If the content of section doesn't adequately address its title,
    generate {number_of_follow_up_queries} follow-up search queries to gather missing information.
    </task>
    """
    section_grader_message = """
    Evaluate whether the content of section adequately addressses its title,
    and consider follow-up questions for missing information if necessary.
    If the grade is 'pass', return None for the follow-up queries,
    else provide specific search queries to gather missing information.
    """
    section_grader_instructions_formatted = section_grader_instructions.format(
        sections_and_contents=sections_and_contents.content,
        number_of_follow_up_queries=Configuration.NUMBER_OF_QUERIES,
    )
    structured_llm = llm.with_structured_output(EvaluationFeedback)
    evaluation_feedback = structured_llm.invoke([
        SystemMessage(content=section_grader_instructions_formatted),
        HumanMessage(content=section_grader_message),
    ])

    # The sections are passing or the max search depth is reached
    if evaluation_feedback.grade == "pass" or state["search_iterations"] >= Configuration.MAX_SEARCH_DEPTH:
        return Command(
            update={"sections_and_contents": sections_and_contents.content},
            goto="summary",
        )
    # Update search queries
    else:
        return Command(
            update={"search_queries": evaluation_feedback.follow_up_queries},
            goto="web_search",
        )

def summary(state: AgentState) -> Command[Literal[END]]:
    """
    Write sections that don't require research using completed sections as context.
    This node handles sections like "Introduction" or "Conclusion" that build on the
    completed sections rather than requiring direct research.
    """
    summary_writer_instructions = """
    You are an expert technical writer crafting the remaining sections of the report based on the available report content.
    
    <Report Topic>
    {topic}
    </Report Topic>

    <Available Report Content>
    {sections_and_contents}
    </Available Report Content>

    <Task>
    1. Section-Specific Approach:

    For Report Title
    - Use # for report title (Markdown format)
    - The topic of the available report content is used as the report title.

    For Introduction:
    - Use ## For section title (Markdown format)
    - 50-100 word limit
    - Write in simple and clear language
    - Focus on the core motivation for the report in 1-2 paragraphs
    - Use a clear narrative arc to introduce the report
    - Include NO structural elements (no lists or tables)
    - No sources section needed

    For Conclusion/Summary:
    - Use ## for section title (Markdown format)
    - 100-150 word limit
    - For comparative reports:
        * Must included a focused comparsion table using Markdown table syntax
        * Table should distill insights from the report
        * Keep table entries clear and concise
    - For non-comparative reports:
        * Only use ONE structural element if it helps distill the points made in the report:
        * Either a focused table comparing items present in the report (using Markdown table syntax)
        * Or a short list using proper Markdown list syntax:
            - Use `*` or `-` for unordered lists
            - Use `1.` for ordered lists
            - Ensure proper indentation and spacing
    - End with specific next steps or implications
    - No sources section needed

    2. Writing Approach:
    - Use concrete details over general statements
    - Focus on your single most important point
    - Make every word count
    </Task>

    <Quality Checks>
    - For Report Title: # for report title, the topic is used as the report title
    - For Introduction: 100-150 word limit, ## for section title, no structural elements, no sources section
    - For Conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no source section
    </Quality Checks>
    """
    huamn_instructions = """
    Add the remaining sections of the report based on the available report content.
    """
    # Get state
    topic = state["topic"]
    sections_and_contents = state["sections_and_contents"]
    # Format system instructions
    system_instructions = summary_writer_instructions.format(
        topic=topic,
        sections_and_contents=sections_and_contents,
    )
    # Generate the section of ## Introduction and ## Conclusion
    llm = ModelTools.get_llm()
    new_sections = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=huamn_instructions)
    ])
    new_sections = new_sections.content
    # TODO: Combine the original sections with the new sections
    system_instructions = """
    You are an expert technical editor, and your duty is to combine the original sections with the new sections.

    <Original Sections>
    {original_sections}
    </Original Sections>

    <New Sections>
    {new_sections}
    </New Sections>

    <Task>
    1.Add the new sections to the original sections and combine them into the full report.
    2.Make sure to preserve the format and content of the original sections, including the citation numbers in the text.
    3.Adjust the sequence of the sections, such as the section of sources should be the last one.

    An example of the full report structure (Markdown format):
    # Report Title
    ## Introduction
    ## Section Title 1
    ## Section Title 2
    ## Section Title X
    ## Conclusion
    ## Sources

    </Task>

    <Quality Checks>
    1.Whether the full report includes all sections from both the new and original ones.
    2.Whether the sequence of the sections is correct.
    </Quality Checks>
    """
    huamn_instructions = """
    Please combine the original sections with the new sections.
    """
    system_instructions_formatted = system_instructions.format(
        original_sections=sections_and_contents,
        new_sections=new_sections
    )
    full_report_content = llm.invoke([
        SystemMessage(content=system_instructions_formatted),
        HumanMessage(content=huamn_instructions)
    ])
    full_report_content = full_report_content.content
    # Write the full report content to state
    return Command(
        update={"full_report_content": full_report_content},
        goto=END
    )

workflow = StateGraph(state_schema=AgentState)
workflow.add_node(node="generate_queries", action=generate_queries)
workflow.add_node(node="human_feedback", action=human_feedback)
workflow.add_node(node="web_search", action=web_search)
workflow.add_node(node="evaluate", action=evaluate)
workflow.add_node(node="summary", action=summary)
workflow.add_edge(start_key=START, end_key="generate_queries")
graph = workflow.compile()
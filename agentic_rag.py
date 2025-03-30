from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence, Literal, Union, Final
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
import json
from IPython.display import Image, display
import pprint
from dotenv import load_dotenv
import os
# from langgraph.prebuilt import ToolNode
# from langgraph.prebuilt import tools_condition

load_dotenv()

# we index 3 blog posts
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_overlap=100,
    chunk_size=100,
)
docs_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=docs_list,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

# Then we create a retriever tool
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="retrieve_blog_posts",
    description="Search and return information about Lilian Weng blog posts " + 
    "on LLM agents, prompt enginerring, and adversarial attacks on LLMs."       
)

tools = [retriever_tool]

class ModelTools():
    DEFAULT_MODEL: Final = "openai"
    MODEL_TEMPERATURE: Final = 0
    CHATGPT_MODEL_NAME: Final = "gpt-4o" # or gpt-4o-mini

    @staticmethod
    def get_llm(
        model_name: str = DEFAULT_MODEL, 
        with_tools: bool = False
    ) -> Union[ChatOpenAI]:
        if model_name == "openai":
            llm = ChatOpenAI(
                model_name=ModelTools.CHATGPT_MODEL_NAME,
                temperature=ModelTools.MODEL_TEMPERATURE,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        else:
            raise ValueError(f"Model {model_name} not found")
        if with_tools:
            llm = llm.bind_tools(tools)
        return llm

# Define the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define the nodes and edges
class grade(BaseModel):
    """Binary score for relevance check"""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

class ToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        print(f"retrieve all docs ---> {outputs}\n")
        return {"messages": outputs}

def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """Determines whether the retrieved documents are relevant to the question"""
    # LLM
    llm = ModelTools.get_llm()
    # LLM with tool and validation
    llm_with_tool = llm.with_structured_output(schema=grade)
    # Prompt
    prompt = PromptTemplate(
        template="""
        You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
       
        If the document contains keyword(s) or semantic meaning related to the user question, 
        grade it as relevant. \n
        
        Give a binary score 'yes' or 'no' score to indicate 
        whether the document is relevant to the question.
        """,
        input_variables=["context", "question"],
    )
    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]
    question, docs = messages[0].content, last_message.content # !!!
    # print(f"retrieve specifical content ---> {docs}\n")

    scorted_result = chain.invoke({"question": question, "context": docs})
    score = scorted_result.binary_score
    if score == "yes":
        print(f"grade_documents ---> docs relevant\n")
        return "generate"
    else:
        print(f"grade_documents ---> docs not relevant\n")
        return "rewrite"

def tools_condition(
    state: AgentState, 
    messages_key: str = "messages"
) -> Literal["retrieve", "__end__"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "retrieve" # tools
    return "__end__"

def agent(state: AgentState):
    """
    Invokes the agent model to generate a response based on the current state.
    Given the question, it will decide to retrieve using the retriever tool,
    or simple end.
    """
    messages = state["messages"]
    llm = ModelTools.get_llm(with_tools=True)
    response = llm.invoke(messages)
    print(f"\nagent ---> {response}\n")
    return Command(
        update={"messages": response}
    )

def rewrite(state: AgentState) -> Command[Literal["agent"]]:
    """Transform the query to produce a better question"""
    print(f"rewrite ---> transform query")
    messages = state["messages"]
    question = messages[0].content
    msg = [
        HumanMessage(content=f"""
        Look at the input and try to reason about the underlying semantic intent / meaning.\n
        Here is the initial question:
        {question}.

        Formulate an improved question:
        """
        )
    ]
    llm = ModelTools.get_llm()
    response = llm.invoke(input=msg)
    print(f"rewrite ---> {response}\n")
    return Command(
        goto="agent",
        update={"messages": response}
    )

def generate(state: AgentState) -> Command[Literal[END]]:
    """Generate answer"""
    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    # Prompt
    """rlm/rag-prompt: 
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = hub.pull("rlm/rag-prompt")
    # LLM
    llm = ModelTools.get_llm(with_tools=True)
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    # Run
    response = rag_chain.invoke({"question": question,"context": docs})
    print(f"generate ---> {response}\n")
    return Command(
        goto=END,
        update={"messages": response},
    )

# Define a new graph
workflow = StateGraph(state_schema=AgentState)
# Define the nodes we will cycle between
workflow.add_node(node="agent", action=agent)
retrieve = ToolNode(tools=tools)
workflow.add_node(node="retrieve", action=retrieve)
workflow.add_node(node="rewrite", action=rewrite)
workflow.add_node(node="generate", action=generate)
workflow.add_edge(start_key=START, end_key="agent")
workflow.add_conditional_edges(
    source="agent",
    path=tools_condition,
)
workflow.add_conditional_edges( 
    source="retrieve",
    path=grade_documents,
)
# Compile
graph = workflow.compile()

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except:
    pass

inputs = {
    "messages": [("user", "What does Lilian Weng say about the types of agent memeory?"),]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pass
    #     pprint.pprint(f"Output from node '{key}':")
    #     pprint.pprint("---")
    #     pprint.pprint(value, indent=2, width=80, depth=None)
    # pprint.pprint("\n---\n")
#################################################################
# Adaptive RAG: Adaptive RAG is a strategy for RAG that unites: #
# (1).query analysis                                            #
# (2).active / self-corrective RAG                              #
#                                                               #
# Paper: https://arxiv.org/abs/2403.14403                       #
#################################################################
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Annotated, Sequence, Literal, Union, Final
from dotenv import load_dotenv
import os
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List
from typing_extensions import TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, StateGraph, START
from IPython.display import Image, display
from langchain.schema import Document
import time

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

################
# Create Index #
################
# Set embeddings
embd = ModelTools.get_embed()

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Splict
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=0,
)

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=docs_list,
    collection_name="rag-chroma",
    embedding=embd,
)

retriever = vectorstore.as_retriever(
    # Fixed the bug: It always retrieves all the data from the VectorStore, even if they might not be relevant to the question.
    # Solution: Only retrieve documents that have a relevance score above a certain threshold
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.8}
)

#######################################
# Web Search Tool: Question Re-writer #
#######################################
web_search_tool = TavilySearchResults(k=3)

######################
# Define Graph State #
######################
class GraphState(TypedDict):
    """Represents the state of our graph.

    Args:
        question (str): question
        generation (str): LLM generation
        document (str): list of documents
    """
    question: str
    generation: str
    documents: List[str]
    # Set a retrieval count to avoid going into an endless loop among the nodes:
    # Retrieve, grade_documents and transform_query
    retrieval_count: int
    # Feedback on the generated question
    feedback_on_question: str

#####################
# Define Graph Flow #
#####################
# BUG:
# It's will go to death loop among the nodes: Retrieve, grade_documents and transform_query, 
# if the question includes some specified keywords that appears in the system prompt of route_question node,
# such as the question: "what's the agent, and how the build it with langgraph?"
# SOLUTION: 
# 1.Add the huamn_feedback node, and build the relationship with the transform_query node
# 2.Limit the retrieval count, define the max_retrieval_count
# 3.If max_retrieval_count: go END, else: continue to loop
def retrieve(state: GraphState) -> Command[Literal["grade_documents", END]]:
    """Retrieve documents"""
    print(f"retrieve ————>")
    MAX_RETRIEVAL_COUNT = 2 # the max retrieval count
    question = state["question"]
    retrieval_count = state["retrieval_count"]
    # Judge the retrieval_count
    if retrieval_count >= MAX_RETRIEVAL_COUNT:
        return Command(goto=END)
    else:
        # Retrieval
        documents = retriever.invoke(question)
        print(f"Retrieve Node ---> documents: {documents}")
        # return {"documents": documents, "question": question}
        return Command(
            update={"documents": documents, "question": question, "retrieval_count": retrieval_count + 1}, 
            goto="transform_query"
        )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def grade_documents(state: GraphState):
    # Prompt
    system = """
    You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    """
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    llm = ModelTools.get_llm()
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader

    """Determines whether the retrieved documents are relevant to the question"""
    print(f"grade_documents ————>")
    question = state["question"]
    documents = state["documents"] # TODO: bug here: KeyError('documents')
    print(f"GradeDocuments Node ---> documents: {documents}")
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print(f"grade_documents ————> document relevant")
            filtered_docs.append(d)
        else:
            print(f"grade_documents ————> document not relevant")
            continue
    return {"documents": filtered_docs, "question": question}

def generate(state: GraphState):
    """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't konw.
    Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs) # Post-processing
    llm = ModelTools.get_llm()
    rag_chain = prompt | llm | StrOutputParser()

    """Generate answer"""
    print(f"generate ————>")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    # TODO: bug here: NameError("name 'generation' is not defined")
    return {"documents": documents, "question": question, "generation": generation}

def transform_query(state: GraphState):
    system = """
    You a question re-writer that converts an input question to a better version that is optimized \n
    for vectorstore retrieval. Look at the input question and try to reason about the underlying semantic intent / meaning.

    <Feedback>
    Here is feedback on the generated question from human review,
    If the feedback value is not None, please regenerate the question according to the feedback.
    The feedback value: {feedback}
    </Feedback>
    """
    feedback = state.get("feedback_on_question", None)
    system = system.format(feedback=feedback)

    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
    ])
    llm = ModelTools.get_llm()
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    """Transform the query to produce a better question"""
    print(f"transform_query ————>")
    question = state["question"]
    documents = state["documents"] # TODO: Delete the unnecessary code
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question} # Always go the human_feedback node

# TODO
def human_feedback(state: GraphState) -> Command[Literal["transform_query", "retrieve"]]:
    new_question = state["question"]
    interrupt_message = f"""
    Please provide feedback on the following generated question:
    \n\n{new_question}\n\n
    Dodes the generated question meet your need?\n
    Pass 'true' to approve the generated question.\n
    Or, Provide feedback to regenerate the question:
    """
    feedback = interrupt(value=interrupt_message)
    # If the user approves the generated question, kick off the retrieval process
    if isinstance(feedback, bool) and feedback is True:
        return Command(
            goto="retrieve"
        )
    # If the user provides feedback, regenerate the question
    elif isinstance(feedback, str):
        return Command(
            update={"feedback_on_question": feedback},
            goto="transform_query"
        )
    # Catch the exception
    else:
        raise TypeError(f"Interrput value of type {type(feedback)} is not supported.")

def web_search(state):
    """Web search based on the re-phrased question"""
    print(f"web_search ————>")
    question = state["question"]
    # Web search
    docs = web_search_tool.invoke({"query": question})
    contents = []
    for d in docs:
        if isinstance(d, str): # Fixed the TypeError: string indices must be integers, not 'str'
            contents.append(d)
        elif "content" in d:
            contents.append(d["content"])
    web_results = "\n".join(contents)
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "question": question}

################
# Define Edges #
################
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore."
    )

def route_question(state: GraphState) -> Command[Literal["web_search", "retrieve"]]:
    system = """
    You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. Otherwise, use web-search.
    """
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
    llm = ModelTools.get_llm()
    structured_llm_router = llm.with_structured_output(RouteQuery)
    question_router = route_prompt | structured_llm_router

    """Route question to web search or RAG"""
    print(f"route_question ————>")
    retrieval_count = 1 # init the retrieval count
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print(f"route_question ————> go to web search")
        # return "web_search"
        return Command(goto="web_search")
    elif source.datasource == "vectorstore":
        print(f"route_question ————> go to RAG")
        # return "vectorstore"
        return Command(update={"retrieval_count": retrieval_count}, goto="retrieve")

def decide_to_generate(state: GraphState):
    """Determines whether to generate an answer, or re-generate a question"""
    print(f"route_question ————>")
    state["question"]
    filtered_documents = state["documents"]
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(f"decide_to_generate ————> go to re-generate a new query")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print(f"decide_to_generate ————> go to generate answer")
        return "generate"

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addressses question"""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

def grade_generation_v_documents_and_question(state: GraphState):
    system = """
    You are a grader assessing whether an LLM generation is grounded in / 
    supported by a set of retrieved facts. \n

    Give a binary score 'yes' or 'no', 
    'Yes' means that the answer is grounded in / supported by the set of facts.
    """
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ])
    llm = ModelTools.get_llm()
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    hallucination_grader = hallucination_prompt | structured_llm_grader

    """
    Determines whether the generation is grounded in the document 
    and answers question.
    """
    print(f"grade_generation_v_documents_and_question ————>")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({
        "documents": documents,
        "generation": generation,
    })
    grade = score.binary_score

    # 1.Check hallucination
    if grade == "yes":
        print(f"grade_generation_v_documents_and_question ————> \
                generation is grounded in documents")
        # 2.Check question-answering
        # ---------------------------------------------------------------------------------
        system = """
        You are a grader assessing whether an answer addresses / resolves a question \n
        Give a binary score 'yes' or 'no'.
        'Yes' means that the answer resolves the question.
        """
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ])
        llm = ModelTools.get_llm()
        structured_llm_grader = llm.with_structured_output(GradeAnswer)
        answer_grader = answer_prompt | structured_llm_grader
        # ---------------------------------------------------------------------------------
        score = answer_grader.invoke({
            "question": question,
            "generation": generation,
        })
        grade = score.binary_score
        if grade == "yes":
            print(f"grade_generation_v_documents_and_question ————> \
                    generation addresses question")
            return "useful"
        else:
            print(f"grade_generation_v_documents_and_question ————> \
                    generation does not addresses question")
            return "not useful"
    else:
        print(f"grade_generation_v_documents_and_question ————> \
                generation is not grounded in documents")
        return "not supported"

#################
# Compile Graph #
#################
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("route_question", route_question)
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("human_feedback", human_feedback)

# Build graph
workflow.add_edge(START, "route_question")
# workflow.add_conditional_edges(
#     START,
#     route_question,
#     {
#         "web_search": "web_search",
#         "vectorstore": "retrieve",
#     }
# )
workflow.add_edge("web_search", "generate")
# workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query",
    }
)
workflow.add_edge("transform_query", "human_feedback")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    }
)

# Compile
graph = workflow.compile()
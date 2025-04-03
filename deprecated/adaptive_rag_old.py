#################################################################
# Adaptive RAG: Adaptive RAG is a strategy for RAG that unites: #
# (1).query analysis                                            #
# (2).active / self-corrective RAG                              #
#                                                               #
# Paper: https://arxiv.org/abs/2403.14403                       #
#################################################################

#######
# LLM #
#######
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Annotated, Sequence, Literal, Union, Final
from dotenv import load_dotenv
import os

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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

# Set embeddings
# embd = OpenAIEmbeddings()
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
retriever = vectorstore.as_retriever()

#######################
# LLMs 1: Tool Router #
#######################
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasoure: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore."
    )

# LLM with function call
# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
# )
llm = ModelTools.get_llm()
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search.
"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

question_router = route_prompt | structured_llm_router

print(question_router.invoke({"question": "Who will the Bears draft first in the NFL draft?"}))
print(question_router.invoke({"question": "What are the types of agent memory?"}))

############################
# LLMs 2: Retrieval Grader #
############################
# Date model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM with function call
# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
# )
llm = ModelTools.get_llm()
structured_llm_grader = llm.with_structured_output(GradeDocuments)

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

retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

####################
# LLMs 3: Generate #
####################
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
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
# LLM
# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
# )
llm = ModelTools.get_llm()
# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# Chain
rag_chain = prompt | llm | StrOutputParser()
# Run
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

################################
# LLMs 4: Hallucination Grader #
################################
# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# LLM with function call
# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
# )
llm = ModelTools.get_llm()
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
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

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader.invoke({"documents": docs, "generation": generation})

#########################
# LLMs 5: Answer Grader #
#########################
# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addressses question"""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# LLM with function call
# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
# )
llm = ModelTools.get_llm()
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """
You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'.
'Yes' means that the answer resolves the question.
"""
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
])

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question, "generation": generation})

##############################
# LLMs 6: Question Re-writer #
##############################
# LLM
# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
# )
llm = ModelTools.get_llm()
# Prompt
system = """
You a question re-writer that converts an input question to a better version that is optimized \n
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
"""
re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
])

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

#######################################
# Web Search Tool: Question Re-writer #
#######################################
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

######################
# Define Graph State #
######################
from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """Represents the state of our graph.

    Args:
        question (str): question
        generation (str): LLM generation
        document (str): list of documents
    """
    question: str
    generation: str
    document: List[str]

#####################
# Define Graph Flow #
#####################
from langchain.schema import Document

def retrieve(state: GraphState):
    """Retrieve documents"""
    print(f"retrieve ————>")
    question = state["question"]
    # Retrieval
    documents = retriever.invoke(question)
    return {"document": documents, "question": question}

def grade_documents(state: GraphState):
    """Determines whether the retrieved documents are relevant to the question"""
    print(f"grade_documents ————>")
    question = state["question"]
    documents = state["documents"]
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
    """Generate answer"""
    print(f"generate ————>")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def transform_query(state: GraphState):
    """Transform the query to produce a better question"""
    print(f"transform_query ————>")
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state):
    """Web search based on the re-phrased question"""
    print(f"web_search ————>")
    question = state["question"]
    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "question": question}

################
# Define Edges #
################
def route_question(state: GraphState):
    """Route question to web search or RAG"""
    print(f"route_question ————>")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print(f"route_question ————> go to web search")
        return "web_search"
    elif source.datasource == "vectorstore":
        print(f"route_question ————> go to RAG")
        return "vectorstore"

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

def grade_generation_v_documents_and_question(state: GraphState):
    """
    Determines whether the generation is grounded in the document 
    and answers question.
    """
    print(f"grade_generation_v_documents_and_question ————>")
    question = state["question"]
    documents = state["documents"]
    generate = state["generation"]

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
from langgraph.graph import END, StateGraph, START
from IPython.display import Image, display

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query",
    }
)
workflow.add_edge("transform_query", "retrieve")
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

# Draw graph
# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except:
#     pass
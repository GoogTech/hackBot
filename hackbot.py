from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Final, Union, Literal, List
from dotenv import load_dotenv
from IPython.display import Image, display
import os

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
def web_reasearch() -> TavilySearchResults:
    tool = TavilySearchResults(max_results=2)
    return tool

tools: List = [web_reasearch()]

class ModelTools():
    DEFAULT_MODEL: Final = "openai"
    MODEL_TEMPERATURE: Final = 0
    CHATGPT_MODEL_NAME: Final = "gpt-4o"

    @staticmethod
    def _get_llm_with_tools(model_name: str = DEFAULT_MODEL) -> Union[ChatOpenAI]:
        if model_name == "openai":
            llm = ChatOpenAI(
                model_name=ModelTools.CHATGPT_MODEL_NAME,
                temperature=ModelTools.MODEL_TEMPERATURE,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        else:
            raise ValueError(f"Model {model_name} not found")
        llm_with_tools = llm.bind_tools(tools)
        return llm_with_tools

class AgentState(TypedDict):
    messages: Annotated[list, add_messages, "the agent messages"]
    user_input: Annotated[str, "the user input information"]
    tasks: Annotated[list, "the generated task list according to the Human Message"]
    pass

class PlannerOuputSchema(TypedDict):
    messages: Annotated[str, "the chat response"]
    tasks: Annotated[list, "the list stored all tasks"]
    pass

class ExecutorOutputSchema(TypedDict):
    tools: Annotated[list, "the name of each hack tool will be used according to the corresponding task"]
    command: Annotated[list, "each command will be generated based on the corresponding task"]
    pass

class SummarizerOutputSchema(TypedDict):
    pass

class Agent():
    def planner(state: AgentState) -> Command[Literal["executor", END]]:
        PLANNER_SYSTEM_PROMPT: Final = SystemMessage(content="""
        You are a planner in penetration testing processes,
        Your duty is generate precise and concise task list according to the Human Message
        if the Huamn Message included the information that required you to hack, else just reponse: 
        "Sorry, I am a planner agent and only respond to hack requirements".
        
        Notes: 
        1.Please include the series number in the task header.
        2.Please try to finish as soon as possible by using the quickest way.
        """
        )
        user_input_message = HumanMessage(content=str(state["user_input"]))
        messages = [PLANNER_SYSTEM_PROMPT, user_input_message]
        llm_with_tools = ModelTools._get_llm_with_tools().with_structured_output(schema=PlannerOuputSchema)
        response = llm_with_tools.invoke(messages)
        print(f"\nplanner ---> {response}")
        if tasks := response['tasks']:
            return Command(
                goto="executor", 
                update={
                    "tasks": tasks,
                    "messages": [AIMessage(content=f"go to executor agent")]
                },
            )
        return Command(goto=END)
    
    def executor(state: AgentState) -> Command[Literal["tools", "summarizer"]]:
        EXECUTOR_SYSTEM_PROMPT: Final = SystemMessage(content=f"""
        You are a executor in penetration testing processes,
        Your duty is to generate each attack command based on every task in the task list
        then execute them by calling the Tool Agent until every task is done.
        
        The task list: {state['tasks']}
        """
        )
        print(EXECUTOR_SYSTEM_PROMPT)
        messages = [EXECUTOR_SYSTEM_PROMPT]
        llm_with_tools = ModelTools._get_llm_with_tools().with_structured_output(schema=ExecutorOutputSchema)
        response = llm_with_tools.invoke(messages)
        print(f"\nexecutor ---> {response}")
        return Command(
            goto="tools",
        )

    def summarizer(state: AgentState) -> Command[Literal["planner"]]:
        return Command(
            goto="planner",
        )

graph_builder = StateGraph(state_schema=AgentState)
graph_builder.add_node(node="planner", action=Agent.planner)
graph_builder.add_node(node="executor", action=Agent.executor)
graph_builder.add_node(node="summarizer", action=Agent.summarizer)
tools_node = ToolNode(tools=tools)
graph_builder.add_node(node="tools", action=tools_node)
graph_builder.add_edge(start_key=START, end_key="planner")
# graph_builder.add_edge(start_key="tools", end_key="executor")
# graph_builder.add_conditional_edges(source="executor", path=tools_condition)
graph = graph_builder.compile()

class Test():
    @staticmethod
    def draw_graph():
        try:
            display(Image(graph.get_graph().draw_mermaid_png()))
        except (ImportError, OSError) as e:
            print("Unable to display graph visualization")
        except Exception as e:
            print("An unexpected error occurred while trying to display the graph")

    @staticmethod
    def _stream_graph_updates(user_input: str) -> None:
        user_input = {"user_input": [{"role": "user", "content": user_input}]}
        for event in graph.stream(input=user_input):
            for value in event.values():
                # print("Assistant: ", value["messages"][-1].content)
                # print("Assistant: ", value)
                pass

    @staticmethod
    def chat() -> None:
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in {"quit", "exit"}:
                    print("Goodbye!")
                    break
                Test._stream_graph_updates(user_input=user_input)
            except KeyboardInterrupt:
                print("\nProgram interrupted by user. Goodbye!")
                break
            except ValueError as ve:
                print(f"Invalid input: {ve}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

Test.draw_graph()
Test.chat()
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import ShellTool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Final, Union, Literal, List, Dict
from dotenv import load_dotenv
from IPython.display import Image, display
import os
import warnings

warnings.filterwarnings(
    action="ignore", 
    message="The shell tool has no safeguards by default. Use at your own risk.", 
    category=UserWarning
)

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

def web_reasearch() -> TavilySearchResults:
    web_search_tool = TavilySearchResults(max_results=2)
    return web_search_tool

def shell_executor() -> ShellTool:
    shell_tool = ShellTool()
    return shell_tool

tools: List = [web_reasearch(), shell_executor()]

class ModelTools():
    DEFAULT_MODEL: Final = "openai"
    MODEL_TEMPERATURE: Final = 0
    CHATGPT_MODEL_NAME: Final = "gpt-4o" # or gpt-4o-mini

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
    tools_name: Annotated[list, "the name of each hack tool will be used according to the corresponding task"]
    commands: Annotated[list, "each command will be generated based on the corresponding task"]
    command_results: Annotated[Dict[str, str], "each result of every executed command"]
    pass

class PlannerOuputSchema(TypedDict):
    messages: Annotated[str, "the chat response"]
    tasks: Annotated[list, "the list stored all tasks"]
    pass

class ExecutorOutputSchema(TypedDict):
    # message: Annotated[str, "the thought process"]
    tools_name: Annotated[list, "the name of each hack tool will be used according to the corresponding task"]
    commands: Annotated[list, "each command will be generated based on the corresponding task"]
    # command_result: Annotated[list, "the result of each executed command"]
    pass

# class ToolsOuputSchema(TypedDict):
#     command_results: Annotated[Dict[str, str], "each result of every executed command"]

class SummarizerOutputSchema(TypedDict):
    fail_solution: Annotated[str, "the solution for the failures"]

class Agent():
    def planner(state: AgentState) -> Command[Literal["executor", "reporter"]]:
        PLANNER_SYSTEM_PROMPT: Final = SystemMessage(content="""
        You are a planner in penetration testing processes,
        Your duty is generate precise and concise task list according to the Human Message
        if the Huamn Message included the information that required you to hack, else just reponse: 
        "Sorry, I am a planner agent and only respond to hack requirements".
        
        Notes: 
        1.Please include the series number in the task header.
        2.Please try to finish as soon as possible by using the quickest way.
        3.Do not generate tasks that cannot be translated into attack commands, 
          For example, documenting the findings and preparing a report.
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
        Your duty is to generate each attack command based on every task in the task list,
        Of course, the prerequisite is that the task can generate the attack command.

        The task list: {state['tasks']}        
        """
        )
        if "command_results" not in state:
            messages = [EXECUTOR_SYSTEM_PROMPT]
            llm_with_tools = ModelTools._get_llm_with_tools().with_structured_output(schema=ExecutorOutputSchema)
            response = llm_with_tools.invoke(messages)
            print(f"\nexecutor ---> {response}")
            return Command(
                goto="tools",
                update={
                    "tools_name": response["tools_name"],
                    "commands": response["commands"],
                    "messages": [AIMessage(content=f"go to tools")]
                },
            )
        return Command(
            goto="summarizer",
            update={
                # "command_results": state["command_results"],
                "messages": [AIMessage(content=f"go to summarizer")]
            }
        )

    def tools(state: AgentState) -> Command[Literal["executor"]]:
        TOOLS_SYSTEM_PROMPT: Final = SystemMessage(content=f"""
        You are a command executor in penetration testing processes,
        Your duty is to use the one of hack tools in {state["tools_name"]}
        to execute every attack command in {state["commands"]},
        and you have access to bash shell using the shell_tool.     
        """
        )
        messages = [TOOLS_SYSTEM_PROMPT]
        llm_with_tools = ModelTools._get_llm_with_tools()
        response = llm_with_tools.invoke(messages)
        
        tools_dict = {tool.name: tool for tool in tools}
        print(f"\ntools_dict ---> {tools_dict}")

        messages.append(response)
        for tool_call in response.tool_calls:
            selected_tool = tool_call["name"].lower()
            print(f"selected_tool ---> {selected_tool}")
            selected_tool_obj = tools_dict[selected_tool]
            selected_tool_response = selected_tool_obj.invoke(tool_call)
            messages.append(selected_tool_response)
            print(f"selected_tool_response ---> {selected_tool_response}")

        return Command(
            goto="executor",
            update={
                    # "command_results": response["command_results"],
                    "command_results": messages,
                    "messages": [AIMessage(content=f"go to executor")]
            },
        )

    def summarizer(state: AgentState) -> Command[Literal["planner"]]:
        SUMMARIZER_SYSTEM_PROMPT: Final = SystemMessage(content=f"""
        You are a summarizer for command execution results in penetration testing processes,
        Your duty is summarize the command execution results concisely, 
        and if the command execution process fails, please provide the solution.

        The command execution results: {state["command_results"]}
        """
        )
        messages = [SUMMARIZER_SYSTEM_PROMPT]
        llm_with_tools = ModelTools._get_llm_with_tools().with_structured_output(schema=SummarizerOutputSchema)
        response = llm_with_tools.invoke(messages)
        print(f"\nsummarizer ---> {response}")
        return Command(
            goto="planner",
        )

    def reporter(state: AgentState) -> Command[Literal[END]]:
        return Command(
            goto=END
        )

graph_builder = StateGraph(state_schema=AgentState)
graph_builder.add_node(node="planner", action=Agent.planner)
graph_builder.add_node(node="executor", action=Agent.executor)
graph_builder.add_node(node="summarizer", action=Agent.summarizer)
graph_builder.add_node(node="tools", action=Agent.tools)
graph_builder.add_node(node="reporter", action=Agent.reporter)
# tools_node = ToolNode(tools=tools)
# graph_builder.add_node(node="tools", action=tools_node)
graph_builder.add_edge(start_key=START, end_key="planner")
# graph_builder.add_edge(start_key="tools", end_key="executor")
# graph_builder.add_conditional_edges(
#     source="executor", 
#     path=tools_condition,
#     path_map={"tools": "tools", END: "summarizer"}
# )
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
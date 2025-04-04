from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import ShellTool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_core.tools import tool
from typing import Final, Union, Literal, List, Dict
from dotenv import load_dotenv
from IPython.display import Image, display
import os
import warnings
import platform
from enum import Enum
from colorama import init, Fore, Style

init(autoreset=True) # Initialize colorama

warnings.filterwarnings(
    action="ignore", 
    message="The shell tool has no safeguards by default. Use at your own risk.", 
    category=UserWarning
)

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

def web_search() -> TavilySearchResults:
    """Tool for web search"""
    web_search_tool = TavilySearchResults(max_results=2)
    return web_search_tool

def shell_executor() -> ShellTool:
    """Tool to run shell commands"""
    shell_tool = ShellTool()
    return shell_tool

tools: List = [web_search(), shell_executor()]

class OSType(Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "darwin"
    UNKNOWN = "unknown"

def get_os_type() -> str:
    try:
        # Get lowercase OS name
        os_name = platform.system().lower()
        # Map system name to OSType
        if os_name.startswith("win"):
            return OSType.WINDOWS.value
        elif os_name.startswith("linux"):
            return OSType.LINUX.value
        elif os_name.startswith("darwin"):
            return OSType.MACOS.value
        else:
            return OSType.UNKNOWN.value
    except Exception as e:
        print(f"Error detecting OS: {e}")
        return OSType.UNKNOWN.value

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
    messages: Annotated[list, add_messages]
    # messages: Annotated[list, add_messages, "all the agent messages"]
    user_input: Annotated[str, "the user input information"]
    tasks: Annotated[list, "the generated task list according to the Human Message"]
    tools_name: Annotated[list, "the name of each hack tool will be used according to the corresponding task"]
    commands: Annotated[list, "each command will be generated based on the corresponding task"]
    command_results: Annotated[Dict[str, str], "the result of every executed command"]
    summary: Annotated[str, "summarize the command execution results"]
    # failure_flag: Annotated[bool, "whether command execution failures exist; the default value is False"]
    failure_flag: Annotated[bool, "whether there are failures, the default value is False"]
    pass

class PlannerOuputSchema(TypedDict):
    # messages: Annotated[str, "the PlannerAgent thought process"]
    tasks: Annotated[list, "the list stored all tasks"]
    pass

class ExecutorOutputSchema(TypedDict):
    tools_name: Annotated[list, "the name of each hack tool will be used according to the corresponding task"]
    commands: Annotated[list, "each command will be generated based on the corresponding task and OS type"]
    # command_result: Annotated[list, "the result of each executed command"]
    pass

class ToolsOutputSchema(TypedDict):
    # failure_flag: Annotated[bool, "whether there are failures, the default value is False"]
    # failure_flag: Annotated[bool, "whether there are failures, such as command execution failures or \
    # failures in the execution results, the default value is False"]
    pass

class SummarizerOutputSchema(TypedDict):
    summary: Annotated[str, "summarize the command execution results concisely, \
    and give the the solution for command execution failures if they occur"]
    # failure_flag: Annotated[bool, "whether command execution failures exist; the default value is False"]
    failure_flag: Annotated[bool, "whether there are failures, such as command execution failures or \
    failures in the execution results, the default value is False"]
    pass

class Agent():
    def planner(state: AgentState) -> Command[Literal["executor", "reporter"]]:
        # For the first call
        if "summary" not in state:
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
            print(f"\n{Fore.GREEN}planner ---> {response}{Style.RESET_ALL}")
            # print(f"{Fore.GREEN}planner ---> state['messages']: {state["messages"]}{Style.RESET_ALL}")

            new_message = f"PlannerAgent: {[messages, response]}, then go to ExecutorAgent"

            if tasks := response['tasks']:
                return Command(
                    goto="executor", 
                    update={
                        "tasks": tasks,
                        "messages": new_message,
                        # "messages": [AIMessage(
                        #     content=f"PlannerAgent: {new_message}, then go to ExecutorAgent"
                        # )],
                    },
                )
            return Command(goto=END)

        # For the second, third...call
        if state["failure_flag"] == True:
            PLANNER_SYSTEM_PROMPT: Final = SystemMessage(content=f"""
            You are a planner in penetration testing processes,
            Your duty is generate precise and concise task list according to the SummarizerAgent's Messages
            if the SummarizerAgent's Messages included the failure information that required you to solve, 
            else just goto the ReporterAgent.

            The SummarizerAgent's Messages: {state["summary"]}

            Notes: 
            1.Please include the series number in the task header.
            2.Please try to finish as soon as possible by using the quickest way.
            3.Do not generate tasks that cannot be translated into the commands.
            """
            )
            messages = [PLANNER_SYSTEM_PROMPT]
            llm_with_tools = ModelTools._get_llm_with_tools().with_structured_output(schema=PlannerOuputSchema)
            response = llm_with_tools.invoke(messages)
            print(f"\n{Fore.GREEN}planner ---> {response}{Style.RESET_ALL}")
            # print(f"{Fore.GREEN}planner ---> state['messages']: {state["messages"]}{Style.RESET_ALL}")

            # new_message = [messages, response]
            new_message = f"PlannerAgent: {[messages, response]}, then go to ExecutorAgent"

            if tasks := response['tasks']:
                return Command(
                    goto="executor",
                    update={
                        "tasks": tasks,
                        "messages": new_message,
                        # "messages": [AIMessage(
                        #     content=f"PlannerAgent: {new_message}, then go to ExecutorAgent"
                        # )],
                    },
                )
            return Command(goto=END)

        # End the loop(planner, executor, summarizer), 
        # try to generate the report
        print(f"{Fore.GREEN}planner ---> state['messages']: {state["messages"]}{Style.RESET_ALL}")
        new_message = f"PlannerAgent: just go to ReporterAgent"
        return Command(
            goto="reporter",
            update={
                "messages": new_message
                # "messages": [AIMessage(
                #     content=f"PlannerAgent: just go to ReporterAgent"
                # )],
            },
        )

    # def executor(state: AgentState) -> Command[Literal["tools", "summarizer"]]:
    # def executor(state: AgentState) -> Command[Literal["human_reviewer", "summarizer"]]:
    def executor(state: AgentState) -> Command[Literal["human_reviewer"]]:
        EXECUTOR_SYSTEM_PROMPT: Final = SystemMessage(content=f"""
        You are a executor in penetration testing processes,
        Your duty is to generate the attack command for each task in the task list, considering the OS type,
        Of course, the prerequisite is that the task can generate the attack command.

        Notes: 
        1.The task list: {state['tasks']}
        2.The operation system type: {get_os_type()}
        3.Don't use the shell to run an interactive session, such as sudo -s.
        """
        )
        # 1.First call: ("command_results" not in state)
        # 2.Second or third...call: (state["failure_flag"] is True)
        # if ("command_results" not in state) or (state["failure_flag"] is True):
        messages = [EXECUTOR_SYSTEM_PROMPT]
        llm_with_tools = ModelTools._get_llm_with_tools().with_structured_output(schema=ExecutorOutputSchema)
        response = llm_with_tools.invoke(messages)
        print(f"\n{Fore.YELLOW}executor ---> {response}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}executor ---> state['messages']: {state["messages"]}{Style.RESET_ALL}")

        # new_message = [messages, response]
        new_message = f"ExecutorAgent: {[messages, response]}, then go to HumanReviewerAgent"

        return Command(
            goto="human_reviewer",
            update={
                "tools_name": response["tools_name"],
                "commands": response["commands"],
                "messages": new_message,
                # "messages": [AIMessage(
                #     content=f"ExecutorAgent: {new_message}, then go to HumanReviewerAgent"
                # )],
            },
        )
        # return Command(
        #     goto="summarizer",
        #     update={
        #         "messages": [AIMessage(content=f"go to summarizer")]
        #     }
        # )

    def tools(state: AgentState) -> Command[Literal["summarizer"]]:
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
        print(f"\n{Fore.RED}[tools]tools_dict ---> {tools_dict}{Style.RESET_ALL}")

        messages.append(response)
        for tool_call in response.tool_calls:
            selected_tool = tool_call["name"].lower()
            print(f"{Fore.RED}[tools]selected_tool ---> {selected_tool}{Style.RESET_ALL}")
            selected_tool_obj = tools_dict[selected_tool]
            selected_tool_response = selected_tool_obj.invoke(tool_call) # Execute the command
            messages.append(selected_tool_response)
            print(f"{Fore.RED}[tools]selected_tool_response ---> {selected_tool_response}{Style.RESET_ALL}")

        # TODO: fix the bad code
        # messages_2 = [TOOLS_SYSTEM_PROMPT]
        # llm_with_tools_2 = ModelTools._get_llm_with_tools().with_structured_output(schema=ToolsOutputSchema)
        # response_2 = llm_with_tools_2.invoke(messages)
        # print(f"{Fore.RED}[tools]failure_flag ---> {response_2["failure_flag"]}{Style.RESET_ALL}")

        new_message = f"ToolsAgent: {[messages]}, then go to SummarizerAgent"

        return Command(
            goto="summarizer",
            update={
                "command_results": messages,
                "messages": new_message,
                # "failure_flag": response_2["failure_flag"],
                # "messages": [AIMessage(
                #     content=f"ToolsAgent: {messages}, then go to SummarizerAgent"
                # )],
            },
        )

    def summarizer(state: AgentState) -> Command[Literal["planner"]]:
        SUMMARIZER_SYSTEM_PROMPT: Final = SystemMessage(content=f"""
        You are a summarizer for command execution results in penetration testing processes,
        Your duty is summarize the command execution results concisely and provide the solution
        if someone command execution process fails or failures in the execution results.

        The command execution results: {state["command_results"]}
        """
        )
        messages = [SUMMARIZER_SYSTEM_PROMPT]
        llm_with_tools = ModelTools._get_llm_with_tools().with_structured_output(schema=SummarizerOutputSchema)
        response = llm_with_tools.invoke(messages)
        print(f"\n{Fore.BLUE}summarizer ---> {response}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}summarizer ---> {response["failure_flag"]}{Style.RESET_ALL}")

        # new_message = [messages, response]
        new_message = f"SummarizerAgent: {[messages, response]}, then go to PlannerAgent"

        return Command(
            goto="planner",
            update={
                "summary": response["summary"],
                "failure_flag": response["failure_flag"],
                "messages": new_message,
                # "messages": [AIMessage(
                #     content=f"SummarizerAgent: {new_message}, then go to PlannerAgent"
                # )],
            }
        )

    def reporter(state: AgentState) -> Command[Literal[END]]:
        REPORTER_SYSTEM_PROMPT: Final = SystemMessage(content=f"""
        You are a reporter for generating report in the end of penetration testing processes,
        Your duty is to generate a concise report on the complete penetration testing processes and results.

        The complete penetration testing processes and results: {state["messages"]},
        that includes four agents' processing procedures: 
        PlannerAgent, ExecutorAgent, ToolsAgent, and SummarizerAgent.

        Notes: 
        1.The generated report only concisely shows each agent’s process and the call sequence.
        """
        )

        print(f"\n{Fore.MAGENTA}reporter ---> state['messages']: {state["messages"]}{Style.RESET_ALL}")

        messages = [REPORTER_SYSTEM_PROMPT]
        llm_with_tools = ModelTools._get_llm_with_tools()
        response = llm_with_tools.invoke(messages)
        print(f"{Fore.MAGENTA}reporter ---> {response}{Style.RESET_ALL}")
        return Command(goto=END)

    def human_reviewer(state: AgentState) -> Command[Literal["executor", "tools"]]:
        while True:
            print(f"\n{Fore.CYAN}human_reviewer ---> commands: {state["commands"]}{Style.RESET_ALL}")
            user_input = input(f"{Fore.CYAN}human_reviewer ---> ensure to execute these commands as below: {Style.RESET_ALL}")
            if user_input in ["yes", "continue"]:
                return Command(goto="tools")
            elif user_input in ["no", "cancel"]:
                # TODO: it needs to go back to the executorAgent instead of END
                return Command(goto=END)

graph_builder = StateGraph(state_schema=AgentState)
graph_builder.add_node(node="planner", action=Agent.planner)
graph_builder.add_node(node="executor", action=Agent.executor)
graph_builder.add_node(node="summarizer", action=Agent.summarizer)
graph_builder.add_node(node="tools", action=Agent.tools)
graph_builder.add_node(node="reporter", action=Agent.reporter)
graph_builder.add_node(node="human_reviewer", action=Agent.human_reviewer)
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
        user_input = {
            "user_input": [{"role": "user", "content": user_input}], 
            "failure_flag": False
        }
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
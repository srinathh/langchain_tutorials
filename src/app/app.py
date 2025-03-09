from dotenv import load_dotenv
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import logging
from pydantic import BaseModel
from src.utils.singleton import Singleton
import json
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
import uuid
from langchain_core.tools import tool
from typing import Dict

StrDict = Dict[str,str]


load_dotenv()
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class Globals(metaclass=Singleton):
    llm: BaseChatModel
    memory : MemorySaver
    chat_threads:dict[str:str]
    active_thread:str|None

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.memory = MemorySaver()
        # chat_threads should probably be persisted as well in db
        self.chat_threads={}
        self.active_thread=None

    def bind_tools(self, tools):
        # noinspection PyTypeChecker
        logging.info(f"binding the following tools: {",".join([t.name for t in tools])}")
        self.llm = self.llm.bind_tools(tools)

    def create_new_chat_thread(self, topic:str)->None:
        while True:
            new_id = uuid.uuid4()
            if new_id not in self.chat_threads:
                break

        self.chat_threads[new_id.hex] = topic
        self.active_thread = new_id.hex
        logging.info(f"created a new chat thread {new_id.hex}:{topic}")

    def switch_chat_threads(self, thread_id:str)->bool:
        if thread_id in self.chat_threads:
            logging.info(f"switched to chat thread: {thread_id}")
            self.active_thread = thread_id
            return True

        logging.info(f"did not find the chat thread: {thread_id}")
        return False


    def get_config(self) -> dict:
        if self.active_thread is None:
            raise RuntimeError("Active thread has not been instantiated!")

        return {"configurable": {"thread_id": self.active_thread}}

@tool
def tool_create_new_chat_thread(
        topic:Annotated[str, "a short sentence describing the topic useful to recognize the chat thread later"]
)->None:
    """This tool creates a new chat thread_id for persistence about the given topic and activates it as the
    active chat thread_id. Use this if the user asks to create a new chat thread to discuss a new topic.
    This function call will always succeed and create a new thread.
    """
    Globals().create_new_chat_thread(topic)

@tool
def tool_fetch_chat_threads()->Annotated[dict[str:str],"a dictionary of thread_id mapped to topics. Both keys and values will be strings"]:
    """This tool fetches a dictionary of previous chat thread_id mapped to topics that are being persisted
    Use this tool to fetch a list of the thread_ids mapped to topics if the user wants to switch conversation
    to a different topic. You can choose to either show the list of thread_id and topic to the user and ask
    them to pick or use the list to recognize which thread the conversation should switch to. You can then
    use the tool_switch_chat_threads tool to change topics"""
    return Globals().chat_threads

@tool
def tool_switch_chat_threads(thread_id:Annotated[str,"the chat thread_id to switch conversation to"])->Annotated[bool,"whether the thread_id was found and we successfully switched conversation threads or not"]:
    """this tool switches to a conversation specified by the provided thread_id given. Note that this tool expects
    the thread_id, not the topic so be sure to provide a thread_id. You can get a dictionary of thread_id mapped
    to topics from the tool_fetch_chat_threads tool. This tool returns True if the switch succeeded and False if
    the switch failed"""
    return Globals().switch_chat_threads(thread_id)

class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot(state: State):
    logging.info("In Chatbot: Documenting current state")
    for message in state.messages:
        logging.info(f"{type(message)} : {message}")

    return {"messages": [Globals().llm.invoke(state.messages)]}


def main():
    # check that the API Keys are set
    assert os.environ.get("OPENAI_API_KEY") != ""
    assert os.environ.get("TAVILY_API_KEY") != ""

    tools = [
        TavilySearchResults(max_results=10),
        tool_switch_chat_threads,
        tool_fetch_chat_threads,
        tool_create_new_chat_thread]

    tool_node = ToolNode(tools=tools)
    Globals().bind_tools(tools)


    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "chatbot")
    # graph_builder.add_edge("chatbot", END)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph = graph_builder.compile(checkpointer=Globals().memory)

    done = False


    while not done:
        user_input = ""
        try:
            user_input = input("User: ")
            if user_input.lower() in ["q", "quit", "exit"]:
                print("Goodbye")
                break
        except:
            user_input = "What do you know about LangGraph?"

        if Globals().active_thread is None:
            Globals().create_new_chat_thread(user_input)


        result = graph.invoke({"messages": HumanMessage(user_input)}, Globals().get_config())
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

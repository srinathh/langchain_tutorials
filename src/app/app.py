from dotenv import load_dotenv
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import logging
from pydantic import BaseModel
from src.utils.singleton import Singleton
import json

load_dotenv()
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class Globals(metaclass=Singleton):
    llm: BaseLanguageModel

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


class BasicToolNode:

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State):
        # should validate that hte last message is an AIMessage
        message = state.messages[-1]

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )

        return {"messages": outputs}


def chatbot(state: State):
    for message in state.messages:
        logging.debug(f"{type(message)} : {message}")

    return {"messages": [Globals().llm.invoke(state.messages)]}


def route_tools(state: State):
    #should check that there is an AI Message
    ai_message = state.messages[-1]

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def main():
    # check that the API Keys are set
    assert os.environ.get("OPENAI_API_KEY") != ""
    assert os.environ.get("TAVILY_API_KEY") != ""

    tool = TavilySearchResults(max_results=10)
    tool_node = BasicToolNode(tools=[tool])

    Globals().llm = Globals().llm.bind_tools([tool])

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "chatbot")
    # graph_builder.add_edge("chatbot", END)
    graph_builder.add_conditional_edges("chatbot", route_tools)
    graph_builder.add_edge("tools", "chatbot")
    graph = graph_builder.compile()

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

        result = graph.invoke({"messages": HumanMessage(user_input)})
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

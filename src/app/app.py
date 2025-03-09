from dotenv import load_dotenv
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import logging
from pydantic import BaseModel
from src.utils.singleton import Singleton
load_dotenv()
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class Globals(metaclass=Singleton):
    llm: BaseLanguageModel

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")

class State(BaseModel):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    for message in state.messages:
        logging.debug(f"{type(message)} : {message}")

    return {"messages": [Globals().llm.invoke(state.messages)]}

def main():
    # llm = ChatOpenAI(model="gpt-4o-mini")
    assert os.environ.get("OPENAI_API_KEY") != ""
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
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

        result = graph.invoke({"messages":HumanMessage(user_input)})
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

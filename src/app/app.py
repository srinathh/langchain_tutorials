from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
assert os.environ.get("OPENAI_API_KEY") != ""


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages":[{"role":"user","content":user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
#
# from utils.sample import sample_util
def main():

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["q", "quit", "exit"]:
                print("Goodbye")
                break

            stream_graph_updates(user_input)
        except:
            # if input() is missing
            user_input = "What do you know about LangGraph?"
            print(f"User: {user_input}")
            stream_graph_updates(user_input)
            break


    pass

if __name__ == "__main__":
    main()

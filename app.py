import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import Annotated, List
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
import time


class State(TypedDict):
    blog: str
    topic: str
    feedback: str
    works_or_not: str


class Feedback(BaseModel):
    grade: Literal["good", "bad"] = Field(
        description="Decide if the blog good or not.",
    )
    feedback: str = Field(
        description="If the blog is not good, provide feedback on how to improve it.",
    )


graph = StateGraph(State)
llm = ChatOllama(model="llama3.2:1b")
# llmrev = ChatOllama(model="gemma2:2b")
evaluator = llm.with_structured_output(Feedback)


def writerAgent(state: State):
    """First LLM call to generate initial Blog"""
    st.write("✍️ Writer called ")
    if state.get("feedback"):
        st.write(f"write is working on {state["feedback"]} in the blog")
        msg = llm.invoke(
            f"Write a blog about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write an exciting blog about {state['topic']}")

    st.markdown(msg.content)
    return {"blog": msg.content}


def testerAgent(state: State):
    """Second LLM call to review provided Blog"""
    st.write("Tester is here")
    grade = evaluator.invoke(
        f"""You are an expert blog reviewer.
    - Grade the blog as either 'good' or 'bad' (only these two values).
    - If the blog is 'bad', provide constructive feedback on how to improve it but take it easy.
    
    Blog content:
    {state['blog']}
    """
    )

    # st.write(grade["feedback"])
    st.write(f"📝 Tester said: {grade.feedback} ")
    st.write(f"📝 Tester rates {grade.grade} to the Blog")
    return {"works_or_not": grade.grade, "feedback": grade.feedback}


def route_blog(state: State):
    """Route back to blog generator or end based upon feedback from the evaluator"""

    if state["works_or_not"] == "good":
        return "Accepted"
    elif state["works_or_not"] == "bad":
        return "Rejected + Feedback"


graph.add_node("writerAgent", writerAgent)
graph.add_node("testerAgent", testerAgent)

graph.add_conditional_edges(
    "testerAgent",
    route_blog,
    {
        "Accepted": END,
        "Rejected + Feedback": "writerAgent",
    },
)

graph.add_edge(START, "writerAgent")
graph.add_edge("writerAgent", "testerAgent")


Compiled = graph.compile()
sidebar = st.sidebar
sidebar.title("Workflow Diagram")
with sidebar:
    try:
        Compiled.get_graph(xray=True).draw_mermaid_png(output_file_path="graph.png")
        sidebar.image("graph.png", use_container_width=True)
    except Exception as e:
        print(f"{e}")


user_input = st.text_area("Enter your Topic:")

if st.button("Run Workflow") and user_input:
    state = {"topic": user_input}
    response = Compiled.invoke(state)
    st.header("Results Just Came Out")
    with st.spinner("Wait for it..."):
        time.sleep(3)
    st.markdown(response["blog"], unsafe_allow_html=True)

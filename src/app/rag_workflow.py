
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict
from langfuse import get_client

from app.clients import app_params, llm
from app.vector_store import get_retriever

load_dotenv()

# langfuse client
langfuse = get_client()

# load system prompt
system_prompt = langfuse.get_prompt(
    name="the_rag_app_system_prompt",
    type="text",
    label=app_params.prompt_label
)

class RAGState(TypedDict):

    query: str
    retrieved_docs: list[Document]
    context: str
    prompt: ChatPromptTemplate
    response: str

def retrieve(state: RAGState) -> dict:
    query = state["query"]
    retriever = get_retriever()
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"retrieved_docs": retrieved_docs, "context": context}

def augmentation(state: RAGState) -> dict:

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.prompt),
        ("human", "context: {context}\n\nquery: {query}")
    ])
    
    return {"prompt": prompt}


def generation(state: RAGState) -> dict:

    query = state['query']
    context = state['context']
    prompt = state['prompt']

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({'context': context, 'query': query})

    return {"response": response}

graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("augmentation", augmentation)
graph_builder.add_node("generation", generation)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "augmentation")
graph_builder.add_edge("augmentation", "generation")
graph_builder.add_edge("generation", END)

graph = graph_builder.compile()

# result = graph.invoke({"query": "How does golden synthesizer generate context-grounded RAG goldens from transcripts to minimize hallucinations?"})

# print(result['query'])
# print(result['retrieved_docs'])
# print(result['context'])
# print(result['response'])



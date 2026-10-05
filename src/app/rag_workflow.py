from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict
import os

load_dotenv()

ROOT_PATH = Path(__file__).parent.parent.parent
DOCS_PATH = ROOT_PATH / "data" / "processed"
EMBEDDING_PATH = ROOT_PATH / "saved-embeddings"
class RAGState(TypedDict):

    query: str
    retrieved_docs: list[Document]
    context: str
    prompt: ChatPromptTemplate
    response: str

# llm = ChatOpenAI(model = "gpt-4o-mini")
llm = ChatOpenRouter(model="deepseek/deepseek-v4-flash-0731",
                     api_key=os.environ["OPENROUTER_API_KEY"])

embedder = OpenAIEmbeddings(model = "text-embedding-3-small", dimensions=1024)

loader = DirectoryLoader(path=Path(DOCS_PATH).as_posix(),
                         loader_cls=TextLoader,
                         show_progress=True)

docs = loader.load()

chunker = RecursiveCharacterTextSplitter(chunk_size = 500,
                                         chunk_overlap = 50)

chunks = chunker.split_documents(docs)

vs = Chroma(collection_name='rag_demo',
            embedding_function=embedder,
            persist_directory=Path(EMBEDDING_PATH).as_posix())

vs.add_documents(chunks)

retriever = vs.as_retriever(search_kwargs = {'k':3},search_type = 'similarity')

def retrieve(state: RAGState) -> dict:
    query = state["query"]

    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"retrieved_docs": retrieved_docs, "context": context}

def augmentation(state: RAGState) -> dict:

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the user query
                      based on the given context only. If you do not know the answer
                      say I don't know. Do not add any preamble to the response"""),
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



from fastapi import APIRouter

from app.clients import llm
from app.vector_store import get_retriever

from api.schemas import AppHealthResponse, DependencyHealthResponce

router = APIRouter(prefix="/health", tags=["health"])

@router.get("", response_model=AppHealthResponse)
def applicarion_health() -> AppHealthResponse:
    return AppHealthResponse(message="Welcome to campusx chatbot")

@router.get("/dependencies", response_model=DependencyHealthResponce)
def depedency_health() -> DependencyHealthResponce:
    llm_health: str = "healthy"
    llm_error: str | None = None
    try:
        llm.invoke("hi")
    except Exception as e:
        llm_health = "unhealthy"
        llm_error = str(e)

    retriever_health: str = "healthy"
    retriever_error: str | None = None
    try:
        get_retriever().invoke("what is faithfulness")
    except Exception as e:
        retriever_health = "Unhealthy"
        retriever_error = str(e)

    return DependencyHealthResponce(
        llm_health= llm_health,
        llm_error= llm_error,
        retriever_health= retriever_health,
        retriever_error= retriever_error
    )
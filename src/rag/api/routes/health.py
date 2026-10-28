from fastapi import APIRouter, Request


health_router = APIRouter()


@health_router.get("/api/health")
def health(request: Request):
    return {
        "status": "ok",
        "llm_configured": request.app.state.llm_configured,
        "embedding_model": request.app.state.embedding_model,
    }
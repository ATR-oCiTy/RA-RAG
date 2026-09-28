import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import engine

app = FastAPI(title="RA-RAG playground")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionRequest(BaseModel):
    source_names: list[str]
    top_k_src: int | None = None


class RoundRequest(BaseModel):
    query: str
    contexts: dict[str, str] = {}
    manual_answers: dict[str, str] = {}


def _session_or_404(session_id: str) -> engine.Session:
    try:
        return engine.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")


@app.post("/api/session")
def create_session(req: CreateSessionRequest):
    try:
        session_id = engine.create_session(req.source_names, top_k_src=req.top_k_src)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"session_id": session_id, **engine.get_session(session_id).state()}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    return _session_or_404(session_id).state()


@app.post("/api/session/{session_id}/calibration_round")
def add_calibration_round(session_id: str, req: RoundRequest):
    session = _session_or_404(session_id)
    try:
        return session.add_calibration_round(req.query, req.contexts, req.manual_answers)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/session/{session_id}/lock")
def lock_session(session_id: str):
    session = _session_or_404(session_id)
    try:
        return session.lock()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/session/{session_id}/inference_round")
def add_inference_round(session_id: str, req: RoundRequest):
    session = _session_or_404(session_id)
    try:
        return session.add_inference_round(req.query, req.contexts, req.manual_answers)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

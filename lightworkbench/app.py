from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import RESOURCES, STATIC_ROOT
from .core import BrowseService, EpisodeService, WorkbenchError
from .operations import OperationManager


class OperationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mode: Literal["trim", "no_trim"]
    source_root: str = Field(alias="sourceRoot")
    episode: str
    output_root: str = Field(alias="outputRoot")
    operator: str
    source_token: str = Field(alias="sourceToken")
    ranges: list[list[int]] = Field(default_factory=list)
    overwrite: bool = False


class OperationSettingsRequest(BaseModel):
    concurrency: int = Field(ge=1, le=4)


app = FastAPI(title="极简高性能剪切工作台", version="1.0.0")
browse_service = BrowseService()
episode_service = EpisodeService()
operation_manager = OperationManager(episode_service)
SSE_POLL_SECONDS = 0.2
SSE_KEEPALIVE_SECONDS = 15.0


async def operation_snapshot_events(manager: OperationManager, request: Request):
    observed_version = -1
    observed_payload: str | None = None
    last_sent = asyncio.get_running_loop().time()
    while not await request.is_disconnected():
        version, snapshot = manager.snapshot()
        if version != observed_version:
            payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
            observed_version = version
            if payload != observed_payload:
                yield f"event: snapshot\ndata: {payload}\n\n"
                observed_payload = payload
                last_sent = asyncio.get_running_loop().time()
        now = asyncio.get_running_loop().time()
        if now - last_sent >= SSE_KEEPALIVE_SECONDS:
            yield ": keep-alive\n\n"
            last_sent = now
        await asyncio.sleep(SSE_POLL_SECONDS)


@app.exception_handler(WorkbenchError)
async def workbench_error(_: Request, exc: WorkbenchError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "请求参数不完整或格式错误", "errors": exc.errors()})


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "resources": RESOURCES}


@app.get("/api/browse")
def browse(root: str, path: str = "", refresh: bool = False) -> dict[str, Any]:
    return browse_service.browse(root, path, refresh)


@app.get("/api/episodes/{episode:path}/media/{stream}")
def media(episode: str, stream: str, root: str) -> FileResponse:
    path = episode_service.media(root, episode, stream)
    media_type = "video/mp4" if path.suffix.casefold() == ".mp4" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=None, content_disposition_type="inline")


@app.get("/api/episodes/{episode:path}")
def episode_detail(episode: str, root: str) -> dict[str, Any]:
    value = episode_service.detail(root, episode)
    for stream in value["streams"]:
        stream["mediaUrl"] = (
            f"/api/episodes/{quote(episode, safe='/')}/media/{quote(stream['name'], safe='')}"
            f"?root={quote(str(Path(root).expanduser().resolve()), safe='')}"
        )
    return value


@app.post("/api/operations", status_code=202)
def create_operation(request: OperationRequest) -> dict[str, Any]:
    state = operation_manager.create(request.model_dump(by_alias=True))
    return {
        "id": state.id,
        "status": "queued",
        "progress": 0.0,
        "message": f"队列中，第 {state.submitted_queue_position} 位",
        "queuePosition": state.submitted_queue_position,
        "episode": state.episode,
        "mode": state.mode,
        "submittedAt": state.submitted_at,
        "startedAt": None,
        "ffmpegSlots": 0,
        "operationId": state.id,
        "eventsUrl": f"/api/operations/{state.id}/events",
    }


@app.get("/api/operations")
def operations() -> dict[str, Any]:
    return operation_manager.list_operations()


@app.get("/api/operations/events")
async def operations_events(request: Request) -> StreamingResponse:
    return StreamingResponse(
        operation_snapshot_events(operation_manager, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/operations/settings")
def operation_settings() -> dict[str, int]:
    return operation_manager.settings()


@app.patch("/api/operations/settings")
def update_operation_settings(request: OperationSettingsRequest) -> dict[str, int]:
    return operation_manager.set_concurrency(request.concurrency)


@app.get("/api/operations/{operation_id}")
def operation(operation_id: str) -> dict[str, Any]:
    return operation_manager.get(operation_id).event()


@app.post("/api/operations/{operation_id}/retry-csv")
def retry_csv(operation_id: str) -> dict[str, Any]:
    return operation_manager.retry_csv(operation_id).event()


@app.get("/api/operations/{operation_id}/events")
async def operation_events(operation_id: str, request: Request) -> StreamingResponse:
    state = operation_manager.get(operation_id)

    async def events():
        observed = -1
        last_sent = asyncio.get_running_loop().time()
        while not await request.is_disconnected():
            payload = state.event()
            version = state.version
            if version != observed:
                yield f"event: progress\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                observed = version
                last_sent = asyncio.get_running_loop().time()
            if payload["status"] in {"completed", "completed_csv_failed", "failed"}:
                break
            now = asyncio.get_running_loop().time()
            if now - last_sent >= SSE_KEEPALIVE_SECONDS:
                yield ": keep-alive\n\n"
                last_sent = now
            await asyncio.sleep(SSE_POLL_SECONDS)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/assets", StaticFiles(directory=STATIC_ROOT), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")

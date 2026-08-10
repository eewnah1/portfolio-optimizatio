"""FastAPI dashboard for the ETF next-day signal screener."""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Add repo root to path so `etf_signals` can be imported as a namespace package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "etf_signals" / "output"
LATEST_JSON = OUTPUT_DIR / "latest_signals.json"

app = FastAPI(title="ETF Next-Day Signal Screener", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job status store. Single uvicorn worker is sufficient.
jobs: dict[str, dict] = {}


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "latest_available": LATEST_JSON.exists(),
            "running_jobs": sum(1 for j in jobs.values() if j.get("status") == "running"),
        }
    )


@app.get("/api/v1/signals/latest")
async def latest_signals() -> JSONResponse:
    if not LATEST_JSON.exists():
        raise HTTPException(status_code=404, detail="No signals have been generated yet.")

    def _load() -> dict:
        return json.loads(LATEST_JSON.read_text())

    payload = await asyncio.to_thread(_load)
    return JSONResponse(payload)


@app.post("/api/v1/predict")
async def predict() -> JSONResponse:
    """Trigger a fresh signal generation run in the background."""
    for job_id, status in jobs.items():
        if status.get("status") == "running":
            return JSONResponse({"job_id": job_id, "status": "running"})

    job_id = str(uuid.uuid4())[:8]

    async def _run() -> None:
        jobs[job_id] = {"status": "running", "started": time.time()}
        try:
            from etf_signals.generate_signals import generate

            summary = await asyncio.to_thread(generate)
            jobs[job_id] = {
                "status": "completed",
                "finished": time.time(),
                "result": {
                    "backtest_high_conviction_accuracy": summary.get(
                        "backtest_high_conviction_accuracy"
                    ),
                    "backtest_high_conviction_signals": summary.get(
                        "backtest_high_conviction_signals"
                    ),
                    "processed_count": summary.get("processed_count"),
                },
            }
        except Exception as exc:
            logger.exception("Signal generation failed")
            jobs[job_id] = {"status": "failed", "error": str(exc)}

    asyncio.create_task(_run())
    return JSONResponse({"job_id": job_id, "status": "running"})


@app.get("/api/v1/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    status = jobs.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(status)


# API routes above must be registered before the static mount so they take precedence.
app.mount("/", StaticFiles(directory=str(ROOT / "app" / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)

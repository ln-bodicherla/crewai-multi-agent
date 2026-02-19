"""
FastAPI REST interface for the CrewAI multi-agent system.

Provides endpoints for starting crew executions, monitoring progress,
and retrieving results via HTTP.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.crew import DataPipelineCrew, CrewExecutionResult, ExecutionStatus


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CrewAI Multi-Agent API",
    description=(
        "REST API for orchestrating AI agent crews. Submit tasks to specialized "
        "agents (Researcher, Developer, Reviewer, Data Engineer) that collaborate "
        "to solve complex software engineering and data pipeline challenges."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

crew_manager = DataPipelineCrew(config_path="configs/config.yaml")


class CrewRunRequest(BaseModel):
    """Request schema for starting a crew execution."""
    topic: str = Field(
        description="The topic or task for the crew to work on",
        min_length=10,
        max_length=2000,
        examples=["Design a real-time fraud detection data pipeline"],
    )
    process_type: str = Field(
        default="sequential",
        description="Execution strategy: 'sequential' or 'hierarchical'",
    )
    include_review: bool = Field(
        default=True,
        description="Whether to include code review in the workflow",
    )
    include_docs: bool = Field(
        default=True,
        description="Whether to include documentation generation",
    )


class CrewRunResponse(BaseModel):
    """Response schema for crew execution initiation."""
    execution_id: str
    status: str
    topic: str
    process_type: str
    message: str


class CrewStatusResponse(BaseModel):
    """Response schema for execution status queries."""
    execution_id: str
    status: str
    topic: str
    process_type: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    current_task: Optional[str] = None
    tasks_completed: int = 0
    error: Optional[str] = None


class CrewResultResponse(BaseModel):
    """Response schema for completed execution results."""
    execution_id: str
    status: str
    topic: str
    process_type: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    raw_output: str = ""
    task_outputs: list[dict[str, Any]] = []
    token_usage: dict[str, int] = {}
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response schema for health checks."""
    status: str
    timestamp: str
    version: str
    active_executions: int
    total_executions: int


def _run_crew_task(
    execution_id: str,
    topic: str,
    process_type: str,
    include_review: bool,
    include_docs: bool,
) -> None:
    """Background task that runs the crew execution."""
    logger.info(f"Background execution started: {execution_id}")
    try:
        if process_type == "hierarchical":
            crew_manager.run_hierarchical(
                topic=topic,
                include_review=include_review,
                include_docs=include_docs,
            )
        else:
            crew_manager.run_sequential(
                topic=topic,
                include_review=include_review,
                include_docs=include_docs,
            )
    except Exception as e:
        logger.error(f"Background execution failed: {execution_id} - {e}")


@app.post(
    "/crew/run",
    response_model=CrewRunResponse,
    status_code=202,
    summary="Start a crew execution",
    description=(
        "Initiates an asynchronous crew execution with the specified topic "
        "and configuration. Returns immediately with an execution ID for "
        "polling status and results."
    ),
)
async def run_crew(
    request: CrewRunRequest,
    background_tasks: BackgroundTasks,
) -> CrewRunResponse:
    """Start a new crew execution in the background."""
    active_count = sum(
        1
        for r in crew_manager.executions.values()
        if r.status == ExecutionStatus.RUNNING
    )
    if active_count >= 3:
        raise HTTPException(
            status_code=429,
            detail="Maximum concurrent executions reached (3). Please wait for active runs to complete.",
        )

    execution = CrewExecutionResult(
        topic=request.topic,
        process_type=request.process_type,
        status=ExecutionStatus.PENDING,
        started_at=datetime.utcnow().isoformat(),
    )
    crew_manager.executions[execution.execution_id] = execution

    background_tasks.add_task(
        _run_crew_task,
        execution.execution_id,
        request.topic,
        request.process_type,
        request.include_review,
        request.include_docs,
    )

    logger.info(
        f"Crew execution queued: {execution.execution_id} "
        f"(topic: {request.topic[:50]}...)"
    )

    return CrewRunResponse(
        execution_id=execution.execution_id,
        status="pending",
        topic=request.topic,
        process_type=request.process_type,
        message="Crew execution started. Poll status endpoint for updates.",
    )


@app.get(
    "/crew/{execution_id}/status",
    response_model=CrewStatusResponse,
    summary="Check execution status",
    description="Returns the current status of a crew execution including progress information.",
)
async def get_crew_status(execution_id: str) -> CrewStatusResponse:
    """Get the current status of a crew execution."""
    result = crew_manager.get_results(execution_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Execution not found: {execution_id}",
        )

    progress = crew_manager.get_progress(execution_id)

    return CrewStatusResponse(
        execution_id=result.execution_id,
        status=result.status.value,
        topic=result.topic,
        process_type=result.process_type,
        started_at=result.started_at,
        completed_at=result.completed_at,
        duration_seconds=result.duration_seconds,
        current_task=progress.get("current_task") if progress else None,
        tasks_completed=progress.get("tasks_completed", 0) if progress else 0,
        error=result.error,
    )


@app.get(
    "/crew/{execution_id}/result",
    response_model=CrewResultResponse,
    summary="Get execution results",
    description="Returns the full results of a completed crew execution.",
)
async def get_crew_result(execution_id: str) -> CrewResultResponse:
    """Get the full results of a completed crew execution."""
    result = crew_manager.get_results(execution_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Execution not found: {execution_id}",
        )

    if result.status == ExecutionStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail="Execution is still running. Check status endpoint for progress.",
        )

    if result.status == ExecutionStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail="Execution has not started yet.",
        )

    return CrewResultResponse(
        execution_id=result.execution_id,
        status=result.status.value,
        topic=result.topic,
        process_type=result.process_type,
        started_at=result.started_at,
        completed_at=result.completed_at,
        duration_seconds=result.duration_seconds,
        raw_output=result.raw_output,
        task_outputs=result.task_outputs,
        token_usage=result.token_usage,
        error=result.error,
    )


@app.get(
    "/crew/executions",
    summary="List all executions",
    description="Returns a summary list of all crew executions.",
)
async def list_executions() -> list[dict[str, Any]]:
    """List all crew executions with status summaries."""
    return crew_manager.list_executions()


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the API health status and execution statistics.",
)
async def health_check() -> HealthResponse:
    """Check API health and return operational metrics."""
    active = sum(
        1
        for r in crew_manager.executions.values()
        if r.status == ExecutionStatus.RUNNING
    )

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        active_executions=active,
        total_executions=len(crew_manager.executions),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

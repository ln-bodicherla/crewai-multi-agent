"""
Crew orchestration for the CrewAI multi-agent system.

Assembles agents and tasks into crews with configurable execution
strategies (sequential, hierarchical) and progress tracking.
"""

import os
import uuid
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import yaml
from crewai import Crew, Process
from pydantic import BaseModel, Field

from src.agents import (
    create_researcher,
    create_developer,
    create_reviewer,
    create_data_engineer,
    create_all_agents,
)
from src.tasks import (
    create_research_task,
    create_implementation_task,
    create_review_task,
    create_pipeline_task,
    create_documentation_task,
)


logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Execution status for crew runs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CrewExecutionResult(BaseModel):
    """Structured result from a crew execution."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    topic: str = ""
    process_type: str = "sequential"
    raw_output: str = ""
    task_outputs: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None


class ProgressCallback:
    """Callback handler for tracking crew execution progress."""

    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.events: list[dict[str, Any]] = []
        self.current_task: Optional[str] = None
        self.tasks_completed: int = 0

    def on_task_start(self, task_description: str) -> None:
        """Called when a task begins execution."""
        self.current_task = task_description[:100]
        event = {
            "type": "task_start",
            "timestamp": datetime.utcnow().isoformat(),
            "task": self.current_task,
            "tasks_completed": self.tasks_completed,
        }
        self.events.append(event)
        logger.info(
            f"[{self.execution_id}] Task started: {self.current_task}"
        )

    def on_task_complete(self, task_description: str, output: str) -> None:
        """Called when a task completes successfully."""
        self.tasks_completed += 1
        event = {
            "type": "task_complete",
            "timestamp": datetime.utcnow().isoformat(),
            "task": task_description[:100],
            "output_length": len(output),
            "tasks_completed": self.tasks_completed,
        }
        self.events.append(event)
        logger.info(
            f"[{self.execution_id}] Task completed: {task_description[:100]} "
            f"({self.tasks_completed} total)"
        )

    def on_error(self, error: str) -> None:
        """Called when an error occurs during execution."""
        event = {
            "type": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": error,
        }
        self.events.append(event)
        logger.error(f"[{self.execution_id}] Error: {error}")

    def get_progress(self) -> dict[str, Any]:
        """Return current execution progress."""
        return {
            "execution_id": self.execution_id,
            "current_task": self.current_task,
            "tasks_completed": self.tasks_completed,
            "events": self.events[-10:],
        }


class DataPipelineCrew:
    """
    Orchestrates a crew of AI agents for data pipeline development tasks.

    Supports sequential and hierarchical execution strategies with
    configurable agents, tasks, and progress tracking.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.agents = {}
        self.executions: dict[str, CrewExecutionResult] = {}
        self.callbacks: dict[str, ProgressCallback] = {}

    def _load_config(self) -> dict:
        """Load crew configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_crew(
        self,
        topic: str,
        process: Process = Process.sequential,
        include_review: bool = True,
        include_docs: bool = True,
    ) -> Crew:
        """
        Assemble agents and tasks into a crew for execution.

        Configures the agent roster, creates appropriate tasks with
        dependencies, and returns a ready-to-run Crew instance.
        """
        self.agents = create_all_agents(config_path=self.config_path)

        researcher = self.agents["researcher"]
        developer = self.agents["developer"]
        reviewer = self.agents["reviewer"]
        data_engineer = self.agents["data_engineer"]

        research_task = create_research_task(
            agent=researcher,
            topic=topic,
        )

        pipeline_task = create_pipeline_task(
            agent=data_engineer,
            pipeline_description=topic,
            context=[research_task],
        )

        implementation_task = create_implementation_task(
            agent=developer,
            specification=f"Implement the data pipeline for: {topic}",
            context=[research_task, pipeline_task],
            language="Python",
        )

        tasks = [research_task, pipeline_task, implementation_task]

        if include_review:
            review_task = create_review_task(
                agent=reviewer,
                context=[implementation_task],
                review_focus="data_quality",
            )
            tasks.append(review_task)

        if include_docs:
            doc_agent = researcher
            doc_task = create_documentation_task(
                agent=doc_agent,
                context=tasks[-2:],
                doc_type="technical",
                audience="developers",
            )
            tasks.append(doc_task)

        crew_config = self.config.get("crew", {})

        crew = Crew(
            agents=[researcher, developer, reviewer, data_engineer],
            tasks=tasks,
            process=process,
            verbose=crew_config.get("verbose", True),
            memory=crew_config.get("memory", True),
            max_rpm=crew_config.get("max_rpm", 10),
            full_output=crew_config.get("full_output", True),
        )

        return crew

    def run(
        self,
        topic: str,
        include_review: bool = True,
        include_docs: bool = True,
    ) -> CrewExecutionResult:
        """
        Execute the crew with default sequential processing.

        Creates the crew, runs all tasks in sequence, and returns
        structured results with execution metadata.
        """
        process_config = self.config.get("crew", {}).get("process", "sequential")
        if process_config == "hierarchical":
            return self.run_hierarchical(topic, include_review, include_docs)
        return self.run_sequential(topic, include_review, include_docs)

    def run_sequential(
        self,
        topic: str,
        include_review: bool = True,
        include_docs: bool = True,
    ) -> CrewExecutionResult:
        """
        Execute the crew with sequential task processing.

        Tasks execute in order, each building on the outputs of
        previous tasks through context passing.
        """
        execution = CrewExecutionResult(
            topic=topic,
            process_type="sequential",
        )
        self.executions[execution.execution_id] = execution
        callback = ProgressCallback(execution.execution_id)
        self.callbacks[execution.execution_id] = callback

        execution.status = ExecutionStatus.RUNNING
        execution.started_at = datetime.utcnow().isoformat()

        logger.info(
            f"Starting sequential crew execution: {execution.execution_id}"
        )

        try:
            crew = self.setup_crew(
                topic=topic,
                process=Process.sequential,
                include_review=include_review,
                include_docs=include_docs,
            )

            result = crew.kickoff()

            execution.status = ExecutionStatus.COMPLETED
            execution.completed_at = datetime.utcnow().isoformat()

            if hasattr(result, "raw"):
                execution.raw_output = str(result.raw)
            else:
                execution.raw_output = str(result)

            if hasattr(result, "tasks_output"):
                for i, task_output in enumerate(result.tasks_output):
                    execution.task_outputs.append(
                        {
                            "task_index": i,
                            "output": str(task_output.raw) if hasattr(task_output, "raw") else str(task_output),
                            "description": str(task_output.description)[:200] if hasattr(task_output, "description") else "",
                        }
                    )

            if hasattr(result, "token_usage"):
                execution.token_usage = dict(result.token_usage)

            started = datetime.fromisoformat(execution.started_at)
            completed = datetime.fromisoformat(execution.completed_at)
            execution.duration_seconds = (completed - started).total_seconds()

            logger.info(
                f"Crew execution completed: {execution.execution_id} "
                f"({execution.duration_seconds:.1f}s)"
            )

        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.completed_at = datetime.utcnow().isoformat()
            execution.error = str(e)
            callback.on_error(str(e))
            logger.error(
                f"Crew execution failed: {execution.execution_id} - {e}"
            )

        return execution

    def run_hierarchical(
        self,
        topic: str,
        include_review: bool = True,
        include_docs: bool = True,
    ) -> CrewExecutionResult:
        """
        Execute the crew with hierarchical (manager-delegated) processing.

        A manager agent coordinates task delegation to specialists,
        deciding which agent handles each subtask based on the
        agent's role and capabilities.
        """
        execution = CrewExecutionResult(
            topic=topic,
            process_type="hierarchical",
        )
        self.executions[execution.execution_id] = execution
        callback = ProgressCallback(execution.execution_id)
        self.callbacks[execution.execution_id] = callback

        execution.status = ExecutionStatus.RUNNING
        execution.started_at = datetime.utcnow().isoformat()

        logger.info(
            f"Starting hierarchical crew execution: {execution.execution_id}"
        )

        try:
            crew = self.setup_crew(
                topic=topic,
                process=Process.hierarchical,
                include_review=include_review,
                include_docs=include_docs,
            )

            result = crew.kickoff()

            execution.status = ExecutionStatus.COMPLETED
            execution.completed_at = datetime.utcnow().isoformat()

            if hasattr(result, "raw"):
                execution.raw_output = str(result.raw)
            else:
                execution.raw_output = str(result)

            if hasattr(result, "tasks_output"):
                for i, task_output in enumerate(result.tasks_output):
                    execution.task_outputs.append(
                        {
                            "task_index": i,
                            "output": str(task_output.raw) if hasattr(task_output, "raw") else str(task_output),
                        }
                    )

            started = datetime.fromisoformat(execution.started_at)
            completed = datetime.fromisoformat(execution.completed_at)
            execution.duration_seconds = (completed - started).total_seconds()

        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.completed_at = datetime.utcnow().isoformat()
            execution.error = str(e)
            callback.on_error(str(e))
            logger.error(
                f"Hierarchical execution failed: {execution.execution_id} - {e}"
            )

        return execution

    def get_results(
        self, execution_id: str
    ) -> Optional[CrewExecutionResult]:
        """
        Retrieve structured results for a completed execution.

        Returns the full execution result including raw output,
        individual task outputs, token usage, and timing metadata.
        """
        return self.executions.get(execution_id)

    def get_progress(self, execution_id: str) -> Optional[dict[str, Any]]:
        """
        Get real-time progress for an active execution.

        Returns the current task, completed count, and recent events.
        """
        callback = self.callbacks.get(execution_id)
        if callback:
            return callback.get_progress()
        return None

    def list_executions(self) -> list[dict[str, Any]]:
        """List all executions with their status summaries."""
        return [
            {
                "execution_id": eid,
                "status": result.status.value,
                "topic": result.topic,
                "process_type": result.process_type,
                "started_at": result.started_at,
                "duration_seconds": result.duration_seconds,
            }
            for eid, result in self.executions.items()
        ]

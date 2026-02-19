"""
Agent definitions for the CrewAI multi-agent system.

Defines specialized agents with distinct roles, goals, backstories,
and tool configurations for collaborative task execution.
"""

import os
from typing import Optional

import yaml
from crewai import Agent
from langchain_openai import ChatOpenAI

from tools.custom_tools import (
    CodeAnalyzerTool,
    DatabaseSchemaTool,
    DataProfilerTool,
    APITesterTool,
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load agent configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _create_llm(config: dict) -> ChatOpenAI:
    """Create the language model instance from configuration."""
    llm_config = config.get("llm", {})
    return ChatOpenAI(
        model=llm_config.get("model", "gpt-4"),
        temperature=llm_config.get("temperature", 0.2),
        max_tokens=llm_config.get("max_tokens", 4096),
        request_timeout=llm_config.get("request_timeout", 120),
    )


def create_researcher(
    config: Optional[dict] = None,
    config_path: str = "configs/config.yaml",
) -> Agent:
    """
    Create the Senior Research Analyst agent.

    This agent specializes in gathering information from multiple sources,
    analyzing technical documentation, and producing structured research
    summaries. Equipped with web search and document analysis tools.

    The researcher can delegate subtasks to other agents when deep
    technical analysis is needed beyond its research scope.
    """
    if config is None:
        config = load_config(config_path)

    llm = _create_llm(config)
    agent_config = config.get("agents", {}).get("researcher", {})

    code_analyzer = CodeAnalyzerTool()
    data_profiler = DataProfilerTool()

    return Agent(
        role=agent_config.get("role", "Senior Research Analyst"),
        goal=agent_config.get(
            "goal",
            "Conduct thorough research and provide comprehensive analysis on technical topics",
        ),
        backstory=agent_config.get(
            "backstory",
            "You are a seasoned technology research analyst with 15 years of "
            "experience in software architecture, data engineering, and ML systems. "
            "You excel at synthesizing information from multiple sources into clear, "
            "actionable insights.",
        ),
        tools=[code_analyzer, data_profiler],
        llm=llm,
        allow_delegation=agent_config.get("allow_delegation", True),
        max_iter=agent_config.get("max_iter", 5),
        verbose=agent_config.get("verbose", True),
        memory=True,
    )


def create_developer(
    config: Optional[dict] = None,
    config_path: str = "configs/config.yaml",
) -> Agent:
    """
    Create the Senior Software Developer agent.

    This agent implements code based on research and specifications.
    Equipped with code generation, file writing, and code analysis tools.
    Follows SOLID principles and produces well-tested, maintainable code.

    The developer does not delegate -- it implements directly based on
    inputs from research and specifications.
    """
    if config is None:
        config = load_config(config_path)

    llm = _create_llm(config)
    agent_config = config.get("agents", {}).get("developer", {})

    code_analyzer = CodeAnalyzerTool()
    db_schema = DatabaseSchemaTool()
    api_tester = APITesterTool()

    return Agent(
        role=agent_config.get("role", "Senior Software Developer"),
        goal=agent_config.get(
            "goal",
            "Write clean, efficient, and well-tested code based on specifications",
        ),
        backstory=agent_config.get(
            "backstory",
            "You are a senior software developer with deep expertise in Python, "
            "Java, and distributed systems. You follow SOLID principles, write "
            "comprehensive tests, and produce performant, maintainable code.",
        ),
        tools=[code_analyzer, db_schema, api_tester],
        llm=llm,
        allow_delegation=agent_config.get("allow_delegation", False),
        max_iter=agent_config.get("max_iter", 10),
        verbose=agent_config.get("verbose", True),
        memory=True,
    )


def create_reviewer(
    config: Optional[dict] = None,
    config_path: str = "configs/config.yaml",
) -> Agent:
    """
    Create the Code Review Specialist agent.

    This agent reviews code for bugs, security vulnerabilities,
    performance issues, and adherence to best practices. Equipped
    with static analysis and testing tools.

    The reviewer operates independently, providing detailed feedback
    on submitted code without delegating to other agents.
    """
    if config is None:
        config = load_config(config_path)

    llm = _create_llm(config)
    agent_config = config.get("agents", {}).get("reviewer", {})

    code_analyzer = CodeAnalyzerTool()
    api_tester = APITesterTool()

    return Agent(
        role=agent_config.get("role", "Code Review Specialist"),
        goal=agent_config.get(
            "goal",
            "Ensure code quality, security, and adherence to best practices",
        ),
        backstory=agent_config.get(
            "backstory",
            "You are a meticulous code reviewer with expertise in software security, "
            "performance optimization, and clean code practices. You catch subtle bugs, "
            "identify security vulnerabilities, and suggest improvements for robustness.",
        ),
        tools=[code_analyzer, api_tester],
        llm=llm,
        allow_delegation=agent_config.get("allow_delegation", False),
        max_iter=agent_config.get("max_iter", 5),
        verbose=agent_config.get("verbose", True),
        memory=True,
    )


def create_data_engineer(
    config: Optional[dict] = None,
    config_path: str = "configs/config.yaml",
) -> Agent:
    """
    Create the Data Pipeline Engineer agent.

    This agent designs and implements data processing pipelines
    with proper error handling, data quality checks, and monitoring.
    Equipped with database schema, data profiling, and analysis tools.

    The data engineer can delegate research subtasks to the researcher
    when exploring unfamiliar data sources or technologies.
    """
    if config is None:
        config = load_config(config_path)

    llm = _create_llm(config)
    agent_config = config.get("agents", {}).get("data_engineer", {})

    code_analyzer = CodeAnalyzerTool()
    db_schema = DatabaseSchemaTool()
    data_profiler = DataProfilerTool()

    return Agent(
        role=agent_config.get("role", "Data Pipeline Engineer"),
        goal=agent_config.get(
            "goal",
            "Design and implement robust, scalable data processing pipelines",
        ),
        backstory=agent_config.get(
            "backstory",
            "You are a data pipeline engineer specializing in ETL/ELT workflows, "
            "stream processing, and data quality. You design systems that handle "
            "data at scale with proper error handling, monitoring, and governance.",
        ),
        tools=[code_analyzer, db_schema, data_profiler],
        llm=llm,
        allow_delegation=agent_config.get("allow_delegation", True),
        max_iter=agent_config.get("max_iter", 8),
        verbose=agent_config.get("verbose", True),
        memory=True,
    )


def create_all_agents(
    config_path: str = "configs/config.yaml",
) -> dict[str, Agent]:
    """
    Create all agents from configuration.

    Returns a dictionary mapping agent names to Agent instances,
    enabling flexible crew assembly based on task requirements.
    """
    config = load_config(config_path)

    return {
        "researcher": create_researcher(config=config),
        "developer": create_developer(config=config),
        "reviewer": create_reviewer(config=config),
        "data_engineer": create_data_engineer(config=config),
    }

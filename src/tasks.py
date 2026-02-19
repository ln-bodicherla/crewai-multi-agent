"""
Task definitions for the CrewAI multi-agent system.

Defines structured tasks with descriptions, expected outputs, and
agent assignments for the crew orchestration pipeline.
"""

from typing import Optional

from crewai import Agent, Task


def create_research_task(
    agent: Agent,
    topic: str,
    context: Optional[list[Task]] = None,
    additional_instructions: str = "",
) -> Task:
    """
    Create a research task for gathering and analyzing information.

    The research task instructs the agent to investigate a topic,
    gather relevant sources, evaluate trade-offs, and produce
    a structured summary with actionable recommendations.
    """
    description = f"""Research the following topic thoroughly and provide a comprehensive analysis:

Topic: {topic}

Your research should cover:
1. Current state of the art and industry best practices
2. Available tools, frameworks, and technologies relevant to the topic
3. Architecture patterns and design considerations
4. Common pitfalls and how to avoid them
5. Performance considerations and scalability factors
6. Security implications and compliance requirements

Provide specific, actionable recommendations based on your findings.
Include concrete examples and code patterns where relevant.
{additional_instructions}"""

    expected_output = """A structured research report containing:
1. Executive Summary (2-3 paragraphs)
2. Technology Landscape (tools, frameworks, versions)
3. Architecture Recommendations (with diagrams described in text)
4. Implementation Strategy (phased approach)
5. Risk Assessment (identified risks with mitigations)
6. References and Resources (specific documentation links)"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context or [],
    )


def create_implementation_task(
    agent: Agent,
    specification: str,
    context: Optional[list[Task]] = None,
    language: str = "Python",
    additional_requirements: str = "",
) -> Task:
    """
    Create an implementation task for code generation.

    The implementation task instructs the developer agent to write
    production-quality code based on a specification, incorporating
    any context from prior research or review tasks.
    """
    description = f"""Implement the following specification in {language}:

Specification: {specification}

Requirements:
1. Write clean, production-quality code following {language} best practices
2. Include proper error handling and input validation
3. Add type hints and comprehensive docstrings
4. Implement logging at appropriate levels
5. Write the code to be testable with dependency injection where appropriate
6. Follow SOLID principles and relevant design patterns
7. Include configuration management (no hardcoded values)
8. Handle edge cases gracefully

Code Organization:
- Use clear module and class structure
- Separate concerns (data access, business logic, presentation)
- Include constants and configuration at the top of modules
- Group related functions and classes logically
{additional_requirements}"""

    expected_output = f"""Complete, runnable {language} code including:
1. Main implementation module(s) with full functionality
2. Data models / schemas with validation
3. Configuration handling
4. Error handling with custom exception classes
5. Inline documentation explaining design decisions
6. Example usage in a main block or separate example script"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context or [],
    )


def create_review_task(
    agent: Agent,
    context: Optional[list[Task]] = None,
    review_focus: str = "general",
    additional_criteria: str = "",
) -> Task:
    """
    Create a code review task for quality assurance.

    The review task instructs the reviewer agent to analyze code
    produced by the developer, checking for bugs, security issues,
    performance problems, and adherence to standards.
    """
    focus_areas = {
        "general": """
- Code correctness and logic errors
- Error handling completeness
- Input validation and sanitization
- Code readability and maintainability
- Adherence to language idioms and conventions
- Test coverage adequacy""",
        "security": """
- Input validation and sanitization (SQL injection, XSS, etc.)
- Authentication and authorization patterns
- Sensitive data handling (credentials, PII)
- Dependency vulnerabilities
- Secure communication (TLS, encryption)
- Logging of sensitive information""",
        "performance": """
- Algorithm complexity and optimization opportunities
- Memory allocation patterns and potential leaks
- Database query efficiency (N+1 queries, missing indexes)
- Caching opportunities
- Concurrency and thread safety
- I/O optimization (batching, streaming)""",
        "data_quality": """
- Data validation at ingestion points
- Null handling and default value strategy
- Type safety and schema enforcement
- Duplicate handling logic
- Data lineage and audit trail
- Error recovery and data reconciliation""",
    }

    focus_description = focus_areas.get(review_focus, focus_areas["general"])

    description = f"""Conduct a thorough code review of the implementation provided in the context.

Review Focus: {review_focus.replace('_', ' ').title()}

Evaluate the code against these criteria:
{focus_description}

For each issue found:
1. Classify severity (Critical, Major, Minor, Suggestion)
2. Identify the specific location in the code
3. Explain the issue clearly
4. Provide a concrete fix or improvement
{additional_criteria}

Also provide positive feedback on well-implemented aspects of the code."""

    expected_output = """A structured code review report containing:
1. Review Summary (overall assessment, key findings)
2. Critical Issues (must fix before deployment)
3. Major Issues (should fix in current iteration)
4. Minor Issues (nice to fix, low priority)
5. Suggestions (improvements for future iterations)
6. Positive Observations (well-implemented patterns)
7. Revised Code Snippets (for critical and major issues)"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context or [],
    )


def create_pipeline_task(
    agent: Agent,
    pipeline_description: str,
    context: Optional[list[Task]] = None,
    data_sources: str = "various",
    output_format: str = "parquet",
    additional_requirements: str = "",
) -> Task:
    """
    Create a data pipeline design and implementation task.

    The pipeline task instructs the data engineer to design a complete
    data processing pipeline with ingestion, transformation, quality
    checks, and output stages.
    """
    description = f"""Design and implement a data pipeline based on the following requirements:

Pipeline Description: {pipeline_description}

Data Sources: {data_sources}
Output Format: {output_format}

Pipeline Requirements:
1. Data Ingestion
   - Source connectivity with retry logic
   - Schema discovery and validation
   - Incremental loading support
   - Raw data archival

2. Data Transformation
   - Cleaning (null handling, deduplication, outlier treatment)
   - Type enforcement and schema evolution handling
   - Business rule application
   - Derived column computation

3. Data Quality
   - Schema validation at ingestion
   - Completeness checks (null rates, row counts)
   - Freshness monitoring
   - Distribution drift detection
   - Referential integrity checks

4. Pipeline Operations
   - Idempotent execution (safe re-runs)
   - Checkpoint and recovery
   - Monitoring and alerting
   - Logging with structured context
   - Configuration-driven behavior

5. Output
   - Partitioned storage in {output_format} format
   - Metadata and lineage tracking
   - Output validation before publishing
{additional_requirements}"""

    expected_output = """A complete data pipeline implementation including:
1. Pipeline Architecture Diagram (described in text)
2. Source code for all pipeline stages
3. Configuration files (YAML/JSON)
4. Data quality rule definitions
5. Monitoring and alerting setup
6. Deployment instructions
7. Runbook for common operational scenarios"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context or [],
    )


def create_documentation_task(
    agent: Agent,
    context: Optional[list[Task]] = None,
    doc_type: str = "technical",
    audience: str = "developers",
    additional_sections: str = "",
) -> Task:
    """
    Create a documentation generation task.

    The documentation task generates comprehensive technical or
    user-facing documentation based on the implementation context
    from previous tasks.
    """
    doc_structures = {
        "technical": """
1. Architecture Overview
2. System Components and Interactions
3. Data Flow Diagrams (described in text/ASCII)
4. API Reference (endpoints, parameters, responses)
5. Configuration Reference
6. Deployment Guide
7. Monitoring and Troubleshooting""",
        "user": """
1. Getting Started Guide
2. Installation and Setup
3. Quick Start Tutorial
4. Feature Walkthrough
5. Configuration Options
6. FAQ and Common Issues
7. Support Resources""",
        "api": """
1. Authentication
2. Rate Limiting
3. Endpoint Reference (grouped by resource)
4. Request/Response Examples
5. Error Codes and Handling
6. SDKs and Client Libraries
7. Changelog""",
        "runbook": """
1. System Overview
2. Normal Operations
3. Deployment Procedures
4. Health Check Procedures
5. Incident Response Playbooks
6. Scaling Procedures
7. Backup and Recovery""",
    }

    doc_structure = doc_structures.get(doc_type, doc_structures["technical"])

    description = f"""Generate comprehensive {doc_type} documentation for {audience}.

Based on the implementation and research context provided, create
documentation that covers:
{doc_structure}
{additional_sections}

Documentation Guidelines:
- Use clear, concise language appropriate for {audience}
- Include code examples and configuration snippets
- Provide ASCII diagrams for architecture visualization
- Include troubleshooting sections with common issues
- Reference specific files, classes, and functions from the implementation
- Follow standard documentation formatting (Markdown)"""

    expected_output = f"""Complete {doc_type} documentation in Markdown format including:
1. All sections listed above with substantive content
2. Code examples with syntax highlighting markers
3. Configuration examples with comments
4. Architecture diagrams in ASCII art
5. Table of contents for navigation
6. Cross-references between sections"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context or [],
    )

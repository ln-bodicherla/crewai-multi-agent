# CrewAI Multi-Agent System

Role-based AI agent collaboration using CrewAI for complex software engineering and data pipeline tasks. Specialized agents (Researcher, Developer, Reviewer, Data Engineer) work together with automatic delegation and configurable execution strategies.

## Architecture

```
                    ┌──────────────┐
                    │   Crew API   │
                    │  (FastAPI)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │     Crew     │
                    │ Orchestrator │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
   ┌─────▼─────┐   ┌──────▼──────┐   ┌─────▼─────┐
   │ Researcher │   │  Developer  │   │  Reviewer  │
   │   Agent    │   │    Agent    │   │   Agent    │
   └─────┬─────┘   └──────┬──────┘   └─────┬─────┘
         │                 │                 │
   ┌─────▼─────┐   ┌──────▼──────┐   ┌─────▼─────┐
   │  Search &  │   │  Code Gen & │   │  Analysis  │
   │  Analysis  │   │  File Tools │   │  & Testing │
   │   Tools    │   │             │   │   Tools    │
   └───────────┘   └─────────────┘   └───────────┘
```

### Agent Roles

- **Senior Research Analyst**: Gathers information, analyzes documentation, and produces structured research summaries
- **Senior Software Developer**: Implements code based on specifications with best practices and design patterns
- **Code Review Specialist**: Reviews code for bugs, security issues, performance, and adherence to standards
- **Data Pipeline Engineer**: Designs and implements data processing pipelines with quality checks

## Why CrewAI Over LangGraph

This project explores CrewAI as an alternative to LangGraph for multi-agent systems:

| Feature | CrewAI | LangGraph |
|---------|--------|-----------|
| Abstraction Level | High (role-based) | Low (state machine) |
| Agent Definition | Declarative roles | Programmatic nodes |
| Task Routing | Automatic delegation | Explicit conditional edges |
| Execution Modes | Sequential, Hierarchical | Custom graph topology |
| Learning Curve | Lower | Higher |
| Flexibility | Convention-based | Full control |

**Decision**: LangGraph was chosen for the production system due to its fine-grained control over agent state transitions and conditional routing, which was critical for the complex decision trees in the workflow.

## Project Structure

```
crewai-multi-agent/
├── src/
│   ├── agents.py          # Agent definitions with roles and tools
│   ├── tasks.py           # Task definitions with expected outputs
│   ├── crew.py            # Crew orchestration and execution
│   └── api.py             # FastAPI REST interface
├── tools/
│   └── custom_tools.py    # Custom CrewAI tools
├── configs/
│   └── config.yaml        # Agent and crew configuration
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Run Directly

```python
from src.crew import DataPipelineCrew

crew = DataPipelineCrew(config_path="configs/config.yaml")
result = crew.run(topic="Build a real-time fraud detection pipeline")
print(result)
```

### Run via API

```bash
uvicorn src.api:app --reload --port 8000

# Start a crew execution
curl -X POST http://localhost:8000/crew/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "Design a customer churn prediction pipeline"}'

# Check status
curl http://localhost:8000/crew/{execution_id}/status
```

## Execution Modes

### Sequential
Tasks execute in order, each building on the previous result:
```python
crew.run_sequential(topic="...")
```

### Hierarchical
A manager agent delegates tasks to specialists:
```python
crew.run_hierarchical(topic="...")
```

## Custom Tools

The system includes specialized tools for code and data tasks:

- **CodeAnalyzerTool**: Static analysis for complexity, code smells, and patterns
- **DatabaseSchemaTool**: Introspects database schemas and relationships
- **DataProfilerTool**: Statistical profiling of datasets
- **APITesterTool**: Endpoint testing with assertions

## Tech Stack

- **Framework**: CrewAI
- **LLM**: OpenAI GPT-4 (configurable)
- **API**: FastAPI + Uvicorn
- **Tools**: LangChain tool ecosystem
- **Config**: YAML-based configuration

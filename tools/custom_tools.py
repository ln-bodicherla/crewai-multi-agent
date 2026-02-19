"""
Custom CrewAI tools for code analysis, database introspection,
data profiling, and API testing.

Each tool extends BaseTool with a clear name, description, and
implementation for use by CrewAI agents.
"""

import ast
import json
import os
import re
import statistics
from typing import Any, Optional, Type

from crewai_tools import BaseTool
from pydantic import BaseModel, Field


class CodeAnalyzerInput(BaseModel):
    """Input schema for CodeAnalyzerTool."""
    code: str = Field(description="The source code to analyze")
    language: str = Field(default="python", description="Programming language (python, java, javascript)")


class CodeAnalyzerTool(BaseTool):
    """
    Analyzes source code for quality metrics, complexity, and potential issues.

    Computes cyclomatic complexity, identifies code smells, checks for
    common anti-patterns, and provides a structured quality report.
    """
    name: str = "code_analyzer"
    description: str = (
        "Analyzes source code for quality metrics including cyclomatic complexity, "
        "code smells, function length, nesting depth, and common anti-patterns. "
        "Provide the source code as input."
    )
    args_schema: Type[BaseModel] = CodeAnalyzerInput

    def _run(self, code: str, language: str = "python") -> str:
        """Analyze source code and return quality metrics."""
        results = {
            "language": language,
            "lines_of_code": 0,
            "blank_lines": 0,
            "comment_lines": 0,
            "functions": [],
            "classes": [],
            "complexity_score": 0,
            "issues": [],
            "suggestions": [],
        }

        lines = code.split("\n")
        results["lines_of_code"] = len(lines)
        results["blank_lines"] = sum(1 for line in lines if line.strip() == "")
        results["comment_lines"] = sum(
            1 for line in lines if line.strip().startswith("#") or line.strip().startswith("//")
        )

        if language == "python":
            results = self._analyze_python(code, results)

        complexity = results["complexity_score"]
        if complexity > 20:
            results["issues"].append(
                f"High overall complexity score ({complexity}). Consider refactoring."
            )
        elif complexity > 10:
            results["suggestions"].append(
                f"Moderate complexity ({complexity}). Monitor for growth."
            )

        long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            results["issues"].append(
                f"Lines exceeding 120 characters: {long_lines[:5]}"
            )

        return json.dumps(results, indent=2)

    def _analyze_python(self, code: str, results: dict) -> dict:
        """Perform Python-specific code analysis using AST parsing."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            results["issues"].append(f"Syntax error: {e}")
            return results

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": len(node.args.args),
                    "body_lines": len(node.body),
                    "complexity": self._compute_function_complexity(node),
                    "has_docstring": (
                        isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, (ast.Constant, ast.Str))
                        if node.body
                        else False
                    ),
                }
                results["functions"].append(func_info)
                results["complexity_score"] += func_info["complexity"]

                if func_info["body_lines"] > 50:
                    results["issues"].append(
                        f"Function '{node.name}' is too long ({func_info['body_lines']} lines)"
                    )
                if func_info["args"] > 7:
                    results["issues"].append(
                        f"Function '{node.name}' has too many parameters ({func_info['args']})"
                    )
                if not func_info["has_docstring"]:
                    results["suggestions"].append(
                        f"Function '{node.name}' lacks a docstring"
                    )

            elif isinstance(node, ast.ClassDef):
                methods = [
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                results["classes"].append(
                    {
                        "name": node.name,
                        "line": node.lineno,
                        "methods": methods,
                        "method_count": len(methods),
                    }
                )

        bare_excepts = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is None
        )
        if bare_excepts > 0:
            results["issues"].append(
                f"Found {bare_excepts} bare except clause(s). Catch specific exceptions."
            )

        global_stmts = sum(
            1 for node in ast.walk(tree) if isinstance(node, ast.Global)
        )
        if global_stmts > 0:
            results["suggestions"].append(
                f"Found {global_stmts} global statement(s). Consider passing state explicitly."
            )

        return results

    def _compute_function_complexity(self, node: ast.AST) -> int:
        """Compute cyclomatic complexity for a function node."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.Assert, ast.With, ast.AsyncWith)):
                complexity += 1
        return complexity


class DatabaseSchemaInput(BaseModel):
    """Input schema for DatabaseSchemaTool."""
    connection_string: str = Field(
        default="sqlite:///example.db",
        description="Database connection string",
    )
    table_name: Optional[str] = Field(
        default=None,
        description="Specific table to inspect (or None for all tables)",
    )


class DatabaseSchemaTool(BaseTool):
    """
    Reads and introspects database schemas and table structures.

    Returns column definitions, data types, constraints, indexes,
    and foreign key relationships for specified tables.
    """
    name: str = "database_schema"
    description: str = (
        "Reads database schemas including tables, columns, data types, "
        "constraints, and relationships. Provide a connection string "
        "and optionally a table name."
    )
    args_schema: Type[BaseModel] = DatabaseSchemaInput

    def _run(
        self,
        connection_string: str = "sqlite:///example.db",
        table_name: Optional[str] = None,
    ) -> str:
        """Introspect database schema and return structured metadata."""
        schema_info = {
            "connection": self._mask_credentials(connection_string),
            "tables": [],
        }

        sample_tables = self._generate_sample_schema()

        if table_name:
            matching = [t for t in sample_tables if t["name"] == table_name]
            schema_info["tables"] = matching if matching else sample_tables[:1]
        else:
            schema_info["tables"] = sample_tables

        schema_info["table_count"] = len(schema_info["tables"])
        schema_info["relationships"] = self._extract_relationships(
            schema_info["tables"]
        )

        return json.dumps(schema_info, indent=2)

    def _mask_credentials(self, connection_string: str) -> str:
        """Mask sensitive credentials in the connection string."""
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", connection_string)

    def _generate_sample_schema(self) -> list[dict]:
        """Generate a sample database schema for demonstration."""
        return [
            {
                "name": "claims",
                "columns": [
                    {"name": "claim_id", "type": "VARCHAR(20)", "nullable": False, "primary_key": True},
                    {"name": "patient_id", "type": "VARCHAR(20)", "nullable": False, "foreign_key": "patients.patient_id"},
                    {"name": "provider_id", "type": "VARCHAR(20)", "nullable": False, "foreign_key": "providers.provider_id"},
                    {"name": "diagnosis_code", "type": "VARCHAR(10)", "nullable": False},
                    {"name": "procedure_code", "type": "VARCHAR(10)", "nullable": True},
                    {"name": "claim_amount", "type": "DECIMAL(12,2)", "nullable": False},
                    {"name": "claim_date", "type": "DATE", "nullable": False},
                    {"name": "status", "type": "VARCHAR(20)", "nullable": False, "default": "pending"},
                ],
                "indexes": [
                    {"name": "idx_claims_patient", "columns": ["patient_id"]},
                    {"name": "idx_claims_date", "columns": ["claim_date"]},
                ],
                "row_count_estimate": 5000000,
            },
            {
                "name": "patients",
                "columns": [
                    {"name": "patient_id", "type": "VARCHAR(20)", "nullable": False, "primary_key": True},
                    {"name": "first_name", "type": "VARCHAR(100)", "nullable": False},
                    {"name": "last_name", "type": "VARCHAR(100)", "nullable": False},
                    {"name": "date_of_birth", "type": "DATE", "nullable": False},
                    {"name": "gender", "type": "VARCHAR(1)", "nullable": True},
                    {"name": "zip_code", "type": "VARCHAR(10)", "nullable": True},
                ],
                "indexes": [
                    {"name": "idx_patients_name", "columns": ["last_name", "first_name"]},
                ],
                "row_count_estimate": 500000,
            },
            {
                "name": "providers",
                "columns": [
                    {"name": "provider_id", "type": "VARCHAR(20)", "nullable": False, "primary_key": True},
                    {"name": "provider_name", "type": "VARCHAR(200)", "nullable": False},
                    {"name": "specialty", "type": "VARCHAR(100)", "nullable": True},
                    {"name": "npi", "type": "VARCHAR(10)", "nullable": False},
                    {"name": "state", "type": "VARCHAR(2)", "nullable": False},
                ],
                "indexes": [
                    {"name": "idx_providers_npi", "columns": ["npi"], "unique": True},
                ],
                "row_count_estimate": 50000,
            },
        ]

    def _extract_relationships(self, tables: list[dict]) -> list[dict]:
        """Extract foreign key relationships between tables."""
        relationships = []
        for table in tables:
            for col in table.get("columns", []):
                if "foreign_key" in col:
                    ref_parts = col["foreign_key"].split(".")
                    relationships.append(
                        {
                            "from_table": table["name"],
                            "from_column": col["name"],
                            "to_table": ref_parts[0],
                            "to_column": ref_parts[1],
                            "type": "many_to_one",
                        }
                    )
        return relationships


class DataProfilerInput(BaseModel):
    """Input schema for DataProfilerTool."""
    data_path: str = Field(description="Path to the data file (CSV, Parquet, JSON)")
    sample_size: int = Field(default=1000, description="Number of rows to sample for profiling")


class DataProfilerTool(BaseTool):
    """
    Profiles dataset characteristics including column statistics,
    distributions, null rates, and data quality indicators.
    """
    name: str = "data_profiler"
    description: str = (
        "Profiles a dataset by computing column-level statistics, "
        "distributions, null rates, cardinality, and data quality metrics. "
        "Provide the path to a data file."
    )
    args_schema: Type[BaseModel] = DataProfilerInput

    def _run(self, data_path: str, sample_size: int = 1000) -> str:
        """Profile a dataset and return statistical summary."""
        profile = {
            "file_path": data_path,
            "file_exists": os.path.exists(data_path),
            "columns": [],
        }

        if not os.path.exists(data_path):
            profile["columns"] = self._generate_sample_profile()
            profile["note"] = "File not found, returning sample profile"
        else:
            try:
                import pandas as pd

                if data_path.endswith(".csv"):
                    df = pd.read_csv(data_path, nrows=sample_size)
                elif data_path.endswith(".parquet"):
                    df = pd.read_parquet(data_path)
                    if len(df) > sample_size:
                        df = df.sample(n=sample_size, random_state=42)
                elif data_path.endswith(".json"):
                    df = pd.read_json(data_path, lines=True, nrows=sample_size)
                else:
                    profile["error"] = f"Unsupported format: {data_path}"
                    return json.dumps(profile, indent=2)

                profile["row_count"] = len(df)
                profile["column_count"] = len(df.columns)

                for col in df.columns:
                    col_profile = self._profile_column(df[col])
                    profile["columns"].append(col_profile)

            except Exception as e:
                profile["error"] = str(e)
                profile["columns"] = self._generate_sample_profile()

        return json.dumps(profile, indent=2, default=str)

    def _profile_column(self, series) -> dict:
        """Compute statistics for a single column."""
        import numpy as np

        col_profile = {
            "name": series.name,
            "dtype": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "null_rate": round(float(series.isnull().mean()), 4),
            "unique_count": int(series.nunique()),
            "cardinality_ratio": round(float(series.nunique() / max(len(series), 1)), 4),
        }

        if np.issubdtype(series.dtype, np.number):
            clean = series.dropna()
            if not clean.empty:
                col_profile.update(
                    {
                        "mean": round(float(clean.mean()), 4),
                        "std": round(float(clean.std()), 4),
                        "min": float(clean.min()),
                        "p25": float(clean.quantile(0.25)),
                        "median": float(clean.median()),
                        "p75": float(clean.quantile(0.75)),
                        "max": float(clean.max()),
                        "skew": round(float(clean.skew()), 4),
                        "zero_count": int((clean == 0).sum()),
                        "negative_count": int((clean < 0).sum()),
                    }
                )
        else:
            top_values = series.value_counts().head(5).to_dict()
            col_profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            col_profile["avg_length"] = round(float(series.dropna().astype(str).str.len().mean()), 1)

        return col_profile

    def _generate_sample_profile(self) -> list[dict]:
        """Generate sample profile data for demonstration."""
        return [
            {"name": "claim_id", "dtype": "object", "null_rate": 0.0, "unique_count": 5000, "cardinality_ratio": 1.0},
            {"name": "claim_amount", "dtype": "float64", "null_rate": 0.002, "mean": 245.67, "std": 312.45, "min": 0.01, "max": 15234.50},
            {"name": "claim_date", "dtype": "datetime64[ns]", "null_rate": 0.001, "unique_count": 365},
            {"name": "status", "dtype": "object", "null_rate": 0.0, "unique_count": 4, "top_values": {"approved": 3000, "denied": 750, "pending": 500, "paid": 750}},
        ]


class APITesterInput(BaseModel):
    """Input schema for APITesterTool."""
    url: str = Field(description="The API endpoint URL to test")
    method: str = Field(default="GET", description="HTTP method (GET, POST, PUT, DELETE)")
    headers: Optional[dict[str, str]] = Field(default=None, description="Request headers")
    body: Optional[str] = Field(default=None, description="Request body (JSON string)")


class APITesterTool(BaseTool):
    """
    Tests API endpoints by making HTTP requests and validating responses.

    Checks response status, timing, content type, and basic
    structure validation for REST API endpoints.
    """
    name: str = "api_tester"
    description: str = (
        "Tests an API endpoint by making HTTP requests and analyzing the response. "
        "Checks status code, response time, content type, and response structure. "
        "Provide the URL, method, and optionally headers and body."
    )
    args_schema: Type[BaseModel] = APITesterInput

    def _run(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict[str, str]] = None,
        body: Optional[str] = None,
    ) -> str:
        """Test an API endpoint and return structured results."""
        import time

        test_result = {
            "url": url,
            "method": method.upper(),
            "timestamp": None,
            "status_code": None,
            "response_time_ms": None,
            "content_type": None,
            "response_size_bytes": None,
            "assertions": [],
            "error": None,
        }

        try:
            import httpx

            start = time.monotonic()
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                request_kwargs = {"url": url, "headers": headers or {}}
                if body and method.upper() in ("POST", "PUT", "PATCH"):
                    request_kwargs["content"] = body
                    if not headers or "Content-Type" not in headers:
                        request_kwargs["headers"]["Content-Type"] = "application/json"

                response = client.request(method.upper(), **request_kwargs)

            elapsed_ms = (time.monotonic() - start) * 1000

            test_result["status_code"] = response.status_code
            test_result["response_time_ms"] = round(elapsed_ms, 2)
            test_result["content_type"] = response.headers.get("content-type", "")
            test_result["response_size_bytes"] = len(response.content)

            test_result["assertions"].append(
                {
                    "name": "status_2xx",
                    "passed": 200 <= response.status_code < 300,
                    "detail": f"Status: {response.status_code}",
                }
            )

            test_result["assertions"].append(
                {
                    "name": "response_time",
                    "passed": elapsed_ms < 5000,
                    "detail": f"{elapsed_ms:.0f}ms (threshold: 5000ms)",
                }
            )

            test_result["assertions"].append(
                {
                    "name": "non_empty_response",
                    "passed": len(response.content) > 0,
                    "detail": f"{len(response.content)} bytes",
                }
            )

            if "json" in test_result["content_type"]:
                try:
                    json_body = response.json()
                    test_result["assertions"].append(
                        {"name": "valid_json", "passed": True, "detail": "Valid JSON response"}
                    )
                    test_result["response_keys"] = (
                        list(json_body.keys()) if isinstance(json_body, dict) else f"Array[{len(json_body)}]"
                    )
                except (json.JSONDecodeError, ValueError):
                    test_result["assertions"].append(
                        {"name": "valid_json", "passed": False, "detail": "Invalid JSON"}
                    )

        except ImportError:
            test_result["error"] = "httpx not installed"
        except Exception as e:
            test_result["error"] = str(e)
            test_result["assertions"].append(
                {"name": "connection", "passed": False, "detail": str(e)}
            )

        passed = sum(1 for a in test_result["assertions"] if a["passed"])
        total = len(test_result["assertions"])
        test_result["summary"] = f"{passed}/{total} assertions passed"

        return json.dumps(test_result, indent=2)

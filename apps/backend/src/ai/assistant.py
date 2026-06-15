"""NLP-to-SQL assistant using LlamaIndex Text-to-SQL query engine.

Provides an async ``ask_assistant`` function that translates natural
language questions into SQL, executes them safely against a read-only
database, and synthesises a human-readable response.  Exposes the
underlying ``NLSQLTableQueryEngine`` factory via
``get_assistant_query_engine`` for advanced use cases.
"""

from llama_index.core import SQLDatabase
from llama_index.core.base.response.schema import (
    AsyncStreamingResponse,
    PydanticResponse,
    Response,
    StreamingResponse,
)
from llama_index.core.query_engine import NLSQLTableQueryEngine

from src.ai.llm import LLMFactory
from src.db.readonly import get_readonly_engine

AssistantResponse = (
    Response | StreamingResponse | AsyncStreamingResponse | PydanticResponse
)


def get_assistant_query_engine() -> NLSQLTableQueryEngine:
    """Build an ``NLSQLTableQueryEngine`` configured for forensic queries.

    Uses the SQL-optimised LLM profile (120 s timeout) via
    ``LLMFactory.create("sql")`` and the read-only database connection.
    Table lists are explicitly declared to limit the schema context
    window and prevent out-of-memory errors on large schemas.

    Returns:
        A ready-to-use ``NLSQLTableQueryEngine``.
    """
    llm = LLMFactory.create("sql")
    engine = get_readonly_engine()

    sql_database = SQLDatabase(
        engine,
        include_tables=["peritajes", "evidencia"],
    )

    return NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["peritajes", "evidencia"],
        llm=llm,
        synthesize_response=True,
    )


async def ask_assistant(query: str) -> str:
    """Ask a natural-language question and receive a synthesised answer.

    The pipeline:
    1. Creates an LLM tuned for SQL generation.
    2. Connects to the database via the read-only engine.
    3. Builds an ``NLSQLTableQueryEngine`` scoped to forensic tables.
    4. Executes the query and returns the synthesised response.

    Args:
        query: Natural language question in Spanish, e.g.
               ``"¿Cuántos peritajes hay?"``.

    Returns:
        A synthesised text response from the LLM.
    """
    query_engine = get_assistant_query_engine()
    response: AssistantResponse = await query_engine.aquery(query)
    return str(response)

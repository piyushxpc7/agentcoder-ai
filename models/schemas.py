"""Pydantic models for the AgentCoder AI application."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    OTHER = "other"


class Severity(str, Enum):
    """Issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AgentRole(str, Enum):
    """Agent roles in the system."""
    CODE_GENERATOR = "code_generator"
    CODE_ANALYZER = "code_analyzer"
    RAG_AGENT = "rag_agent"


# ============== Code Generation Models ==============

class CodeGenerationRequest(BaseModel):
    """Request model for code generation."""
    prompt: str = Field(..., description="User's code generation request")
    language: Optional[Language] = Field(None, description="Target programming language")
    context: Optional[str] = Field(None, description="Additional context or requirements")
    include_tests: bool = Field(False, description="Whether to include unit tests")
    include_docs: bool = Field(True, description="Whether to include documentation")


class CodeGenerationResponse(BaseModel):
    """Response model for generated code."""
    code: str = Field(..., description="Generated code")
    language: Language = Field(..., description="Programming language used")
    explanation: str = Field(..., description="Explanation of the code")
    filename: Optional[str] = Field(None, description="Suggested filename")
    dependencies: list[str] = Field(default_factory=list, description="Required dependencies")
    tests: Optional[str] = Field(None, description="Generated unit tests if requested")
    usage_example: Optional[str] = Field(None, description="Example of how to use the code")


# ============== Code Analysis Models ==============

class CodeIssue(BaseModel):
    """Model for a code issue/suggestion."""
    line: Optional[int] = Field(None, description="Line number where issue occurs")
    severity: Severity = Field(..., description="Issue severity")
    category: str = Field(..., description="Issue category (security, performance, style, etc.)")
    message: str = Field(..., description="Issue description")
    suggestion: Optional[str] = Field(None, description="Suggested fix")


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis."""
    code: str = Field(..., description="Code to analyze")
    language: Optional[Language] = Field(None, description="Programming language")
    focus_areas: list[str] = Field(
        default_factory=lambda: ["security", "performance", "best_practices"],
        description="Areas to focus analysis on"
    )


class CodeAnalysisResult(BaseModel):
    """Result model for code analysis."""
    summary: str = Field(..., description="Overall analysis summary")
    quality_score: int = Field(..., ge=0, le=100, description="Code quality score (0-100)")
    issues: list[CodeIssue] = Field(default_factory=list, description="Found issues")
    strengths: list[str] = Field(default_factory=list, description="Code strengths")
    recommendations: list[str] = Field(default_factory=list, description="Improvement recommendations")
    refactored_code: Optional[str] = Field(None, description="Refactored version if applicable")


# ============== Document/RAG Models ==============

class DocumentMetadata(BaseModel):
    """Metadata for an uploaded document."""
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Path to stored file")
    upload_time: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    page_count: int = Field(0, description="Number of pages")
    chunk_count: int = Field(0, description="Number of indexed chunks")
    file_size_bytes: int = Field(0, description="File size in bytes")


class DocumentChunk(BaseModel):
    """Model for a document chunk with embedding."""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    content: str = Field(..., description="Chunk text content")
    page_number: int = Field(..., description="Source page number")
    chunk_index: int = Field(..., description="Index within document")
    embedding: Optional[list[float]] = Field(None, description="Vector embedding")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class RAGQuery(BaseModel):
    """Query model for RAG retrieval."""
    query: str = Field(..., description="User's question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    document_filter: Optional[list[str]] = Field(None, description="Filter by document IDs")


class RAGResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str = Field(..., description="Generated answer")
    sources: list[DocumentChunk] = Field(default_factory=list, description="Source chunks used")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


# ============== Agent Workflow Models ==============

class StepType(str, Enum):
    """Types of agent steps."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class AgentStep(BaseModel):
    """Model for tracking agent reasoning steps."""
    step_number: int = Field(..., description="Step sequence number")
    step_type: StepType = Field(..., description="Type of step")
    content: str = Field(..., description="Step content/description")
    timestamp: datetime = Field(default_factory=datetime.now, description="Step timestamp")
    agent_role: AgentRole = Field(..., description="Which agent performed this step")
    tool_used: Optional[str] = Field(None, description="Tool used if applicable")
    tool_input: Optional[dict] = Field(None, description="Tool input parameters")
    tool_output: Optional[str] = Field(None, description="Tool output if applicable")
    duration_ms: Optional[int] = Field(None, description="Step duration in milliseconds")


class AgentWorkflow(BaseModel):
    """Complete workflow execution record."""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    agent_role: AgentRole = Field(..., description="Primary agent handling the workflow")
    start_time: datetime = Field(default_factory=datetime.now, description="Workflow start time")
    end_time: Optional[datetime] = Field(None, description="Workflow end time")
    steps: list[AgentStep] = Field(default_factory=list, description="Execution steps")
    status: str = Field("running", description="Workflow status")
    error: Optional[str] = Field(None, description="Error message if failed")


# ============== Chat Models ==============

class MessageRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Model for chat messages."""
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ChatHistory(BaseModel):
    """Chat history container."""
    messages: list[ChatMessage] = Field(default_factory=list, description="Chat messages")
    context_documents: list[str] = Field(default_factory=list, description="Active document IDs")

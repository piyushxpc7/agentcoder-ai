"""RAG agent for documentation Q&A using Pydantic AI."""

import os
import asyncio
from typing import Optional, List, Tuple
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv

from models.schemas import RAGResponse, AgentStep, StepType, AgentRole
from tools.retrieval import RetrievalTools

load_dotenv()

# Fallback models to try if primary model is rate-limited
FALLBACK_MODELS = [
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "meta-llama/llama-3-8b-instruct:free",
]


class RAGAgent:
    """Agent for answering questions using document retrieval."""
    
    def __init__(self, model_name: Optional[str] = None, retrieval_tools: Optional[RetrievalTools] = None):
        self.model_name = model_name or os.getenv("DEFAULT_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free")
        self.fallback_models = [m for m in FALLBACK_MODELS if m != self.model_name]
        self.retrieval = retrieval_tools or RetrievalTools()
        self.model = OpenAIModel(
            self.model_name,
            provider='openrouter',
        )
        self.agent = Agent(
            model=self.model,
            output_type=RAGResponse,
            system_prompt=self._get_system_prompt(),
        )
        self.steps: List[AgentStep] = []
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_str = str(error).lower()
        return "429" in str(error) or "rate" in error_str or "rate-limited" in error_str or "limit" in error_str
    
    def _create_agent_for_model(self, model_name: str) -> Agent:
        """Create an agent instance for a specific model."""
        model = OpenAIModel(model_name, provider='openrouter')
        return Agent(
            model=model,
            output_type=RAGResponse,
            system_prompt=self._get_system_prompt(),
        )
    
    def _get_system_prompt(self) -> str:
        return """You are a helpful assistant that answers questions using provided documentation.

Guidelines:
1. Only use information from the provided context
2. If info isn't in context, say so clearly
3. Cite sources with document name and page
4. Be accurate and concise
5. Provide confidence level (0-1) based on how well context answers the question

Respond with RAGResponse format: answer, sources used, confidence score."""

    def _log_step(self, step_type: StepType, content: str, **kwargs):
        step = AgentStep(
            step_number=len(self.steps) + 1,
            step_type=step_type, content=content,
            agent_role=AgentRole.RAG_AGENT, **kwargs
        )
        self.steps.append(step)
    
    async def answer(
        self, query: str, top_k: int = 5, document_ids: Optional[List[str]] = None
    ) -> Tuple[RAGResponse, List[AgentStep]]:
        self.steps = []
        self._log_step(StepType.THOUGHT, f"Query: {query[:80]}...")
        self._log_step(StepType.ACTION, f"Retrieving top {top_k} chunks", tool_used="vector_search")
        
        # Get context
        context = self.retrieval.get_context(query, top_k, document_ids)
        search_results = self.retrieval.search(query, top_k, document_ids)
        
        self._log_step(StepType.OBSERVATION, f"Found {len(search_results)} relevant chunks")
        
        prompt = f"""Question: {query}

Context from documents:
{context}

Answer the question using only the context above."""

        # Try primary model first, then fallback models
        models_to_try = [self.model_name] + self.fallback_models
        last_error = None
        
        for model_idx, model_name in enumerate(models_to_try):
            try:
                # Create agent for this model if not the primary one
                if model_idx == 0:
                    agent = self.agent
                else:
                    self._log_step(
                        StepType.THOUGHT,
                        f"Primary model rate-limited, trying fallback: {model_name}"
                    )
                    agent = self._create_agent_for_model(model_name)
                    await asyncio.sleep(1)
                
                result = await agent.run(prompt)
                response = result.output
                
                # Add source chunks
                from models.schemas import DocumentChunk
                response.sources = [
                    DocumentChunk(
                        id=r["chunk_id"], document_id=r["document_id"],
                        content=r["content"], page_number=r["page_number"], chunk_index=0
                    ) for r in search_results[:3]
                ]
                
                self._log_step(StepType.FINAL_ANSWER, f"Answered with {response.confidence:.0%} confidence")
                return response, self.steps
                
            except Exception as e:
                last_error = e
                if self._is_rate_limit_error(e):
                    self._log_step(
                        StepType.OBSERVATION,
                        f"Model {model_name} rate-limited: {str(e)}"
                    )
                    continue
                else:
                    self._log_step(
                        StepType.OBSERVATION,
                        f"Error with model {model_name}: {str(e)}"
                    )
                    raise
        
        # All models failed
        raise Exception(f"All free models are rate-limited. Please try again in a few minutes. Last error: {str(last_error)}")
    
    def answer_sync(self, query: str, top_k: int = 5, document_ids: Optional[List[str]] = None):
        import asyncio
        return asyncio.run(self.answer(query, top_k, document_ids))

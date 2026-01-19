"""Code analysis agent using Pydantic AI."""

import os
import asyncio
from typing import Optional, List, Tuple
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv

from models.schemas import (
    CodeAnalysisRequest, CodeAnalysisResult, Severity,
    AgentStep, StepType, AgentRole,
)
from tools.code_tools import CodeTools

load_dotenv()

# Fallback models to try if primary model is rate-limited
FALLBACK_MODELS = [
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "meta-llama/llama-3-8b-instruct:free",
]


class CodeAnalysisAgent:
    """Agent for analyzing code quality and providing feedback."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("DEFAULT_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free")
        self.fallback_models = [m for m in FALLBACK_MODELS if m != self.model_name]
        self.model = OpenAIModel(
            self.model_name,
            provider='openrouter',
        )
        self.agent = Agent(
            model=self.model,
            output_type=CodeAnalysisResult,
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
            output_type=CodeAnalysisResult,
            system_prompt=self._get_system_prompt(),
        )
    
    def _get_system_prompt(self) -> str:
        return """You are an expert code reviewer. Analyze code for:
1. Security vulnerabilities
2. Performance issues
3. Best practices violations
4. Code quality and readability
5. Error handling

Provide quality score (0-100), issues list, strengths, and recommendations.
Always respond with CodeAnalysisResult format."""
    
    def _log_step(self, step_type: StepType, content: str, **kwargs):
        step = AgentStep(
            step_number=len(self.steps) + 1,
            step_type=step_type, content=content,
            agent_role=AgentRole.CODE_ANALYZER, **kwargs
        )
        self.steps.append(step)
    
    async def analyze(self, request: CodeAnalysisRequest) -> Tuple[CodeAnalysisResult, List[AgentStep]]:
        self.steps = []
        language = request.language or CodeTools.detect_language(request.code)
        self._log_step(StepType.THOUGHT, f"Analyzing {language.value} code")
        
        prompt = f"""Analyze this {language.value} code:
```{CodeTools.get_syntax_name(language)}
{request.code}
```
Focus: {', '.join(request.focus_areas)}"""
        
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
                
                self._log_step(StepType.ACTION, "Running code analysis")
                result = await agent.run(prompt)
                self._log_step(StepType.FINAL_ANSWER, f"Score: {result.output.quality_score}/100")
                return result.output, self.steps
                
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
    
    def analyze_sync(self, request: CodeAnalysisRequest):
        import asyncio
        return asyncio.run(self.analyze(request))

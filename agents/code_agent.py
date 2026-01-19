"""Code generation agent using Pydantic AI."""

import os
import asyncio
import time
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import uuid

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv

from models.schemas import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    Language,
    AgentStep,
    StepType,
    AgentRole,
)
from tools.code_tools import CodeTools

load_dotenv()

# Fallback models to try if primary model is rate-limited
FALLBACK_MODELS = [
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "meta-llama/llama-3-8b-instruct:free",
]


class CodeGenerationAgent:
    """Agent for generating code based on user requirements."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the code generation agent.
        
        Args:
            model_name: Model to use via OpenRouter.
        """
        self.model_name = model_name or os.getenv("DEFAULT_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free")
        self.fallback_models = [m for m in FALLBACK_MODELS if m != self.model_name]
        
        # Configure OpenRouter
        self.model = OpenAIModel(
            self.model_name,
            provider='openrouter',
        )
        
        # Create the agent
        self.agent = Agent(
            model=self.model,
            output_type=CodeGenerationResponse,
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
            output_type=CodeGenerationResponse,
            system_prompt=self._get_system_prompt(),
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for code generation."""
        return """You are an expert software developer assistant. Your task is to generate high-quality, 
production-ready code based on user requirements.

Guidelines:
1. Write clean, well-documented, and maintainable code
2. Follow best practices and design patterns for the chosen language
3. Include proper error handling and edge case management
4. Add meaningful comments and docstrings
5. Consider security implications
6. Optimize for readability and performance
7. If tests are requested, write comprehensive unit tests
8. Suggest appropriate filenames and dependencies

When generating code:
- Start with a brief explanation of your approach
- Consider the requirements carefully before coding
- Provide complete, runnable code that users can use immediately
- Include usage examples when helpful

Always respond with the structured CodeGenerationResponse format."""
    
    def _log_step(
        self,
        step_type: StepType,
        content: str,
        tool_used: Optional[str] = None,
        tool_input: Optional[Dict] = None,
        tool_output: Optional[str] = None,
    ):
        """Log an agent reasoning step."""
        step = AgentStep(
            step_number=len(self.steps) + 1,
            step_type=step_type,
            content=content,
            agent_role=AgentRole.CODE_GENERATOR,
            tool_used=tool_used,
            tool_input=tool_input,
            tool_output=tool_output,
        )
        self.steps.append(step)
    
    async def generate(
        self,
        request: CodeGenerationRequest,
    ) -> Tuple[CodeGenerationResponse, List[AgentStep]]:
        """Generate code based on the request.
        
        Args:
            request: Code generation request with requirements.
            
        Returns:
            Tuple of (response, steps list).
        """
        self.steps = []
        
        # Log thought step
        self._log_step(
            StepType.THOUGHT,
            f"Analyzing request: {request.prompt[:100]}..."
        )
        
        # Build the prompt
        prompt_parts = [f"Generate code for the following requirement:\n\n{request.prompt}"]
        
        if request.language:
            prompt_parts.append(f"\nTarget language: {request.language.value}")
        
        if request.context:
            prompt_parts.append(f"\nAdditional context: {request.context}")
        
        if request.include_tests:
            prompt_parts.append("\nPlease include comprehensive unit tests.")
        
        if request.include_docs:
            prompt_parts.append("\nInclude detailed documentation and docstrings.")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Log action step
        self._log_step(
            StepType.ACTION,
            "Generating code using LLM",
            tool_used="llm_generate",
            tool_input={"prompt_length": len(full_prompt)},
        )
        
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
                    # Small delay before retry
                    await asyncio.sleep(1)
                
                # Run the agent
                result = await agent.run(full_prompt)
                response = result.output
                
                # Auto-detect language if not specified
                if response.language == Language.OTHER or not response.language:
                    detected = CodeTools.detect_language(response.code)
                    if detected != Language.OTHER:
                        response.language = detected
                
                # Generate filename if not provided
                if not response.filename:
                    response.filename = CodeTools.suggest_filename(
                        request.prompt,
                        response.language
                    )
                
                # Log observation
                if model_idx > 0:
                    self._log_step(
                        StepType.OBSERVATION,
                        f"Successfully used fallback model: {model_name}"
                    )
                
                self._log_step(
                    StepType.OBSERVATION,
                    f"Generated {CodeTools.count_lines(response.code)} lines of {response.language.value} code"
                )
                
                # Log final answer
                self._log_step(
                    StepType.FINAL_ANSWER,
                    f"Code generation complete: {response.filename}"
                )
                
                return response, self.steps
                
            except Exception as e:
                last_error = e
                if self._is_rate_limit_error(e):
                    self._log_step(
                        StepType.OBSERVATION,
                        f"Model {model_name} rate-limited: {str(e)}"
                    )
                    # Continue to next model
                    continue
                else:
                    # Non-rate-limit error, log and re-raise
                    self._log_step(
                        StepType.OBSERVATION,
                        f"Error with model {model_name}: {str(e)}"
                    )
                    raise
        
        # All models failed
        self._log_step(
            StepType.OBSERVATION,
            "All models rate-limited or failed"
        )
        raise Exception(f"All free models are rate-limited. Please try again in a few minutes. Last error: {str(last_error)}")
    
    def generate_sync(
        self,
        request: CodeGenerationRequest,
    ) -> Tuple[CodeGenerationResponse, List[AgentStep]]:
        """Synchronous wrapper for generate.
        
        Args:
            request: Code generation request.
            
        Returns:
            Tuple of (response, steps list).
        """
        import asyncio
        return asyncio.run(self.generate(request))

# mistral_integration.py
"""
Mistral 7B OpenOrca Integration for AI Web Agent (llama-cpp backend)

This file provides a backwards-compatible wrapper that exposes the original
MistralLLMParser interface but delegates parsing to a local llama-cpp-based
implementation (MistralLlamaParser) present in
llama_cpp_integration_and_patches.py.

Expectations:
- llama_cpp_integration_and_patches.py must be in the same directory.
- The GGUF path defaults to: D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import the llama-cpp-based implementation (ensure file is in the same directory)
from llama_cpp_integration_and_patches import MistralLlamaParser as _MistralLlamaParser

@dataclass
class TaskStep:
    step_id: int
    description: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[int] = None
    completed: bool = False
    result: Any = None

class MistralLLMParser:
    """
    Backwards-compatible wrapper around the new MistralLlamaParser (llama-cpp).
    Keeps the same method names used by the rest of your codebase.
    """

    def __init__(self,
                 model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf",
                 demo_mode: bool = False,
                 n_ctx: int = 2048):
        self._impl = _MistralLlamaParser(model_path=model_path, demo_mode=demo_mode, n_ctx=n_ctx)

    def is_ollama_available(self) -> bool:
        """
        Legacy name kept for compatibility: returns True when local GGUF client
        can be initialized (same semantic as 'available').
        """
        return self._impl.setup_model()

    def setup_model(self) -> bool:
        return self._impl.setup_model()

    async def parse_complex_instruction(self, instruction: str, context: List[Dict] = None) -> List[TaskStep]:
        """
        Delegate to the underlying Llama parser and convert the returned dicts
        to TaskStep dataclass instances so the rest of your code continues to work.
        """
        steps_dicts = await self._impl.parse_complex_instruction(instruction, context)
        task_steps: List[TaskStep] = []

        for idx, s in enumerate(steps_dicts, start=1):
            if not isinstance(s, dict):
                # Fallback: wrap simple text step
                task_steps.append(TaskStep(
                    step_id=idx,
                    description=str(s),
                    action="navigate",
                    parameters={},
                    dependencies=[]
                ))
                continue

            step_id = s.get("step_id", idx)
            description = s.get("description", "") or s.get("desc", "")
            action = s.get("action", "navigate")
            parameters = s.get("parameters", {})
            dependencies = s.get("dependencies", []) or []

            task_steps.append(TaskStep(
                step_id=step_id,
                description=description,
                action=action,
                parameters=parameters,
                dependencies=dependencies
            ))

        return task_steps


# Integration with existing WebAgent (optional convenience wrapper)
class MistralWebAgent:
    """
    Simple wrapper that composes your existing WebAgent with the Mistral parser.
    It uses the compatibility MistralLLMParser above and your existing ai_web_agent.WebAgent.
    """

    def __init__(self,
                 model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf",
                 demo_mode: bool = False,
                 n_ctx: int = 2048):
        # Lazy import to avoid heavy imports at module load time
        from ai_web_agent import WebAgent, TaskMemory
        self.base_agent = WebAgent()
        self.llm_parser = MistralLLMParser(model_path=model_path, demo_mode=demo_mode, n_ctx=n_ctx)
        self.memory = TaskMemory()

    async def start(self):
        """Start the base browser agent and ensure model is ready."""
        await self.base_agent.start()
        if self.llm_parser.setup_model():
            print("🚀 Mistral (llama-cpp) model ready for advanced parsing!")
        else:
            print("⚠️ Mistral model not available — falling back to rule-based parsing.")

    async def execute_instruction(self, instruction: str):
        """
        Parse instruction using Mistral and then call base_agent.execute_instruction().
        The base agent still receives the original instruction; this wrapper mainly
        provides parsing + optional orchestration hooks (you can expand execution logic).
        """
        print(f"\n🎯 Processing: {instruction}")

        recent_tasks = self.memory.get_recent_tasks(3)
        steps = await self.llm_parser.parse_complex_instruction(instruction, recent_tasks)

        print(f"📋 Parsed into {len(steps)} steps:")
        for step in steps:
            print(f"  {step.step_id}. {step.description} ({step.action})")

        # For now, keep compatibility by letting base agent handle the top-level instruction.
        # You may change this to run the step workflow directly using base_agent capabilities.
        result = await self.base_agent.execute_instruction(instruction)

        if result.success:
            # Save raw result (keeping old memory API)
            self.memory.save_task(instruction, result.data)

        return result

    async def stop(self):
        await self.base_agent.stop()


# A small test harness for local checks
async def test_mistral_integration():
    """Run a few example instructions through the wrapper to verify flow."""
    agent = MistralWebAgent()
    await agent.start()

    test_instructions = [
        "search for laptops under $50K and list top 5",
        "find wireless headphones under $100 with good ratings and export to CSV",
        "search for smartphones, compare prices, and show the best 3 options",
        "get running shoes under $150, filter by Nike brand, and save results"
    ]

    for instruction in test_instructions:
        print("\n" + "=" * 60)
        print(f"Testing: {instruction}")
        print("=" * 60)
        result = await agent.execute_instruction(instruction)
        if getattr(result, "success", False):
            print("✅ Success!")
        else:
            print(f"❌ Failed: {getattr(result, 'error_message', 'No error message')}")

    await agent.stop()


if __name__ == "__main__":
    print("🤖 Testing Mistral (llama-cpp) Integration")
    asyncio.run(test_mistral_integration())

# advanced_features.py
"""
Advanced Features for AI Web Agent
Multi-step reasoning, task chaining, and enhanced LLM integration (llama-cpp backend)
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ai_web_agent import WebAgent, TaskResult
from datetime import datetime

# Use the llama-cpp-based advanced parser (ensure llama_cpp_integration_and_patches.py is in same folder)
from llama_cpp_integration_and_patches import AdvancedLlamaParser as _AdvancedLlamaParser

@dataclass
class TaskStep:
    """Individual step in a multi-step task"""
    step_id: int
    description: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[int] = None
    completed: bool = False
    result: Any = None

class AdvancedLLMParser:
    """
    Compatibility wrapper that delegates to AdvancedLlamaParser (llama-cpp) when available,
    and falls back to the rule-based parser when not.
    """

    def __init__(self, model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf", demo_mode: bool = False, n_ctx: int = 2048):
        self.model_path = model_path
        self.demo_mode = demo_mode
        self.n_ctx = n_ctx
        self._impl = None
        try:
            if not demo_mode:
                self._impl = _AdvancedLlamaParser(model_path=model_path, demo_mode=demo_mode, n_ctx=n_ctx)
        except Exception as e:
            # Keep failing quietly; fallback rule-based will be used.
            print(f"[AdvancedLLMParser] Could not initialize llama-cpp parser: {e}")
            self._impl = None

    def is_ollama_available(self) -> bool:
        """
        Legacy compatibility name. Returns True if the local llama-cpp parser is ready.
        """
        if self._impl is None:
            return False
        try:
            # _impl has no explicit setup_model in this design, but its client will raise if missing.
            return True if self._impl else False
        except Exception:
            return False

    async def parse_complex_instruction(self, instruction: str, context: List[Dict] = None) -> List[TaskStep]:
        """
        Parse complex multi-step instructions.
        Prefer llama-cpp parser if initialized; otherwise use rule-based fallback.
        Returns a list of TaskStep objects.
        """
        # Use llama-cpp powered parser if available
        if self._impl and not self.demo_mode:
            try:
                steps_data = await self._impl.parse_complex_instruction(instruction, context)
                # steps_data is expected to be a list of dicts; convert to TaskStep
                task_steps: List[TaskStep] = []
                for i, s in enumerate(steps_data, start=1):
                    if not isinstance(s, dict):
                        # fallback: wrap simple textual step
                        task_steps.append(TaskStep(step_id=i, description=str(s), action="navigate", parameters={}, dependencies=[]))
                        continue
                    step_id = s.get("step_id", i)
                    description = s.get("description", "") or s.get("desc", "")
                    action = s.get("action", "navigate")
                    parameters = s.get("parameters", {}) or {}
                    dependencies = s.get("dependencies", []) or []
                    task_steps.append(TaskStep(
                        step_id=step_id,
                        description=description,
                        action=action,
                        parameters=parameters,
                        dependencies=dependencies
                    ))
                if task_steps:
                    return task_steps
            except Exception as e:
                print(f"[AdvancedLLMParser] llama-cpp parsing failed, falling back to rule-based: {e}")

        # Fallback to rule-based parsing
        return self._rule_based_parse_instruction(instruction, context)

    # ---------- Original rule-based parsing logic (kept as fallback) ----------
    def _rule_based_parse_instruction(self, instruction: str, context: List[Dict] = None) -> List[TaskStep]:
        """Enhanced rule-based instruction parsing"""
        steps = []
        instruction_lower = instruction.lower()

        # Detect comparison tasks
        if any(word in instruction_lower for word in ['compare', 'best', 'cheapest', 'vs']):
            return self._parse_comparison_task(instruction)

        # Detect multi-criteria search
        elif 'and' in instruction_lower and any(word in instruction_lower for word in ['under', 'over', 'with', 'without']):
            return self._parse_multi_criteria_task(instruction)

        # Detect data collection tasks
        elif any(word in instruction_lower for word in ['collect', 'gather', 'save', 'export', 'download']):
            return self._parse_data_collection_task(instruction)

        # Default to simple search
        else:
            search_term = self._extract_search_term(instruction)
            limit = self._extract_number(instruction) or 5

            steps.append(TaskStep(
                step_id=1,
                description=f"Search for {search_term}",
                action="search",
                parameters={
                    "search_term": search_term,
                    "limit": limit,
                    "site": "amazon.com"
                },
                dependencies=[]
            ))

        return steps

    def _parse_comparison_task(self, instruction: str) -> List[TaskStep]:
        """Parse comparison tasks"""
        steps = []
        search_term = self._extract_search_term(instruction)

        # Step 1: Search for products
        steps.append(TaskStep(
            step_id=1,
            description=f"Search for {search_term}",
            action="search",
            parameters={
                "search_term": search_term,
                "limit": 10,
                "site": "amazon.com"
            },
            dependencies=[]
        ))

        # Step 2: Compare and rank
        steps.append(TaskStep(
            step_id=2,
            description="Compare products and find best options",
            action="compare",
            parameters={
                "criteria": ["price", "rating", "reviews"],
                "output_top": 5
            },
            dependencies=[1]
        ))

        return steps

    def _parse_multi_criteria_task(self, instruction: str) -> List[TaskStep]:
        """Parse multi-criteria filtering tasks"""
        steps = []
        search_term = self._extract_search_term(instruction)

        # Extract filters
        filters = {}

        # Price filters
        price_match = re.search(r'under\s+\$?(\d+(?:,\d+)*(?:k|K)?)', instruction.lower())
        if price_match:
            filters["max_price"] = price_match.group(1)

        # Rating filters
        rating_match = re.search(r'rating\s+(?:above|over)\s+(\d+(?:\.\d+)?)', instruction.lower())
        if rating_match:
            try:
                filters["min_rating"] = float(rating_match.group(1))
            except:
                pass

        # Feature requirements
        if 'with' in instruction.lower():
            features = re.findall(r'with\s+([^,\s]+(?:\s+[^,\s]+)*)', instruction.lower())
            filters["required_features"] = features

        # Step 1: Search
        steps.append(TaskStep(
            step_id=1,
            description=f"Search for {search_term}",
            action="search",
            parameters={
                "search_term": search_term,
                "limit": 20,
                "site": "amazon.com"
            },
            dependencies=[]
        ))

        # Step 2: Apply filters
        steps.append(TaskStep(
            step_id=2,
            description=f"Apply filters: {filters}",
            action="filter",
            parameters=filters,
            dependencies=[1]
        ))

        return steps

    def _parse_data_collection_task(self, instruction: str) -> List[TaskStep]:
        """Parse data collection and export tasks"""
        steps = []
        search_term = self._extract_search_term(instruction)

        # Determine export format
        export_format = "csv"
        if "json" in instruction.lower():
            export_format = "json"
        elif "excel" in instruction.lower():
            export_format = "xlsx"

        # Step 1: Search and collect
        steps.append(TaskStep(
            step_id=1,
            description=f"Collect data for {search_term}",
            action="search",
            parameters={
                "search_term": search_term,
                "limit": 50,
                "site": "amazon.com",
                "detailed": True
            },
            dependencies=[]
        ))

        # Step 2: Export data
        steps.append(TaskStep(
            step_id=2,
            description=f"Export results as {export_format}",
            action="export",
            parameters={
                "format": export_format,
                "filename": f"{search_term.replace(' ', '_')}_results"
            },
            dependencies=[1]
        ))

        return steps

    def _extract_search_term(self, instruction: str) -> str:
        """Extract main search term from instruction"""
        # Remove common words and extract main subject
        instruction_lower = instruction.lower()

        # Common patterns
        patterns = [
            r'search for ([^"]+?)(?:\s+and|$)',
            r'find ([^"]+?)(?:\s+and|$)',
            r'look for ([^"]+?)(?:\s+and|$)',
            r'get ([^"]+?)(?:\s+and|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                return match.group(1).strip()

        # Fallback: take first few meaningful words
        words = instruction_lower.split()
        meaningful_words = [w for w in words if len(w) > 2 and w not in ['the', 'and', 'for', 'with']]
        return ' '.join(meaningful_words[:3])

    def _extract_number(self, instruction: str) -> Optional[int]:
        """Extract number from instruction"""
        number_match = re.search(r'(?:top|first)\s+(\d+)', instruction.lower())
        return int(number_match.group(1)) if number_match else None


class AdvancedWebAgent(WebAgent):
    """Extended WebAgent with multi-step reasoning"""

    def __init__(self, model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf", demo_mode: bool = False):
        super().__init__()
        self.advanced_parser = AdvancedLLMParser(model_path=model_path, demo_mode=demo_mode)
        self.current_workflow = []

    async def execute_complex_instruction(self, instruction: str) -> TaskResult:
        """Execute multi-step instructions"""
        try:
            # Parse instruction into steps
            steps = await self.advanced_parser.parse_complex_instruction(instruction)

            if not steps:
                return TaskResult(
                    success=False,
                    data=None,
                    error_message="Could not parse instruction into executable steps"
                )

            self.current_workflow = steps
            results = {}
            execution_log = []

            # Execute steps in dependency order
            for step in self._sort_by_dependencies(steps):
                execution_log.append(f"Executing: {step.description}")

                try:
                    # Wait for dependencies
                    if step.dependencies:
                        for dep_id in step.dependencies:
                            if not self._is_step_completed(dep_id):
                                raise Exception(f"Dependency step {dep_id} not completed")

                    # Execute step
                    if step.action == "search":
                        result = await self._execute_search_step(step)
                    elif step.action == "filter":
                        result = await self._execute_filter_step(step, results)
                    elif step.action == "compare":
                        result = await self._execute_compare_step(step, results)
                    elif step.action == "export":
                        result = await self._execute_export_step(step, results)
                    else:
                        # default handler - use base agent to run a search
                        result = await super().execute_instruction(f"search for {step.parameters.get('search_term', '')}")

                    # Mark as completed
                    step.completed = True
                    step.result = result.data if result.success else None
                    results[step.step_id] = result.data if result.success else None

                    execution_log.append(f"✅ Completed: {step.description}")

                except Exception as e:
                    execution_log.append(f"❌ Failed: {step.description} - {str(e)}")
                    return TaskResult(
                        success=False,
                        data=None,
                        error_message=f"Step {step.step_id} failed: {str(e)}",
                        steps_taken=execution_log
                    )

            # Combine results
            final_result = self._combine_step_results(results, steps)

            return TaskResult(
                success=True,
                data=final_result,
                steps_taken=execution_log
            )

        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                error_message=f"Workflow execution failed: {str(e)}"
            )

    def _sort_by_dependencies(self, steps: List[TaskStep]) -> List[TaskStep]:
        """Sort steps by dependency order"""
        sorted_steps = []
        remaining_steps = steps.copy()

        while remaining_steps:
            # Find steps with no unfulfilled dependencies
            ready_steps = []
            for step in remaining_steps:
                if not step.dependencies or all(
                    dep_id in [s.step_id for s in sorted_steps]
                    for dep_id in step.dependencies
                ):
                    ready_steps.append(step)

            if not ready_steps:
                raise Exception("Circular dependency detected in workflow")

            # Add ready steps to sorted list
            for step in ready_steps:
                sorted_steps.append(step)
                remaining_steps.remove(step)

        return sorted_steps

    def _is_step_completed(self, step_id: int) -> bool:
        """Check if a step is completed"""
        for step in self.current_workflow:
            if step.step_id == step_id:
                return step.completed
        return False

    async def _execute_search_step(self, step: TaskStep) -> TaskResult:
        """Execute search step"""
        params = step.parameters
        search_instruction = f"search for {params.get('search_term', '')} and list top {params.get('limit', 5)}"
        return await super().execute_instruction(search_instruction)

    async def _execute_filter_step(self, step: TaskStep, results: Dict) -> TaskResult:
        """Execute filter step"""
        # Get data from previous step
        dep_data = results.get(step.dependencies[0]) if step.dependencies else None
        if not dep_data or 'products' not in dep_data:
            return TaskResult(success=False, data=None, error_message="No data to filter")

        products = dep_data['products']
        filters = step.parameters
        filtered_products = []

        for product in products:
            # Apply price filter
            if 'max_price' in filters:
                price_str = product.get('price', '').replace('$', '').replace(',', '')
                try:
                    price = float(re.search(r'\d+(?:\.\d+)?', price_str).group(0))
                    max_price = float(str(filters['max_price']).replace('k', '000').replace('K', '000').replace(',', ''))
                    if price > max_price:
                        continue
                except:
                    continue

            # Apply rating filter
            if 'min_rating' in filters:
                rating_str = product.get('rating', '')
                try:
                    rating = float(re.search(r'\d+(?:\.\d+)?', rating_str).group(0))
                    if rating < filters['min_rating']:
                        continue
                except:
                    continue

            filtered_products.append(product)

        return TaskResult(
            success=True,
            data={
                'products': filtered_products,
                'total_filtered': len(filtered_products),
                'original_count': len(products),
                'filters_applied': filters
            }
        )

    async def _execute_compare_step(self, step: TaskStep, results: Dict) -> TaskResult:
        """Execute comparison step"""
        # Get data from previous step
        dep_data = results.get(step.dependencies[0]) if step.dependencies else None
        if not dep_data or 'products' not in dep_data:
            return TaskResult(success=False, data=None, error_message="No data to compare")

        products = dep_data['products']
        criteria = step.parameters.get('criteria', ['price', 'rating'])
        output_top = step.parameters.get('output_top', 5)

        # Score products based on criteria
        scored_products = []
        for product in products:
            score = 0

            # Price score (lower is better)
            if 'price' in criteria:
                price_str = product.get('price', '').replace('$', '').replace(',', '')
                try:
                    price = float(re.search(r'\d+(?:\.\d+)?', price_str).group(0))
                    # Normalize price score (assuming max reasonable price)
                    score += max(0, 1000 - price) / 1000 * 0.4
                except:
                    pass

            # Rating score (higher is better)
            if 'rating' in criteria:
                rating_str = product.get('rating', '')
                try:
                    rating = float(re.search(r'\d+(?:\.\d+)?', rating_str).group(0))
                    score += rating / 5.0 * 0.6
                except:
                    pass

            scored_products.append((score, product))

        # Sort by score and return top products
        scored_products.sort(key=lambda x: x[0], reverse=True)
        top_products = [product for score, product in scored_products[:output_top]]

        return TaskResult(
            success=True,
            data={
                'products': top_products,
                'comparison_criteria': criteria,
                'total_compared': len(products)
            }
        )

    async def _execute_export_step(self, step: TaskStep, results: Dict) -> TaskResult:
        """Execute export step"""
        # Get data from previous step
        dep_data = results.get(step.dependencies[0]) if step.dependencies else None
        if not dep_data:
            return TaskResult(success=False, data=None, error_message="No data to export")

        format_type = step.parameters.get('format', 'csv')
        filename = step.parameters.get('filename', 'export')

        # Generate export content
        if format_type == 'csv':
            content = self._generate_csv(dep_data)
            full_filename = f"{filename}.csv"
        elif format_type == 'json':
            content = json.dumps(dep_data, indent=2)
            full_filename = f"{filename}.json"
        else:
            return TaskResult(success=False, data=None, error_message=f"Unsupported format: {format_type}")

        # In a real implementation, you would save to file
        # For demo, we'll return the content

        return TaskResult(
            success=True,
            data={
                'exported_file': full_filename,
                'content_preview': content[:200] + "..." if len(content) > 200 else content,
                'full_content': content,
                'format': format_type
            }
        )

    def _generate_csv(self, data: Dict) -> str:
        """Generate CSV content from data"""
        if 'products' in data:
            products = data['products']
            if not products:
                return "No products to export"

            # CSV header
            headers = list(products[0].keys())
            csv_lines = [','.join(headers)]

            # CSV rows
            for product in products:
                row = [str(product.get(header, '')).replace(',', ';') for header in headers]
                csv_lines.append(','.join(row))

            return '\n'.join(csv_lines)

        return str(data)

    def _combine_step_results(self, results: Dict, steps: List[TaskStep]) -> Dict:
        """Combine results from all steps"""
        final_step_id = max(step.step_id for step in steps)
        final_result = results.get(final_step_id, {})

        # Add workflow metadata
        workflow_summary = {
            'workflow_steps': len(steps),
            'steps_completed': sum(1 for step in steps if step.completed),
            'execution_time': datetime.now().isoformat(),
            'step_descriptions': [step.description for step in steps]
        }

        if isinstance(final_result, dict):
            final_result['workflow_metadata'] = workflow_summary
        else:
            final_result = {
                'data': final_result,
                'workflow_metadata': workflow_summary
            }

        return final_result


# Example usage and testing
async def test_advanced_features():
    """Test advanced features"""
    agent = AdvancedWebAgent()
    await agent.start()

    # Test complex instruction
    instruction = "search for laptops under $50K, filter by good ratings, compare prices, and export top 5 to CSV"

    print(f"Executing: {instruction}")
    result = await agent.execute_complex_instruction(instruction)

    if result.success:
        print("✅ Success!")
        print("Steps taken:")
        for step in result.steps_taken:
            print(f"  • {step}")

        print("\nFinal results:")
        print(json.dumps(result.data, indent=2))
    else:
        print(f"❌ Failed: {result.error_message}")

    await agent.stop()


if __name__ == "__main__":
    asyncio.run(test_advanced_features())

# ai_web_agent.py
"""
AI Web Agent - Hackathon Solution
Autonomous web browsing agent with local LLM integration (llama-cpp backend)
"""

import asyncio
import json
import sqlite3
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from playwright.async_api import async_playwright, Page, Browser
from pathlib import Path

# Use local llama-cpp wrapper for parsing instructions
from llama_cpp_integration_and_patches import LocalLlamaLLM as _LocalLlamaLLM

# Task Memory Storage
class TaskMemory:
    def __init__(self, db_path: str = "task_memory.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instruction TEXT,
                result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def save_task(self, instruction: str, result: Any):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tasks (instruction, result) VALUES (?, ?)",
            (instruction, json.dumps(result))
        )
        conn.commit()
        conn.close()
    
    def get_recent_tasks(self, limit: int = 5) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT instruction, result, timestamp FROM tasks ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        tasks = []
        for row in rows:
            try:
                result_parsed = json.loads(row[1])
            except Exception:
                result_parsed = row[1]
            tasks.append({"instruction": row[0], "result": result_parsed, "timestamp": row[2]})
        conn.close()
        return tasks

@dataclass
class TaskResult:
    success: bool
    data: Any
    error_message: Optional[str] = None
    steps_taken: List[str] = None

# ---------------------------------------------------------------------
# LocalLLM compatibility wrapper
# This keeps the same synchronous interface `parse_instruction(...)` used
# across the codebase, while delegating heavy parsing to the llama-cpp
# implementation (LocalLlamaLLM).
# ---------------------------------------------------------------------
class LocalLLM:
    def __init__(self, model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf", demo_mode: bool = False):
        """
        model_path: path to your GGUF file (default set to your provided path)
        demo_mode: if True, skip initializing llama-cpp and use rule-based parsing
        """
        try:
            if demo_mode:
                self._impl = None
            else:
                # instantiate the LocalLlamaLLM which internally chooses fast/advanced parser
                self._impl = _LocalLlamaLLM(model_path=model_path, demo_mode=demo_mode)
        except Exception as e:
            print(f"[LocalLLM] Failed to initialize LocalLlamaLLM: {e}")
            self._impl = None

    def parse_instruction(self, instruction: str, context: List[Dict] = None) -> Dict[str, Any]:
        """
        Return a dict that matches the legacy contract:
        - action: "search_and_extract" | "fill_form" | "navigate_and_extract" | "complex_workflow"
        - other keys depending on action
        """
        # If llama-cpp powered parser is available, use it for complex instructions.
        try:
            if self._impl:
                # LocalLlamaLLM.parse_instruction is synchronous (it may internally call asyncio.run)
                parsed = self._impl.parse_instruction(instruction, context)
                if parsed:
                    return parsed
        except Exception as e:
            print(f"[LocalLLM] Error using llama-cpp parser, falling back to rule-based: {e}")

        # Fallback: use a simple rule-based parser (similar to the prior implementation)
        instruction_lower = instruction.lower()
        
        if "search" in instruction_lower:
            # Extract search terms and requirements
            search_match = re.search(r'search for ([^"]+?)(?:\s+and|$)', instruction_lower)
            search_term = search_match.group(1).strip() if search_match else ""
            
            # Extract number requirements
            number_match = re.search(r'(?:top|first)\s+(\d+)', instruction_lower)
            limit = int(number_match.group(1)) if number_match else 5
            
            # Extract price/filter requirements
            price_match = re.search(r'under\s+\$?(\d+(?:,\d+)*(?:k|K)?)', instruction_lower)
            max_price = price_match.group(1) if price_match else None
            
            return {
                "action": "search_and_extract",
                "search_term": search_term,
                "limit": limit,
                "filters": {"max_price": max_price} if max_price else {},
                "site": "amazon.com"  # Default site
            }
        
        elif "fill form" in instruction_lower:
            return {
                "action": "fill_form",
                "url": self._extract_url(instruction),
                "form_data": self._extract_form_data(instruction)
            }
        
        else:
            return {
                "action": "navigate_and_extract",
                "url": self._extract_url(instruction),
                "extract_text": True
            }
    
    def _extract_url(self, instruction: str) -> Optional[str]:
        """Extract URL from instruction"""
        url_pattern = r'https?://[^\s]+'
        match = re.search(url_pattern, instruction)
        return match.group(0) if match else None
    
    def _extract_form_data(self, instruction: str) -> Dict[str, str]:
        """Extract form data from instruction"""
        # Simplified form data extraction
        return {}

# ---------------------------------------------------------------------
# WebAgent
# ---------------------------------------------------------------------
class WebAgent:
    """Main web automation agent"""
    
    def __init__(self):
        self.llm = LocalLLM()
        self.memory = TaskMemory()
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
    
    async def start(self):
        """Initialize browser"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=False)  # Set to True for headless
        self.page = await self.browser.new_page()
        
        # Set user agent and viewport
        await self.page.set_viewport_size({"width": 1920, "height": 1080})
        await self.page.set_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    
    async def stop(self):
        """Clean up browser"""
        if self.browser:
            await self.browser.close()
    
    async def execute_instruction(self, instruction: str) -> TaskResult:
        """Execute natural language instruction"""
        try:
            # Parse instruction using LLM
            parsed_task = self.llm.parse_instruction(instruction)
            print(f"Parsed task: {parsed_task}")
            
            # Execute based on action type
            if parsed_task["action"] == "search_and_extract":
                result = await self._search_and_extract(parsed_task)
            elif parsed_task["action"] == "fill_form":
                result = await self._fill_form(parsed_task)
            elif parsed_task["action"] == "navigate_and_extract":
                result = await self._navigate_and_extract(parsed_task)
            elif parsed_task["action"] == "complex_workflow":
                # If parser returned a complex workflow (list of steps), you can either
                # handle it here or delegate to an AdvancedWebAgent. For now, fallback to a
                # simple behavior: run the top-level instruction using existing code.
                result = await self._search_and_extract({
                    "search_term": instruction,
                    "limit": 5,
                    "site": "amazon.com"
                })
            else:
                return TaskResult(
                    success=False,
                    data=None,
                    error_message=f"Unknown action: {parsed_task['action']}"
                )
            
            # Save to memory
            self.memory.save_task(instruction, result.data if result.success else result.error_message)
            
            return result
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                error_message=f"Execution error: {str(e)}"
            )
    
    async def _search_and_extract(self, task: Dict[str, Any]) -> TaskResult:
        """Search for items and extract results"""
        steps = []
        
        try:
            search_term = task["search_term"]
            limit = task["limit"]
            site = task.get("site", "amazon.com")
            
            # Navigate to search site
            search_url = f"https://www.{site}/s?k={search_term.replace(' ', '+')}"
            steps.append(f"Navigating to: {search_url}")
            await self.page.goto(search_url)
            await self.page.wait_for_load_state("networkidle")
            
            # Extract product information
            steps.append("Extracting product information")
            
            if "amazon" in site:
                products = await self._extract_amazon_products(limit)
            else:
                products = await self._generic_product_extraction(limit)
            
            steps.append(f"Found {len(products)} products")
            
            return TaskResult(
                success=True,
                data={
                    "search_term": search_term,
                    "products": products[:limit],
                    "total_found": len(products)
                },
                steps_taken=steps
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                error_message=str(e),
                steps_taken=steps
            )
    
    async def _extract_amazon_products(self, limit: int) -> List[Dict[str, str]]:
        """Extract Amazon product information"""
        products = []
        
        # Wait for products to load
        await self.page.wait_for_selector('[data-component-type="s-search-result"]', timeout=10000)
        
        # Extract product elements
        product_elements = await self.page.query_selector_all('[data-component-type="s-search-result"]')
        
        for element in product_elements[:limit]:
            try:
                # Extract title
                title_element = await element.query_selector('h2 a span, h2 span')
                title = await title_element.text_content() if title_element else "N/A"
                
                # Extract price
                price_element = await element.query_selector('.a-price-whole, .a-offscreen')
                price = await price_element.text_content() if price_element else "N/A"
                
                # Extract rating
                rating_element = await element.query_selector('.a-icon-alt')
                rating = await rating_element.get_attribute('title') if rating_element else "N/A"
                
                # Extract link
                link_element = await element.query_selector('h2 a')
                link = await link_element.get_attribute('href') if link_element else "N/A"
                if link and not link.startswith('http'):
                    link = f"https://www.amazon.com{link}"
                
                products.append({
                    "title": title.strip(),
                    "price": price.strip(),
                    "rating": rating,
                    "link": link
                })
                
            except Exception as e:
                print(f"Error extracting product: {e}")
                continue
        
        return products
    
    async def _generic_product_extraction(self, limit: int) -> List[Dict[str, str]]:
        """Generic product extraction for other sites"""
        # Simplified generic extraction
        products = []
        
        # Look for common product selectors
        selectors = [
            '.product', '.item', '.result', '[data-testid*="product"]'
        ]
        
        for selector in selectors:
            elements = await self.page.query_selector_all(selector)
            if elements:
                for element in elements[:limit]:
                    try:
                        text = await element.text_content()
                        if text and len(text.strip()) > 10:
                            products.append({
                                "title": text.strip()[:100],
                                "price": "N/A",
                                "rating": "N/A",
                                "link": "N/A"
                            })
                    except:
                        continue
                break
        
        return products
    
    async def _fill_form(self, task: Dict[str, Any]) -> TaskResult:
        """Fill form with provided data"""
        steps = []
        
        try:
            url = task["url"]
            form_data = task["form_data"]
            
            steps.append(f"Navigating to: {url}")
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
            
            # Fill form fields
            for field, value in form_data.items():
                steps.append(f"Filling field: {field}")
                await self.page.fill(f'input[name="{field}"], input[id="{field}"]', value)
            
            return TaskResult(
                success=True,
                data={"message": "Form filled successfully", "fields": list(form_data.keys())},
                steps_taken=steps
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                error_message=str(e),
                steps_taken=steps
            )
    
    async def _navigate_and_extract(self, task: Dict[str, Any]) -> TaskResult:
        """Navigate to URL and extract content"""
        steps = []
        
        try:
            url = task["url"]
            
            steps.append(f"Navigating to: {url}")
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
            
            # Extract page content
            steps.append("Extracting page content")
            title = await self.page.title()
            content = await self.page.text_content('body')
            
            return TaskResult(
                success=True,
                data={
                    "title": title,
                    "content": content[:1000] + "..." if len(content) > 1000 else content,
                    "url": url
                },
                steps_taken=steps
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                error_message=str(e),
                steps_taken=steps
            )

# CLI Interface
async def main():
    """Main CLI interface"""
    agent = WebAgent()
    await agent.start()
    
    print("🤖 AI Web Agent Started!")
    print("Type 'quit' to exit, 'memory' to see recent tasks")
    print("Example: 'search for laptops under $50K and list top 5'")
    print("-" * 50)
    
    try:
        while True:
            instruction = input("\n💬 Enter instruction: ").strip()
            
            if instruction.lower() == 'quit':
                break
            elif instruction.lower() == 'memory':
                tasks = agent.memory.get_recent_tasks()
                print("\n📝 Recent Tasks:")
                for i, task in enumerate(tasks, 1):
                    print(f"{i}. {task['instruction'][:50]}... ({task['timestamp']})")
                continue
            elif not instruction:
                continue
            
            print(f"\n🔄 Processing: {instruction}")
            result = await agent.execute_instruction(instruction)
            
            if result.success:
                print("✅ Success!")
                if result.steps_taken:
                    print("Steps taken:")
                    for step in result.steps_taken:
                        print(f"  • {step}")
                
                print("\n📊 Results:")
                if isinstance(result.data, dict) and 'products' in result.data:
                    products = result.data['products']
                    for i, product in enumerate(products, 1):
                        print(f"{i}. {product['title']}")
                        print(f"   Price: {product['price']}")
                        print(f"   Rating: {product['rating']}")
                        print(f"   Link: {product['link'][:50]}...")
                        print()
                else:
                    print(json.dumps(result.data, indent=2))
            else:
                print(f"❌ Error: {result.error_message}")
                if result.steps_taken:
                    print("Steps completed:")
                    for step in result.steps_taken:
                        print(f"  • {step}")
    
    finally:
        await agent.stop()

if __name__ == "__main__":
    # Required dependencies:
    # pip install playwright asyncio sqlite3
    # playwright install chromium
    
    print("Starting AI Web Agent...")
    asyncio.run(main())

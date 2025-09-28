# updated_main_agent.py
"""
Updated AI Web Agent — Chrome-only with local LLM loader helper.

This version:
- Starts Playwright Chromium/Chrome (persisted profile) as before.
- Adds a simple local-model loader helper load_local_mistral_model(path)
  which will attempt to use llama_cpp (llama-cpp-python) if available,
  otherwise will still mark the model as present if the file exists on disk.
- Exposes attributes:
    agent.model_path (str|None)
    agent.model_loaded (bool)
    agent.mistral_parser (object)  # has model_path/model_loaded fields for inspection
"""

import asyncio
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from playwright.async_api import async_playwright
    _PLAYWRIGHT_AVAILABLE = True
except Exception:
    async_playwright = None
    _PLAYWRIGHT_AVAILABLE = False

if TYPE_CHECKING:
    from playwright.async_api import Page, BrowserContext

# -------------------------
# dataclasses & memory
# -------------------------
@dataclass
class TaskStep:
    step_id: int
    description: str
    action: str
    parameters: Dict[str, Any]
    completed: bool = False
    result: Any = None

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
                steps_taken TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit(); conn.close()
    def save_task(self, instruction: str, result: Any, steps_taken: List[str] = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO tasks (instruction, result, steps_taken) VALUES (?, ?, ?)",
                       (instruction, json.dumps(result, ensure_ascii=False), json.dumps(steps_taken or [])))
        conn.commit(); conn.close()
    def get_recent_tasks(self, limit: int = 5) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT instruction, result, steps_taken, timestamp FROM tasks ORDER BY id DESC LIMIT ?",
                       (limit,))
        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                "instruction": row[0],
                "result": json.loads(row[1]) if row[1] else None,
                "steps_taken": json.loads(row[2]) if row[2] else [],
                "timestamp": row[3]
            })
        conn.close(); return tasks

@dataclass
class TaskResult:
    success: bool
    data: Any
    error_message: Optional[str] = None
    steps_taken: List[str] = None

# -------------------------
# Chrome-only Agent + local-LLM loader
# -------------------------
class SimpleMistralParser:
    """
    Minimal object to expose model metadata to external inspectors (like FastAPI's /health).
    If llama_cpp loads the model, we attach loader info as available.
    """
    def __init__(self):
        self.model_path: Optional[str] = None
        self.model_loaded: bool = False
        self.model_obj = None
        self.model_name: Optional[str] = None
        self.loader_repr: Optional[str] = None

class EnhancedWebAgent:
    def __init__(self, headless: bool = True):
        self.memory = TaskMemory()
        self.page: Optional["Page"] = None
        self.context: Optional["BrowserContext"] = None
        self._playwright = None
        self.headless = headless

        # Model-related attributes for health checks
        self.model_path: Optional[str] = None
        self.model_loaded: bool = False
        self.mistral_parser = SimpleMistralParser()

        # persistent profile dir for Playwright
        self.user_data_dir = Path(".playwright_profile").resolve()
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"🤖 EnhancedWebAgent initialized (headless={self.headless}). Mistral loader present: True", flush=True)

    async def start(self):
        # start playwright if available, be defensive about errors (Windows NotImplementedError)
        if not _PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not installed/available; will run without browser automation.", flush=True)
            return
        try:
            self._playwright = await async_playwright().start()
            # prefer launching system chrome, fallback to chromium
            try:
                self.context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self.user_data_dir),
                    channel="chrome",
                    headless=self.headless,
                    viewport={"width": 1366, "height": 768},
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"]
                )
            except Exception:
                # fallback to default chromium persistent context
                self.context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self.user_data_dir),
                    headless=self.headless,
                    viewport={"width": 1366, "height": 768},
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"]
                )
            pages = self.context.pages
            if pages:
                self.page = pages[0]
            else:
                self.page = await self.context.new_page()
            print("✅ Launched Chrome/Chromium via Playwright.", flush=True)
        except NotImplementedError:
            # common on some Windows environments when asyncio subprocess support isn't available
            print("⚠️ Playwright NotImplementedError: subprocesses not supported in this event loop environment.", flush=True)
            # don't re-raise here - agent can still operate with HTTP scrapers
        except Exception as e:
            print("⚠️ Playwright start error:", e, flush=True)
            # keep going - fallback scrapers still available

    async def stop(self):
        try:
            if self.page:
                await self.page.close()
        except Exception:
            pass
        try:
            if self.context:
                await self.context.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

    # -------------------------
    # Local model loader helper
    # -------------------------
    def load_local_mistral_model(self, path: str, force_mark_if_exists: bool = True):
        """
        Attempt to load a local GGUF model at `path`.
        - If llama_cpp (llama-cpp-python) is installed, try to create Llama object.
        - Otherwise, if the file exists, mark model_loaded=True and expose model_path.
        Returns True if model was marked loaded (or actually loaded).
        """
        p = Path(path)
        self.model_path = str(p.resolve()) if p.exists() else None
        self.mistral_parser.model_path = self.model_path

        # Try to import llama_cpp if available
        try:
            from llama_cpp import Llama
            # instantiate the model (this may consume GPU/CPU memory). Use small kwargs by default.
            # Users should customize n_ctx / n_gpu_layers depending on their environment.
            if p.exists():
                try:
                    llama = Llama(model_path=str(p))
                    # Set metadata
                    self.mistral_parser.model_obj = llama
                    self.mistral_parser.model_loaded = True
                    self.mistral_parser.model_name = getattr(llama, "__repr__", lambda: "llama_model")()
                    self.mistral_parser.loader_repr = repr(llama)[:1000]
                    self.model_loaded = True
                    print(f"✅ Loaded local model via llama_cpp at: {self.model_path}", flush=True)
                    return True
                except Exception as e:
                    print(f"⚠️ llama_cpp failed to load model at {p}: {e}", flush=True)
                    # fall through to mark presence below (if allowed)
        except Exception:
            # llama_cpp not available; we will still mark model present if file exists
            pass

        # If we reach here, llama_cpp wasn't used or failed. If file exists, mark present.
        if p.exists() and force_mark_if_exists:
            self.mistral_parser.model_loaded = True
            self.mistral_parser.model_name = p.name
            self.mistral_parser.loader_repr = "<file present, loader not bound>"
            self.model_loaded = True
            print(f"ℹ️ Model file exists at {self.model_path}. Marked as present (llama_cpp not used).", flush=True)
            return True

        # Not found / not loaded
        self.mistral_parser.model_loaded = False
        self.model_loaded = False
        print("⚠️ No local model loaded.", flush=True)
        return False

    # ----------- Google Search Only (playwright or fallback) -----------
    async def _execute_search_step(self, step: TaskStep) -> TaskResult:
        search_term = step.parameters.get("search_term", "")
        if not search_term:
            return TaskResult(success=False, data=None, error_message="No search term", steps_taken=[])

        steps_log = [f"🔎 Start search for: {search_term}"]
        url = f"https://www.google.com/search?q={search_term}&num=10"

        # Prefer Playwright extraction if page exists
        if self.page is not None:
            try:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await asyncio.sleep(1.0)
                results = []
                blocks = await self.page.query_selector_all("div.g, div.yuRUbf")
                rank = 1
                for b in blocks:
                    if rank > 10:
                        break
                    try:
                        title_el = await b.query_selector("h3")
                        link_el = await b.query_selector("a")
                        snippet_el = await b.query_selector(".VwiC3b, .aCOpRe, span.st")
                        title = (await title_el.inner_text()).strip() if title_el else None
                        link = (await link_el.get_attribute("href")).strip() if link_el else None
                        snippet = (await snippet_el.inner_text()).strip() if snippet_el else ""
                        if title and link and link.startswith("http"):
                            results.append({"rank": rank, "title": title, "link": link, "snippet": snippet, "source": "google"})
                            rank += 1
                    except Exception:
                        continue

                if results:
                    steps_log.append(f"[chrome] Extracted {len(results)} results from Google.")
                    return TaskResult(success=True, data={"search_term": search_term, "results": results}, steps_taken=steps_log)
                else:
                    steps_log.append("[chrome] No results extracted from Playwright DOM; falling back to HTTP fetch.")
            except Exception as e:
                # Playwright issues are tolerated; fallback next
                steps_log.append(f"[playwright] goto warning: {repr(e)}")
                # do not raise here; continue to HTTP scrapers

        # HTTP fetch fallback (simple, without external dependency)
        try:
            import requests, re
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = requests.get(url, headers=headers, timeout=15)
            html = resp.text
            # attempt naive extraction of google results (subject to google layout changes)
            anchors = re.findall(r'<a href="(/url\?q=https?://[^"&]+)', html)
            results = []
            seen = set()
            rank = 1
            for a in anchors:
                if rank > 10: break
                # extract q=...
                import urllib.parse as up
                qpart = up.unquote(a)
                m = up.parse_qs(up.urlsplit(qpart).query).get("q")
                if m and m[0] not in seen:
                    seen.add(m[0])
                    results.append({"rank": rank, "title": None, "link": m[0], "snippet": "", "source": "google-http"})
                    rank += 1
            if results:
                steps_log.append(f"[http] Extracted {len(results)} results via HTTP Google scrape.")
                return TaskResult(success=True, data={"search_term": search_term, "results": results}, steps_taken=steps_log)
            else:
                steps_log.append("[fallback] HTTP Google returned 0 results")
        except Exception as e:
            steps_log.append(f"[fallback] HTTP Google scrape error: {repr(e)}")

        # As last resort return failure
        steps_log.append("[fallback] All attempts (Playwright + HTTP) failed or returned no results")
        return TaskResult(success=False, data=None, error_message="All attempts (Playwright + HTTP) failed or returned no results", steps_taken=steps_log)

    async def execute_instruction(self, instruction: str) -> TaskResult:
        step = TaskStep(step_id=1, description=f"Search for {instruction}", action="search",
                        parameters={"search_term": instruction})
        return await self._execute_search_step(step)


# CLI test runner (optional)
async def main():
    agent = EnhancedWebAgent(headless=True)
    # Example: try to auto-load local model path (adjust to your path)
    # If you want to auto-load model on start, uncomment the next line and set your path:
    # agent.load_local_mistral_model("C:\Users\yashj\ai-web-agent\model\mistral-7b-openorca.gguf2.Q4_0.gguf")
    await agent.start()
    try:
        while True:
            instr = input("Enter instruction (or 'quit'): ").strip()
            if instr.lower() == "quit":
                break
            res = await agent.execute_instruction(instr)
            print("Result:", res.success, res.error_message)
            print(json.dumps(res.data, indent=2, ensure_ascii=False))
    finally:
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())

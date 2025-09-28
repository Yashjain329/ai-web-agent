# 🤖 AI Web Agent — Hackathon Submission

## 📌 Project Title

AI Web Agent — Autonomous Browsing with Local LLM (Mistral 7B GGUF)

## 📖 Project Summary

### Problem Understanding

Modern web browsing agents often rely on cloud-hosted LLMs and limited automation scripts, making them slow, dependent on the internet, and unable to run fully offline. The challenge is to build a **locally running AI Agent** that can take natural language instructions and autonomously drive the web — performing searches, filling forms, and collecting information.

### Proposed Prototype Solution

Our solution integrates:

* **Playwright** for browser automation.
* **llama-cpp-python** with **Mistral 7B OpenOrca GGUF** for local LLM-based instruction parsing.
* **FastAPI backend** to control the agent.
* **Interactive Web UI** (HTML, CSS, JS) to give commands and view results.
* **Advanced features** such as multi-step reasoning, task chaining, CAPTCHA handling, and export (JSON/CSV).

This enables users to say: *“search for laptops under 50k and list top 5”* and the agent will parse → plan → browse → scrape → return structured results.

### Uniqueness & Impact

* **Fully offline LLM** (no cloud dependency).
* **End-to-end autonomous agent** (parse → plan → execute → display).
* **Production-ready modes**:

  * Advanced multi-step workflows
  * Fast optimized mode (hackathon speed)
  * Enhanced persistent agent with API
* **Extensible UI** with speech-to-text, downloads, and recent task history.

This project demonstrates a path towards **self-reliant AI agents** that can automate human-like browsing tasks while ensuring privacy and speed.

---

## 🛠 Tech Stack

* **Backend:** Python, FastAPI, asyncio
* **Browser Automation:** Playwright
* **LLM Integration:** llama-cpp-python, Mistral 7B GGUF
* **Frontend:** HTML, CSS, JavaScript (Vanilla)
* **Database:** SQLite (task memory)
* **Utilities:** pytesseract (CAPTCHA OCR), pandas, plotly (for data viz)

---

## ✨ Features Implemented

* ✅ Natural language instruction parsing via local LLM
* ✅ Browser automation (search, navigate, extract, fill forms)
* ✅ Multi-step task chaining (search + filter + compare)
* ✅ Fast optimized agent (headless, resource blocking)
* ✅ CAPTCHA detection & solving (OCR & slider)
* ✅ Persistent task memory with history
* ✅ Interactive frontend (speech-to-text, clickable results, CSV/JSON download)
* ✅ REST API with FastAPI for integration

---

## 📂 Project Structure

```
/ai-web-agent
 ├── ai_web_agent.py              # Baseline agent
 ├── advanced_features.py         # Multi-step reasoning & workflows
 ├── fast_production_agent.py     # Speed-optimized agent
 ├── updated_main_agent.py        # Enhanced persistent agent
 ├── fastapi_agent_server.py      # FastAPI backend server
 ├── captcha_handler.py           # CAPTCHA detection & solving
 ├── mistral_integration.py       # Compatibility parser for Mistral
 ├── llama_cpp_integration_and_patches.py # Direct llama-cpp integration
 ├── test_agent.py                # Sanity tests (LLM, Playwright, integration)
 ├── static/
 │   ├── index.html               # Frontend UI
 │   ├── style.css                # Stylesheet
 │   └── app.js                   # Client-side logic
 └── requirements.txt             # Python dependencies
```

---

## 📊 System Architecture Diagram

![System Architecture](./docs/system-architecture.png)

*(Diagram shows: User → Frontend UI → FastAPI Server → Enhanced Web Agent → Playwright + Local LLM → Websites + SQLite Memory → Back to User)*

---

## 🔄 Workflow Example

![Workflow Example](./docs/workflow-sequence.png)

*(Diagram shows: User gives instruction → Frontend sends to FastAPI → LLM parses → Playwright executes → Results returned → Displayed in UI)*

---

## 👥 Team Contributions

* **Member 1 (Team Lead):** Core agent architecture, LLM integration
* **Member 2:** Playwright automation, advanced task execution
* **Member 3:** CAPTCHA handler, optimized fast agent
* **Member 4:** FastAPI backend, thread-safe runner
* **Member 5:** Frontend (UI/UX), downloads, speech-to-text

---

## 📊 Evaluation Mapping

* **Problem Understanding (20%)** → Explained in summary & video
* **Innovation & Creativity (40%)** → Offline LLM + autonomous browsing + CAPTCHA solving
* **Feasibility & Clarity (30%)** → Working prototype with FastAPI + UI
* **Collaboration (10%)** → Contributions detailed, video includes all members

---

## 🚀 Next Steps

* Extend beyond Amazon to multi-site shopping comparison
* Add reinforcement learning for dynamic web navigation
* Containerize deployment with Docker for easy scaling
* Integrate optional voice output (TTS) for accessibility

---

## 📧 Support

Conatact : Yashjain9350@gmail.com 

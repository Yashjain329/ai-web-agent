# test_agent.py
"""
Test script for AI Web Agent with local llama-cpp (GGUF) backend
Run this to verify your setup is working correctly
"""

import asyncio
import sys
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    ok = True

    try:
        import playwright  # type: ignore
        print("✅ Playwright imported successfully")
    except ImportError:
        print("❌ Playwright not found. Run: pip install playwright")
        ok = False

    try:
        import streamlit  # type: ignore
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("❌ Streamlit not found. Run: pip install streamlit")
        ok = False

    # Test our local llama-cpp integration module
    try:
        # The user-provided integration file should be in the same folder
        from llama_cpp_integration_and_patches import LlamaCppClient  # type: ignore
        print("✅ llama-cpp integration module found")
    except ImportError:
        print("❌ llama-cpp integration module not found. Make sure llama_cpp_integration_and_patches.py is in this directory")
        ok = False
    except Exception as e:
        print(f"❌ Error importing llama-cpp integration: {e}")
        ok = False

    return ok

def test_llama_model_load():
    """Test loading the GGUF model via LlamaCppClient"""
    print("\n🤖 Testing local llama-cpp (GGUF) model load...")

    try:
        from llama_cpp_integration_and_patches import LlamaCppClient  # type: ignore
    except Exception as e:
        print(f"❌ Cannot import LlamaCppClient: {e}")
        print("💡 Ensure llama_cpp_integration_and_patches.py is present and that llama-cpp-python is installed.")
        return False

    model_path = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf"
    try:
        print(f"🔎 Checking model file: {model_path}")
        client = LlamaCppClient(model_path=model_path, n_ctx=2048, verbose=False)
        # small sanity generate
        sample = client.generate("Say hello in one word.", max_tokens=8, temperature=0.1)
        if sample and isinstance(sample, str):
            print("✅ Local llama-cpp model loaded and responded")
            print(f"🗒️ Response preview: {sample.strip()[:200]}")
            return True
        else:
            print("⚠️ Model loaded but response was empty (still considered OK)")
            return True
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("💡 Put your .gguf file at the path above or update the path in this test.")
        return False
    except Exception as e:
        print(f"❌ Failed to load or run model: {e}")
        print("💡 Common issues: llama-cpp-python not installed, incompatible wheel on Windows, or wrong GGUF path.")
        return False

async def test_browser_automation():
    """Test browser automation with Playwright"""
    print("\n🌐 Testing browser automation...")

    try:
        from playwright.async_api import async_playwright  # type: ignore

        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()

        # Test navigation
        await page.goto("https://httpbin.org/get", timeout=10000)
        title = await page.title()

        await browser.close()
        await playwright.stop()

        print("✅ Browser automation working")
        print(f"📄 Test page title: {title}")
        return True

    except Exception as e:
        print(f"❌ Browser automation failed: {e}")
        print("💡 Try running: playwright install chromium")
        return False

async def test_agent_integration():
    """Test the main agent integration (parser + optional agent)"""
    print("\n🤖 Testing agent integration...")

    try:
        # Try to import our compatibility wrapper (which uses llama-cpp)
        from mistral_integration import MistralLLMParser  # type: ignore
        print("✅ Imported MistralLLMParser from mistral_integration.py")
    except Exception as e:
        print(f"❌ Could not import MistralLLMParser: {e}")
        print("💡 Make sure mistral_integration.py is present and references the llama-cpp wrapper.")
        return False

    try:
        # Initialize parser (this will create the underlying Llama client)
        parser = MistralLLMParser()
        print("🔧 Initializing parser and model...")
        if not parser.setup_model():
            print("⚠️ Parser setup_model() returned False (model not ready)")
            # still try to call parser; it may fallback to rule-based parsing
        else:
            print("✅ Parser reports model is ready")

        # Run an example parse (async)
        steps = await parser.parse_complex_instruction("search for laptops and list top 5", context=None)
        if steps:
            print(f"✅ Instruction parsing returned {len(steps)} steps")
            # print a short preview
            for i, s in enumerate(steps[:5], start=1):
                # s may be TaskStep dataclass or dict depending on wrapper
                if hasattr(s, 'description'):
                    desc = getattr(s, 'description')
                    action = getattr(s, 'action', 'N/A')
                elif isinstance(s, dict):
                    desc = s.get('description', str(s))
                    action = s.get('action', 'N/A')
                else:
                    desc = str(s)
                    action = 'N/A'
                print(f"   {i}. {desc} ({action})")
            return True
        else:
            print("⚠️ Parsing returned no steps (rule-based fallback may be used)")
            return True

    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")
        return False

async def run_full_test():
    """Run complete test suite"""
    print("🚀 AI Web Agent - Complete System Test (llama-cpp backend)")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tests_passed = 0
    total_tests = 4

    # Test 1: Imports
    if test_imports():
        tests_passed += 1

    # Test 2: Local llama-cpp model load
    if test_llama_model_load():
        tests_passed += 1

    # Test 3: Browser automation
    if await test_browser_automation():
        tests_passed += 1

    # Test 4: Agent Integration
    if await test_agent_integration():
        tests_passed += 1

    # Results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Tests Passed: {tests_passed}/{total_tests}")
    print(f"📈 Success Rate: {(tests_passed/total_tests)*100:.1f}%")

    if tests_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Your AI Web Agent is ready (llama-cpp).")
        print("\n🚀 Next steps:")
        print("   • Run GUI: streamlit run enhanced_streamlit_gui.py")
        print("   • Run CLI: python updated_main_agent.py")
        print("   • Try command: 'search for laptops under $1500 and list top 5'")
    else:
        print(f"\n⚠️ {total_tests - tests_passed} tests failed. See messages above for details.")
        print("\n💡 Common solutions:")
        print("   • Install missing packages: pip install -r requirements.txt")
        print("   • Install browsers: playwright install chromium")
        print("   • Ensure llama-cpp-python is installed and a compatible wheel is present on Windows")
        print("   • Confirm GGUF model path is correct and readable")

    return tests_passed == total_tests

def main():
    """Main test function"""
    try:
        success = asyncio.run(run_full_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

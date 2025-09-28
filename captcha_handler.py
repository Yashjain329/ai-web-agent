"""
CAPTCHA Detection and Resolution Module (async-first)
This version is designed to work with Playwright's async Page API.
It also exposes small sync wrappers if you prefer to call it from synchronous code.
"""

import asyncio
import time
import os
import base64
from typing import Dict, List, Optional, Tuple, Any
import io
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# pytesseract might not be available in every environment; guard it
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# Helper to detect awaitability of a call
def _is_coroutine(obj) -> bool:
    return asyncio.iscoroutine(obj) or asyncio.iscoroutinefunction(obj)

class CaptchaDetector:
    """Enhanced CAPTCHA detection and solving (async-first)"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.captcha_types = [
            "recaptcha", "hcaptcha", "text_captcha",
            "image_captcha", "puzzle_captcha", "slider_captcha"
        ]

        # Common CAPTCHA selectors
        self.selectors = {
            "recaptcha": [
                'iframe[src*="recaptcha"]',
                '.g-recaptcha',
                '[data-sitekey]',
                '#recaptcha'
            ],
            "hcaptcha": [
                'iframe[src*="hcaptcha"]',
                '.h-captcha',
                '[data-sitekey*="hcaptcha"]'
            ],
            "text_captcha": [
                'img[alt*="captcha"]',
                'img[src*="captcha"]',
                '.captcha-image',
                '#captcha-img'
            ],
            "slider_captcha": [
                '.slider-captcha',
                '.slide-verify',
                '[class*="slider"]'
            ]
        }

    # ---------- Detection ----------
    async def detect_captcha_type_async(self, page: Any) -> Optional[str]:
        """Detect type of CAPTCHA present on page (async Playwright Page)"""
        try:
            for captcha_type, selectors in self.selectors.items():
                for selector in selectors:
                    try:
                        el = await page.query_selector(selector)
                        if el:
                            # ensure visible if possible
                            try:
                                visible = await el.is_visible()
                                if visible:
                                    return captcha_type
                            except Exception:
                                return captcha_type
                    except Exception:
                        continue

            # Check for common CAPTCHA keywords in page content
            try:
                content = await page.content()
                page_content = content.lower()
            except Exception:
                try:
                    page_content = (await page.evaluate("() => document.body.innerText")).lower()
                except Exception:
                    page_content = ""

            captcha_keywords = ["captcha", "verify", "i'm not a robot", "prove you're human", "are you human"]

            for keyword in captcha_keywords:
                if keyword in page_content:
                    return "generic_captcha"

            return None

        except Exception as e:
            print(f"[captcha] Detection error: {e}")
            return None

    # ---------- OCR-based text CAPTCHA solving ----------
    async def solve_text_captcha_async(self, page: Any) -> bool:
        """Solve text-based CAPTCHA using OCR (async Playwright Page)"""
        if not TESSERACT_AVAILABLE:
            print("[captcha] pytesseract not available")
            return False

        try:
            captcha_img_handle = None
            for selector in self.selectors.get("text_captcha", []):
                try:
                    el = await page.query_selector(selector)
                    if el:
                        captcha_img_handle = el
                        break
                except Exception:
                    continue

            if not captcha_img_handle:
                return False

            # Playwright screenshot: returns bytes when awaited
            try:
                screenshot_bytes = await captcha_img_handle.screenshot()
            except Exception:
                # fallback: screenshot the whole page and crop bounding box
                try:
                    box = await captcha_img_handle.bounding_box()
                    full_bytes = await page.screenshot()
                    img = Image.open(io.BytesIO(full_bytes)).convert("RGB")
                    if box:
                        left = int(box["x"])
                        top = int(box["y"])
                        right = int(box["x"] + box["width"])
                        bottom = int(box["y"] + box["height"])
                        cropped = img.crop((left, top, right, bottom))
                        processed = self.preprocess_captcha_image(cropped)
                        text = pytesseract.image_to_string(
                            processed,
                            config='--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        ).strip()
                        if text:
                            return await self._submit_captcha_text(page, text)
                    return False
                except Exception as e:
                    print(f"[captcha] Fallback screenshot failed: {e}")
                    return False

            # Convert bytes to PIL Image
            from io import BytesIO
            image = Image.open(BytesIO(screenshot_bytes))
            processed_image = self.preprocess_captcha_image(image)

            captcha_text = pytesseract.image_to_string(
                processed_image,
                config='--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            ).strip()

            if captcha_text:
                print(f"[captcha] OCR result: {captcha_text}")
                return await self._submit_captcha_text(page, captcha_text)

            return False

        except Exception as e:
            print(f"[captcha] Text CAPTCHA solving failed: {e}")
            return False

    async def _submit_captcha_text(self, page: Any, text: str) -> bool:
        """Fill input and submit the OCR result (async)"""
        try:
            input_selectors = [
                'input[name*="captcha"]',
                'input[id*="captcha"]',
                'input[placeholder*="captcha"]',
                'input[type="text"][name*="code"]',
                'input[aria-label*="captcha"]'
            ]

            for selector in input_selectors:
                try:
                    input_field = await page.query_selector(selector)
                    if input_field:
                        await input_field.fill(text)
                        # find submit
                        submit_selectors = [
                            'button[type="submit"]',
                            'input[type="submit"]',
                            'button:has-text("Submit")',
                            'button:has-text("Verify")',
                            'button:has-text("Check")'
                        ]
                        for submit_selector in submit_selectors:
                            try:
                                submit_btn = await page.query_selector(submit_selector)
                                if submit_btn:
                                    await submit_btn.click()
                                    await page.wait_for_timeout(1500)
                                    return True
                            except Exception:
                                continue
                except Exception:
                    continue
            return False
        except Exception as e:
            print(f"[captcha] Submitting captcha text failed: {e}")
            return False

    # ---------- image preprocessing ----------
    def preprocess_captcha_image(self, image: Image.Image) -> Image.Image:
        """Preprocess CAPTCHA image for better OCR accuracy (PIL)"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')

            width, height = image.size
            if width < 150 or height < 50:
                scale_factor = max(150 / max(1, width), 50 / max(1, height))
                new_size = (int(width * scale_factor), int(height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)

            image = image.filter(ImageFilter.MedianFilter(3))

            # apply simple threshold
            threshold = 140
            image = image.point(lambda p: 255 if p > threshold else 0)

            return image

        except Exception as e:
            print(f"[captcha] Image preprocessing failed: {e}")
            return image

    # ---------- slider captcha attempt ----------
    async def solve_slider_captcha_async(self, page: Any) -> bool:
        """Attempt to solve slider CAPTCHA (async)"""
        try:
            slider = None
            for selector in self.selectors.get("slider_captcha", []):
                try:
                    el = await page.query_selector(selector)
                    if el:
                        slider = el
                        break
                except Exception:
                    continue

            if not slider:
                return False

            box = await slider.bounding_box()
            if not box:
                return False

            start_x = box["x"] + 10
            start_y = box["y"] + box["height"] / 2
            end_x = box["x"] + box["width"] - 10

            # simulate drag
            await page.mouse.move(start_x, start_y)
            await page.mouse.down()
            steps = 18
            for i in range(steps):
                progress = (i + 1) / steps
                current_x = start_x + (end_x - start_x) * progress
                current_y = start_y + (np.random.normal(0, 1) if np else 0)
                await page.mouse.move(current_x, current_y)
                await asyncio.sleep(0.05 + (np.random.uniform(0, 0.02) if np else 0))
            await page.mouse.up()
            await page.wait_for_timeout(2000)
            return True

        except Exception as e:
            print(f"[captcha] Slider CAPTCHA solving failed: {e}")
            return False

    # ---------- reCAPTCHA handling (best-effort / manual) ----------
    async def handle_recaptcha_async(self, page: Any) -> bool:
        """Handle reCAPTCHA (detect and allow manual solving)"""
        try:
            success_indicators = ['.recaptcha-checkbox-checked', '[aria-checked="true"]']
            for indicator in success_indicators:
                try:
                    el = await page.query_selector(indicator)
                    if el:
                        return True
                except Exception:
                    continue

            # try to click simple checkbox
            checkbox_selectors = ['.recaptcha-checkbox', '[role="checkbox"]', 'div.recaptcha-checkbox']
            for sel in checkbox_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        try:
                            await el.click()
                        except Exception:
                            pass
                        await page.wait_for_timeout(1500)
                        for indicator in success_indicators:
                            try:
                                el2 = await page.query_selector(indicator)
                                if el2:
                                    return True
                            except:
                                continue

                        # If challenge appears, wait a short time for manual solve
                        for _ in range(30):
                            await asyncio.sleep(1)
                            for indicator in success_indicators:
                                try:
                                    el3 = await page.query_selector(indicator)
                                    if el3:
                                        return True
                                except:
                                    continue
                        return False
                except Exception:
                    continue

            return False
        except Exception as e:
            print(f"[captcha] reCAPTCHA handling failed: {e}")
            return False

    # ---------- main solver dispatcher ----------
    async def solve_captcha_async(self, page: Any, captcha_type: Optional[str] = None) -> bool:
        """Main async solver entrypoint"""
        try:
            ctype = captcha_type
            if not ctype:
                ctype = await self.detect_captcha_type_async(page)
            if not ctype:
                return False

            if ctype == "text_captcha":
                return await self.solve_text_captcha_async(page)
            if ctype == "slider_captcha":
                return await self.solve_slider_captcha_async(page)
            if ctype == "recaptcha":
                return await self.handle_recaptcha_async(page)
            if ctype == "hcaptcha":
                # hCaptcha often requires manual solving
                print("[captcha] hCaptcha detected — manual solve may be required")
                # Optionally wait for manual solve
                for _ in range(30):
                    await asyncio.sleep(1)
                    typ = await self.detect_captcha_type_async(page)
                    if not typ:
                        return True
                return False

            # generic fallback: attempt OCR
            if ctype == "generic_captcha":
                return await self.solve_text_captcha_async(page)

            return False

        except Exception as e:
            print(f"[captcha] Solving failed: {e}")
            return False

    # ---------- SYNC wrappers for convenience ----------
    def detect_captcha_type(self, page) -> Optional[str]:
        """Sync wrapper (calls async implementation)"""
        return asyncio.get_event_loop().run_until_complete(self.detect_captcha_type_async(page))

    def solve_captcha(self, page, captcha_type: Optional[str] = None) -> bool:
        """Sync wrapper for solve_captcha_async"""
        return asyncio.get_event_loop().run_until_complete(self.solve_captcha_async(page, captcha_type))

# ErrorRecoveryManager is unchanged but provided in sync + async variants for Playwright usage
class ErrorRecoveryManager:
    """Handle various error scenarios and recovery strategies"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)

    # The following recovery methods expect an async Playwright Page and are async.
    async def handle_page_error_async(self, page, error_type: str, context: Dict = None) -> bool:
        try:
            ctx = context or {}
            if error_type == "timeout":
                return await self.handle_timeout_error_async(page, ctx)
            if error_type == "network":
                return await self.handle_network_error_async(page, ctx)
            if error_type == "element_not_found":
                return await self.handle_element_error_async(page, ctx)
            if error_type == "javascript":
                return await self.handle_javascript_error_async(page, ctx)
            if error_type == "navigation":
                return await self.handle_navigation_error_async(page, ctx)
            return False
        except Exception as e:
            print(f"[recovery] Error handling failed: {e}")
            return False

    async def handle_timeout_error_async(self, page, context: Dict) -> bool:
        try:
            await page.reload(wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[recovery] Timeout recovery failed: {e}")
            return False

    async def handle_network_error_async(self, page, context: Dict) -> bool:
        try:
            await asyncio.sleep(self.retry_delay)
            try:
                await page.evaluate("() => document.readyState")
                return True
            except Exception:
                try:
                    await page.goto("about:blank")
                except Exception:
                    pass
                await asyncio.sleep(1)
                return False
        except Exception as e:
            print(f"[recovery] Network recovery failed: {e}")
            return False

    async def handle_element_error_async(self, page, context: Dict) -> bool:
        try:
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)
            alternatives = context.get("alternative_selectors", [])
            for sel in alternatives:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        return True
                except Exception:
                    continue
            return False
        except Exception as e:
            print(f"[recovery] Element recovery failed: {e}")
            return False

    async def handle_javascript_error_async(self, page, context: Dict) -> bool:
        try:
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[recovery] JavaScript recovery failed: {e}")
            return False

    async def handle_navigation_error_async(self, page, context: Dict) -> bool:
        try:
            target_url = context.get("target_url")
            if not target_url:
                return False
            try:
                await page.goto(target_url, wait_until="load")
                return True
            except Exception as e:
                print(f"[recovery] Navigation attempt failed: {e}")
                return False
        except Exception as e:
            print(f"[recovery] Navigation recovery failed: {e}")
            return False

    # Sync wrappers (convenience)
    def handle_page_error(self, page, error_type: str, context: Dict = None) -> bool:
        return asyncio.get_event_loop().run_until_complete(self.handle_page_error_async(page, error_type, context or {}))

__all__ = ["CaptchaDetector", "ErrorRecoveryManager"]
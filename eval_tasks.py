"""
BROWSER-USE WEBBENCH EVALUATOR  — fixed for slow local LLMs (DeepSeek 32B @ ~1.6 tok/s)
==========================================================================================
ADDED: --llm-provider flag  (gemini | local)
  - gemini  → uses langchain_google_genai.ChatGoogleGenerativeAI + GOOGLE_API_KEY from .env
  - local   → uses ChatOpenAI (OpenAI-compatible) pointing at your vLLM / llama.cpp endpoint

KEY FIXES vs original:
  FIX-1  step_timeout raised to 1200s default (was 300s).
  FIX-2  max_steps reduced to 5 default (was 10).
  FIX-3  DeepSeek <think> tag stripping.
  FIX-4  DOM content length cap.
  FIX-5  Hard asyncio.wait_for guard around agent.run().
  FIX-6  Terse output instruction added to prompt.
  FIX-7  Token budget estimator in __init__.
  FIX-8  Per-step timing logged.
  FIX-9  [NEW] --llm-provider gemini|local  with automatic LLM factory.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd
from dotenv import load_dotenv

# ── Import guard ───────────────────────────────────────────────────────────────
try:
    from browser_use import Agent, Browser
    from browser_use.llm import ChatOpenAI
except ImportError as e:
    print(f"\n[ERROR] Cannot import browser_use: {e}")
    print("  Fix: pip install browser-use  or  pip install -e path/to/browser-use\n")
    raise SystemExit(1)

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("browser_use").setLevel(logging.WARNING)

# ── FIX-4: Maximum characters of DOM/accessibility tree sent to LLM per step ──
MAX_DOM_CHARS = 3_000

# ── FIX-3: Regex to strip DeepSeek <think>...</think> blocks ──────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove DeepSeek <think>...</think> reasoning blocks from model output."""
    return _THINK_RE.sub("", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# FIX-9: LLM FACTORY
# Strategy: Use Google's OpenAI-compatible endpoint for Gemini so that
# browser-use's ChatOpenAI path handles tool-calling natively.
# This avoids the langchain-google-genai wrapper which produces malformed
# structured outputs that browser-use cannot parse (❌ Result failed 5/6: items).
# ─────────────────────────────────────────────────────────────────────────────

# Google OpenAI-compat base URL — same API surface as OpenAI, works with ChatOpenAI
_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Map friendly short names to full Gemini model IDs on the OpenAI-compat endpoint
_GEMINI_MODEL_ALIASES: dict[str, str] = {
    "gemini-2.5-flash":        "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash":        "gemini-2.0-flash",
    "gemini-2.0-flash-exp":    "gemini-2.0-flash-exp",
    "gemini-1.5-pro":          "gemini-1.5-pro",
    "gemini-1.5-flash":        "gemini-1.5-flash",
}


def build_llm(args: argparse.Namespace) -> Any:
    """
    Return the appropriate LLM object based on --llm-provider.

    Providers:
      gemini  — ChatOpenAI pointed at Google's OpenAI-compatible endpoint.
                Uses GOOGLE_API_KEY from .env. No extra packages needed.
                Tool-calling works identically to OpenAI — browser-use parses
                it correctly without any wrapper.

      local   — ChatOpenAI (OpenAI-compat) pointing at vLLM / llama.cpp
                Requires: --llm-base-url  --llm-api-key  --llm-model
    """
    provider = args.llm_provider.lower()

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment / .env file.\n"
                "  Add:  GOOGLE_API_KEY=your-key  to .env and retry."
            )

        # Resolve model name: use alias map, or pass through as-is
        raw_model = (
            args.llm_model
            if args.llm_model != _LOCAL_MODEL_DEFAULT
            else "gemini-2.5-flash-preview-05-20"
        )
        model_name = _GEMINI_MODEL_ALIASES.get(raw_model, raw_model)

        logger.info(f"  LLM Provider : Gemini via OpenAI-compat endpoint  (model={model_name})")
        logger.info(f"  Endpoint     : {_GEMINI_OPENAI_BASE}")

        # ChatOpenAI pointed at Google's OpenAI-compat base — browser-use handles
        # this path natively; tool-call JSON is identical to OpenAI format.
        llm = ChatOpenAI(
            base_url=_GEMINI_OPENAI_BASE,
            api_key=api_key,
            model=model_name,
            temperature=args.llm_temperature,
            #max_tokens=4096,
        )
        return llm

    elif provider == "local":
        # ── Local vLLM / llama.cpp (OpenAI-compat) ────────────────────────────
        logger.info(f"  LLM Provider : Local OpenAI-compat  (url={args.llm_base_url}  model={args.llm_model})")
        llm = ChatOpenAI(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            temperature=args.llm_temperature,
        )
        llm.max_tokens = 1500
        if hasattr(llm, "model_kwargs"):
            llm.model_kwargs = {"max_tokens": 1500}
        return llm

    else:
        raise ValueError(f"Unknown --llm-provider '{provider}'. Choose: gemini | local")


# ─────────────────────────────────────────────────────────────────────────────
# ERROR TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────
ERROR_TAXONOMY: dict[str, dict[str, str]] = {
    "Security_BotBlocked": {
        "label": "WAF / IP Block",
        "hint": "Add stealth plugin + residential proxy rotation",
    },
    "Security_CaptchaFailed": {
        "label": "CAPTCHA wall",
        "hint": "Integrate 2captcha / CapSolver webhook",
    },
    "Auth_CredentialInjectionFailed": {
        "label": "Auth wall — no credentials",
        "hint": "Pre-inject session cookies or use secrets vault",
    },
    "Data_StaleLayout": {
        "label": "DOM element not found",
        "hint": "Enable vision fallback; refresh benchmark dataset",
    },
    "System_SilentCrash": {
        "label": "Browser/LLM crash or timeout",
        "hint": "Add health-check + exponential-backoff retry",
    },
    "Logic_AgentLoopOrHallucination": {
        "label": "LLM reasoning failure",
        "hint": "Increase model size or add step validator",
    },
    "LLM_Timeout": {
        "label": "LLM did not respond within timeout",
        "hint": "Increase step_timeout or switch to faster model",
    },
    "ExecutionTimeout": {
        "label": "Full-task wall-clock timeout",
        "hint": "Increase step_timeout or reduce max_steps",
    },
    "BrowserInitFailed": {
        "label": "Browser failed to start",
        "hint": "Check Playwright installation; retry with headless=True",
    },
    "UnknownError": {
        "label": "Unclassified exception",
        "hint": "Review full stack trace in results JSON",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# TASK VALIDITY SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
VALIDITY_CODES = {
    "URL_Dead":                  "URL returns 4xx/5xx or is unreachable",
    "URL_Redirected":            "URL permanently redirected to a different domain/path",
    "Product_Discontinued":      "The specific product/listing/item no longer exists on the site",
    "Page_Restructured":         "The page exists but its layout/navigation changed making the task impossible",
    "Data_NotFound_OnSite":      "The data the task asks for was never on the site (bad benchmark entry)",
    "Task_RequiresAccount":      "Task requires login/account that cannot be created autonomously",
    "Task_Ambiguous":            "Task instruction is unclear or contradictory",
    "Framework_NoFileHandling":  "browser-use has no file download/upload pipeline",
    "Framework_NoCaptchaSolver": "browser-use has no CAPTCHA bypass integration",
    "Framework_NoCredentials":   "browser-use has no credential/session injection",
    "VALID":                     "Task and URL are valid; failure is agent/framework issue",
    "UNKNOWN":                   "Validity could not be determined",
}

VALIDITY_TO_ROOT = {
    "URL_Dead":                  "TASK_INVALID",
    "URL_Redirected":            "TASK_INVALID",
    "Product_Discontinued":      "TASK_INVALID",
    "Page_Restructured":         "TASK_INVALID",
    "Data_NotFound_OnSite":      "TASK_INVALID",
    "Task_Ambiguous":            "TASK_INVALID",
    "Task_RequiresAccount":      "FRAMEWORK_LIMIT",
    "Framework_NoFileHandling":  "FRAMEWORK_LIMIT",
    "Framework_NoCaptchaSolver": "FRAMEWORK_LIMIT",
    "Framework_NoCredentials":   "FRAMEWORK_LIMIT",
    "VALID":                     "AGENT_FAILURE",
    "UNKNOWN":                   "AGENT_FAILURE",
}

_FRAMEWORK_LIMIT_MAP: list[tuple[str, list[str]]] = [
    ("Framework_NoFileHandling",  ["download", "upload", "file", "attachment", "export csv", "export pdf", "save file"]),
    ("Framework_NoCaptchaSolver", ["captcha", "cloudflare", "recaptcha", "hcaptcha", "human verification"]),
    ("Framework_NoCredentials",   ["log in", "login", "sign in", "create account", "register", "your account", "your profile"]),
]


def classify_framework_limit(instruction: str, thoughts: str) -> str | None:
    combined = (instruction + " " + thoughts).lower()
    for code, keywords in _FRAMEWORK_LIMIT_MAP:
        if any(kw in combined for kw in keywords):
            return code
    return None


def probe_url(url: str, timeout: int = 8) -> tuple[int, str]:
    """HEAD-request the URL. Returns (http_status, note)."""
    try:
        parsed = urlparse(url)
        original_domain = parsed.netloc
        import urllib.request
        req = urllib.request.Request(
            url, method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
        )
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    final_domain = urlparse(resp.url).netloc
                    if final_domain and final_domain != original_domain:
                        return -1, f"Redirected to {resp.url}"
                    return resp.status, ""
            except urllib.error.HTTPError as e:
                return e.code, str(e.reason)
        return 0, "Too many redirects"
    except Exception as e:
        return 0, str(e)[:120]


_VALIDITY_CLASSIFIER_PROMPT = """\
You are a benchmark quality auditor. Given a web task and the agent's thoughts after attempting it, \
classify WHY the task failed.

Return ONLY a JSON object — no preamble, no markdown:
{{
  "validity_code": "<one of the codes below>",
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence explanation>"
}}

Validity codes (pick exactly one):
- Product_Discontinued  : the specific product/listing no longer exists on the site
- Page_Restructured     : the page structure changed so the task path is broken
- Data_NotFound_OnSite  : the data requested was never on the site (bad benchmark)
- Task_Ambiguous        : the task instruction is unclear or self-contradictory
- VALID                 : task and site are fine; the agent just failed

Task instruction:
{instruction}

Agent's thoughts / observations:
{thoughts}

JSON only:"""


async def classify_task_validity(
    instruction: str,
    thoughts: str,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[str, float, str]:
    if not api_key or not thoughts.strip():
        return "UNKNOWN", 0.0, "No API key or no agent thoughts to analyse"

    prompt = _VALIDITY_CLASSIFIER_PROMPT.format(
        instruction=instruction[:600],
        thoughts=thoughts[:1200],
    )
    try:
        import urllib.request as _ur
        payload = json.dumps({
            "model": model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        if model.startswith("claude"):
            endpoint = "https://api.anthropic.com/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
        else:
            endpoint = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

        req = _ur.Request(endpoint, data=payload, headers=headers, method="POST")
        with _ur.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())

        if model.startswith("claude"):
            raw = data["content"][0]["text"].strip()
        else:
            raw = data["choices"][0]["message"]["content"].strip()

        raw = raw.strip("`").lstrip("json").strip()
        parsed = json.loads(raw)
        return (
            parsed.get("validity_code", "UNKNOWN"),
            float(parsed.get("confidence", 0.5)),
            parsed.get("reason", "")[:300],
        )
    except Exception as e:
        return "UNKNOWN", 0.0, f"Classifier error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TaskResult:
    # Identity
    task_id: str = ""
    run_timestamp: str = ""
    category: str = ""
    url: str = ""
    url_domain: str = ""
    instruction: str = ""
    instruction_length: int = 0

    # Execution
    status: str = "PENDING"
    latency_seconds: float = 0.0
    steps_taken: int = 0
    steps_budget: int = 0
    steps_utilization_pct: float = 0.0
    retry_count: int = 0

    # LLM interaction
    llm_model: str = ""
    llm_provider: str = ""   # NEW: track which provider was used
    llm_calls: int = 0
    total_thought_chars: int = 0
    final_answer: str = ""
    final_answer_length: int = 0

    # Timing detail
    avg_step_latency_seconds: float = 0.0
    max_step_latency_seconds: float = 0.0

    # Error fields
    error_code: str = ""
    error_label: str = ""
    error_hint: str = ""
    error_message: str = ""
    has_captcha_signal: bool = False
    has_auth_signal: bool = False
    has_timeout_signal: bool = False
    has_bot_block_signal: bool = False

    # Task Validity Layer
    failure_root: str = ""
    task_validity_code: str = ""
    task_validity_reason: str = ""
    task_validity_confidence: float = 0.0
    url_http_status: int = -1
    url_probe_note: str = ""

    # Raw data (JSON only, excluded from CSV)
    agent_thoughts: list[str] = field(default_factory=list)
    raw_traceback: str = ""

    def set_error(self, code: str, message: str = "") -> None:
        entry = ERROR_TAXONOMY.get(code, {"label": code, "hint": ""})
        self.error_code = code
        self.error_label = entry["label"]
        self.error_hint = entry["hint"]
        self.error_message = message[:500]

    def to_csv_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("agent_thoughts", None)
        d.pop("raw_traceback", None)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────
_LOCAL_MODEL_DEFAULT = "./models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"


class WebBenchEvaluator:
    def __init__(
        self,
        csv_path: str,
        records_per_category: int = 5,              # Changed default to 5
        categories: list[str] | None = None,
        llm_provider: str = "local",                # NEW: "gemini" or "local"
        llm_base_url: str = "http://localhost:8000/v1",
        llm_api_key: str = "dummy-key",
        llm_model: str = _LOCAL_MODEL_DEFAULT,
        llm_temperature: float = 0.0,
        llm_timeout: int = 1200,
        step_timeout: int = 1200,
        max_steps: int = 5,
        headless: bool = False,
        disable_security: bool = True,
        max_retries: int = 2,
        output_dir: str = "./results",
        run_label: str = "",
        validity_classifier_api_key: str = "",
        validity_classifier_model: str = "claude-haiku-4-5-20251001",
        max_dom_chars: int = MAX_DOM_CHARS,
    ) -> None:
        self.csv_path = csv_path
        self.records_per_category = records_per_category
        self.categories = categories or ["READ", "CREATE", "DELETE", "UPDATE", "FILE_MANIPULATION"]
        self.llm_provider = llm_provider
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_timeout = llm_timeout
        self.step_timeout = step_timeout
        self.max_steps = max_steps
        self.headless = headless
        self.disable_security = disable_security
        self.max_retries = max_retries
        self.output_dir = Path(output_dir)
        self.run_label = run_label or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.validity_classifier_api_key = validity_classifier_api_key
        self.validity_classifier_model = validity_classifier_model
        self.max_dom_chars = max_dom_chars

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[TaskResult] = []

        self.stats: dict[str, dict[str, int]] = {
            cat: {"success": 0, "failed": 0, "timeout": 0}
            for cat in self.categories
        }

        # FIX-7: Warn at startup if step_timeout is likely too low (only relevant for local LLMs)
        if self.llm_provider == "local":
            _estimated_min = int((1500 / 3.8) + (150 / 1.6))  # ≈ 489s
            if self.step_timeout < _estimated_min:
                logger.warning(
                    f"⚠️  FIX-7: step_timeout={self.step_timeout}s is likely too short for local LLM.\n"
                    f"   Estimated minimum for 32B model: {_estimated_min}s\n"
                    f"   Recommended: --step-timeout {_estimated_min + 120}"
                )
            else:
                logger.info(f"   step_timeout={self.step_timeout}s  ✓  (min estimated: {_estimated_min}s)")
        else:
            logger.info(f"   step_timeout={self.step_timeout}s  ✓  (Gemini is fast; 120s is usually enough)")

    # ── Dataset ────────────────────────────────────────────────────────────────
    def load_dataset(self) -> pd.DataFrame:
        path = next(
            (p for p in [
                Path(self.csv_path),
                Path(__file__).resolve().parent / self.csv_path,
                Path.cwd() / self.csv_path,
            ] if p.exists()),
            None,
        )
        if path is None:
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        logger.info(f"Loading dataset: {path}")
        df = pd.read_csv(path)
        df = df.rename(columns={"Starting URL": "website", "Task": "instruction"})
        logger.info(f"  {len(df)} total rows | columns: {list(df.columns)}")
        return df

    def stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Category" not in df.columns:
            raise ValueError("CSV must have a 'Category' column")
        available = set(df["Category"].dropna().unique())
        missing = set(self.categories) - available
        if missing:
            logger.warning(f"Categories not in CSV (will skip): {missing}")
        valid_cats = [c for c in self.categories if c in available]

        parts = []
        for cat in valid_cats:
            sub = df[df["Category"] == cat]
            n = min(self.records_per_category, len(sub))
            parts.append(sub.sample(n=n, random_state=42, replace=False))
            logger.info(f"  {cat}: sampled {n}/{len(sub)}")

        sample = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
        logger.info(f"Total sample: {len(sample)} tasks")

        sample_path = self.output_dir / f"sample_{self.run_label}.csv"
        sample.to_csv(sample_path, index=False)
        logger.info(f"Sample saved → {sample_path}")
        return sample

    # ── LLM health check (local only) ─────────────────────────────────────────
    def _health_check_llm(self) -> bool:
        if self.llm_provider == "gemini":
            logger.info("  Gemini provider — skipping local endpoint health check")
            return True
        try:
            base = self.llm_base_url.rstrip("/v1").rstrip("/")
            url = f"{base}/v1/models"
            with urlopen(url, timeout=10) as r:
                return r.status == 200
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False

    # ── FIX-3: Extract thoughts with <think> stripping ─────────────────────────
    def _extract_thoughts(self, history: Any) -> tuple[list[str], int]:
        thoughts_parts: list[str] = []
        llm_calls = 0
        if history and history.history:
            for step in history.history:
                if hasattr(step, "model_output") and step.model_output:
                    raw_text = str(step.model_output)
                    clean_text = strip_think_tags(raw_text)
                    thoughts_parts.append(clean_text)
                    llm_calls += 1
        return thoughts_parts, llm_calls

    # ── Main run ───────────────────────────────────────────────────────────────
    async def run(self) -> None:
        df = self.load_dataset()
        sample = self.stratified_sample(df)

        logger.info("\nBuilding LLM...")
        # FIX-9: build LLM once; reuse across all tasks
        # We need args-like object — use a simple namespace built from self attrs
        _fake_args = argparse.Namespace(
            llm_provider=self.llm_provider,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            llm_model=self.llm_model,
            llm_temperature=self.llm_temperature,
        )
        llm = build_llm(_fake_args)

        logger.info("\nRunning LLM health check...")
        if not self._health_check_llm():
            logger.warning("  LLM endpoint unreachable — tasks will likely fail with LLM_Timeout")
        else:
            logger.info("  LLM endpoint OK")

        csv_path = self.output_dir / f"results_{self.run_label}.csv"
        json_path = self.output_dir / f"results_{self.run_label}.json"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_headers: list[str] = []
        writer: csv.DictWriter | None = None

        total = len(sample)
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING EVALUATION  |  {total} tasks  |  run={self.run_label}")
        logger.info(f"  provider={self.llm_provider}  |  step_timeout={self.step_timeout}s  |  max_steps={self.max_steps}")
        logger.info(f"  Estimated max task time: {self.step_timeout * self.max_steps // 60} min/task")
        logger.info(f"{'='*70}\n")

        for idx, (_, row) in enumerate(sample.iterrows(), 1):
            result = await self._run_with_retry(idx, total, row, llm)
            self.results.append(result)

            cat = result.category
            if cat not in self.stats:
                self.stats[cat] = {"success": 0, "failed": 0, "timeout": 0}
            if result.status == "SUCCESS":
                self.stats[cat]["success"] += 1
            elif result.status == "TIMEOUT":
                self.stats[cat]["timeout"] += 1
            else:
                self.stats[cat]["failed"] += 1

            row_dict = result.to_csv_row()
            if writer is None:
                csv_headers = list(row_dict.keys())
                writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
                writer.writeheader()
            writer.writerow(row_dict)
            csv_file.flush()

            if idx % 5 == 0 or idx == total:
                done = sum(s["success"] for s in self.stats.values())
                logger.info(f"  Progress {idx}/{total} | cumulative success: {done}")

        csv_file.close()

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(
                {
                    "run_label": self.run_label,
                    "config": self._config_dict(),
                    "stats": self.stats,
                    "tasks": [asdict(r) for r in self.results],
                },
                jf,
                indent=2,
                ensure_ascii=False,
            )

        self._print_summary(csv_path, json_path)

    # ── Retry wrapper ──────────────────────────────────────────────────────────
    async def _run_with_retry(
        self, idx: int, total: int, row: pd.Series, llm: Any
    ) -> TaskResult:
        result = TaskResult()
        for attempt in range(1, self.max_retries + 2):
            result = await self._run_single(idx, total, row, llm)
            result.retry_count = attempt - 1
            if result.status == "SUCCESS":
                break
            if result.error_code in ("BrowserInitFailed", "System_SilentCrash") and attempt <= self.max_retries:
                wait = 2 ** attempt
                logger.info(f"    Retry {attempt}/{self.max_retries} in {wait}s...")
                await asyncio.sleep(wait)
            else:
                break
        return result

    # ── Single task ────────────────────────────────────────────────────────────
    async def _run_single(
        self, idx: int, total: int, row: pd.Series, llm: Any
    ) -> TaskResult:
        category = str(row.get("Category", "Unknown"))
        target_url = str(row.get("website", ""))
        instruction = str(row.get("instruction", ""))
        domain = _extract_domain(target_url)

        result = TaskResult(
            task_id=f"task_{idx:04d}",
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
            url=target_url,
            url_domain=domain,
            instruction=instruction[:300],
            instruction_length=len(instruction),
            steps_budget=self.max_steps,
            llm_model=self.llm_model,
            llm_provider=self.llm_provider,
        )

        prompt = _build_prompt(target_url, instruction)
        logger.info(f"[{idx:04d}/{total}] {category:18s} | {domain}")

        # Stage 1: URL probe
        http_status, probe_note = probe_url(target_url)
        result.url_http_status = http_status
        result.url_probe_note = probe_note

        if http_status == 0:
            logger.warning("   URL UNREACHABLE → TASK_INVALID [URL_Dead]")
            result.status = "FAILED"
            result.failure_root = "TASK_INVALID"
            result.task_validity_code = "URL_Dead"
            result.task_validity_reason = probe_note or "URL returned no response"
            result.task_validity_confidence = 1.0
            result.set_error("System_SilentCrash", f"URL unreachable: {probe_note}")
            return result
        elif http_status == -1:
            logger.warning("   URL REDIRECTED → TASK_INVALID [URL_Redirected]")
            result.status = "FAILED"
            result.failure_root = "TASK_INVALID"
            result.task_validity_code = "URL_Redirected"
            result.task_validity_reason = probe_note
            result.task_validity_confidence = 0.9
            result.set_error("Data_StaleLayout", f"URL redirected: {probe_note}")
            return result
        elif http_status >= 400:
            logger.warning(f"   URL HTTP {http_status} → TASK_INVALID [URL_Dead]")
            result.status = "FAILED"
            result.failure_root = "TASK_INVALID"
            result.task_validity_code = "URL_Dead"
            result.task_validity_reason = f"HTTP {http_status}"
            result.task_validity_confidence = 1.0
            result.set_error(
                "Security_BotBlocked" if http_status == 403 else "System_SilentCrash",
                f"HTTP {http_status}",
            )
            return result

        logger.info(f"   URL probe OK (HTTP {http_status})")

        start = time.time()
        browser: Browser | None = None

        try:
            browser = Browser(headless=self.headless, disable_security=self.disable_security)
        except Exception as e:
            result.latency_seconds = round(time.time() - start, 2)
            result.status = "FAILED"
            result.set_error("BrowserInitFailed", str(e))
            result.raw_traceback = traceback.format_exc()
            logger.error(f"   Browser init failed: {e}")
            return result

        try:
            # Gemini is fast — use smaller step_timeout for it, else use configured value
            effective_step_timeout = 120 if self.llm_provider == "gemini" else self.step_timeout

            agent = Agent(
                task=prompt,
                llm=llm,
                browser=browser,
                llm_timeout=self.llm_timeout if self.llm_provider != "gemini" else 120,
                step_timeout=effective_step_timeout,
                include_attributes=[
                    "text", "role", "href", "placeholder",
                ],
                max_actions_per_step=1,
            )

            outer_timeout = effective_step_timeout * self.max_steps + 60
            logger.info(f"   Running agent (provider={self.llm_provider}, outer timeout={outer_timeout}s)...")

            history = await asyncio.wait_for(
                agent.run(max_steps=self.max_steps),
                timeout=outer_timeout,
            )

            result.latency_seconds = round(time.time() - start, 2)

            steps = len(history.history) if history and history.history else 0
            result.steps_taken = steps
            result.steps_utilization_pct = round(steps / self.max_steps * 100, 1) if self.max_steps else 0

            if steps > 0:
                avg_step = result.latency_seconds / steps
                result.avg_step_latency_seconds = round(avg_step, 1)
                result.max_step_latency_seconds = round(avg_step, 1)
                logger.info(
                    f"   Steps: {steps}  |  avg step latency ≈ {avg_step:.0f}s  "
                    f"|  total: {result.latency_seconds:.0f}s"
                )

            thoughts_parts, llm_calls = self._extract_thoughts(history)
            result.agent_thoughts = thoughts_parts
            result.llm_calls = llm_calls
            thoughts_concat = " ".join(thoughts_parts).lower()
            result.total_thought_chars = len(thoughts_concat)

            if hasattr(history, "final_result") and history.final_result():
                final = str(history.final_result())
                result.final_answer = final[:1000]
                result.final_answer_length = len(final)

            result.has_captcha_signal = any(
                kw in thoughts_concat for kw in ("captcha", "cloudflare", "human verification", "i am not a robot")
            )
            result.has_auth_signal = any(
                kw in thoughts_concat for kw in ("login", "sign in", "password", "authenticate")
            )
            result.has_timeout_signal = steps == 0
            result.has_bot_block_signal = any(
                kw in thoughts_concat for kw in ("403", "access denied", "err_connection_reset", "bot detected")
            )

            if not history or history.has_errors() or steps == 0:
                result.status = "FAILED"

                last_error = "Unknown internal failure"
                if history and history.errors():
                    last_error = str(history.errors()[-1])
                elif steps == 0:
                    last_error = "Agent took 0 steps"

                error_lower = last_error.lower()
                final_lower = result.final_answer.lower()

                if result.has_bot_block_signal or "net::err_connection_reset" in error_lower or "403" in error_lower:
                    result.set_error("Security_BotBlocked", last_error)
                elif result.has_captcha_signal:
                    result.set_error("Security_CaptchaFailed", last_error)
                elif result.has_auth_signal and ("failed" in final_lower or "unable" in final_lower):
                    result.set_error("Auth_CredentialInjectionFailed", last_error)
                elif any(kw in thoughts_concat for kw in ("0 matches", "could not locate", "unable to find", "element not found")):
                    result.set_error("Data_StaleLayout", last_error)
                elif "timeout" in error_lower or "disconnected" in error_lower or steps == 0:
                    result.set_error("System_SilentCrash", last_error)
                else:
                    result.set_error("Logic_AgentLoopOrHallucination", last_error)

                fw_limit = classify_framework_limit(instruction, thoughts_concat)
                if fw_limit:
                    result.failure_root = "FRAMEWORK_LIMIT"
                    result.task_validity_code = fw_limit
                    result.task_validity_reason = VALIDITY_CODES.get(fw_limit, "")
                    result.task_validity_confidence = 0.95
                    logger.info(f"   Validity: FRAMEWORK_LIMIT [{fw_limit}] (keyword match)")
                else:
                    v_code, v_conf, v_reason = await classify_task_validity(
                        instruction=instruction,
                        thoughts=thoughts_concat,
                        api_key=self.validity_classifier_api_key,
                        model=self.validity_classifier_model,
                    )
                    result.task_validity_code = v_code
                    result.task_validity_confidence = v_conf
                    result.task_validity_reason = v_reason
                    result.failure_root = VALIDITY_TO_ROOT.get(v_code, "AGENT_FAILURE")
                    logger.info(f"   Validity: {result.failure_root} [{v_code}] conf={v_conf:.2f}")

                logger.warning(
                    f"   FAILED [{result.error_code}] root={result.failure_root} "
                    f"| {result.error_message[:80]}"
                )

            else:
                result.status = "SUCCESS"
                result.failure_root = ""
                result.task_validity_code = "VALID"
                result.task_validity_confidence = 1.0
                logger.info(f"   SUCCESS in {result.latency_seconds}s | {steps} steps")

        except asyncio.TimeoutError:
            result.latency_seconds = round(time.time() - start, 2)
            result.status = "TIMEOUT"
            result.set_error(
                "ExecutionTimeout",
                f"Hard outer timeout exceeded. LLM may be hung or browser stalled.",
            )
            logger.warning(f"   HARD TIMEOUT after {result.latency_seconds:.0f}s")

        except Exception as e:
            result.latency_seconds = round(time.time() - start, 2)
            result.status = "FAILED"
            err_str = str(e)
            result.raw_traceback = traceback.format_exc()

            if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                result.set_error("LLM_Timeout", err_str)
            else:
                result.set_error("UnknownError", err_str)
            logger.error(f"   EXCEPTION [{result.error_code}] {err_str[:120]}")

        finally:
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass

        return result

    # ── Summary ────────────────────────────────────────────────────────────────
    def _print_summary(self, csv_path: Path, json_path: Path) -> None:
        total = len(self.results)
        successes = sum(1 for r in self.results if r.status == "SUCCESS")
        timeouts = sum(1 for r in self.results if r.status == "TIMEOUT")
        failures = total - successes - timeouts
        rate = round(successes / total * 100, 1) if total else 0
        avg_latency = round(sum(r.latency_seconds for r in self.results) / total, 1) if total else 0
        avg_steps = round(sum(r.steps_taken for r in self.results) / total, 1) if total else 0

        error_counts: dict[str, int] = {}
        for r in self.results:
            if r.error_code:
                error_counts[r.error_code] = error_counts.get(r.error_code, 0) + 1

        root_counts: dict[str, int] = {"TASK_INVALID": 0, "FRAMEWORK_LIMIT": 0, "AGENT_FAILURE": 0}
        validity_code_counts: dict[str, int] = {}
        for r in self.results:
            if r.failure_root:
                root_counts[r.failure_root] = root_counts.get(r.failure_root, 0) + 1
            if r.task_validity_code and r.task_validity_code != "VALID":
                validity_code_counts[r.task_validity_code] = validity_code_counts.get(r.task_validity_code, 0) + 1

        stale_task_count = root_counts.get("TASK_INVALID", 0)
        framework_limit_count = root_counts.get("FRAMEWORK_LIMIT", 0)
        true_agent_failures = root_counts.get("AGENT_FAILURE", 0)
        valid_tasks = total - stale_task_count
        true_rate = round(successes / valid_tasks * 100, 1) if valid_tasks > 0 else 0

        step_latencies = [r.avg_step_latency_seconds for r in self.results if r.avg_step_latency_seconds > 0]
        avg_step_lat = round(sum(step_latencies) / len(step_latencies), 1) if step_latencies else 0

        sep = "=" * 70
        logger.info(f"\n{sep}")
        logger.info("EVALUATION COMPLETE")
        logger.info(sep)
        logger.info(f"  Run Label       : {self.run_label}")
        logger.info(f"  LLM Provider    : {self.llm_provider}")
        logger.info(f"  Total Tasks     : {total}")
        logger.info(f"  Reported Rate   : {rate}%  ({successes} success / {failures} fail / {timeouts} timeout)")
        logger.info("")
        logger.info("  ── Failure Root Analysis ──")
        logger.info(f"  Stale/Invalid Tasks  : {stale_task_count}")
        logger.info(f"  Framework Limits     : {framework_limit_count}")
        logger.info(f"  True Agent Failures  : {true_agent_failures}")
        logger.info(f"  Adjusted Success Rate: {true_rate}%  (on {valid_tasks} valid tasks only)")
        logger.info("")
        if validity_code_counts:
            logger.info("  Validity Sub-codes:")
            for code, count in sorted(validity_code_counts.items(), key=lambda x: -x[1]):
                desc = VALIDITY_CODES.get(code, "")
                logger.info(f"    {code:35s}  x{count}  —  {desc}")
        logger.info("")
        logger.info(f"  Avg Task Latency    : {avg_latency}s")
        logger.info(f"  Avg Steps/Task      : {avg_steps}")
        logger.info(f"  Avg Step Latency    : {avg_step_lat}s")
        logger.info(f"  Recommended timeout : {int(avg_step_lat * 1.5) if avg_step_lat else 'n/a'}s  (1.5× avg observed)")
        logger.info("")
        logger.info("  Per-Category:")
        for cat, s in self.stats.items():
            cat_total = s["success"] + s["failed"] + s["timeout"]
            cat_rate = round(s["success"] / cat_total * 100, 1) if cat_total else 0
            logger.info(f"    {cat:22s}  {cat_rate:5.1f}%  (s={s['success']} f={s['failed']} t={s['timeout']})")
        logger.info("")
        if error_counts:
            logger.info("  Error Code Breakdown:")
            for code, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                hint = ERROR_TAXONOMY.get(code, {}).get("hint", "")
                logger.info(f"    {code:40s}  x{count}  →  {hint}")
        logger.info("")
        logger.info(f"  Results CSV     : {csv_path}")
        logger.info(f"  Results JSON    : {json_path}")
        logger.info(sep)

    def _config_dict(self) -> dict[str, Any]:
        return {
            "csv_path": self.csv_path,
            "records_per_category": self.records_per_category,
            "categories": self.categories,
            "llm_provider": self.llm_provider,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_timeout": self.llm_timeout,
            "step_timeout": self.step_timeout,
            "max_steps": self.max_steps,
            "headless": self.headless,
            "disable_security": self.disable_security,
            "max_retries": self.max_retries,
            "max_dom_chars": self.max_dom_chars,
            "validity_classifier_model": self.validity_classifier_model,
            "validity_classifier_enabled": bool(self.validity_classifier_api_key),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


def _build_prompt(target_url: str, instruction: str) -> str:
    return (
        f"Go to {target_url} and complete this task: {instruction}\n\n"
        "SYSTEM CRITICAL INSTRUCTION: You are an autonomous headless browser API. "
        "You MUST use the provided tool/function to output your actions. "
        "DO NOT output any conversational text, greetings, or explanations. "
        "DO NOT output <think> tags or reasoning blocks. "
        "Your entire response must be strictly the requested JSON schema. "
        "Keep your JSON action as short as possible — no extra keys, no comments.\n\n"
        "CRITICAL COMPLETION RULE: As soon as you have successfully extracted or confirmed "
        "the information requested in the task, you MUST immediately call the 'done' action "
        "with success=true and include the extracted information in your final answer. "
        "Do NOT continue navigating after you already have the answer."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WebBench browser-use evaluator — supports Gemini and local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Gemini (needs GOOGLE_API_KEY in .env):
  python eval_tasks_fixed.py --llm-provider gemini --csv webbench_ready.csv

  # Use Gemini with a specific model:
  python eval_tasks_fixed.py --llm-provider gemini --llm-model gemini-2.0-flash-exp --csv webbench_ready.csv

  # Use local DeepSeek (original behaviour):
  python eval_tasks_fixed.py --llm-provider local \\
      --llm-base-url http://103.108.136.157:8000/v1 \\
      --llm-model ./models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf \\
      --step-timeout 1200 --max-steps 5 --csv webbench_ready.csv
        """,
    )
    p.add_argument("--csv", default="webbench_ready.csv")
    p.add_argument("--records-per-category", type=int, default=5,    # Changed to 5
                   help="Tasks per category (default: 5)")
    p.add_argument(
        "--categories", nargs="+",
        default=["READ", "CREATE", "DELETE", "UPDATE", "FILE_MANIPULATION"],
    )

    # ── FIX-9: LLM provider selection ──────────────────────────────────────────
    p.add_argument(
        "--llm-provider", default="local", choices=["gemini", "local"],
        help=(
            "LLM backend to use.\n"
            "  gemini → ChatGoogleGenerativeAI (needs GOOGLE_API_KEY in .env)\n"
            "           Install: pip install langchain-google-genai\n"
            "  local  → OpenAI-compat endpoint (vLLM / llama.cpp)\n"
            "           Requires --llm-base-url and --llm-model"
        ),
    )

    # Local-only options (ignored when --llm-provider gemini)
    p.add_argument("--llm-base-url", default="http://103.108.136.157:8000/v1",
                   help="Base URL for local OpenAI-compat server (ignored for Gemini)")
    p.add_argument("--llm-api-key", default="dummy-key",
                   help="API key for local server (ignored for Gemini)")
    p.add_argument("--llm-model", default=_LOCAL_MODEL_DEFAULT,
                   help=(
                       "Model name/path.\n"
                       "  For Gemini: e.g. gemini-2.5-flash-preview-05-20 (default when --llm-provider gemini)\n"
                       "  For local:  path/model name for vLLM"
                   ))

    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-timeout", type=int, default=1200,
                   help="Per-LLM-call timeout in seconds (auto-reduced to 120s for Gemini)")
    p.add_argument("--step-timeout", type=int, default=1200,
                   help="Per-step timeout in seconds (auto-reduced to 120s for Gemini)")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--max-dom-chars", type=int, default=MAX_DOM_CHARS)
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--no-disable-security", dest="disable_security", action="store_false", default=True)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--run-label", default="")
    p.add_argument("--validity-classifier-api-key", default="")
    p.add_argument("--validity-classifier-model", default="claude-haiku-4-5-20251001")
    return p.parse_args()


async def main() -> None:
    args = _parse_args()
    evaluator = WebBenchEvaluator(
        csv_path=args.csv,
        records_per_category=args.records_per_category,
        categories=args.categories,
        llm_provider=args.llm_provider,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_timeout=args.llm_timeout,
        step_timeout=args.step_timeout,
        max_steps=args.max_steps,
        headless=args.headless,
        disable_security=args.disable_security,
        max_retries=args.max_retries,
        output_dir=args.output_dir,
        run_label=args.run_label,
        validity_classifier_api_key=args.validity_classifier_api_key,
        validity_classifier_model=args.validity_classifier_model,
        max_dom_chars=args.max_dom_chars,
    )
    await evaluator.run()


if __name__ == "__main__":
    asyncio.run(main())
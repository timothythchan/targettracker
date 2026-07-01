"""
llm_extractor.py — LLM-based semantic target extractor for EarningsLens.

This is the NB03-v2 rewrite. It replaces the older char-count chunker and
flat per-component loop with:

  * Token-aware chunking (_chunk_by_tokens) using tiktoken + sentence boundaries.
  * Component aggregation in extract_transcript: Presentation/Question/Answer
    text is concatenated per type before chunking, so a typical Q&A-heavy
    transcript drops from ~84 calls to ~3-6.
  * Resumable JSONL writer (extract_corpus_to_jsonl) — one JSON line per
    finished transcript; rerunning skips already-written transcript_ids.
  * Proactive RPM + TPM rate limiter (TokenBucketLimiter) so we can run
    max_concurrent=20 without 429 storms.
  * Typed telemetry: total_requests, total_input_tokens, total_output_tokens,
    total_tokens_used, http_failures, empty_responses, schema_invalid_drops,
    retries.
  * Auth failures (openai.AuthenticationError) propagate immediately so
    callers don't waste a corpus run on bad credentials.

Public surface used by NB03 cell 31 / cell 41 / NB04:
  - LLMTargetExtractor(backend, model, api_key, max_concurrent, temperature,
                       max_input_tokens_per_chunk, request_timeout_s,
                       max_retries, rpm_cap, ...)
  - extractor.count_tokens(text) -> int
  - extractor._chunk_by_tokens(text) -> List[str]
  - extractor.extract_targets(text, component_type) -> List[Dict]
  - extractor.extract_transcript(transcript) -> List[Dict]
  - extractor.extract_corpus(transcripts, max_concurrent) -> Dict[tid, List[Dict]]
  - extractor.extract_corpus_to_jsonl(transcripts, out_path,
                                      max_concurrent, progress_cb) -> None
  - extractor.telemetry -> Dict[str, int]   (8 keys, see above)
  - extractor.reset_telemetry() -> None
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple, Union

import httpx

try:
    import openai
    from openai import AsyncOpenAI, AuthenticationError
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENAI_AVAILABLE = False

    # Distinct stub class so ``except AuthenticationError`` blocks remain
    # *narrow* when the openai package isn't installed. If we aliased to
    # ``Exception`` here, those except blocks would silently swallow every
    # exception type — including bugs we want to surface.
    class AuthenticationError(Exception):  # type: ignore[no-redef]
        """Stub used when the openai package is not installed."""

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False

from .prompts import (
    build_extraction_prompt,
    SYSTEM_PROMPT,
    canonicalize_metric,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_LOCAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"

# Approximate characters-per-token ratio when tiktoken is unavailable.
_CHARS_PER_TOKEN = 4

# Regex fallback for extracting JSON from polluted LLM output.
_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Reserved tokens for SYSTEM_PROMPT + extraction-prompt skeleton + completion.
# Subtracted from max_input_tokens_per_chunk before sentence packing.
_PROMPT_OVERHEAD_TOKENS = 1500


# ---------------------------------------------------------------------------
# Token counting (module-level so it can be cached / reused)
# ---------------------------------------------------------------------------

_ENCODER_CACHE: Dict[str, Any] = {}


def _get_encoder(model: str) -> Optional[Any]:
    if not _TIKTOKEN_AVAILABLE:
        return None
    enc = _ENCODER_CACHE.get(model)
    if enc is not None:
        return enc
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
    _ENCODER_CACHE[model] = enc
    return enc


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in *text* using tiktoken when available, else char heuristic."""
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff capped at *cap* seconds."""
    return min(base * (2 ** attempt), cap)


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (RPM + TPM aware)
# ---------------------------------------------------------------------------

class TokenBucketLimiter:
    """
    Async token-bucket limiter that enforces both a request-per-minute (RPM)
    cap and a token-per-minute (TPM) cap. Calling ``acquire(tokens)`` blocks
    until both buckets have capacity, then debits them.

    This is proactive: it prevents 429 storms instead of waiting for them to
    happen and reactively backing off, which is critical when running 20-50
    concurrent extractions across a 10k+ transcript corpus.
    """

    def __init__(
        self,
        rpm: int = 500,
        tpm: int = 200_000,
        burst_rpm: Optional[int] = None,
        burst_tpm: Optional[int] = None,
    ) -> None:
        if rpm <= 0 or tpm <= 0:
            raise ValueError("rpm and tpm must be positive")

        self._req_rate = rpm / 60.0           # requests per second
        self._tok_rate = tpm / 60.0           # tokens per second
        self._req_capacity = float(burst_rpm if burst_rpm is not None else rpm)
        self._tok_capacity = float(burst_tpm if burst_tpm is not None else tpm)

        self._req_tokens = self._req_capacity
        self._tok_tokens = self._tok_capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._req_tokens = min(
            self._req_capacity, self._req_tokens + elapsed * self._req_rate
        )
        self._tok_tokens = min(
            self._tok_capacity, self._tok_tokens + elapsed * self._tok_rate
        )
        self._last_refill = now

    async def acquire(self, tokens: int) -> None:
        """Block until 1 request slot AND ``tokens`` token slots are available."""
        # Cap a single huge prompt at the bucket size so we never deadlock.
        tokens = max(1, min(int(tokens), int(self._tok_capacity)))

        while True:
            async with self._lock:
                self._refill()
                if self._req_tokens >= 1.0 and self._tok_tokens >= tokens:
                    self._req_tokens -= 1.0
                    self._tok_tokens -= tokens
                    return
                req_wait = (1.0 - self._req_tokens) / self._req_rate if self._req_rate > 0 else 0.0
                tok_wait = (tokens - self._tok_tokens) / self._tok_rate if self._tok_rate > 0 else 0.0
                wait = max(req_wait, tok_wait, 0.0) + 0.005
            await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMTargetExtractor:
    """
    LLM-powered performance-target extractor for earnings call transcripts.

    Parameters
    ----------
    backend : {"openai", "local"}
        Inference backend. ``"local"`` targets a vLLM/Ollama OpenAI-compatible
        REST server.
    model : str, optional
        Model identifier. Defaults to ``gpt-4o-mini`` for OpenAI.
    api_key : str, optional
        OpenAI API key (falls back to ``OPENAI_API_KEY`` env var).
    base_url : str, optional
        Base URL for the local backend.
    max_concurrent : int
        Default semaphore size for ``extract_corpus`` /
        ``extract_corpus_to_jsonl``. Each call can override.
    temperature : float
        Sampling temperature; ``0.0`` is the deterministic default.
    max_input_tokens_per_chunk : int
        Token budget applied by ``_chunk_by_tokens`` before splitting at
        sentence boundaries. Default 12 000 — well under gpt-4o-mini's
        128k window after accounting for system prompt and completion.
    request_timeout_s : float
        Per-request HTTP timeout in seconds.
    max_retries : int
        Maximum retry attempts on transient API errors before returning ``[]``.
    rpm_cap : int or None
        Requests-per-minute cap for the token-bucket limiter. ``None`` or
        0 disables the limiter (use only for local backends).
    tpm_cap : int
        Tokens-per-minute cap. Defaults to a generous 2 000 000 which is
        roughly the OpenAI tier-2 ceiling for gpt-4o-mini.
    rate_limiter : TokenBucketLimiter, optional
        Pre-built limiter (overrides ``rpm_cap`` / ``tpm_cap``).
    """

    def __init__(
        self,
        backend: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        max_concurrent: int = 20,
        temperature: float = 0.0,
        max_input_tokens_per_chunk: int = 12_000,
        max_completion_tokens: int = 2048,
        request_timeout_s: float = 90.0,
        max_retries: int = 3,
        rpm_cap: Optional[int] = 4_500,
        tpm_cap: int = 2_000_000,
        rate_limiter: Optional[TokenBucketLimiter] = None,
    ) -> None:
        if backend not in ("openai", "local"):
            raise ValueError(f"backend must be 'openai' or 'local', got {backend!r}")

        self.backend = backend
        self.model = model or (
            DEFAULT_OPENAI_MODEL if backend == "openai" else DEFAULT_LOCAL_MODEL
        )
        self.max_concurrent = int(max(1, max_concurrent))
        self.temperature = float(temperature)
        self.max_input_tokens_per_chunk = int(max(512, max_input_tokens_per_chunk))
        self.max_completion_tokens = int(max(64, max_completion_tokens))

        # Warn early if the chunk window leaves almost no room for actual
        # content after the prompt overhead is subtracted. Below this
        # threshold ``_chunk_by_tokens`` will hard-split sentences mid-token,
        # which silently corrupts prompts.
        _budget_after_overhead = self.max_input_tokens_per_chunk - _PROMPT_OVERHEAD_TOKENS
        if _budget_after_overhead < 1024:
            logger.warning(
                "max_input_tokens_per_chunk=%d leaves only %d tokens after "
                "prompt overhead (%d). Sentences exceeding this budget will "
                "be hard-split mid-token, which can corrupt the prompt. "
                "Consider raising max_input_tokens_per_chunk to >= %d.",
                self.max_input_tokens_per_chunk,
                max(0, _budget_after_overhead),
                _PROMPT_OVERHEAD_TOKENS,
                _PROMPT_OVERHEAD_TOKENS + 2048,
            )
        self.request_timeout_s = float(request_timeout_s)
        self.max_retries = int(max(1, max_retries))

        # ── OpenAI client ─────────────────────────────────────────────────
        if backend == "openai":
            if not _OPENAI_AVAILABLE:
                raise ImportError(
                    "openai package is required for the 'openai' backend. "
                    "Install with: pip install openai"
                )
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            # base_url=None lets the SDK use its default (api.openai.com).
            # When set (e.g. Gemini's OpenAI-compatible endpoint), route there.
            self._client = AsyncOpenAI(
                api_key=resolved_key,
                base_url=base_url,
                timeout=self.request_timeout_s,
            )
            self._use_sdk_for_local = False
            self._is_gemini_compat = bool(base_url and "googleapis.com" in str(base_url))

        # ── Local backend (vLLM / Ollama) ─────────────────────────────────
        else:
            resolved_url = base_url or VLLM_DEFAULT_BASE_URL
            if _OPENAI_AVAILABLE:
                self._client = AsyncOpenAI(
                    api_key="not-needed",
                    base_url=resolved_url,
                    timeout=self.request_timeout_s,
                )
                self._use_sdk_for_local = True
            else:
                self._client = None
                self._http_base_url = resolved_url
                self._use_sdk_for_local = False
            self._is_gemini_compat = False

        # ── Rate limiter ──────────────────────────────────────────────────
        if rate_limiter is not None:
            self._limiter: Optional[TokenBucketLimiter] = rate_limiter
        elif backend == "openai" and rpm_cap and rpm_cap > 0:
            self._limiter = TokenBucketLimiter(rpm=int(rpm_cap), tpm=int(tpm_cap))
        else:
            # No limiter for local or when explicitly disabled.
            self._limiter = None

        # ── Telemetry counters ────────────────────────────────────────────
        self._total_requests: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_tokens_used: int = 0
        self._http_failures: int = 0
        self._empty_responses: int = 0
        self._schema_invalid_drops: int = 0
        self._retries: int = 0

    # ------------------------------------------------------------------
    # Public token utilities
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text* under this extractor's model."""
        return _count_tokens(text, self.model)

    # ------------------------------------------------------------------
    # Public extraction API
    # ------------------------------------------------------------------

    async def extract_targets(
        self, text: str, component_type: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Extract performance targets from a single text segment.

        Long inputs are split via ``_chunk_by_tokens`` and per-chunk results
        are merged with ``_dedup``.
        """
        if not text or not text.strip():
            return []

        chunks = self._chunk_by_tokens(text)
        logger.debug(
            "extract_targets | %d chunk(s) for %d chars (component=%d)",
            len(chunks), len(text), component_type,
        )

        per_chunk: List[List[Dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
            logger.debug("  chunk %d/%d (%d tokens)",
                         idx + 1, len(chunks), self.count_tokens(chunk))
            per_chunk.append(await self._call_llm_with_retry(chunk, component_type))

        merged = self._dedup(per_chunk)
        # Stamp component_type on every survivor for downstream joins.
        for t in merged:
            t.setdefault("component_type", component_type)
        return merged

    async def extract_transcript(
        self, transcript: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract targets for one transcript dict, aggregating components by
        type before chunking.

        Performance: a typical Q&A-heavy transcript has ~80 components; the
        aggregation step collapses these into <= 4 logical inputs (one per
        component_type), which the chunker then breaks into ~3-6 actual API
        calls. This is roughly a 15-20x reduction in calls/transcript.
        """
        components: List[Dict[str, Any]] = transcript.get("components", []) or []
        if not components:
            return []

        # Aggregate by component_type, preserving order within each bucket.
        bucket: Dict[int, List[str]] = {}
        for comp in components:
            text = (comp.get("text") or "").strip()
            if not text:
                continue
            ctype = int(comp.get("component_type", 0))
            bucket.setdefault(ctype, []).append(text)

        if not bucket:
            return []

        all_targets: List[Dict[str, Any]] = []
        for ctype, texts in bucket.items():
            agg_text = "\n\n".join(texts)
            try:
                targets = await self.extract_targets(agg_text, ctype)
            except AuthenticationError:
                # Bad creds — propagate so caller can halt the corpus.
                raise
            for t in targets:
                t.setdefault("component_type", ctype)
            all_targets.extend(targets)

        # Final transcript-level dedup so the same metric mentioned in both
        # presentation and Q&A doesn't double-count.
        return self._dedup([all_targets])

    async def extract_batch(
        self, transcript_components: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Backwards-compatible alias: aggregate a flat component list."""
        fake_transcript = {"components": transcript_components}
        return await self.extract_transcript(fake_transcript)

    async def extract_corpus(
        self,
        transcripts: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """In-memory corpus extraction (use ``extract_corpus_to_jsonl`` for
        long corpora — it is resumable and constant-memory)."""
        sem = asyncio.Semaphore(int(max_concurrent or self.max_concurrent))
        results: Dict[str, List[Dict[str, Any]]] = {}

        async def _one(tr: Dict[str, Any]) -> None:
            tid = str(tr.get("transcript_id", "unknown"))
            async with sem:
                t0 = time.monotonic()
                try:
                    targets = await self.extract_transcript(tr)
                except AuthenticationError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("extract_corpus | %s failed: %s", tid, exc)
                    targets = []
                results[tid] = targets
                logger.info(
                    "extract_corpus | %s | %d target(s) | %.2fs",
                    tid, len(targets), time.monotonic() - t0,
                )

        await asyncio.gather(*[_one(t) for t in transcripts])
        return results

    async def extract_corpus_to_jsonl(
        self,
        transcripts: List[Dict[str, Any]],
        out_path: Union[str, Path],
        max_concurrent: Optional[int] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Resumable corpus extraction → newline-delimited JSON.

        Each line written is a JSON object:
            {"transcript_id": "<id>",
             "company_id": "<id>",
             "quarter": "YYYYQn",
             "targets": [ {...}, ... ]}

        On rerun, transcripts whose ``transcript_id`` is already present in
        ``out_path`` are skipped — so a Colab disconnect mid-corpus loses at
        most the in-flight transcripts, not the completed ones.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read previously-finished IDs.
        already_done: set[str] = set()
        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        tid = rec.get("transcript_id")
                        if isinstance(tid, str):
                            already_done.add(tid)
                    except json.JSONDecodeError:
                        continue
        if already_done:
            logger.info(
                "extract_corpus_to_jsonl | %d transcript(s) already on disk, skipping",
                len(already_done),
            )

        todo = [t for t in transcripts if str(t.get("transcript_id")) not in already_done]
        total = len(transcripts)
        done_count = len(already_done)

        sem = asyncio.Semaphore(int(max_concurrent or self.max_concurrent))
        write_lock = asyncio.Lock()

        # Use a single append-mode file handle protected by an asyncio lock.
        # Flush after each line so a crash never leaves a half-written record.
        f = out_path.open("a", encoding="utf-8", buffering=1)

        async def _one(tr: Dict[str, Any]) -> None:
            nonlocal done_count
            tid = str(tr.get("transcript_id", "unknown"))
            async with sem:
                t0 = time.monotonic()
                try:
                    targets = await self.extract_transcript(tr)
                except AuthenticationError:
                    # Surface auth failures immediately.
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("extract_corpus_to_jsonl | %s failed: %s", tid, exc)
                    targets = []

                record = {
                    "transcript_id": tid,
                    "company_id": tr.get("company_id"),
                    "quarter": tr.get("quarter"),
                    "n_targets": len(targets),
                    "targets": targets,
                }
                async with write_lock:
                    f.write(json.dumps(record, default=str) + "\n")
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                    done_count += 1
                    if progress_cb is not None:
                        try:
                            progress_cb(done_count, total)
                        except Exception:  # noqa: BLE001
                            pass

                logger.info(
                    "extract_corpus_to_jsonl | %s | %d target(s) | %.2fs",
                    tid, len(targets), time.monotonic() - t0,
                )

        try:
            # Bounded gather: spawn tasks in batches so a 10k corpus doesn't
            # try to allocate 10k coroutines up front.
            BATCH = max(self.max_concurrent * 4, 64)
            for i in range(0, len(todo), BATCH):
                batch = todo[i : i + BATCH]
                await asyncio.gather(*[_one(t) for t in batch])
        finally:
            f.close()

    # ------------------------------------------------------------------
    # Internal: LLM call with retry + rate limiting
    # ------------------------------------------------------------------

    async def _call_llm_with_retry(
        self, text: str, component_type: int
    ) -> List[Dict[str, Any]]:
        prompt = build_extraction_prompt(text, component_type)
        last_exc: Optional[Exception] = None

        # Pre-flight token estimate for the limiter.
        estimated = (
            self.count_tokens(prompt)
            + self.count_tokens(SYSTEM_PROMPT)
            + self.max_completion_tokens
        )

        for attempt in range(self.max_retries):
            try:
                if self._limiter is not None:
                    await self._limiter.acquire(estimated)
                self._total_requests += 1
                raw_text, in_tok, out_tok = await self._send_to_llm(prompt)

                self._total_input_tokens += in_tok
                self._total_output_tokens += out_tok
                self._total_tokens_used += in_tok + out_tok

                if not raw_text or not raw_text.strip():
                    self._empty_responses += 1
                    return []

                parsed = self._parse_llm_response(raw_text)
                if parsed is None:
                    self._schema_invalid_drops += 1
                    return []
                return parsed

            except AuthenticationError as exc:
                # Unrecoverable — surface immediately.
                self._http_failures += 1
                logger.error("_call_llm | auth failure: %s", exc)
                raise

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._http_failures += 1
                if attempt < self.max_retries - 1:
                    self._retries += 1
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "_call_llm | attempt %d/%d failed (%s) — retrying in %.1fs",
                        attempt + 1, self.max_retries, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "_call_llm | all %d retries failed (%s)",
                        self.max_retries, exc,
                    )

        logger.error(
            "_call_llm | all %d retries failed (%s)",
            self.max_retries,
            last_exc,
        )
        return []

    async def _send_to_llm(self, prompt: str) -> Tuple[str, int, int]:
        """Dispatch *prompt* to the active backend.

        Returns
        -------
        (response_text, input_tokens, output_tokens)
        """
        is_gemini = getattr(self, "_is_gemini_compat", False)

        # For Gemini we append a hard JSON-only suffix to the user prompt to
        # suppress chain-of-thought leakage and degenerate looping. Smaller
        # Gemini variants (Flash Lite) tend to dump the Stage 1/2/3 reasoning
        # text instead of the final JSON without this nudge.
        user_content = prompt
        if is_gemini:
            user_content = (
                prompt
                + "\n\n---\nIMPORTANT: Output ONLY the final JSON object "
                  "described above (a single object with a \"targets\" key). "
                  "Do NOT print Stage 1, Stage 2, or Stage 3 reasoning. "
                  "Do NOT add prose or markdown fences. Begin your response "
                  "with the character '{' and nothing before it."
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # ── OpenAI SDK path (also used for local-with-SDK) ────────────────
        if self.backend == "openai" or self._use_sdk_for_local:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_completion_tokens,
            }
            # Both real OpenAI and Gemini's OpenAI-compatible endpoint accept
            # response_format={"type":"json_object"}; we send it for both so
            # Gemini stops emitting Stage 1/2/3 chain-of-thought text. If
            # Gemini ever 400s on this field for a specific model, we'll
            # special-case below.
            if self.backend == "openai":
                kwargs["response_format"] = {"type": "json_object"}

            response = await self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            usage = response.usage
            if usage is not None:
                in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            else:
                in_tok = self.count_tokens(SYSTEM_PROMPT) + self.count_tokens(prompt)
                out_tok = self.count_tokens(content)
            return content, in_tok, out_tok

        # ── Raw httpx path (local without SDK) ────────────────────────────
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_completion_tokens,
        }
        async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
            resp = await client.post(
                f"{self._http_base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {}) or {}
            in_tok = int(usage.get("prompt_tokens") or
                         self.count_tokens(SYSTEM_PROMPT) + self.count_tokens(prompt))
            out_tok = int(usage.get("completion_tokens") or self.count_tokens(content))
            return content, in_tok, out_tok

    # ------------------------------------------------------------------
    # Internal: response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(self, response_text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse the LLM's response into a list of target dicts.

        Returns ``None`` if the response cannot be coerced into a list of
        dicts (caller treats this as a schema-invalid drop). Returns ``[]``
        for an explicit empty target set.
        """
        if not response_text:
            return []

        cleaned = response_text.strip()

        # Strip markdown fences.
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        # 1. Direct parse.
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = None

        # 2. Regex array fallback.
        if parsed is None:
            m = _JSON_ARRAY_RE.search(cleaned)
            if m:
                try:
                    parsed = json.loads(m.group())
                except json.JSONDecodeError:
                    parsed = None

        # 3. Last-resort object scan.
        if parsed is None:
            objs: List[Dict[str, Any]] = []
            for m in _JSON_OBJ_RE.finditer(cleaned):
                try:
                    o = json.loads(m.group())
                    if isinstance(o, dict):
                        objs.append(o)
                except json.JSONDecodeError:
                    continue
            if objs:
                return [o for o in objs if self._looks_like_target(o)]
            logger.warning(
                "_parse_llm_response | could not parse JSON (len=%d), first 200: %r",
                len(response_text), response_text[:200],
            )
            return None

        # Coerce to list of dicts.
        if isinstance(parsed, list):
            return [d for d in parsed if isinstance(d, dict)]
        if isinstance(parsed, dict):
            # Preferred shape: {"targets": [...]}
            if "targets" in parsed and isinstance(parsed["targets"], list):
                return [d for d in parsed["targets"] if isinstance(d, dict)]
            # Generic: any value that is a list of dicts.
            for v in parsed.values():
                if isinstance(v, list):
                    list_of_dicts = [d for d in v if isinstance(d, dict)]
                    if list_of_dicts:
                        return list_of_dicts
            # Single-target dict.
            if self._looks_like_target(parsed):
                return [parsed]
        return None

    @staticmethod
    def _looks_like_target(obj: Dict[str, Any]) -> bool:
        """Light schema sniff so we don't surface random JSON dicts as targets."""
        keys = {"metric_name", "raw_text", "numerical_value",
                "trend_direction", "unit", "time_horizon", "confidence"}
        return any(k in obj for k in keys)

    # ------------------------------------------------------------------
    # Internal: token-aware chunking
    # ------------------------------------------------------------------

    def _chunk_by_tokens(self, text: str) -> List[str]:
        """
        Split *text* into chunks fitting within ``max_input_tokens_per_chunk``,
        breaking at sentence boundaries (``. `` / ``? `` / ``! ``).

        Always returns at least one chunk (even for empty input it returns
        ``[""]`` so callers can iterate uniformly).
        """
        if not text:
            return [""]

        budget = max(256, self.max_input_tokens_per_chunk - _PROMPT_OVERHEAD_TOKENS)

        if self.count_tokens(text) <= budget:
            return [text]

        sentences = _SENT_SPLIT_RE.split(text)
        chunks: List[str] = []
        cur: List[str] = []
        cur_tokens = 0

        for sent in sentences:
            if not sent:
                continue
            stoks = self.count_tokens(sent)
            # If a single sentence blows the budget, hard-split by tokens.
            if stoks > budget:
                if cur:
                    chunks.append(" ".join(cur))
                    cur, cur_tokens = [], 0
                chunks.extend(self._hard_split_by_tokens(sent, budget))
                continue
            if cur_tokens + stoks > budget and cur:
                chunks.append(" ".join(cur))
                cur, cur_tokens = [sent], stoks
            else:
                cur.append(sent)
                cur_tokens += stoks

        if cur:
            chunks.append(" ".join(cur))

        return chunks if chunks else [text]

    def _hard_split_by_tokens(self, text: str, budget: int) -> List[str]:
        """Last-resort character-level split for sentences exceeding budget."""
        enc = _get_encoder(self.model)
        if enc is None:
            # Fall back to a char-budget proxy.
            char_budget = budget * _CHARS_PER_TOKEN
            return [text[i : i + char_budget] for i in range(0, len(text), char_budget)]
        ids = enc.encode(text)
        out: List[str] = []
        for i in range(0, len(ids), budget):
            out.append(enc.decode(ids[i : i + budget]))
        return out

    # ------------------------------------------------------------------
    # Internal: deduplication / merging
    # ------------------------------------------------------------------

    _CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}

    def _dedup(
        self, chunk_results: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Merge per-chunk target lists, deduplicating on canonical metric name.

        Each surviving target is annotated with ``canonical_name`` so NB03
        Section 10's INNER-join with the spaCy panel works without rerunning
        the canonicaliser downstream.
        """
        seen: Dict[str, Dict[str, Any]] = {}
        for chunk in chunk_results:
            for target in chunk:
                if not isinstance(target, dict):
                    continue
                metric_name = str(target.get("metric_name") or "").strip()
                canon = canonicalize_metric(metric_name)
                if not canon:
                    continue
                # Work on a shallow copy so the caller's per-chunk debug
                # data isn't retroactively mutated with canonical_name.
                annotated = dict(target)
                annotated["canonical_name"] = canon
                existing = seen.get(canon)
                if existing is None:
                    seen[canon] = annotated
                    continue
                # Prefer higher confidence; tie-break on having a numerical value.
                old_rank = self._CONFIDENCE_RANK.get(
                    str(existing.get("confidence", "")).lower(), 0
                )
                new_rank = self._CONFIDENCE_RANK.get(
                    str(annotated.get("confidence", "")).lower(), 0
                )
                if new_rank > old_rank:
                    seen[canon] = annotated
                elif new_rank == old_rank:
                    if existing.get("numerical_value") in (None, "", "null") and \
                       annotated.get("numerical_value") not in (None, "", "null"):
                        seen[canon] = annotated

        return list(seen.values())

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    @property
    def telemetry(self) -> Dict[str, int]:
        """Return accumulated usage statistics (8 typed keys)."""
        return {
            "total_requests": self._total_requests,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens_used": self._total_tokens_used,
            "http_failures": self._http_failures,
            "empty_responses": self._empty_responses,
            "schema_invalid_drops": self._schema_invalid_drops,
            "retries": self._retries,
        }

    def reset_telemetry(self) -> None:
        """Zero out all telemetry counters (call before a new corpus run)."""
        self._total_requests = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_tokens_used = 0
        self._http_failures = 0
        self._empty_responses = 0
        self._schema_invalid_drops = 0
        self._retries = 0

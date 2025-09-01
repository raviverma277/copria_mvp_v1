# core/parsing/classifier.py
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import re
import os
from collections import defaultdict


# --------- Utility: safe import of OpenAI for optional fallback ----------
def _get_openai_client():
    """
    Try to create an OpenAI client if the package + env are present.
    Supports either 'openai' (legacy) or 'openai>=1' (client).
    Returns (client, mode) where mode in {"chat_legacy","responses_v1"}.
    Returns (None, None) if unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        return None, None
    try:
        # new SDK (openai>=1.x)
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        return client, "responses_v1"
    except Exception:
        pass
    try:
        # legacy SDK
        import openai  # type: ignore

        openai.api_key = api_key
        return openai, "chat_legacy"
    except Exception:
        return None, None


# ------------------------------------------------------------------------


class DocumentClassifier:
    """
    Heuristic-first classifier with optional LLM fallback.
    Buckets: submission | sov | loss_run | questionnaire | email_body | other
    """

    def __init__(self) -> None:
        # Keyword sets (tune freely)
        self.keywords: Dict[str, List[re.Pattern]] = {
            "submission": [
                re.compile(r"\bmarket reform contract\b", re.I),
                re.compile(r"\bumr\b", re.I),
                re.compile(r"\binsured name\b", re.I),
                re.compile(r"\bperiod of insurance\b", re.I),
                re.compile(r"\bslip\b", re.I),
                re.compile(r"\bmrc\b", re.I),
            ],
            "sov": [
                re.compile(r"\bschedule of values\b", re.I),
                re.compile(r"\b(tiv|total insured value)\b", re.I),
                re.compile(r"\bconstruction\b", re.I),
                re.compile(r"\boccupancy\b", re.I),
                re.compile(r"\byear built\b", re.I),
                re.compile(r"\bsprinkler(s)?\b", re.I),
            ],
            "loss_run": [
                re.compile(r"\bloss run\b", re.I),
                re.compile(r"\bbordereau\b", re.I),
                re.compile(r"\bdate of loss\b", re.I),
                re.compile(r"\bgross (paid|outstanding)\b", re.I),
                re.compile(r"\bclaim (no\.|number)\b", re.I),
                re.compile(r"\bcause of loss\b", re.I),
            ],
            "questionnaire": [
                re.compile(r"\bproposal form\b", re.I),
                re.compile(r"\bquestionnaire\b", re.I),
                re.compile(r"\b(fill|complete) (all|the) questions\b", re.I),
                re.compile(r"\bdeclaration\b", re.I),
                re.compile(r"\bplease provide details\b", re.I),
            ],
            "email_body": [
                re.compile(r"^from: ", re.I),
                re.compile(r"^to: ", re.I),
                re.compile(r"^subject: ", re.I),
                re.compile(r"\bplease find attached\b", re.I),
            ],
        }

        # Extension hints (soft priors)
        self.ext_hints: Dict[str, Dict[str, float]] = {
            "pdf": {},
            "xlsx": {"sov": 0.6, "loss_run": 0.6},
            "xls": {"sov": 0.6, "loss_run": 0.6},
            "docx": {"submission": 0.5, "questionnaire": 0.4},
            "eml": {"email_body": 1.0},
            "msg": {"email_body": 1.0},
        }

        # Confidence threshold for falling back to LLM
        self.llm_threshold = 0.55

        # Allowed buckets
        self.allowed = [
            "submission",
            "sov",
            "loss_run",
            "questionnaire",
            "email_body",
            "other",
        ]

    # --------------- helpers ----------------

    def _norm(self, s: str) -> str:
        return (s or "").strip()

    def _ext_prior(self, filename: Optional[str]) -> Dict[str, float]:
        if not filename:
            return {}
        name = filename.lower()
        for ext, prior in self.ext_hints.items():
            if name.endswith(f".{ext}"):
                return dict(prior)
        return {}

    def _score_keywords(self, text: str) -> Dict[str, float]:
        """
        Count keyword hits per bucket; return normalized scores.
        """
        text = text or ""
        scores: Dict[str, float] = {k: 0.0 for k in self.keywords.keys()}
        if not text:
            return scores
        for bucket, pats in self.keywords.items():
            hits = 0
            for p in pats:
                if p.search(text):
                    hits += 1
            scores[bucket] = float(hits)
        # normalize to 0..1 by dividing by max hits > 0
        mx = max(scores.values()) if scores else 0.0
        if mx > 0:
            for k in scores:
                scores[k] = scores[k] / mx
        return scores

    def _score_filename(self, filename: Optional[str]) -> Dict[str, float]:
        """
        Simple filename heuristics (e.g., 'sov.xlsx', 'loss_run.xlsx').
        """
        if not filename:
            return {}
        fn = filename.lower()
        out: Dict[str, float] = {}
        if re.search(r"\bsov\b|schedule[_\- ]?of[_\- ]?values", fn):
            out["sov"] = 1.0
        if re.search(r"\bloss(_| )?run\b|bordereau", fn):
            out["loss_run"] = max(out.get("loss_run", 0.0), 1.0)
        if re.search(r"\bsubmission\b|mrc|slip", fn):
            out["submission"] = max(out.get("submission", 0.0), 0.8)
        if re.search(r"\bquestionnaire\b|proposal[_\- ]?form", fn):
            out["questionnaire"] = max(out.get("questionnaire", 0.0), 0.9)
        if re.search(r"\.eml$|\.msg$", fn):
            out["email_body"] = 1.0
        return out

    def _blend_scores(self, parts: List[Dict[str, float]]) -> Dict[str, float]:
        out: Dict[str, float] = defaultdict(float)
        for d in parts:
            for k, v in d.items():
                out[k] += float(v or 0.0)
        return dict(out)

    def _pick(self, scores: Dict[str, float]) -> Tuple[str, float]:
        if not scores:
            return "other", 0.0
        # ensure all allowed buckets exist
        for b in self.allowed:
            scores.setdefault(b, 0.0)
        bucket, score = max(scores.items(), key=lambda kv: kv[1])
        return bucket, float(score)

    # --------------- public: text (PDF/DOCX/unknown) ----------------

    def classify_text(
        self,
        text: str,
        filename: Optional[str] = None,
        page_tags: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Classify free text (PDF full text or DOCX text).
        Optional page_tags from PDF: [{"page":0,"type":"sov","confidence":0.92}, ...]
        """
        parts: List[Dict[str, float]] = []

        # 1) If we have page_tags, treat them as a weighted vote
        if page_tags:
            tag_scores: Dict[str, float] = defaultdict(float)
            for t in page_tags:
                typ = (t.get("type") or "").lower()
                conf = float(t.get("confidence", 0) or 0)
                if typ in {"submission", "sov", "loss_run", "questionnaire"}:
                    tag_scores[typ] += conf
            if tag_scores:
                # normalize
                mx = max(tag_scores.values())
                if mx > 0:
                    for k in list(tag_scores.keys()):
                        tag_scores[k] = tag_scores[k] / mx
                parts.append(dict(tag_scores))

        # 2) Keyword scores
        parts.append(self._score_keywords(text or ""))

        # 3) Filename hints
        parts.append(self._score_filename(filename))

        # 4) Extension prior
        parts.append(self._ext_prior(filename))

        # Blend & pick
        scores = self._blend_scores(parts)
        bucket, conf = self._pick(scores)

        # If weak, consider LLM fallback (best-effort)
        if conf < self.llm_threshold:
            fb = self._llm_fallback_text(text=text, filename=filename)
            if fb:
                bucket = fb.get("bucket", bucket)
                # do not overstate confidence; take max of (heuristic_conf, llm_conf*0.95)
                llm_conf = float(fb.get("confidence", 0.75))
                conf = max(conf, min(0.95, llm_conf))

        return {"bucket": bucket, "confidence": conf, "scores": scores}

    # --------------- public: excel ----------------

    def classify_excel(
        self, sheets: List[Dict[str, Any]], filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify an Excel workbook based on headers/values across sheets.
        """
        # Header-derived hints
        sov_hits = 0
        loss_hits = 0

        sov_hdrs = [
            r"\btiv\b",
            r"total insured value",
            r"\bconstruction\b",
            r"\boccupancy\b",
            r"\byear built\b",
            r"\bsprinkler",
            r"\balarm\b",
            r"\bpostcode\b|\bzip\b",
        ]
        loss_hdrs = [
            r"\bdate of loss\b",
            r"\bcause of loss\b",
            r"\bclaim (no\.|number)\b",
            r"\bgross (paid|outstanding)\b",
            r"\bnet (paid|outstanding)\b",
            r"\bstatus\b",
        ]
        sov_re = [re.compile(p, re.I) for p in sov_hdrs]
        loss_re = [re.compile(p, re.I) for p in loss_hdrs]

        for sh in sheets or []:
            headers = [str(h) for h in (sh.get("headers") or [])]
            for h in headers:
                for p in sov_re:
                    if p.search(h):
                        sov_hits += 1
                for p in loss_re:
                    if p.search(h):
                        loss_hits += 1

        # Normalize header scores
        mx = max(sov_hits, loss_hits, 1)
        sov_score = sov_hits / mx
        loss_score = loss_hits / mx

        parts: List[Dict[str, float]] = [
            {"sov": sov_score, "loss_run": loss_score},
            self._score_filename(filename),
            self._ext_prior(filename),
        ]
        scores = self._blend_scores(parts)
        bucket, conf = self._pick(scores)

        # Weak? Try LLM fallback with header-only context
        if conf < self.llm_threshold:
            sample_headers = []
            for sh in sheets[:2]:
                sample_headers.extend([str(h) for h in (sh.get("headers") or [])][:12])
            fb = self._llm_fallback_text(
                text="\n".join(sample_headers), filename=filename or "workbook.xlsx"
            )
            if fb:
                bucket = fb.get("bucket", bucket)
                llm_conf = float(fb.get("confidence", 0.75))
                conf = max(conf, min(0.95, llm_conf))

        return {"bucket": bucket, "confidence": conf, "scores": scores}

    # --------------- optional LLM fallback ----------------

    def _llm_fallback_text(
        self, text: str, filename: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Ask an LLM to classify into one of the allowed buckets.
        Only used if OPENAI_API_KEY is present. Returns None if LLM is unavailable.
        """
        client, mode = _get_openai_client()
        if not client or not mode:
            return None

        # keep prompt tiny; we only need a bucket
        system = (
            "You categorize insurance-related documents. "
            "Allowed labels: submission, sov, loss_run, questionnaire, email_body, other. "
            'Reply strictly as JSON: {"bucket": "<label>", "confidence": 0.0-1.0}.'
        )
        user = (
            f"Filename: {filename or '(unknown)'}\n\n"
            f"Content sample (first ~1500 chars):\n{text[:1500] if text else ''}\n\n"
            "Pick the single best label."
        )

        try:
            if mode == "responses_v1":
                # openai>=1 client
                resp = client.responses.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                content = resp.output_text  # SDK helper to get the string
            else:
                # legacy chat.completions
                content = client.ChatCompletion.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4-0613"),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                )["choices"][0]["message"]["content"]

            import json

            data = json.loads(content)
            bucket = (data.get("bucket") or "other").lower()
            if bucket not in self.allowed:
                bucket = "other"
            conf = float(data.get("confidence", 0.75))
            # bound confidence
            conf = max(0.0, min(1.0, conf))
            return {"bucket": bucket, "confidence": conf}
        except Exception:
            # Be silent; heuristics already returned something
            return None

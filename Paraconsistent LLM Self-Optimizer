#!/usr/bin/env python3
"""
Shadow-HoTT LLM Self-Edit Optimizer (v2)
=======================================

A practical paraconsistent code optimizer:
- Patch-based proposals (exact-match patch ops preferred)
- Bilateral evidence state in Sigma_pre = [0,1]^2 (truth-support / falsity-support)
- Belnap 4-valued classification (T, F, B, U)
- Language-bridge profiles (task-specific thresholds, weights, costs)
- Classical evaluator worlds (syntax, static risk, optional runtime/test/lint/typecheck commands)
- Determinization (root collapse) + classical patch selection
- Contradiction-kernel telemetry and provenance artifacts

This script is designed to run without external dependencies (stdlib-only). If requests is
installed, it will be used for Anthropic API calls; otherwise urllib is used.

IMPORTANT SAFETY NOTE:
- Code execution is OFF by default.
- Runtime execution / command worlds require --allow-exec.
- For stronger isolation, pass --sandbox-cmd (e.g., firejail / bwrap / docker wrapper).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
from dataclasses import dataclass, field
import difflib
import enum
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional requests for HTTP
try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    import resource  # Unix-only
except Exception:
    resource = None


# ============================================================
# 1. Core Paraconsistent Types / Utilities
# ============================================================
class Val(enum.Enum):
    T = "True_Only"
    F = "False_Only"
    B = "Both_Glut"
    U = "Neither_Gap"

    def symbol(self) -> str:
        return {Val.T: "✓", Val.F: "✗", Val.B: "⚡", Val.U: "∅"}[self]


def belnap_classify(t_score: float, f_score: float, theta: float) -> Val:
    t = t_score >= theta
    f = f_score >= theta
    if t and not f:
        return Val.T
    if (not t) and f:
        return Val.F
    if t and f:
        return Val.B
    return Val.U


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


@dataclass
class PatchOp:
    """
    Patch operation with exact-match semantics for robustness.

    Supported ops:
      - replace: target -> replacement
      - replace_once: same as replace (single replacement)
      - replace_all: replace all occurrences
      - insert_after: insert text immediately after target
      - insert_before: insert text immediately before target
      - append: append text at EOF
      - prepend: prepend text at BOF
      - regex_replace: regex pattern (target) -> replacement (use sparingly)
    """
    op: str
    target: Optional[str] = None
    replacement: Optional[str] = None
    text: Optional[str] = None
    count: Optional[int] = None


@dataclass
class PatchProposal:
    label: str
    description: str
    patch_ops: List[PatchOp] = field(default_factory=list)
    proposed_code: Optional[str] = None  # fallback / alternative path
    confidence: float = 0.5
    rationale: str = ""


@dataclass
class WorldEvidence:
    world: str
    t_support: float = 0.0
    f_support: float = 0.0
    notes: str = ""
    hard_fail: bool = False
    skipped: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)

    def classify(self, theta: float) -> Val:
        return belnap_classify(self.t_support, self.f_support, theta)


@dataclass
class DiffStats:
    lines_added: int = 0
    lines_removed: int = 0
    changed_line_ratio: float = 0.0
    unified_diff: str = ""
    changed_functions_est: int = 0
    dangerous_pattern_hits: int = 0
    patch_op_count: int = 0


@dataclass
class ProposalEvaluation:
    proposal: PatchProposal
    candidate_code: str
    patch_applied: bool
    patch_notes: str
    diff_stats: DiffStats
    world_evidence: List[WorldEvidence] = field(default_factory=list)
    t_score: float = 0.0
    f_score: float = 0.0
    cost_score: float = 0.0
    utility: float = 0.0
    overall_class: Val = Val.U
    collapsed_class: Optional[Val] = None
    hard_fail: bool = False

    @property
    def effective_class(self) -> Val:
        return self.collapsed_class or self.overall_class


@dataclass
class BaselineContext:
    code: str
    file_path: Optional[str]
    repo_path: Optional[str]
    task: str
    baseline_worlds: Dict[str, WorldEvidence] = field(default_factory=dict)


# ============================================================
# 2. Patch Engine (patch-based proposals)
# ============================================================
class PatchEngine:
    @staticmethod
    def apply_ops(code: str, ops: Sequence[PatchOp]) -> Tuple[bool, str, str]:
        """Apply exact-match patch ops in order. Returns (ok, new_code, notes)."""
        cur = code
        notes: List[str] = []

        for i, op in enumerate(ops, start=1):
            kind = op.op.strip().lower()
            try:
                if kind in {"replace", "replace_once"}:
                    if op.target is None or op.replacement is None:
                        return False, code, f"op#{i} {kind}: missing target/replacement"
                    if op.target not in cur:
                        return False, code, f"op#{i} {kind}: target not found"
                    occurrences = cur.count(op.target)
                    if occurrences != 1 and kind == "replace":
                        return False, code, f"op#{i} replace: target occurs {occurrences} times (must be exact unique)"
                    cur = cur.replace(op.target, op.replacement, 1)
                    notes.append(f"{kind}:1")

                elif kind == "replace_all":
                    if op.target is None or op.replacement is None:
                        return False, code, f"op#{i} replace_all: missing target/replacement"
                    occurrences = cur.count(op.target)
                    if occurrences == 0:
                        return False, code, f"op#{i} replace_all: target not found"
                    cur = cur.replace(op.target, op.replacement)
                    notes.append(f"replace_all:{occurrences}")

                elif kind == "insert_after":
                    if op.target is None or op.text is None:
                        return False, code, f"op#{i} insert_after: missing target/text"
                    idx = cur.find(op.target)
                    if idx < 0:
                        return False, code, f"op#{i} insert_after: target not found"
                    idx2 = idx + len(op.target)
                    cur = cur[:idx2] + op.text + cur[idx2:]
                    notes.append("insert_after")

                elif kind == "insert_before":
                    if op.target is None or op.text is None:
                        return False, code, f"op#{i} insert_before: missing target/text"
                    idx = cur.find(op.target)
                    if idx < 0:
                        return False, code, f"op#{i} insert_before: target not found"
                    cur = cur[:idx] + op.text + cur[idx:]
                    notes.append("insert_before")

                elif kind == "append":
                    if op.text is None:
                        return False, code, f"op#{i} append: missing text"
                    if not cur.endswith("\n") and op.text:
                        cur += "\n"
                    cur += op.text
                    notes.append("append")

                elif kind == "prepend":
                    if op.text is None:
                        return False, code, f"op#{i} prepend: missing text"
                    cur = op.text + cur
                    notes.append("prepend")

                elif kind == "regex_replace":
                    if op.target is None or op.replacement is None:
                        return False, code, f"op#{i} regex_replace: missing pattern/replacement"
                    count = op.count if op.count is not None else 1
                    new_cur, n = re.subn(op.target, op.replacement, cur, count=count, flags=re.MULTILINE)
                    if n == 0:
                        return False, code, f"op#{i} regex_replace: pattern not matched"
                    cur = new_cur
                    notes.append(f"regex_replace:{n}")

                else:
                    return False, code, f"op#{i}: unsupported op '{op.op}'"
            except re.error as e:
                return False, code, f"op#{i} regex error: {e}"

        return True, cur, ", ".join(notes) if notes else "no-op"


# ============================================================
# 3. Safer command execution hooks (classical worlds runtime/test/lint/etc.)
# ============================================================
@dataclass
class RunResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False
    cmd_display: str = ""


class CommandRunner:
    def __init__(
        self,
        allow_exec: bool = False,
        sandbox_cmd: Optional[str] = None,
        timeout_default: int = 10,
        cpu_seconds: Optional[int] = None,
        memory_mb: Optional[int] = None,
    ):
        self.allow_exec = allow_exec
        self.sandbox_cmd = shlex.split(sandbox_cmd) if sandbox_cmd else []
        self.timeout_default = timeout_default
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb

    def _preexec(self):
        if resource is None:
            return
        try:
            if self.cpu_seconds:
                resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds))
            if self.memory_mb:
                mem = self.memory_mb * 1024 * 1024
                # RLIMIT_AS may not be enforced on all systems, but still useful where supported.
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except Exception:
            pass

    def run(
        self,
        cmd: Sequence[str],
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> RunResult:
        if not self.allow_exec:
            return RunResult(
                ok=False,
                returncode=126,
                stdout="",
                stderr="Execution disabled (enable with --allow-exec)",
                duration_sec=0.0,
                cmd_display=" ".join(cmd),
            )

        full_cmd = list(self.sandbox_cmd) + list(cmd)
        start = time.time()
        try:
            proc = subprocess.run(
                full_cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout_default,
                env=env,
                preexec_fn=self._preexec if (os.name != "nt" and (self.cpu_seconds or self.memory_mb)) else None,
            )
            dur = time.time() - start
            return RunResult(
                ok=proc.returncode == 0,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_sec=dur,
                timed_out=False,
                cmd_display=" ".join(full_cmd),
            )
        except subprocess.TimeoutExpired as e:
            dur = time.time() - start
            return RunResult(
                ok=False,
                returncode=124,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + "\n[timeout]",
                duration_sec=dur,
                timed_out=True,
                cmd_display=" ".join(full_cmd),
            )
        except FileNotFoundError as e:
            dur = time.time() - start
            return RunResult(
                ok=False,
                returncode=127,
                stdout="",
                stderr=str(e),
                duration_sec=dur,
                cmd_display=" ".join(full_cmd),
            )
        except Exception as e:
            dur = time.time() - start
            return RunResult(
                ok=False,
                returncode=1,
                stdout="",
                stderr=str(e),
                duration_sec=dur,
                cmd_display=" ".join(full_cmd),
            )

    def run_python_code(
        self,
        code: str,
        timeout: int = 5,
        workdir: Optional[str] = None,
        filename: str = "candidate.py",
    ) -> RunResult:
        if not self.allow_exec:
            return RunResult(
                ok=False,
                returncode=126,
                stdout="",
                stderr="Execution disabled (enable with --allow-exec)",
                duration_sec=0.0,
                cmd_display=f"python {filename}",
            )
        tmpdir_obj = tempfile.TemporaryDirectory(dir=workdir)
        tmpdir = tmpdir_obj.name
        path = os.path.join(tmpdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        rr = self.run([sys.executable, path], cwd=tmpdir, timeout=timeout)
        tmpdir_obj.cleanup()
        return rr


# ============================================================
# 4. LLM Interface (Anthropic, optional)
# ============================================================
class LLMInterface:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.available = bool(self.api_key)

    def _http_post_json(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        if requests is not None:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        # urllib fallback
        import urllib.request
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - used for known API endpoint
            return json.loads(resp.read().decode("utf-8"))

    def _call_api(self, system: str, user_msg: str, max_tokens: int = 2048) -> str:
        if not self.available:
            return ""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_msg}],
        }
        try:
            data = self._http_post_json(self.base_url, headers, payload)
            content = data.get("content", [])
            if content and isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                return "\n".join(t for t in text_parts if t)
            return ""
        except Exception as e:
            return ""

    @staticmethod
    def _strip_fences(raw: str) -> str:
        return re.sub(r"```(?:json)?\s*|```", "", raw).strip()

    def speculate_patches(
        self,
        code: str,
        task: str,
        history_context: str = "",
        max_proposals: int = 4,
    ) -> List[PatchProposal]:
        if not self.available:
            return self._heuristic_speculate(code, task)

        system = textwrap.dedent(
            f"""
            You are a code transformation engine.
            Propose {max(2, min(6, max_proposals))} DISTINCT patch-based edits to a Python source file.

            Respond ONLY with a JSON array. No markdown.
            Each item must be an object with keys:
              - label: short snake_case id
              - description: one-line summary
              - confidence: float 0..1
              - patch_ops: array of exact-match patch ops (preferred)
              - proposed_code: optional full-file fallback string (use only if patch_ops are impractical)
              - rationale: short explanation

            Patch ops schema (exact-match semantics):
              replace / replace_once: {{"op":"replace","target":"...","replacement":"..."}}
              replace_all: {{"op":"replace_all","target":"...","replacement":"..."}}
              insert_after: {{"op":"insert_after","target":"...","text":"..."}}
              insert_before: {{"op":"insert_before","target":"...","text":"..."}}
              append: {{"op":"append","text":"..."}}
              prepend: {{"op":"prepend","text":"..."}}
              regex_replace: {{"op":"regex_replace","target":"REGEX","replacement":"...","count":1}}

            Prefer SMALL, safe, reversible patches. Avoid deleting broad code regions.
            If adding imports or helper functions, use patch ops where possible.
            """
        )

        user_msg = textwrap.dedent(
            f"""
            Task: {task}
            {('Previous iteration context: ' + history_context) if history_context else ''}

            Current code:
            ```python
            {code}
            ```

            Return ONLY JSON array.
            """
        )
        raw = self._call_api(system, user_msg, max_tokens=4096)
        if not raw:
            return self._heuristic_speculate(code, task)
        return self._parse_patch_proposals(raw, code, task)

    def judge_edit(self, original: str, candidate: str, task: str) -> Tuple[float, float, str]:
        """
        Returns (t_support, f_support, rationale) from LLM judge.
        f_support here is *risk/concern* support, not merely task-failure.
        """
        if not self.available:
            return self._heuristic_judge(original, candidate, task)

        system = textwrap.dedent(
            """
            You are a strict code review judge.
            Score a candidate edit along two independent axes:
            - t_support: evidence the edit advances the task correctly (0..1)
            - f_support: evidence the edit introduces risk/regressions/quality concerns (0..1)

            Return ONLY JSON:
            {"t_support":0.0,"f_support":0.0,"rationale":"..."}
            """
        )
        user_msg = textwrap.dedent(
            f"""
            Task: {task}

            Original code:
            ```python
            {original}
            ```

            Candidate code:
            ```python
            {candidate}
            ```
            """
        )
        raw = self._call_api(system, user_msg, max_tokens=256)
        if not raw:
            return self._heuristic_judge(original, candidate, task)
        try:
            obj = json.loads(self._strip_fences(raw))
            t = clamp01(obj.get("t_support", 0.5))
            f = clamp01(obj.get("f_support", 0.5))
            rat = str(obj.get("rationale", ""))
            return t, f, rat
        except Exception:
            return self._heuristic_judge(original, candidate, task)

    def _parse_patch_proposals(self, raw: str, code: str, task: str) -> List[PatchProposal]:
        try:
            data = json.loads(self._strip_fences(raw))
            if isinstance(data, dict):
                data = [data]
            out: List[PatchProposal] = []
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                ops: List[PatchOp] = []
                raw_ops = item.get("patch_ops") or []
                if isinstance(raw_ops, list):
                    for ro in raw_ops:
                        if isinstance(ro, dict) and ro.get("op"):
                            ops.append(PatchOp(
                                op=str(ro.get("op")),
                                target=ro.get("target"),
                                replacement=ro.get("replacement"),
                                text=ro.get("text"),
                                count=ro.get("count"),
                            ))
                out.append(PatchProposal(
                    label=str(item.get("label", f"edit_{i}")),
                    description=str(item.get("description", ""))[:240],
                    patch_ops=ops,
                    proposed_code=item.get("proposed_code") if isinstance(item.get("proposed_code"), str) else None,
                    confidence=clamp01(item.get("confidence", 0.5)),
                    rationale=str(item.get("rationale", ""))[:500],
                ))
            # Basic dedupe by label+patch hash
            dedup: Dict[str, PatchProposal] = {}
            for p in out:
                key = p.label + ":" + stable_hash(json.dumps([dataclasses.asdict(op) for op in p.patch_ops], sort_keys=True) + (p.proposed_code or ""))
                dedup[key] = p
            result = list(dedup.values())
            return result if result else self._heuristic_speculate(code, task)
        except Exception:
            return self._heuristic_speculate(code, task)

    def _heuristic_speculate(self, code: str, task: str) -> List[PatchProposal]:
        """Reasonable fallback patch proposals without an API key."""
        proposals: List[PatchProposal] = []
        task_l = task.lower()

        # 1) Context manager for simple open/close patterns
        if "open(" in code and ("context manager" in task_l or "error handling" in task_l or "file" in task_l):
            m = re.search(
                r"(?P<indent>[ \t]*)f\s*=\s*open\((?P<args>[^\n]+)\)\n(?P<body>(?:[ \t]+.*\n)+?)(?P=indent)f\.close\(\)",
                code,
                flags=re.MULTILINE,
            )
            if m:
                indent = m.group("indent")
                body = m.group("body")
                body2 = re.sub(r"\bf\.", "fh.", body)
                replacement = (
                    f"{indent}with open({m.group('args')}) as fh:\n"
                    f"{body2}"
                )
                proposals.append(PatchProposal(
                    label="use_context_manager",
                    description="Replace manual open/close with a context manager",
                    patch_ops=[PatchOp(op="replace", target=m.group(0), replacement=replacement)],
                    confidence=0.72,
                    rationale="Reduces resource-leak risk and improves exception safety.",
                ))

        # 2) Add basic type hints to simple defs (exact replacements only when signatures are one-line)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None
        type_ops: List[PatchOp] = []
        if tree and ("type" in task_l or "hint" in task_l):
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.returns is None and node.args.kwarg is None and node.args.vararg is None:
                        line = code.splitlines()[node.lineno - 1]
                        if line.strip().startswith("def ") and line.rstrip().endswith(":") and "->" not in line:
                            hinted = line[:-1] + " -> None:" if "return " not in ast.get_source_segment(code, node)[:500] else line[:-1] + " -> list:"  # crude fallback
                            # Avoid bad duplicate if line already changed by another op in same proposal
                            if line != hinted:
                                type_ops.append(PatchOp(op="replace_once", target=line, replacement=hinted))
            if type_ops:
                proposals.append(PatchProposal(
                    label="add_basic_return_hints",
                    description="Add basic return type hints to simple function definitions",
                    patch_ops=type_ops[:4],
                    confidence=0.45,
                    rationale="Heuristic hints only; intended as a conservative starting point.",
                ))

        # 3) Add try/except around __main__ block if present
        if "if __name__ == \"__main__\":" in code and ("error handling" in task_l or "robust" in task_l):
            marker = 'if __name__ == "__main__":\n'
            idx = code.find(marker)
            if idx >= 0:
                # Only patch exact main body if simple indentation block exists
                after = code[idx + len(marker):]
                body_match = re.match(r"((?:    .*\n?)+)", after)
                if body_match:
                    body = body_match.group(1)
                    repl = (
                        marker +
                        "    try:\n" +
                        "".join("    " + ln if ln else ln for ln in body.splitlines(True)) +
                        "    except Exception as e:\n"
                        "        print(f\"Error: {e}\", file=sys.stderr)\n"
                        "        raise\n"
                    )
                    target = marker + body
                    # Add import sys if absent
                    ops: List[PatchOp] = []
                    if "import sys" not in code:
                        # insert after first import line or prepend
                        import_line = re.search(r"^(import .+|from .+ import .+)\n", code, flags=re.MULTILINE)
                        if import_line:
                            ops.append(PatchOp(op="insert_after", target=import_line.group(0), text="import sys\n"))
                        else:
                            ops.append(PatchOp(op="prepend", text="import sys\n"))
                    ops.append(PatchOp(op="replace", target=target, replacement=repl))
                    proposals.append(PatchProposal(
                        label="wrap_main_try_except",
                        description="Wrap __main__ execution path in try/except with stderr logging",
                        patch_ops=ops,
                        confidence=0.66,
                        rationale="Adds visible failure handling without changing core logic.",
                    ))

        # 4) Safe no-op/observability patch (docstring)
        if not proposals:
            proposals.append(PatchProposal(
                label="no_change",
                description="No safe heuristic patch identified",
                patch_ops=[],
                proposed_code=code,
                confidence=0.1,
                rationale="Heuristic fallback had no task-specific patch.",
            ))
        return proposals[:4]

    def _heuristic_judge(self, original: str, candidate: str, task: str) -> Tuple[float, float, str]:
        if candidate.strip() == original.strip():
            return 0.05, 0.05, "No change"
        task_l = task.lower()
        t = 0.35
        f = 0.15
        reasons: List[str] = []

        if "with open(" in candidate and "open(" in original and "context" in task_l:
            t += 0.35
            reasons.append("uses context manager")
        if "try:" in candidate and ("error" in task_l or "exception" in task_l):
            t += 0.2
            reasons.append("adds exception handling")
        if "->" in candidate and ("type" in task_l or "hint" in task_l):
            t += 0.15
            reasons.append("adds type hints")

        # risk heuristics
        if len(candidate) > len(original) * 1.6:
            f += 0.2
            reasons.append("large rewrite")
        if any(x in candidate for x in ["eval(", "exec(", "shell=True"]):
            f += 0.6
            reasons.append("dangerous construct")
        return clamp01(t), clamp01(f), "; ".join(reasons) or "heuristic estimate"


# ============================================================
# 5. Language Bridge Profiles (task-specific thresholds / weights / costs)
# ============================================================
@dataclass
class LanguageBridgeProfile:
    name: str
    theta: float = 0.5
    glut_tolerance: int = 1
    det_margin: float = 0.08
    cost_weight: float = 0.25
    max_cost_for_accept: float = 0.85
    hard_worlds: Tuple[str, ...] = ("patch_apply", "syntax")
    world_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def weight(self, world_name: str) -> Tuple[float, float]:
        w = self.world_weights.get(world_name, self.world_weights.get("*", {"t": 0.0, "f": 0.0}))
        return float(w.get("t", 0.0)), float(w.get("f", 0.0))

    def classify(self, t: float, f: float) -> Val:
        return belnap_classify(t, f, self.theta)

    def cost(self, diff_stats: DiffStats) -> float:
        # A practical cost functional C_L for ambiguous regions (B/U) and utility scoring.
        # Penalizes broad rewrites, many changed functions, dangerous patterns, many patch ops.
        c = 0.0
        c += min(diff_stats.changed_line_ratio, 1.0) * 0.45
        c += min(diff_stats.changed_functions_est / 5.0, 1.0) * 0.20
        c += min(diff_stats.dangerous_pattern_hits / 3.0, 1.0) * 0.30
        c += min(diff_stats.patch_op_count / 8.0, 1.0) * 0.10
        return clamp01(c)


def default_profiles() -> Dict[str, LanguageBridgeProfile]:
    base_weights = {
        "*": {"t": 0.0, "f": 0.0},
        "patch_apply": {"t": 0.10, "f": 0.70},
        "syntax": {"t": 0.20, "f": 0.85},
        "compile": {"t": 0.20, "f": 0.60},
        "diff_risk": {"t": 0.05, "f": 0.55},
        "complexity": {"t": 0.10, "f": 0.25},
        "dangerous_patterns": {"t": 0.00, "f": 0.75},
        "task_keyword": {"t": 0.20, "f": 0.05},
        "llm_judge": {"t": 0.55, "f": 0.35},
        "runtime_smoke": {"t": 0.35, "f": 0.60},
        "test_cmd": {"t": 0.75, "f": 0.80},
        "lint_cmd": {"t": 0.30, "f": 0.35},
        "typecheck_cmd": {"t": 0.45, "f": 0.60},
        "custom_cmd": {"t": 0.30, "f": 0.40},
    }
    profiles = {
        "bugfix": LanguageBridgeProfile(
            name="bugfix",
            theta=0.52,
            glut_tolerance=1,
            det_margin=0.10,
            cost_weight=0.30,
            max_cost_for_accept=0.75,
            hard_worlds=("patch_apply", "syntax", "compile"),
            world_weights=base_weights,
        ),
        "refactor": LanguageBridgeProfile(
            name="refactor",
            theta=0.50,
            glut_tolerance=2,
            det_margin=0.06,
            cost_weight=0.22,
            max_cost_for_accept=0.90,
            hard_worlds=("patch_apply", "syntax"),
            world_weights={
                **base_weights,
                "complexity": {"t": 0.30, "f": 0.20},
                "diff_risk": {"t": 0.05, "f": 0.45},
            },
        ),
        "security": LanguageBridgeProfile(
            name="security",
            theta=0.55,
            glut_tolerance=0,
            det_margin=0.12,
            cost_weight=0.35,
            max_cost_for_accept=0.70,
            hard_worlds=("patch_apply", "syntax", "compile", "dangerous_patterns"),
            world_weights={
                **base_weights,
                "dangerous_patterns": {"t": 0.00, "f": 0.95},
                "runtime_smoke": {"t": 0.20, "f": 0.80},
            },
        ),
        "typing": LanguageBridgeProfile(
            name="typing",
            theta=0.50,
            glut_tolerance=1,
            det_margin=0.08,
            cost_weight=0.20,
            max_cost_for_accept=0.85,
            hard_worlds=("patch_apply", "syntax", "compile"),
            world_weights={
                **base_weights,
                "typecheck_cmd": {"t": 0.80, "f": 0.85},
                "task_keyword": {"t": 0.30, "f": 0.05},
            },
        ),
        "docs": LanguageBridgeProfile(
            name="docs",
            theta=0.48,
            glut_tolerance=2,
            det_margin=0.05,
            cost_weight=0.10,
            max_cost_for_accept=0.95,
            hard_worlds=("patch_apply", "syntax"),
            world_weights={
                **base_weights,
                "llm_judge": {"t": 0.35, "f": 0.20},
                "diff_risk": {"t": 0.05, "f": 0.25},
            },
        ),
        "perf": LanguageBridgeProfile(
            name="perf",
            theta=0.52,
            glut_tolerance=1,
            det_margin=0.09,
            cost_weight=0.28,
            max_cost_for_accept=0.85,
            hard_worlds=("patch_apply", "syntax", "compile"),
            world_weights={
                **base_weights,
                "runtime_smoke": {"t": 0.50, "f": 0.70},
                "complexity": {"t": 0.15, "f": 0.30},
            },
        ),
        "prod_safe": LanguageBridgeProfile(
            name="prod_safe",
            theta=0.58,
            glut_tolerance=0,
            det_margin=0.14,
            cost_weight=0.40,
            max_cost_for_accept=0.60,
            hard_worlds=("patch_apply", "syntax", "compile", "dangerous_patterns", "test_cmd"),
            world_weights={
                **base_weights,
                "llm_judge": {"t": 0.35, "f": 0.25},
                "test_cmd": {"t": 0.95, "f": 0.95},
                "typecheck_cmd": {"t": 0.70, "f": 0.85},
                "lint_cmd": {"t": 0.35, "f": 0.45},
                "diff_risk": {"t": 0.02, "f": 0.70},
                "dangerous_patterns": {"t": 0.00, "f": 0.98},
            },
        ),
        "race": LanguageBridgeProfile(
            name="race",
            theta=0.47,
            glut_tolerance=3,
            det_margin=0.03,
            cost_weight=0.12,
            max_cost_for_accept=0.95,
            hard_worlds=("patch_apply", "syntax"),
            world_weights={
                **base_weights,
                "llm_judge": {"t": 0.70, "f": 0.25},
                "task_keyword": {"t": 0.35, "f": 0.03},
                "diff_risk": {"t": 0.08, "f": 0.30},
                "complexity": {"t": 0.18, "f": 0.18},
                "runtime_smoke": {"t": 0.45, "f": 0.45},
                "test_cmd": {"t": 0.85, "f": 0.70},
            },
        ),
    }
    # alias
    profiles["auto"] = profiles["bugfix"]
    return profiles


def infer_profile_name(task: str) -> str:
    t = task.lower()
    if any(k in t for k in ["security", "sanitize", "injection", "auth", "secret"]):
        return "security"
    if any(k in t for k in ["refactor", "clarity", "readability", "cleanup"]):
        return "refactor"
    if any(k in t for k in ["type hint", "typing", "mypy", "annotations"]):
        return "typing"
    if any(k in t for k in ["docstring", "docs", "documentation"]):
        return "docs"
    if any(k in t for k in ["performance", "optimize", "speed", "latency"]):
        return "perf"
    return "bugfix"


def profile_to_dict(profile: LanguageBridgeProfile) -> Dict[str, Any]:
    return {
        "name": profile.name,
        "theta": float(profile.theta),
        "glut_tolerance": int(profile.glut_tolerance),
        "det_margin": float(profile.det_margin),
        "cost_weight": float(profile.cost_weight),
        "max_cost_for_accept": float(profile.max_cost_for_accept),
        "hard_worlds": list(profile.hard_worlds),
        "world_weights": {k: {"t": float(v.get("t", 0.0)), "f": float(v.get("f", 0.0))} for k, v in profile.world_weights.items()},
    }


def apply_profile_json_overrides(profile: LanguageBridgeProfile, path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("profile JSON must be an object")
    data = raw.get("profile", raw)
    if not isinstance(data, dict):
        raise ValueError("profile JSON 'profile' key must be an object")

    scalar_fields = ["theta", "det_margin", "cost_weight", "max_cost_for_accept"]
    int_fields = ["glut_tolerance"]
    for key in scalar_fields:
        if key in data:
            setattr(profile, key, float(data[key]))
    for key in int_fields:
        if key in data:
            setattr(profile, key, int(data[key]))

    if "name" in data:
        profile.name = str(data["name"])

    if "hard_worlds" in data:
        hw = data["hard_worlds"]
        if not isinstance(hw, list) or not all(isinstance(x, str) for x in hw):
            raise ValueError("hard_worlds must be a list of strings")
        profile.hard_worlds = tuple(hw)

    if "world_weights" in data:
        ww = data["world_weights"]
        if not isinstance(ww, dict):
            raise ValueError("world_weights must be an object")
        merged = {k: {"t": float(v.get("t", 0.0)), "f": float(v.get("f", 0.0))} for k, v in profile.world_weights.items()}
        for world, wf in ww.items():
            if not isinstance(world, str) or not isinstance(wf, dict):
                raise ValueError("world_weights entries must be objects keyed by world name")
            cur = merged.get(world, {"t": 0.0, "f": 0.0})
            if "t" in wf:
                cur["t"] = float(wf["t"])
            if "f" in wf:
                cur["f"] = float(wf["f"])
            merged[world] = cur
        profile.world_weights = merged


def parse_weight_override(spec: str) -> Tuple[str, Dict[str, float]]:
    # Supported forms:
    #   world:t=0.7,f=0.2
    #   world:0.7,0.2
    #   world:t=0.7
    s = spec.strip()
    if ":" not in s:
        raise ValueError(f"invalid --set-weight '{spec}' (expected world:t=...,f=...)")
    world, rest = s.split(":", 1)
    world = world.strip()
    if not world:
        raise ValueError(f"invalid --set-weight '{spec}' (missing world name)")
    rest = rest.strip()
    if not rest:
        raise ValueError(f"invalid --set-weight '{spec}' (missing weights)")

    t_val: Optional[float] = None
    f_val: Optional[float] = None
    parts = [p.strip() for p in rest.split(",") if p.strip()]
    if all("=" not in p for p in parts):
        if len(parts) == 1:
            t_val = float(parts[0])
        elif len(parts) >= 2:
            t_val = float(parts[0])
            f_val = float(parts[1])
    else:
        for p in parts:
            if "=" not in p:
                raise ValueError(f"invalid --set-weight segment '{p}' in '{spec}'")
            k, v = p.split("=", 1)
            k = k.strip().lower()
            v = float(v.strip())
            if k in ("t", "truth"):
                t_val = v
            elif k in ("f", "false", "risk"):
                f_val = v
            else:
                raise ValueError(f"unknown weight key '{k}' in '{spec}' (use t or f)")

    if t_val is None and f_val is None:
        raise ValueError(f"invalid --set-weight '{spec}' (no weights parsed)")
    out: Dict[str, float] = {}
    if t_val is not None:
        out["t"] = float(t_val)
    if f_val is not None:
        out["f"] = float(f_val)
    return world, out


def apply_runtime_profile_overrides(profile: LanguageBridgeProfile, args: argparse.Namespace) -> None:
    if getattr(args, "profile_json", None):
        apply_profile_json_overrides(profile, args.profile_json)

    if getattr(args, "theta", None) is not None:
        profile.theta = float(args.theta)
    if getattr(args, "glut_tolerance", None) is not None:
        profile.glut_tolerance = int(args.glut_tolerance)
    if getattr(args, "det_margin", None) is not None:
        profile.det_margin = float(args.det_margin)
    if getattr(args, "cost_weight", None) is not None:
        profile.cost_weight = float(args.cost_weight)
    if getattr(args, "max_cost_for_accept", None) is not None:
        profile.max_cost_for_accept = float(args.max_cost_for_accept)

    if getattr(args, "set_weight", None):
        merged = {k: {"t": float(v.get("t", 0.0)), "f": float(v.get("f", 0.0))} for k, v in profile.world_weights.items()}
        for spec in args.set_weight:
            world, vals = parse_weight_override(spec)
            cur = merged.get(world, {"t": 0.0, "f": 0.0})
            cur.update(vals)
            merged[world] = cur
        profile.world_weights = merged

    hard = set(profile.hard_worlds)
    for w in (getattr(args, "set_hard_world", None) or []):
        hard.add(w)
    for w in (getattr(args, "unset_hard_world", None) or []):
        hard.discard(w)
    profile.hard_worlds = tuple(sorted(hard))


# ============================================================
# 6. Evaluator Worlds (classical fibers)
# ============================================================
class EvaluatorWorld:
    name = "world"

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        raise NotImplementedError


class PatchApplyWorld(EvaluatorWorld):
    name = "patch_apply"

    def __init__(self, patch_ok: bool, notes: str):
        self.patch_ok = patch_ok
        self.notes = notes

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        return WorldEvidence(
            world=self.name,
            t_support=1.0 if self.patch_ok else 0.0,
            f_support=0.0 if self.patch_ok else 1.0,
            notes=self.notes,
            hard_fail=not self.patch_ok,
        )


class SyntaxWorld(EvaluatorWorld):
    name = "syntax"

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        try:
            ast.parse(candidate)
            return WorldEvidence(world=self.name, t_support=1.0, f_support=0.0, notes="AST parse OK")
        except SyntaxError as e:
            return WorldEvidence(world=self.name, t_support=0.0, f_support=1.0, notes=f"SyntaxError: {e}", hard_fail=True)


class CompileWorld(EvaluatorWorld):
    name = "compile"

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        try:
            compile(candidate, baseline.file_path or "<candidate>", "exec")
            return WorldEvidence(world=self.name, t_support=0.8, f_support=0.0, notes="compile() OK")
        except Exception as e:
            return WorldEvidence(world=self.name, t_support=0.0, f_support=1.0, notes=f"Compile error: {e}", hard_fail=True)


class DiffRiskWorld(EvaluatorWorld):
    name = "diff_risk"

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        # Lower change ratio => lower risk. Tiny t_support for small, precise edits.
        ratio = diff_stats.changed_line_ratio
        f = clamp01(ratio * 1.15)
        t = clamp01(0.8 - f) if ratio > 0 else 0.0
        notes = f"changed_ratio={ratio:.3f}, +{diff_stats.lines_added}/-{diff_stats.lines_removed}"
        return WorldEvidence(world=self.name, t_support=t, f_support=f, notes=notes)


class ComplexityWorld(EvaluatorWorld):
    name = "complexity"

    @staticmethod
    def _count_constructs(code: str) -> int:
        try:
            tree = ast.parse(code)
        except Exception:
            return 0
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                count += 1
        return count

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        orig_c = self._count_constructs(baseline.code)
        cand_c = self._count_constructs(candidate)
        if orig_c == 0 and cand_c == 0:
            return WorldEvidence(world=self.name, t_support=0.2, f_support=0.0, notes="No structural constructs")
        if max(orig_c, cand_c, 1) == 0:
            delta = 0.0
        else:
            delta = (cand_c - orig_c) / max(orig_c, cand_c, 1)
        # Added complexity raises risk; reduced complexity gives task-help support for refactor-ish tasks.
        f = clamp01(max(delta, 0.0))
        t = clamp01(max(-delta, 0.0))
        return WorldEvidence(world=self.name, t_support=t, f_support=f, notes=f"constructs {orig_c}->{cand_c} (delta={delta:.2f})")


class DangerousPatternWorld(EvaluatorWorld):
    name = "dangerous_patterns"
    PATTERNS = [
        r"\beval\(",
        r"\bexec\(",
        r"subprocess\.(Popen|run|call)\(.*shell\s*=\s*True",
        r"pickle\.loads\(",
        r"yaml\.load\(",
        r"os\.system\(",
    ]

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        hits = 0
        details = []
        for pat in self.PATTERNS:
            m = re.findall(pat, candidate, flags=re.IGNORECASE | re.DOTALL)
            if m:
                hits += len(m)
                details.append(pat)
        diff_stats.dangerous_pattern_hits = hits
        if hits == 0:
            return WorldEvidence(world=self.name, t_support=0.4, f_support=0.0, notes="No flagged patterns")
        return WorldEvidence(
            world=self.name,
            t_support=0.0,
            f_support=clamp01(0.5 + 0.2 * hits),
            notes=f"Flagged patterns ({hits}): {', '.join(details[:4])}",
            hard_fail=False,
            metrics={"hits": hits},
        )


class TaskKeywordWorld(EvaluatorWorld):
    name = "task_keyword"

    STOPWORDS = {
        "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with", "add",
        "make", "improve", "better", "code", "python", "use", "handling",
    }

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        task_words = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", baseline.task.lower()) if w not in self.STOPWORDS]
        if not task_words:
            return WorldEvidence(world=self.name, t_support=0.1, f_support=0.0, notes="No informative task tokens")
        new_lines = set(candidate.splitlines()) - set(baseline.code.splitlines())
        new_text = "\n".join(new_lines).lower()
        matches = [w for w in set(task_words) if w in new_text]
        t = clamp01(len(matches) / max(1, min(4, len(set(task_words)))))
        return WorldEvidence(world=self.name, t_support=t, f_support=0.0, notes=f"matches={matches[:6]}")


class LLMJudgeWorld(EvaluatorWorld):
    name = "llm_judge"

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        t, f, rat = llm.judge_edit(baseline.code, candidate, baseline.task)
        return WorldEvidence(world=self.name, t_support=t, f_support=f, notes=rat)


class RuntimeSmokeWorld(EvaluatorWorld):
    name = "runtime_smoke"

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        rr = runner.run_python_code(candidate, timeout=self.timeout)
        if rr.returncode == 126 and "Execution disabled" in rr.stderr:
            return WorldEvidence(world=self.name, skipped=True, notes=rr.stderr)
        if rr.ok:
            # Runtime success provides evidence against breakage; small t_support (it *ran*)
            return WorldEvidence(world=self.name, t_support=0.5, f_support=0.0, notes=f"Ran successfully in {rr.duration_sec:.2f}s")
        # Import/module-related failures in non-self-contained code are weaker evidence than actual runtime exceptions.
        stderr = (rr.stderr or "")[-300:]
        weak = any(tok in stderr for tok in ["ModuleNotFoundError", "ImportError", "No module named"])
        return WorldEvidence(
            world=self.name,
            t_support=0.0,
            f_support=0.2 if weak else 0.75,
            notes=f"Runtime failed ({rr.returncode}): {stderr}",
            hard_fail=False,
            metrics={"timeout": rr.timed_out, "duration_sec": rr.duration_sec},
        )


class CommandWorld(EvaluatorWorld):
    """
    External command evaluator world.
    Command can reference:
      {candidate} -> candidate file path
      {workdir}   -> temporary working directory path
      {filename}  -> basename of candidate file
    """
    def __init__(self, name: str, command_template: str, timeout: int = 20, hard_fail: bool = False):
        self.name = name
        self.command_template = command_template
        self.timeout = timeout
        self.hard_fail_on_error = hard_fail

    def evaluate(self, candidate: str, baseline: BaselineContext, runner: CommandRunner, llm: LLMInterface, diff_stats: DiffStats) -> WorldEvidence:
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.basename(baseline.file_path) if baseline.file_path else "candidate.py"
            candidate_path = os.path.join(tmpdir, filename)
            with open(candidate_path, "w", encoding="utf-8") as f:
                f.write(candidate)

            # If repo_path is provided, copy candidate into a temp clone of just the file path layout (best effort)
            cwd = tmpdir
            if baseline.repo_path and baseline.file_path:
                rel_path = None
                try:
                    rel_path = os.path.relpath(os.path.abspath(baseline.file_path), os.path.abspath(baseline.repo_path))
                    if not rel_path.startswith(".."):
                        repo_tmp = os.path.join(tmpdir, "repo")
                        shutil.copytree(baseline.repo_path, repo_tmp, dirs_exist_ok=True)
                        full_candidate_path = os.path.join(repo_tmp, rel_path)
                        os.makedirs(os.path.dirname(full_candidate_path), exist_ok=True)
                        with open(full_candidate_path, "w", encoding="utf-8") as f2:
                            f2.write(candidate)
                        candidate_path = full_candidate_path
                        cwd = repo_tmp
                except Exception:
                    pass

            cmd_text = self.command_template.format(candidate=candidate_path, workdir=cwd, filename=os.path.basename(candidate_path))
            rr = runner.run(shlex.split(cmd_text), cwd=cwd, timeout=self.timeout)
            if rr.returncode == 126 and "Execution disabled" in rr.stderr:
                return WorldEvidence(world=self.name, skipped=True, notes=rr.stderr)

            # Success => support. Failure => risk. Keep scores moderate because command semantics vary.
            if rr.ok:
                return WorldEvidence(
                    world=self.name,
                    t_support=0.8,
                    f_support=0.0,
                    notes=f"OK ({rr.duration_sec:.2f}s): {cmd_text}",
                    metrics={"duration_sec": rr.duration_sec, "cmd": rr.cmd_display},
                )

            stderr = (rr.stderr or rr.stdout or "")[-500:]
            return WorldEvidence(
                world=self.name,
                t_support=0.0,
                f_support=0.8,
                notes=f"Fail ({rr.returncode}): {stderr}",
                hard_fail=self.hard_fail_on_error,
                metrics={"duration_sec": rr.duration_sec, "timeout": rr.timed_out, "cmd": rr.cmd_display},
            )


# ============================================================
# 7. Optimizer state + telemetry (contradiction kernels)
# ============================================================
@dataclass
class IterationTelemetry:
    iteration: int
    signature: Dict[str, int]
    shock_l1: int
    glut_count: int
    gap_count: int
    active_kernel: List[str]
    resolved_from_prev: List[str]
    newly_active: List[str]
    selected_label: Optional[str]
    selected_utility: Optional[float]
    proposals_evaluated: int


class SemanticShadowState:
    def __init__(self, theta: float = 0.5):
        self.theta = theta
        self.bi: Dict[str, Tuple[float, float]] = {}
        self.cache: Dict[str, Val] = {}
        self.evals: Dict[str, ProposalEvaluation] = {}
        self.provenance_iter: int = 0

    def refresh_cache(self) -> None:
        self.cache.clear()
        for label, (t, f) in self.bi.items():
            self.cache[label] = belnap_classify(t, f, self.theta)

    def signature_counts(self) -> Dict[str, int]:
        counts = {"T": 0, "F": 0, "B": 0, "U": 0}
        for v in self.cache.values():
            counts[v.name[0]] += 1  # T/F/B/U
        return counts


# ============================================================
# 8. Diff / stats helpers
# ============================================================
def estimate_changed_functions(code_before: str, code_after: str) -> int:
    try:
        t1 = ast.parse(code_before)
        t2 = ast.parse(code_after)
    except Exception:
        return 0

    def func_map(tree: ast.AST, src: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seg = ast.get_source_segment(src, n) or ""
                out[n.name] = seg
        return out

    m1, m2 = func_map(t1, code_before), func_map(t2, code_after)
    changed = 0
    names = set(m1) | set(m2)
    for name in names:
        if m1.get(name) != m2.get(name):
            changed += 1
    return changed


def compute_diff_stats(original: str, candidate: str, patch_op_count: int = 0) -> DiffStats:
    orig_lines = original.splitlines(keepends=True)
    cand_lines = candidate.splitlines(keepends=True)
    udiff = "".join(difflib.unified_diff(orig_lines, cand_lines, fromfile="original", tofile="candidate", n=2))

    added = 0
    removed = 0
    for line in udiff.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1

    denom = max(1, len([ln for ln in orig_lines if ln.strip()]) or 1)
    changed_ratio = (added + removed) / denom
    stats = DiffStats(
        lines_added=added,
        lines_removed=removed,
        changed_line_ratio=min(changed_ratio, 10.0),
        unified_diff=udiff,
        changed_functions_est=estimate_changed_functions(original, candidate),
        patch_op_count=patch_op_count,
    )
    return stats


# ============================================================
# 9. Main Optimizer
# ============================================================
class ShadowHoTTOptimizerV2:
    def __init__(
        self,
        profile: LanguageBridgeProfile,
        llm: LLMInterface,
        runner: CommandRunner,
        iterations: int = 3,
        max_proposals: int = 4,
        verbose: bool = True,
        provenance_dir: Optional[str] = None,
    ):
        self.profile = profile
        self.llm = llm
        self.runner = runner
        self.iterations = iterations
        self.max_proposals = max(2, min(max_proposals, 8))
        self.verbose = verbose
        self.state = SemanticShadowState(theta=profile.theta)
        self.history: List[IterationTelemetry] = []
        self.prev_active_kernel: set[str] = set()
        self.provenance_dir = provenance_dir
        if provenance_dir:
            os.makedirs(provenance_dir, exist_ok=True)

        # Acceptance guard state (v2.1 hardening)
        self.applied_candidate_hashes: set[str] = set()
        self.applied_patch_fingerprints: set[str] = set()
        self.last_applied_label: Optional[str] = None
        self.last_applied_utility: Optional[float] = None
        self.repeat_label_nonimproving_streak: int = 0
        self.repeat_drop_margin: float = 0.02

    def log(self, msg: str = "") -> None:
        if self.verbose:
            print(msg)

    def build_worlds(self, patch_ok: bool, patch_notes: str, args_ns: argparse.Namespace) -> List[EvaluatorWorld]:
        worlds: List[EvaluatorWorld] = [
            PatchApplyWorld(patch_ok, patch_notes),
            SyntaxWorld(),
            CompileWorld(),
            DiffRiskWorld(),
            ComplexityWorld(),
            DangerousPatternWorld(),
            TaskKeywordWorld(),
            LLMJudgeWorld(),
        ]
        if getattr(args_ns, "runtime_smoke", False):
            worlds.append(RuntimeSmokeWorld(timeout=getattr(args_ns, "runtime_timeout", 5)))
        if getattr(args_ns, "test_cmd", None):
            worlds.append(CommandWorld("test_cmd", args_ns.test_cmd, timeout=getattr(args_ns, "test_timeout", 30), hard_fail=False))
        if getattr(args_ns, "lint_cmd", None):
            worlds.append(CommandWorld("lint_cmd", args_ns.lint_cmd, timeout=getattr(args_ns, "lint_timeout", 20), hard_fail=False))
        if getattr(args_ns, "typecheck_cmd", None):
            worlds.append(CommandWorld("typecheck_cmd", args_ns.typecheck_cmd, timeout=getattr(args_ns, "typecheck_timeout", 30), hard_fail=False))
        for i, c in enumerate(getattr(args_ns, "custom_cmd", []) or []):
            worlds.append(CommandWorld(f"custom_cmd", c, timeout=getattr(args_ns, "custom_timeout", 20), hard_fail=False))
        return worlds

    def _aggregate_scores(self, evidence: List[WorldEvidence], diff_stats: DiffStats) -> Tuple[float, float, float, bool]:
        t_num = t_den = 0.0
        f_num = f_den = 0.0
        hard_fail = False
        for ev in evidence:
            if ev.skipped:
                continue
            wt_t, wt_f = self.profile.weight(ev.world)
            if wt_t > 0:
                t_num += wt_t * clamp01(ev.t_support)
                t_den += wt_t
            if wt_f > 0:
                f_num += wt_f * clamp01(ev.f_support)
                f_den += wt_f
            if ev.hard_fail and ev.world in self.profile.hard_worlds:
                hard_fail = True

        t_score = clamp01((t_num / t_den) if t_den else 0.0)
        f_score = clamp01((f_num / f_den) if f_den else 0.0)
        cost = self.profile.cost(diff_stats)
        return t_score, f_score, cost, hard_fail

    def _patch_to_candidate(self, current_code: str, proposal: PatchProposal) -> Tuple[bool, str, str]:
        if proposal.patch_ops:
            ok, candidate, notes = PatchEngine.apply_ops(current_code, proposal.patch_ops)
            return ok, candidate, notes
        if proposal.proposed_code is not None:
            return True, proposal.proposed_code, "proposed_code fallback"
        return False, current_code, "empty proposal (no patch_ops/proposed_code)"

    def _evaluate_proposal(self, proposal: PatchProposal, current_code: str, baseline: BaselineContext, args_ns: argparse.Namespace) -> ProposalEvaluation:
        patch_ok, candidate, patch_notes = self._patch_to_candidate(current_code, proposal)
        diff_stats = compute_diff_stats(current_code, candidate, patch_op_count=len(proposal.patch_ops))
        worlds = self.build_worlds(patch_ok=patch_ok, patch_notes=patch_notes, args_ns=args_ns)
        evidence: List[WorldEvidence] = []
        for world in worlds:
            try:
                ev = world.evaluate(candidate, baseline, self.runner, self.llm, diff_stats)
            except Exception as e:
                ev = WorldEvidence(world=world.name, t_support=0.0, f_support=0.8, notes=f"Evaluator exception: {e}", hard_fail=False)
            evidence.append(ev)

        t_score, f_score, cost_score, hard_fail = self._aggregate_scores(evidence, diff_stats)
        overall = self.profile.classify(t_score, f_score)
        utility = t_score - f_score - self.profile.cost_weight * cost_score

        pe = ProposalEvaluation(
            proposal=proposal,
            candidate_code=candidate,
            patch_applied=patch_ok,
            patch_notes=patch_notes,
            diff_stats=diff_stats,
            world_evidence=evidence,
            t_score=t_score,
            f_score=f_score,
            cost_score=cost_score,
            utility=utility,
            overall_class=overall,
            hard_fail=hard_fail,
        )
        return pe

    def det_root(self, pe: ProposalEvaluation) -> Val:
        """Det_root: collapse B/U to classical decision using score margin, cost, and safety bias."""
        cls = pe.overall_class
        if pe.hard_fail:
            return Val.F
        if cls == Val.T:
            # reject if too costly under profile
            if pe.cost_score > self.profile.max_cost_for_accept:
                return Val.F
            return Val.T
        if cls == Val.F:
            return Val.F
        if cls == Val.B:
            if pe.t_score > (pe.f_score + self.profile.det_margin) and pe.cost_score <= self.profile.max_cost_for_accept:
                return Val.T
            return Val.F  # safety-biased tie-break
        # cls == U
        if pe.t_score >= (self.profile.theta + self.profile.det_margin) and pe.f_score < self.profile.theta and pe.cost_score <= self.profile.max_cost_for_accept:
            return Val.T
        return Val.F

    def det_class_select(self, evals: List[ProposalEvaluation]) -> Optional[ProposalEvaluation]:
        admissible = [e for e in evals if e.effective_class == Val.T and not e.hard_fail]
        if not admissible:
            return None
        admissible.sort(key=lambda e: (e.utility, e.t_score, -e.f_score, -e.cost_score), reverse=True)
        return admissible[0]

    def _proposal_fingerprint(self, pe: ProposalEvaluation) -> str:
        """Stable fingerprint for an accepted patch proposal (diff + ops + label)."""
        payload = {
            "label": pe.proposal.label,
            "ops": [dataclasses.asdict(op) for op in pe.proposal.patch_ops],
            "diff": pe.diff_stats.unified_diff,
        }
        return stable_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def _main_try_wrapper_count(self, code: str) -> int:
        # Count top-level __main__ wrappers that begin with an immediate indented try.
        pat = re.compile(r'if __name__ == ["\']__main__["\']:\n(?:[ \t]*)try:\n')
        return len(pat.findall(code))

    def _selection_guard_reason(self, pe: ProposalEvaluation, current_code: str) -> Optional[str]:
        """Return rejection reason if proposal should not be accepted, else None."""
        cand_hash = stable_hash(pe.candidate_code)
        if cand_hash in self.applied_candidate_hashes:
            return "duplicate candidate hash (already applied)"

        pfp = self._proposal_fingerprint(pe)
        if pfp in self.applied_patch_fingerprints:
            return "duplicate patch fingerprint (already applied)"

        # Anti-nesting guard for repeated __main__ wrappers (common heuristic failure mode)
        label_l = (pe.proposal.label or "").lower()
        desc_l = (pe.proposal.description or "").lower()
        wrapperish = (
            "wrap_main" in label_l
            or ("__main__" in (pe.diff_stats.unified_diff or "") and "except" in (pe.diff_stats.unified_diff or ""))
            or ("main" in desc_l and "try/except" in desc_l)
        )
        if wrapperish:
            # Fast exact-prefix guard catches repeated wrapper proposals.
            if 'if __name__ == "__main__":\n    try:\n' in current_code:
                return "anti-nesting guard: __main__ already wrapped"
            before_count = self._main_try_wrapper_count(current_code)
            after_count = self._main_try_wrapper_count(pe.candidate_code)
            nested_pat = re.compile(r'if __name__ == ["\']__main__["\']:\n(?:[ \t]*)try:\n(?:[ \t]*)try:\n')
            nested_in_candidate = bool(nested_pat.search(pe.candidate_code))
            already_wrapped = before_count >= 1
            if nested_in_candidate or (already_wrapped and after_count > before_count):
                return "anti-nesting guard: repeated __main__ try/except wrapper"

        # No repeated utility-drop acceptance rule for same patch label across iterations
        if (
            self.last_applied_label is not None
            and pe.proposal.label == self.last_applied_label
            and self.last_applied_utility is not None
            and pe.utility < (self.last_applied_utility - self.repeat_drop_margin)
        ):
            return (
                f"repeat-label utility drop ({pe.utility:.3f} < "
                f"{self.last_applied_utility:.3f} - {self.repeat_drop_margin:.3f})"
            )

        # Extra guard: stop repeated non-improving applications of same label after one streak
        if (
            self.last_applied_label is not None
            and pe.proposal.label == self.last_applied_label
            and self.last_applied_utility is not None
            and pe.utility <= (self.last_applied_utility + 1e-9)
            and self.repeat_label_nonimproving_streak >= 1
        ):
            return "repeat-label non-improving streak guard"

        return None

    def _select_with_guards(self, evals: List[ProposalEvaluation], current_code: str) -> Tuple[Optional[ProposalEvaluation], List[str]]:
        admissible = [e for e in evals if e.effective_class == Val.T and not e.hard_fail]
        if not admissible:
            return None, []
        admissible.sort(key=lambda e: (e.utility, e.t_score, -e.f_score, -e.cost_score), reverse=True)
        rejections: List[str] = []
        for pe in admissible:
            reason = self._selection_guard_reason(pe, current_code)
            if reason is None:
                return pe, rejections
            rejections.append(f"{pe.proposal.label}: {reason}")
        return None, rejections

    def _proposal_kernel_keys(self, pe: ProposalEvaluation) -> List[str]:
        keys: List[str] = []
        if pe.overall_class == Val.B:
            keys.append(f"proposal:{pe.proposal.label}")
        for ev in pe.world_evidence:
            if ev.skipped:
                continue
            if ev.classify(self.profile.theta) == Val.B:
                keys.append(f"world:{pe.proposal.label}:{ev.world}")
        return keys

    def _write_provenance_iteration(self, iteration: int, current_code: str, evals: List[ProposalEvaluation], selected: Optional[ProposalEvaluation]) -> None:
        if not self.provenance_dir:
            return
        idir = os.path.join(self.provenance_dir, f"iter_{iteration:02d}")
        os.makedirs(idir, exist_ok=True)
        with open(os.path.join(idir, "current_before.py"), "w", encoding="utf-8") as f:
            f.write(current_code)
        if selected:
            with open(os.path.join(idir, "selected_after.py"), "w", encoding="utf-8") as f:
                f.write(selected.candidate_code)
            with open(os.path.join(idir, "selected.diff"), "w", encoding="utf-8") as f:
                f.write(selected.diff_stats.unified_diff)

        report = []
        for pe in evals:
            report.append({
                "label": pe.proposal.label,
                "description": pe.proposal.description,
                "patch_applied": pe.patch_applied,
                "patch_notes": pe.patch_notes,
                "t_score": pe.t_score,
                "f_score": pe.f_score,
                "cost_score": pe.cost_score,
                "utility": pe.utility,
                "overall_class": pe.overall_class.name,
                "collapsed_class": pe.effective_class.name,
                "hard_fail": pe.hard_fail,
                "diff_stats": dataclasses.asdict(pe.diff_stats),
                "world_evidence": [dataclasses.asdict(w) for w in pe.world_evidence],
                "patch_ops": [dataclasses.asdict(op) for op in pe.proposal.patch_ops],
                "rationale": pe.proposal.rationale,
            })
        with open(os.path.join(idir, "evaluation_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def optimize(self, code: str, baseline: BaselineContext, args_ns: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
        current_code = code
        best_code = code
        best_utility_seen = -999.0
        sep = "=" * 72

        self.log(sep)
        self.log("Shadow-HoTT Optimizer v2")
        self.log(f"Task: {baseline.task}")
        self.log(f"Profile: {self.profile.name} | theta={self.profile.theta} | glut_tolerance={self.profile.glut_tolerance}")
        self.log(f"Iterations: {self.iterations} | LLM: {'Anthropic API' if self.llm.available else 'Heuristic fallback'}")
        self.log(f"Execution worlds: {'ENABLED' if self.runner.allow_exec else 'DISABLED'}")
        self.log(sep)

        for i in range(1, self.iterations + 1):
            self.state = SemanticShadowState(theta=self.profile.theta)
            self.state.provenance_iter = i
            self.log(f"\n-- Iteration {i}/{self.iterations} --")

            history_ctx = ""
            if self.history:
                last = self.history[-1]
                history_ctx = f"last_selected={last.selected_label}; last_signature={last.signature}; active_kernel={last.active_kernel[:6]}"

            proposals = self.llm.speculate_patches(current_code, baseline.task, history_context=history_ctx, max_proposals=self.max_proposals)
            self.log(f"[1] Proposed {len(proposals)} patches")

            evals: List[ProposalEvaluation] = []
            for p in proposals:
                pe = self._evaluate_proposal(p, current_code, baseline, args_ns)
                self.state.evals[p.label] = pe
                self.state.bi[p.label] = (pe.t_score, pe.f_score)
                evals.append(pe)

            self.state.refresh_cache()

            # Root classifications + determinization
            active_kernel: set[str] = set()
            for pe in evals:
                pe.collapsed_class = self.det_root(pe)
                active_kernel.update(self._proposal_kernel_keys(pe))

            sig = self.state.signature_counts()
            prev_sig = self.history[-1].signature if self.history else {"T": 0, "F": 0, "B": 0, "U": 0}
            shock = sum(abs(sig[k] - prev_sig.get(k, 0)) for k in ["T", "F", "B", "U"])
            glut_count = sig["B"]
            gap_count = sig["U"]
            resolved = sorted(self.prev_active_kernel - active_kernel)
            newly_active = sorted(active_kernel - self.prev_active_kernel)
            self.prev_active_kernel = set(active_kernel)

            self.log(f"[2] Signature [T,F,B,U] = {[sig['T'], sig['F'], sig['B'], sig['U']]} | shock={shock} | gluts={glut_count} | gaps={gap_count}")
            if active_kernel:
                self.log(f"[3] Active contradiction kernel |Kact|={len(active_kernel)}")
            if resolved:
                self.log(f"    Resolved from previous: {resolved[:6]}{' ...' if len(resolved)>6 else ''}")

            # Print proposal summaries
            for pe in sorted(evals, key=lambda x: x.utility, reverse=True):
                self.log(
                    f"    {pe.overall_class.symbol()}->{pe.effective_class.symbol()} {pe.proposal.label:<24} "
                    f"T={pe.t_score:.2f} F={pe.f_score:.2f} C={pe.cost_score:.2f} U={pe.utility:.2f} "
                    f"{'[HARD_FAIL]' if pe.hard_fail else ''}"
                )
                if args_ns.verbose_details:
                    for ev in pe.world_evidence:
                        if ev.skipped:
                            self.log(f"      - {ev.world}: [skipped] {ev.notes}")
                        else:
                            self.log(f"      - {ev.world}: {ev.classify(self.profile.theta).symbol()} t={ev.t_support:.2f} f={ev.f_support:.2f} :: {ev.notes}")

            selected, guard_rejections = self._select_with_guards(evals, current_code)
            if glut_count > self.profile.glut_tolerance:
                self.log(f"[4] Det_root pressure: glut_count {glut_count} > tolerance {self.profile.glut_tolerance}")
            else:
                self.log(f"[4] Root state within glut tolerance")
            if guard_rejections:
                shown = guard_rejections[:6]
                self.log(f"    Selection guards rejected {len(guard_rejections)} candidate(s): {shown}{' ...' if len(guard_rejections) > 6 else ''}")

            if selected is None:
                self.log("[5] No admissible patch selected; stopping early.")
                tel = IterationTelemetry(
                    iteration=i,
                    signature=sig,
                    shock_l1=shock,
                    glut_count=glut_count,
                    gap_count=gap_count,
                    active_kernel=sorted(active_kernel),
                    resolved_from_prev=resolved,
                    newly_active=newly_active,
                    selected_label=None,
                    selected_utility=None,
                    proposals_evaluated=len(evals),
                )
                self.history.append(tel)
                self._write_provenance_iteration(i, current_code, evals, selected)
                break

            # Apply selected patch (Det_class output) and validate final syntax again (rollback guard)
            post_syntax_ok = True
            try:
                ast.parse(selected.candidate_code)
            except Exception:
                post_syntax_ok = False

            if not post_syntax_ok:
                self.log(f"[5] Selected patch failed final syntax guard; rollback. ({selected.proposal.label})")
                tel = IterationTelemetry(
                    iteration=i,
                    signature=sig,
                    shock_l1=shock,
                    glut_count=glut_count,
                    gap_count=gap_count,
                    active_kernel=sorted(active_kernel),
                    resolved_from_prev=resolved,
                    newly_active=newly_active,
                    selected_label=None,
                    selected_utility=None,
                    proposals_evaluated=len(evals),
                )
                self.history.append(tel)
                self._write_provenance_iteration(i, current_code, evals, None)
                break

            self.log(f"[5] Applying patch: {selected.proposal.label} — {selected.proposal.description}")
            current_code = selected.candidate_code

            # Persist acceptance fingerprints / utility history for future guard checks
            self.applied_candidate_hashes.add(stable_hash(selected.candidate_code))
            self.applied_patch_fingerprints.add(self._proposal_fingerprint(selected))
            if self.last_applied_label == selected.proposal.label and self.last_applied_utility is not None:
                if selected.utility <= (self.last_applied_utility + 1e-9):
                    self.repeat_label_nonimproving_streak += 1
                else:
                    self.repeat_label_nonimproving_streak = 0
            else:
                self.repeat_label_nonimproving_streak = 0
            self.last_applied_label = selected.proposal.label
            self.last_applied_utility = selected.utility

            if selected.utility > best_utility_seen:
                best_utility_seen = selected.utility
                best_code = current_code

            tel = IterationTelemetry(
                iteration=i,
                signature=sig,
                shock_l1=shock,
                glut_count=glut_count,
                gap_count=gap_count,
                active_kernel=sorted(active_kernel),
                resolved_from_prev=resolved,
                newly_active=newly_active,
                selected_label=selected.proposal.label,
                selected_utility=selected.utility,
                proposals_evaluated=len(evals),
            )
            self.history.append(tel)
            self._write_provenance_iteration(i, current_code, evals, selected)

        summary = {
            "profile": self.profile.name,
            "iterations_requested": self.iterations,
            "iterations_run": len(self.history),
            "edits_applied": sum(1 for h in self.history if h.selected_label),
            "total_proposals_evaluated": sum(h.proposals_evaluated for h in self.history),
            "glut_collapses_triggered": sum(1 for h in self.history if h.glut_count > self.profile.glut_tolerance),
            "final_code_hash": stable_hash(current_code),
            "best_code_hash": stable_hash(best_code),
            "telemetry": [dataclasses.asdict(h) for h in self.history],
        }
        return current_code, summary


# ============================================================
# 10. Baseline measurement (optional command worlds)
# ============================================================
def collect_baseline_context(code: str, file_path: Optional[str], repo_path: Optional[str], task: str) -> BaselineContext:
    return BaselineContext(code=code, file_path=file_path, repo_path=repo_path, task=task)


# ============================================================
# 11. Demo Code
# ============================================================
DEMO_CODE = '''\
def process_data(data):
    results = []
    for item in data:
        value = item["value"]
        result = value * 2 + 10
        results.append(result)
    return results

def save_results(results, filename):
    f = open(filename, "w")
    for r in results:
        f.write(str(r) + "\\n")
    f.close()

if __name__ == "__main__":
    data = [{"value": 1}, {"value": 2}, {"value": 3}]
    results = process_data(data)
    save_results(results, "output.txt")
    print("Done:", results)
'''


# ============================================================
# 12. CLI / main
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shadow-HoTT LLM Self-Edit Optimizer v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              %(prog)s --demo
              %(prog)s --file mycode.py --task "Add error handling and use context managers"
              %(prog)s --file app.py --task "Refactor for clarity" --profile refactor --iterations 4
              %(prog)s --file app.py --task "Add type hints" --typecheck-cmd "mypy {candidate}" --allow-exec
              %(prog)s --file app.py --task "Harden subprocess usage" --profile security --allow-exec \\
                  --sandbox-cmd "firejail --quiet --net=none --private"
            """
        ),
    )

    src = parser.add_argument_group("Source")
    src.add_argument("--file", "-f", help="Python file to optimize")
    src.add_argument("--task", "-t", help="Task description / optimization goal")
    src.add_argument("--repo", help="Optional repo root (for command worlds working on project context)")
    src.add_argument("--output", "-o", help="Output file (default: overwrite input file, or print in demo)")
    src.add_argument("--demo", action="store_true", help="Run built-in demo")

    engine = parser.add_argument_group("Optimizer")
    engine.add_argument("--iterations", "-n", type=int, default=3, help="Number of iterations")
    engine.add_argument("--max-proposals", type=int, default=4, help="Max proposals per iteration")
    engine.add_argument("--profile", choices=["auto", "bugfix", "refactor", "security", "typing", "docs", "perf", "prod_safe", "race"], default="auto")
    engine.add_argument("--theta", type=float, default=None, help="Override profile threshold theta")
    engine.add_argument("--glut-tolerance", type=int, default=None, help="Override profile glut tolerance")
    engine.add_argument("--det-margin", type=float, default=None, help="Override profile determinization margin")
    engine.add_argument("--cost-weight", type=float, default=None, help="Override profile cost weight")
    engine.add_argument("--max-cost-for-accept", type=float, default=None, help="Override profile acceptance cost cap")
    engine.add_argument("--profile-json", help="Path to JSON profile override file (theta/weights/hard_worlds/etc.)")
    engine.add_argument("--set-weight", action="append", help="Override one world weight, e.g. 'test_cmd:t=0.95,f=0.9' or 'llm_judge:0.7,0.25' (repeatable)")
    engine.add_argument("--set-hard-world", action="append", help="Add a world to hard_worlds (repeatable)")
    engine.add_argument("--unset-hard-world", action="append", help="Remove a world from hard_worlds (repeatable)")
    engine.add_argument("--show-profile", action="store_true", help="Print resolved profile JSON before optimization")

    llm_group = parser.add_argument_group("LLM")
    llm_group.add_argument("--anthropic-api-key", help="Anthropic API key (defaults to ANTHROPIC_API_KEY env)")
    llm_group.add_argument("--anthropic-model", default="claude-sonnet-4-20250514", help="Anthropic model name")

    exec_group = parser.add_argument_group("Execution worlds (disabled by default)")
    exec_group.add_argument("--allow-exec", action="store_true", help="Enable runtime/command execution worlds")
    exec_group.add_argument("--sandbox-cmd", help="Optional sandbox wrapper command prefix (e.g., firejail / bwrap / docker exec)")
    exec_group.add_argument("--cpu-seconds", type=int, default=None, help="Per-command CPU time limit (Unix only)")
    exec_group.add_argument("--memory-mb", type=int, default=None, help="Per-command memory limit MB (best effort, Unix only)")
    exec_group.add_argument("--runtime-smoke", action="store_true", help="Run candidate as script in a smoke-test world")
    exec_group.add_argument("--runtime-timeout", type=int, default=5)
    exec_group.add_argument("--test-cmd", help="External test command template, e.g. 'pytest -q' or 'python -m pytest -q'")
    exec_group.add_argument("--test-timeout", type=int, default=30)
    exec_group.add_argument("--lint-cmd", help="External lint command template, e.g. 'ruff check {candidate}'")
    exec_group.add_argument("--lint-timeout", type=int, default=20)
    exec_group.add_argument("--typecheck-cmd", help="External typecheck command template, e.g. 'mypy {candidate}'")
    exec_group.add_argument("--typecheck-timeout", type=int, default=30)
    exec_group.add_argument("--custom-cmd", action="append", help="Additional command world template (repeatable)")
    exec_group.add_argument("--custom-timeout", type=int, default=20)

    out = parser.add_argument_group("Output / Provenance")
    out.add_argument("--provenance-dir", help="Directory to write iteration snapshots + JSON evidence")
    out.add_argument("--report-json", help="Write final summary report JSON here")
    out.add_argument("--print-final-code", action="store_true", help="Print final code to stdout (non-demo too)")
    out.add_argument("--git-commit-final", action="store_true", help="Attempt git add+commit of final output (if in a repo)")
    out.add_argument("--git-commit-message", default=None, help="Custom git commit message for --git-commit-final")

    ui = parser.add_argument_group("UI")
    ui.add_argument("--quiet", "-q", action="store_true", help="Minimal logging")
    ui.add_argument("--verbose-details", action="store_true", help="Print per-world evidence details")

    return parser


def maybe_git_commit_final(file_path: str, repo_path: Optional[str], message: Optional[str], verbose: bool = True) -> Tuple[bool, str]:
    try:
        target = os.path.abspath(file_path)
        repo = repo_path or os.path.dirname(target)
        # detect repo root
        rr = subprocess.run(["git", "-C", repo, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if rr.returncode != 0:
            return False, "Not a git repo (or git unavailable)"
        root = rr.stdout.strip()
        rel = os.path.relpath(target, root)
        subprocess.run(["git", "-C", root, "add", rel], check=False)
        msg = message or f"Shadow-HoTT optimizer: apply final patch to {rel}"
        cr = subprocess.run(["git", "-C", root, "commit", "-m", msg], capture_output=True, text=True)
        if cr.returncode == 0:
            return True, cr.stdout.strip() or "Committed"
        # no changes / commit failure are both returned here
        return False, (cr.stderr or cr.stdout or "git commit failed").strip()
    except Exception as e:
        return False, str(e)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.demo:
        code = DEMO_CODE
        task = args.task or "Add error handling, type hints, and use context managers for file operations"
        file_path = None
    elif args.file and args.task:
        file_path = os.path.abspath(args.file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            return 1
        task = args.task
    else:
        parser.print_help()
        print("\nError: provide --demo or both --file and --task", file=sys.stderr)
        return 1

    profiles = default_profiles()
    profile_name = args.profile if args.profile != "auto" else infer_profile_name(task)
    profile = dataclasses.replace(profiles[profile_name])
    # detach nested dicts from shared defaults
    profile.world_weights = {k: {"t": float(v.get("t", 0.0)), "f": float(v.get("f", 0.0))} for k, v in profile.world_weights.items()}
    apply_runtime_profile_overrides(profile, args)
    if args.show_profile and not args.quiet:
        print("Resolved profile:")
        print(json.dumps(profile_to_dict(profile), indent=2, sort_keys=True))

    llm = LLMInterface(api_key=args.anthropic_api_key, model=args.anthropic_model)
    runner = CommandRunner(
        allow_exec=args.allow_exec,
        sandbox_cmd=args.sandbox_cmd,
        timeout_default=10,
        cpu_seconds=args.cpu_seconds,
        memory_mb=args.memory_mb,
    )

    provenance_dir = args.provenance_dir
    if provenance_dir is None and not args.demo:
        # Default provenance next to output file if user requested report details via verbose_details.
        if args.verbose_details:
            provenance_dir = os.path.join(tempfile.gettempdir(), f"shadow_hott_provenance_{now_ts()}")

    optimizer = ShadowHoTTOptimizerV2(
        profile=profile,
        llm=llm,
        runner=runner,
        iterations=args.iterations,
        max_proposals=args.max_proposals,
        verbose=not args.quiet,
        provenance_dir=provenance_dir,
    )

    baseline = collect_baseline_context(
        code=code,
        file_path=file_path,
        repo_path=os.path.abspath(args.repo) if args.repo else None,
        task=task,
    )

    final_code, summary = optimizer.optimize(code, baseline, args)

    # Output routing
    wrote_path = None
    if args.output:
        wrote_path = args.output
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_code)
        if not args.quiet:
            print(f"\nOptimized code written to: {args.output}")
    elif args.demo:
        print("\n" + "=" * 72)
        print("FINAL OPTIMIZED CODE")
        print("=" * 72)
        print(final_code)
    elif file_path:
        wrote_path = file_path
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_code)
        if not args.quiet:
            print(f"\nOptimized code written back to: {file_path}")

    if args.print_final_code and not args.demo:
        print("\n" + "=" * 72)
        print("FINAL OPTIMIZED CODE")
        print("=" * 72)
        print(final_code)

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if not args.quiet:
            print(f"Summary report written to: {args.report_json}")

    if optimizer.provenance_dir and not args.quiet:
        print(f"Provenance snapshots: {optimizer.provenance_dir}")

    if args.git_commit_final and wrote_path:
        ok, msg = maybe_git_commit_final(wrote_path, args.repo, args.git_commit_message, verbose=not args.quiet)
        if not args.quiet:
            print(("Git commit: " if ok else "Git commit skipped/failed: ") + msg)

    # Brief final summary for CLI users
    if not args.quiet:
        print("\n" + "=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(json.dumps({k: v for k, v in summary.items() if k != "telemetry"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

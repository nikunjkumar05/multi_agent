"""Task classifier — routes incoming tasks to the right category.

Uses rule-based keyword matching first (fast, no API calls).
Falls back to LLM classification for ambiguous cases (confidence < threshold).

Task types:
  - code_generation: write new code
  - code_review: review existing code
  - refactoring: restructure code
  - debugging: find and fix bugs
  - documentation: write docs/comments
  - testing: write tests
  - explanation: explain code
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


class TaskType(str, Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    EXPLANATION = "explanation"


@dataclass
class ClassificationResult:
    """Result of task classification."""

    task_type: TaskType
    confidence: float  # 0.0 to 1.0
    method: str  # "rule_based" or "llm"


# Keywords for each task type (order matters — first match wins)
_TASK_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.CODE_GENERATION: [
        "write", "create", "implement", "build", "develop", "generate",
        "add a function", "make a script", "code a", "develop a",
        "write a function", "write a class", "write a module",
        "create a file", "implement a", "build a", "generate a",
    ],
    TaskType.CODE_REVIEW: [
        "review", "check", "audit", "inspect", "analyze this code",
        "code review", "look at", "evaluate", "assess", "critique",
        "review this", "check this code", "audit this",
    ],
    TaskType.REFACTORING: [
        "refactor", "restructure", "reorganize", "clean up", "simplify",
        "optimize", "improve the structure", "rewrite", "rework",
        "extract method", "extract function", "rename", "move",
        "refactor this", "clean this up", "simplify this",
    ],
    TaskType.DEBUGGING: [
        "bug", "error", "fix", "broken", "crash", "issue", "problem",
        "not working", "failing", "fails", "debug", "troubleshoot",
        "fix this bug", "fix the error", "why is this broken",
        "find the bug", "debug this", "what's wrong",
    ],
    TaskType.DOCUMENTATION: [
        "document", "docstring", "comment", "readme", "docs",
        "write documentation", "add comments", "explain the code",
        "write a docstring", "write docs", "add documentation",
    ],
    TaskType.TESTING: [
        "test", "spec", "assert", "mock", "pytest", "unittest",
        "write tests", "add tests", "test this", "write a test",
        "create tests", "unit test", "integration test",
        "write pytest", "add coverage",
    ],
    TaskType.EXPLANATION: [
        "explain", "what does", "how does", "describe", "summarize",
        "tell me about", "what is", "how do I", "why does",
        "explain this", "what does this code do",
    ],
}

# Scoring weights for matched keywords
_EXACT_MATCH_WEIGHT = 1.0
_PARTIAL_MATCH_WEIGHT = 0.5
_PHRASE_MATCH_WEIGHT = 1.5  # multi-word phrases are stronger signals


def classify_task(prompt: str, context: dict | None = None) -> ClassificationResult:
    """Classify a task based on its prompt text.

    Uses rule-based keyword matching. Fast, no API calls.

    Args:
        prompt: The task instruction text.
        context: Optional context (language, files, etc.)

    Returns:
        ClassificationResult with task_type, confidence, and method.
    """
    prompt_lower = prompt.lower().strip()

    # Score each task type
    scores: dict[TaskType, float] = {}

    for task_type, keywords in _TASK_KEYWORDS.items():
        score = 0.0

        for keyword in keywords:
            if " " in keyword:
                # Phrase match — stronger signal
                if keyword in prompt_lower:
                    score += _PHRASE_MATCH_WEIGHT
            else:
                # Single keyword — check with word boundaries
                pattern = r"\b" + re.escape(keyword) + r"\b"
                if re.search(pattern, prompt_lower):
                    score += _EXACT_MATCH_WEIGHT

        # Bonus: if file context matches task type
        if context:
            files = context.get("files", [])
            if task_type == TaskType.TESTING:
                # Files named test_* or *_test.py get bonus
                test_files = [f for f in files if "test" in f.lower()]
                score += len(test_files) * 0.3
            if task_type == TaskType.DOCUMENTATION:
                # README or docs files get bonus
                doc_files = [f for f in files if any(x in f.lower() for x in ["readme", "doc", ".md"])]
                score += len(doc_files) * 0.3

        scores[task_type] = score

    # Find the best match
    if not any(s > 0 for s in scores.values()):
        # No keywords matched — default to code_generation with low confidence
        return ClassificationResult(
            task_type=TaskType.CODE_GENERATION,
            confidence=0.3,
            method="rule_based",
        )

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Calculate confidence (0.0 to 1.0)
    total_score = sum(scores.values())
    if total_score > 0:
        confidence = best_score / total_score
    else:
        confidence = 0.3

    # Cap confidence at 0.95 for rule-based (never be too certain)
    confidence = min(confidence, 0.95)

    # Check if confidence is low (ambiguous task)
    if confidence < 0.4:
        log.info(
            "Low confidence classification: type=%s confidence=%.2f prompt='%s'",
            best_type.value, confidence, prompt[:50],
        )

    return ClassificationResult(
        task_type=best_type,
        confidence=round(confidence, 2),
        method="rule_based",
    )


def classify_task_with_llm(
    prompt: str,
    context: dict | None = None,
    llm_call=None,
) -> ClassificationResult:
    """Classify using LLM for ambiguous cases.

    Falls back to rule-based if LLM is not available or fails.

    Args:
        prompt: The task instruction text.
        context: Optional context.
        llm_call: Async function that takes a prompt and returns text.

    Returns:
        ClassificationResult with task_type, confidence, and method.
    """
    # Try rule-based first
    rule_result = classify_task(prompt, context)

    # If confidence is high enough, skip LLM
    if rule_result.confidence >= 0.5:
        return rule_result

    # If no LLM available, return rule-based result
    if llm_call is None:
        return rule_result

    # Ask LLM to classify
    llm_prompt = f"""Classify this coding task into exactly one category.

Task: "{prompt}"

Categories:
- code_generation: writing new code
- code_review: reviewing existing code
- refactoring: restructuring code
- debugging: finding and fixing bugs
- documentation: writing docs/comments
- testing: writing tests
- explanation: explaining code

Reply with ONLY the category name, nothing else."""

    try:
        response = llm_call(llm_prompt)
        response_text = response.strip().lower()

        # Parse LLM response
        for task_type in TaskType:
            if task_type.value in response_text:
                return ClassificationResult(
                    task_type=task_type,
                    confidence=0.7,  # LLM-based confidence
                    method="llm",
                )

        # LLM gave invalid response — fall back to rule-based
        return rule_result

    except Exception as e:
        log.warning("LLM classification failed: %s", e)
        return rule_result


def get_task_type_label(task_type: TaskType | str) -> str:
    """Get a human-readable label for a task type."""
    labels = {
        TaskType.CODE_GENERATION: "Code Generation",
        TaskType.CODE_REVIEW: "Code Review",
        TaskType.REFACTORING: "Refactoring",
        TaskType.DEBUGGING: "Debugging",
        TaskType.DOCUMENTATION: "Documentation",
        TaskType.TESTING: "Testing",
        TaskType.EXPLANATION: "Explanation",
        "code_generation": "Code Generation",
        "code_review": "Code Review",
        "refactoring": "Refactoring",
        "debugging": "Debugging",
        "documentation": "Documentation",
        "testing": "Testing",
        "explanation": "Explanation",
    }
    if isinstance(task_type, str):
        return labels.get(task_type, task_type)
    return labels.get(task_type, task_type.value)

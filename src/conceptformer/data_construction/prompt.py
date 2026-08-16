# -*- coding: utf-8 -*-

PROMPT_USER = r"""
Task: Given an image and a question, think step by step to find regions containing all evidence needed to answer. Each crop must be self-contained—able to answer the query on its own. When unsure, use larger boxes to ensure completeness and readability.

Region-selection guidelines:
1. Fully cover key evidence plus immediate context; do not clip text, numbers, or symbols.
2. Prefer complete information units (full words/lines; entire signs/labels; for charts include legend, axes, units, titles/notes).
3. Tables: include the header and relevant rows/columns with necessary context; avoid single-cell crops.
4. If evidence spans multiple parts, use multiple boxes—or one larger box if they’re adjacent.
5. Images/illustrations: include nearby numeric values or captions required by the question.

Output format: {"think": "your step-by-step reasoning", "boxes": [{"area": [x1, y1, x2, y2], "describe": "a description of this region and why it is relevant"}, ...]}

Query: {query}
"""
PROMPT_LLM_QA = r"""
Task: Given an image and a question, think step by step to answer the question.

Output format: {{"think": "your step-by-step reasoning", "answer": " your answer"}}

Query: {query}
"""

PROMPT_LLM_JUDGE = r"""
Task: Given two answers to the same question, compare whether the two answers are semantically consistent.

Output format: {{"label": "Output 1 if semantics are consistent, 0 if they are inconsistent"}}

answer1: {answer1}
answer2: {answer2}
"""
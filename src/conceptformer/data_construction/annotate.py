"""Annotate query-relevant image regions through an OpenAI-compatible vision API."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from openai import OpenAI

from .prompt import PROMPT_USER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Query parquet file.")
    parser.add_argument("--corpus-root", required=True, type=Path, help="Root for document IDs.")
    parser.add_argument("--output", required=True, type=Path, help="Resumable JSONL output.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("ANNOTATION_MODEL"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def question_from(row: dict[str, Any]) -> str:
    text = str(row.get("query") or row.get("query_text") or "").strip()
    marker = "Query:"
    return text.rsplit(marker, 1)[-1].strip() if marker in text else text


def positive_ids(row: dict[str, Any]) -> list[str]:
    value = row.get("relevant_doc_ids") or []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    return [str(item) for item in value]


def parse_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    value = json.loads(text[start : end + 1])
    boxes = value.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("response JSON does not contain a boxes list")
    return value


def annotate_one(client: OpenAI, args: argparse.Namespace, row: dict[str, Any], doc_id: str):
    image_path = args.corpus_root / doc_id
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    prompt = PROMPT_USER.replace("{query}", question_from(row))
    content = [
        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
        {"type": "text", "text": prompt},
    ]
    error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": content}],
                temperature=args.temperature,
            )
            result = parse_response(response.choices[0].message.content or "")
            return {
                "query_id": str(row["query_id"]),
                "question": question_from(row),
                "document_id": doc_id,
                "bbox_2d": [item.get("area") for item in result["boxes"]],
                "descriptions": [item.get("describe", "") for item in result["boxes"]],
            }
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt < args.max_retries:
                time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"annotation failed after retries: {error}")


def main() -> None:
    args = parse_args()
    if not args.base_url or not args.model or not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY, OPENAI_BASE_URL, and ANNOTATION_MODEL.")
    client = OpenAI(base_url=args.base_url, api_key=os.environ["OPENAI_API_KEY"])
    rows = pq.read_table(args.input).to_pylist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            done.add((str(value["query_id"]), str(value["document_id"])))
    jobs = [
        (row, doc_id)
        for row in rows
        for doc_id in positive_ids(row)
        if (str(row["query_id"]), doc_id) not in done
    ]
    lock = threading.Lock()
    with args.output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = {pool.submit(annotate_one, client, args, row, doc_id): (row, doc_id) for row, doc_id in jobs}
        for future in as_completed(futures):
            row, doc_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"failed query_id={row['query_id']} document_id={doc_id}: {exc}")
                continue
            with lock:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()

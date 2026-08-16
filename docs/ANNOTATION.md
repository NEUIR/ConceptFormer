# Evidence Annotation

The annotation input is an image and its real retrieval question. The API returns one or
more evidence regions with concise descriptions.

## Prompt

```text
Task: Given an image and a question, think step by step to find regions containing all
evidence needed to answer. Each crop must be self-contained, able to answer the query on
its own. When unsure, use larger boxes to ensure completeness and readability.

Region-selection guidelines:
1. Fully cover key evidence plus immediate context; do not clip text, numbers, or symbols.
2. Prefer complete information units. For charts include legend, axes, units, titles, and
   notes when needed.
3. For tables, include the header and relevant rows or columns; avoid isolated cells.
4. If evidence spans multiple parts, use multiple boxes, or one larger adjacent box.
5. For images and illustrations, include nearby values or captions required by the query.

Return JSON only:
{"think":"...","boxes":[{"area":[x1,y1,x2,y2],"describe":"..."}]}

Query: {query}
```

The source of truth is `src/conceptformer/data_construction/prompt.py`.

## API Annotation

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_BASE_URL='https://your-openai-compatible-endpoint/v1'
export ANNOTATION_MODEL='your-vision-model'

python -m conceptformer.data_construction.annotate \
  --input data/eval/infovqa/query.parquet \
  --corpus-root data/opendoc_corpus \
  --output annotations/infovqa.jsonl \
  --temperature 0.2 \
  --concurrency 16
```

Recommended settings are temperature `0.2`, concurrency `16`, JSON validation, bounded
retries, and a resumable JSONL output. Never commit API keys or raw request logs containing
credentials.

## Coordinate Convention

Boxes use `[x1, y1, x2, y2]` in the annotation input image. `image_size` and `input_size`
are retained so the training collator can map boxes back to visual patch coordinates.

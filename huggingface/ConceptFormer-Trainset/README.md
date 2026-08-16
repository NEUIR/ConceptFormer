---
language:
- en
license: other
task_categories:
- visual-document-retrieval
pretty_name: ConceptFormer Trainset
size_categories:
- 10K<n<100K
---

# ConceptFormer Trainset

Grounded visual-document retrieval training data for ConceptFormer. The release contains
37,966 rows and references positive images by `relevant_doc_ids`. Image bytes are resolved
from the upstream OpenDocVQA corpus and are not duplicated here.

```python
from datasets import load_dataset
dataset = load_dataset("hmhm1229/ConceptFormer-Trainset", split="train")
```

Each row contains `query_id`, `query_text`, `relevant_doc_ids`, `describe`, `bbox_2d`,
`image_size`, `input_size`, `answers`, and `dataset_names`. Bounding boxes are stored as
JSON-encoded `xyxy` coordinates. See the repository `docs/DATA.md` and
`docs/ANNOTATION.md` for schema and construction details.

## License and Provenance

Derived annotations are released for research use. Images and source questions retain the
licenses and terms of their upstream datasets. Users are responsible for complying with
those terms.

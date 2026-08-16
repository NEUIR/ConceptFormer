# Release Checklist

1. Confirm the public Hugging Face namespace, citation, and contact metadata.
2. Review every upstream dataset license before uploading images or derived annotations.
3. Run `pytest -q`, `python -m compileall -q src scripts tests`, and
   `python scripts/validate_release.py`.
4. Run `python scripts/build_manifest.py huggingface` after the final file change.
5. Verify both model tokenizers report exactly one project token: `<|lcon|>`.
6. Verify adapter configs use public base-model IDs and contain no local paths.
7. Search for credentials and private paths before publishing.
8. Upload the four directories under `huggingface/` as separate repositories.
9. Run the six-dataset evaluation from a clean environment and archive `metrics.csv`.

Suggested checks:

```bash
rg -n -i 'api[_-]?key|secret|password|sk-[a-z0-9]+' .
rg -n '/mnt1|/home/' .
python scripts/build_manifest.py huggingface
```

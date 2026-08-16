from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from conceptformer.retriever.arguments import DataArguments
from conceptformer.retriever.latent_concepts import parse_bboxes

import logging
logger = logging.getLogger(__name__)


# Candidate dataset columns that can carry per-query bbox supervision. The
# auto-detection runs in this priority order; ``--bbox_field`` overrides it.
_BBOX_FIELD_CANDIDATES = (
    'model_boxes', 'bbox', 'bbox_2d', 'bboxes', 'boxes', 'area', 'regions',
)

_BBOX_IMAGE_SIZE_FIELD_CANDIDATES = (
    'image_size', 'original_image_size', 'orig_size',
)

_BBOX_BY_DOC_ID_FIELD_CANDIDATES = (
    'bbox_by_doc_id', 'bboxes_by_doc_id', 'boxes_by_doc_id',
)

_BBOX_IMAGE_SIZE_BY_DOC_ID_FIELD_CANDIDATES = (
    'image_size_by_doc_id', 'bbox_image_size_by_doc_id',
)

_BBOX_DOC_ID_FIELD_CANDIDATES = (
    'bbox_doc_id', 'bbox_image_doc_id', 'bbox_source_doc_id',
)


def _detect_bbox_field(features, override: str = None) -> str:
    if override:
        if override in features:
            return override
        logger.warning(
            "Configured bbox_field=%s not found in dataset columns %s; "
            "falling back to auto-detection.", override, list(features),
        )
    for name in _BBOX_FIELD_CANDIDATES:
        if name in features:
            return name
    return None


def _detect_bbox_image_size_field(features) -> str:
    for name in _BBOX_IMAGE_SIZE_FIELD_CANDIDATES:
        if name in features:
            return name
    return None


def _detect_bbox_by_doc_id_field(features) -> str:
    for name in _BBOX_BY_DOC_ID_FIELD_CANDIDATES:
        if name in features:
            return name
    return None


def _detect_bbox_image_size_by_doc_id_field(features) -> str:
    for name in _BBOX_IMAGE_SIZE_BY_DOC_ID_FIELD_CANDIDATES:
        if name in features:
            return name
    return None


def _detect_bbox_doc_id_field(features) -> str:
    for name in _BBOX_DOC_ID_FIELD_CANDIDATES:
        if name in features:
            return name
    return None


def _parse_json_mapping(raw):
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode('utf-8', errors='ignore')
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in {'nan', 'none', 'null', '{}'}:
            return {}
        try:
            raw = json.loads(s)
        except Exception:
            try:
                import ast
                raw = ast.literal_eval(s)
            except Exception:
                return {}
    try:
        import numpy as _np
        if isinstance(raw, _np.ndarray):
            raw = raw.tolist()
    except Exception:
        pass
    return raw if isinstance(raw, dict) else {}


def _parse_image_size(raw):
    """Return (width, height) from a dataset size cell, or None."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode('utf-8', errors='ignore')
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in {'nan', 'none', 'null', '[]'}:
            return None
        try:
            raw = json.loads(s)
        except Exception:
            try:
                import ast
                raw = ast.literal_eval(s)
            except Exception:
                return None
    try:
        import numpy as _np
        if isinstance(raw, _np.ndarray):
            raw = raw.tolist()
    except Exception:
        pass
    if isinstance(raw, dict):
        if 'width' in raw and 'height' in raw:
            raw = [raw['width'], raw['height']]
        elif 'w' in raw and 'h' in raw:
            raw = [raw['w'], raw['h']]
        else:
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        w = float(raw[0])
        h = float(raw[1])
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return [w, h]


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        
        if self.data_args.dataset_path and (os.path.exists(self.data_args.dataset_path) or 
                                            os.path.exists(os.path.dirname(self.data_args.dataset_path))):
            if os.path.isdir(self.data_args.dataset_path):
                data_files = os.path.join(self.data_args.dataset_path, '*.parquet')
                if not any(f.endswith('.parquet') for f in os.listdir(self.data_args.dataset_path)):
                    data_files = os.path.join(self.data_args.dataset_path, '*.json')
                self.train_data = load_dataset(
                    'parquet' if '*.parquet' in data_files else 'json',
                    data_files=data_files,
                    split=self.data_args.dataset_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )
            else:
                self.train_data = load_dataset(
                    'parquet' if self.data_args.dataset_path.endswith('.parquet') else 'json',
                    data_files=self.data_args.dataset_path,
                    split=self.data_args.dataset_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )
        else:
            self.train_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )
            
        if not self.data_args.pretrain:
            corpus_dir = self.data_args.corpus_path.replace('/*.parquet', '').replace('*.parquet', '').replace('/*', '').replace('*', '')
            if self.data_args.corpus_path and (os.path.exists(corpus_dir) or os.path.isdir(os.path.dirname(self.data_args.corpus_path))):
                if os.path.isdir(corpus_dir):
                    corpus_files = os.path.join(corpus_dir, '*.parquet')
                else:
                    corpus_files = self.data_args.corpus_path
                    
                self.corpus  = load_dataset(
                    'parquet',
                    data_files=corpus_files,
                    split=self.data_args.corpus_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )
            else:
                self.corpus  = load_dataset(
                    self.data_args.corpus_name,
                    self.data_args.corpus_config,
                    data_files=self.data_args.corpus_path,
                    split=self.data_args.corpus_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )

            from datasets.features import Image as ImageFeature
            if 'image' in self.corpus.features:
                new_features = self.corpus.features.copy()
                new_features['image'] = ImageFeature(decode=True)
                self.corpus = self.corpus.cast(new_features)


            self.docid2idx = {}
            if 'doc_id' in self.corpus.features:
                for idx, docid in enumerate(self.corpus['doc_id']):
                    self.docid2idx[str(docid)] = idx
            else:
                for idx in range(len(self.corpus)):
                    self.docid2idx[str(idx)] = idx

        # Auto-detect bbox column for dynamic LCON visual-patch supervision.
        self.bbox_field = _detect_bbox_field(
            self.train_data.features,
            override=getattr(data_args, 'bbox_field', None),
        )
        self.bbox_image_size_field = _detect_bbox_image_size_field(self.train_data.features)
        self.bbox_by_doc_id_field = _detect_bbox_by_doc_id_field(self.train_data.features)
        self.bbox_image_size_by_doc_id_field = _detect_bbox_image_size_by_doc_id_field(
            self.train_data.features
        )
        self.bbox_doc_id_field = _detect_bbox_doc_id_field(self.train_data.features)
        if self.bbox_field is None:
            if self.bbox_by_doc_id_field is None:
                logger.info("[latent concept] No bbox column detected; LCON MSE supervision will be unavailable.")
        else:
            logger.info("[latent concept] Using bbox column '%s' for LCON MSE supervision.", self.bbox_field)
            if self.bbox_image_size_field is not None:
                logger.info(
                    "[latent concept] Using image-size column '%s' to scale bboxes into Qwen's resized grid.",
                    self.bbox_image_size_field,
                )
        if self.bbox_by_doc_id_field is not None:
            logger.info(
                "[latent concept] Using per-doc bbox column '%s' for LCON supervision.",
                self.bbox_by_doc_id_field,
            )
            if self.bbox_image_size_by_doc_id_field is not None:
                logger.info(
                    "[latent concept] Using per-doc image-size column '%s' to scale bboxes.",
                    self.bbox_image_size_by_doc_id_field,
                )
        if self.bbox_doc_id_field is not None:
            logger.info(
                "[latent concept] Using bbox doc-id column '%s' to keep positive image sampling aligned.",
                self.bbox_doc_id_field,
            )

        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query_text = group.get('query_text') or group.get('query')

        # Training data in the released parquet has used both column names
        # across revisions. Treat either as the textual-description teacher.
        describe = group.get('description') or group.get('describe') or ''
        if 'd_exist' in group:
            d_exist = group['d_exist']
            describe = describe if d_exist == 'yes' else ''
        else:
            d_exist = 'yes' if describe and str(describe).strip() else 'no'
        
        if self.data_args.pretrain:
            image = group['image']
            selected_docid = None
            return query_text, image
        else:
            relevant_docids = group['relevant_doc_ids']
            selected_docid = None

            if not relevant_docids or len(relevant_docids) == 0:
                relevant_doc_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
            else:
                bbox_doc_id = None
                if self.bbox_doc_id_field is not None and self.bbox_by_doc_id_field is None:
                    raw_bbox_doc_id = group.get(self.bbox_doc_id_field)
                    if isinstance(raw_bbox_doc_id, (list, tuple)):
                        raw_bbox_doc_id = raw_bbox_doc_id[0] if raw_bbox_doc_id else None
                    if raw_bbox_doc_id is not None:
                        bbox_doc_id = str(raw_bbox_doc_id).strip() or None

                relevant_docid_set = {str(x) for x in relevant_docids}
                if (
                    bbox_doc_id
                    and bbox_doc_id in self.docid2idx
                    and bbox_doc_id in relevant_docid_set
                ):
                    docid = bbox_doc_id
                elif self.data_args.positive_document_no_shuffle or self.data_args.image_sample_strategy == 'first':
                    docid = relevant_docids[0]
                else:
                    docid = relevant_docids[(_hashed_seed + epoch) % len(relevant_docids)]

                try:
                    selected_docid = str(docid)
                    relevant_doc_image = self.corpus[self.docid2idx[selected_docid]]['image']
                except KeyError:
                    relevant_doc_image = Image.new('RGB', (224, 224), color=(255, 255, 255))

        bboxes = []
        bbox_image_size = None
        if self.bbox_by_doc_id_field is not None and selected_docid:
            bbox_map = _parse_json_mapping(group.get(self.bbox_by_doc_id_field))
            try:
                bboxes = parse_bboxes(bbox_map.get(selected_docid))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to parse per-doc bbox for item %s doc %s: %s", item, selected_docid, exc)
                bboxes = []
            if bboxes and self.bbox_image_size_by_doc_id_field is not None:
                size_map = _parse_json_mapping(group.get(self.bbox_image_size_by_doc_id_field))
                bbox_image_size = _parse_image_size(size_map.get(selected_docid))
        elif self.bbox_field is not None:
            try:
                bboxes = parse_bboxes(group.get(self.bbox_field))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to parse bbox for item %s: %s", item, exc)
                bboxes = []
            if bboxes and self.bbox_image_size_field is not None:
                bbox_image_size = _parse_image_size(group.get(self.bbox_image_size_field))

        return query_text, relevant_doc_image, d_exist, describe, bboxes, bbox_image_size


class EncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        if self.data_args.encode_is_query:
            self.encode_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )
        else:    
            self.encode_data = load_dataset(
                self.data_args.corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )

        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        
    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        data = self.encode_data[item]
        text, image = None, None
        if self.data_args.encode_is_query:
            id = data['query_id']
            text = data['query']
        else:
            id = data['doc_id']
            image = data['image']
        return id, text, image

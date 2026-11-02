"""Hybrid content encoder: original hashed BOW + numeric + semantic branches.

This is an experimental replacement for the original ContentEncoder. It keeps
all downstream HoSig graph construction and Louvain code unchanged.

Semantic branch uses a pretrained SentenceTransformer to embed categorical
field/value phrases, e.g. "network security field service value http".
Numeric fields are read from Event.context (not parsed back from payload text).
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from collections import defaultdict

import numpy as np
from scipy.special import ndtr

from .config import Config


def stable_hash(text, salt="", digest_size=8):
    digest = hashlib.blake2b(
        str(text).encode("utf-8"),
        key=salt.encode("utf-8")[:64],
        digest_size=digest_size,
    ).digest()
    return int.from_bytes(digest, "big")


def payload_tokens(payload):
    return re.findall(r"[\w./:@=+-]+", str(payload).lower())


def _field_key(name: str) -> str:
    """Normalize a field name for type hints without changing its display name."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _to_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _value_text(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


# Protocol/state/service values are categorical even when a dataset stores
# them as integer codes. Add dataset-specific field names here when necessary.
_FORCE_CATEGORICAL = {
    "proto", "protocol", "service", "connstate", "state", "flag",
    "flags", "tcpflags", "application", "app", "direction",
    "protocoltype", "icmp_type", "icmptype", "icmpcode",
    "srcport", "dstport", "sourceport", "destinationport",
    "sport", "dport", "idorigp", "idrespp",
}

# These fields are categorical identifiers/codes, not natural-language
# concepts. They remain represented by the exact BOW branch but are not sent
# to the general-purpose language model for semantic interpretation.
_NON_SEMANTIC_CATEGORICAL_FIELDS = {
    "connstate", "state", "flag", "flags", "tcpflags", "history",
    "srcport", "dstport", "sourceport", "destinationport",
    "sport", "dport", "idorigp", "idrespp", "icmptype", "icmpcode",
}

# High-cardinality identifiers are not useful semantic categories. The original
# BOW branch still retains their payload tokens where present.
_IDENTIFIER_FIELDS = {
    "id", "uid", "flowid", "connectionid", "sessionid", "eventid",
    "srcip", "dstip", "sourceip", "destinationip", "origip", "respip",
}


class HybridContentEncoder:
    """Type-aware semantic-enhanced encoder for structured network events.

    Branches:
      1) BOW/IP: same hashed-token and IP-prefix features as original HoSig.
      2) Numeric: signed-log + robust scaling for numeric context fields.
      3) Semantic: pretrained sentence embeddings for categorical field/value
         pairs; vectors are summed so repeated semantic evidence contributes.

    The concatenated branches are projected to config.embedding_dim, then use
    the same GELU + per-event LayerNorm style transform as the original code.
    """

    def __init__(
        self,
        config: Config,
        semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        bow_weight: float = 1.0 / 3.0,
        numeric_weight: float = 1.0 / 3.0,
        semantic_weight: float = 1.0 / 3.0,
        semantic_device: str | None = None,
        semantic_aliases: dict | None = None,
        max_categories_per_field: int = 256,
        numeric_sample_size: int = 50000,
    ):
        self.config = config
        self.semantic_model_name = semantic_model_name
        self.semantic_device = semantic_device
        self.semantic_aliases = {}
        for field_name, mapping in (semantic_aliases or {}).items():
            if isinstance(mapping, dict):
                self.semantic_aliases[_field_key(field_name)] = {
                    str(value).strip().lower(): str(canonical).strip()
                    for value, canonical in mapping.items()
                }
        self.max_categories_per_field = int(max_categories_per_field)
        self.numeric_sample_size = int(numeric_sample_size)
        if self.max_categories_per_field < 2:
            raise ValueError("max_categories_per_field must be >= 2")
        if self.numeric_sample_size < 100:
            raise ValueError("numeric_sample_size must be >= 100")

        weights = np.asarray([bow_weight, numeric_weight, semantic_weight], dtype=np.float64)
        if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("branch weights must be finite, nonnegative, and not all zero")
        weights /= weights.sum()
        self.bow_weight, self.numeric_weight, self.semantic_weight = map(float, weights)

        self.numeric_fields: list[str] = []
        self.semantic_fields: list[str] = []
        self.skipped_high_cardinality_fields: list[str] = []
        self.numeric_stats: dict[str, tuple[float, float]] = {}
        self.semantic_cache: dict[tuple[str, str], np.ndarray] = {}
        self.semantic_model = None
        self.semantic_dim = 0
        self.projection = None
        self.input_dim = None
        self._fitted = False

    @staticmethod
    def _ip_features(value):
        address = ipaddress.ip_address(value)
        prefixes = (24, 16) if address.version == 4 else (64, 48)
        result = []
        for prefix in prefixes:
            network = ipaddress.ip_network(f"{address}/{prefix}", strict=False)
            hash_value = stable_hash(str(network), digest_size=4)
            result.append(hash_value / (2**32 - 1))
        return result

    @staticmethod
    def _signed_log1p(value: float) -> float:
        return math.copysign(math.log1p(abs(value)), value)

    def _load_semantic_model(self):
        if self.semantic_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Hybrid encoder requires sentence-transformers. Install it with: "
                "python -m pip install -r requirements-hybrid.txt"
            ) from exc
        try:
            kwargs = {}
            if self.semantic_device:
                kwargs["device"] = self.semantic_device
            self.semantic_model = SentenceTransformer(self.semantic_model_name, **kwargs)
            self.semantic_dim = int(self.semantic_model.get_sentence_embedding_dimension())
        except Exception as exc:
            raise RuntimeError(
                f"Could not load semantic model '{self.semantic_model_name}'. "
                "The first run may need internet access to download the model. "
                "You can also pass --semantic-model with a local model directory. "
                f"Original error: {exc}"
            ) from exc

    def _canonical_semantic_value(self, field_name: str, value: str) -> str:
        mapping = self.semantic_aliases.get(_field_key(field_name), {})
        return mapping.get(str(value).strip().lower(), str(value).strip())

    def _semantic_phrase(self, field_name: str, value: str) -> str:
        # Field context helps distinguish values that mean different things in
        # different columns (e.g. "SF" as a connection state vs another field).
        readable_field = re.sub(r"[_\-]+", " ", str(field_name)).strip()
        canonical_value = self._canonical_semantic_value(field_name, value)
        return f"network security field {readable_field} value {canonical_value}"

    def fit(self, events):
        """Learn unsupervised numeric scaling and cache category embeddings.

        This pass uses event content only; labels and timestamps are not read.
        Numeric statistics are estimated from a deterministic reservoir sample
        to keep memory bounded for large CSV files.
        """
        field_total = defaultdict(int)
        field_numeric = defaultdict(int)
        category_values = defaultdict(set)
        reservoirs = {}
        seen_numeric = defaultdict(int)
        rng = np.random.default_rng(self.config.seed + 17)

        for event in events:
            context = event.context if isinstance(event.context, dict) else {}
            for name, value in context.items():
                if not isinstance(name, str) or value is None:
                    continue
                key = _field_key(name)
                if key in _IDENTIFIER_FIELDS or key in {"timestamp", "label", "target", "class"}:
                    continue
                text_value = _value_text(value)
                if not text_value or text_value.lower() in {"nan", "none", "null", "-", "?", "n/a"}:
                    continue

                field_total[name] += 1
                number = _to_number(value)
                if number is not None:
                    field_numeric[name] += 1
                    seen_numeric[name] += 1
                    sample = reservoirs.setdefault(name, [])
                    if len(sample) < self.numeric_sample_size:
                        sample.append(number)
                    else:
                        # Deterministic reservoir sampling: a bounded, unbiased
                        # sample of values seen so far for robust quantiles.
                        j = int(rng.integers(0, seen_numeric[name]))
                        if j < self.numeric_sample_size:
                            sample[j] = number

                values = category_values[name]
                if len(values) <= self.max_categories_per_field:
                    values.add(text_value)

        numeric_fields = []
        categorical_fields = []
        for name in sorted(field_total):
            key = _field_key(name)
            forced_category = key in _FORCE_CATEGORICAL
            all_numeric = field_numeric[name] == field_total[name]
            if all_numeric and not forced_category:
                numeric_fields.append(name)
            else:
                categorical_fields.append(name)

        self.numeric_fields = numeric_fields
        self.numeric_stats = {}
        for name in self.numeric_fields:
            values = np.asarray(reservoirs.get(name, []), dtype=np.float64)
            if values.size == 0:
                median, scale = 0.0, 1.0
            else:
                transformed = np.sign(values) * np.log1p(np.abs(values))
                q25, median, q75 = np.percentile(transformed, [25, 50, 75])
                scale = float(q75 - q25)
                if not math.isfinite(scale) or scale < 1e-8:
                    scale = float(np.std(transformed))
                if not math.isfinite(scale) or scale < 1e-8:
                    scale = 1.0
                median = float(median)
            self.numeric_stats[name] = (median, scale)

        self.semantic_fields = []
        self.skipped_high_cardinality_fields = []
        for name in categorical_fields:
            key = _field_key(name)
            values = category_values.get(name, set())
            if key in _IDENTIFIER_FIELDS or key in _NON_SEMANTIC_CATEGORICAL_FIELDS or len(values) < 2:
                # A constant field or a machine code/identifier cannot provide
                # reliable natural-language semantics. BOW still retains it.
                continue
            if all(_to_number(value) is not None for value in values):
                # Numeric-coded categories (e.g. protocol number 6/17) are
                # exact categories, not natural-language phrases.
                continue
            if len(values) > self.max_categories_per_field:
                self.skipped_high_cardinality_fields.append(name)
                continue
            self.semantic_fields.append(name)

        # The semantic model is only needed when there are categorical values
        # eligible for semantic encoding.
        phrases = []
        phrase_keys = []
        for name in self.semantic_fields:
            for value in sorted(category_values[name]):
                phrase_keys.append((name, value))
                phrases.append(self._semantic_phrase(name, value))

        if phrases:
            self._load_semantic_model()
            vectors = self.semantic_model.encode(
                phrases,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(phrase_keys):
                raise RuntimeError("Semantic model returned an unexpected embedding shape")
            self.semantic_dim = int(vectors.shape[1])
            self.semantic_cache = {
                key: vectors[index].astype(np.float32)
                for index, key in enumerate(phrase_keys)
            }
        else:
            self.semantic_dim = 0
            self.semantic_cache = {}

        bow_dim = self.config.vocabulary_size + 4
        numeric_dim = len(self.numeric_fields)
        self.input_dim = bow_dim + numeric_dim + self.semantic_dim
        generator = np.random.default_rng(self.config.seed)
        self.projection = generator.normal(
            0,
            1 / np.sqrt(self.input_dim),
            (self.input_dim, self.config.embedding_dim),
        ).astype(np.float32)
        self._fitted = True
        return self

    def _encode_unseen_category(self, field_name: str, value: str):
        key = (field_name, value)
        if key in self.semantic_cache:
            return self.semantic_cache[key]
        if self.semantic_model is None:
            return None
        vector = self.semantic_model.encode(
            [self._semantic_phrase(field_name, value)],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        vector = np.asarray(vector, dtype=np.float32)
        self.semantic_cache[key] = vector
        return vector

    def transform(self, events):
        if not self._fitted:
            raise RuntimeError("Call HybridContentEncoder.fit(events) before transform().")
        if not events:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        bow_dim = self.config.vocabulary_size + 4
        numeric_dim = len(self.numeric_fields)
        semantic_dim = self.semantic_dim
        rows = np.zeros((len(events), self.input_dim), dtype=np.float32)

        numeric_index = {name: index for index, name in enumerate(self.numeric_fields)}
        semantic_field_set = set(self.semantic_fields)

        for row_index, event in enumerate(events):
            offset = 0
            # Branch 1: exact original HoSig IP-prefix + hashed BOW features.
            bow = np.zeros(bow_dim, dtype=np.float32)
            bow[:4] = self._ip_features(event.src_ip) + self._ip_features(event.dst_ip)
            tokens = payload_tokens(event.payload)
            for token in tokens:
                slot = 4 + stable_hash(token) % self.config.vocabulary_size
                bow[slot] += 1.0
            bow_norm = float(np.linalg.norm(bow))
            if bow_norm > 0:
                bow /= bow_norm
            rows[row_index, offset:offset + bow_dim] = self.bow_weight * bow
            offset += bow_dim

            # Branch 2: field-aware numeric representation.
            if numeric_dim:
                numeric = np.zeros(numeric_dim, dtype=np.float32)
                context = event.context if isinstance(event.context, dict) else {}
                for name, index in numeric_index.items():
                    value = _to_number(context.get(name))
                    if value is None:
                        continue  # median imputation -> robust z-score 0
                    median, scale = self.numeric_stats[name]
                    transformed = self._signed_log1p(value)
                    numeric[index] = np.clip((transformed - median) / scale, -5.0, 5.0)
                numeric /= np.sqrt(max(1, numeric_dim))
                rows[row_index, offset:offset + numeric_dim] = self.numeric_weight * numeric
            offset += numeric_dim

            # Branch 3: semantic embeddings for categorical field/value pairs.
            # Summation (rather than a set) lets repeated semantic evidence add
            # to the representation. Division by sqrt(count) controls scale.
            if semantic_dim:
                semantic = np.zeros(semantic_dim, dtype=np.float32)
                context = event.context if isinstance(event.context, dict) else {}
                count = 0
                for name in self.semantic_fields:
                    if name not in context:
                        continue
                    value = _value_text(context[name])
                    if not value:
                        continue
                    vector = self._encode_unseen_category(name, value)
                    if vector is not None:
                        semantic += vector
                        count += 1
                if count:
                    semantic /= np.sqrt(count)
                rows[row_index, offset:offset + semantic_dim] = self.semantic_weight * semantic

        projected = rows @ self.projection
        activated = projected * ndtr(projected)  # GELU
        centered = activated - activated.mean(axis=1, keepdims=True)
        variance = activated.var(axis=1, keepdims=True)
        normalized = centered / np.sqrt(variance + 1e-5)
        return normalized.astype(np.float32)

    def info(self):
        return {
            "name": "HoSig hybrid: hashed BOW + robust numeric + semantic embeddings",
            "semantic_model": self.semantic_model_name,
            "semantic_dimension": self.semantic_dim,
            "semantic_alias_mapping_count": sum(len(v) for v in self.semantic_aliases.values()),
            "numeric_fields": list(self.numeric_fields),
            "semantic_fields": list(self.semantic_fields),
            "skipped_high_cardinality_fields": list(self.skipped_high_cardinality_fields),
            "input_dimension": self.input_dim,
            "output_dimension": self.config.embedding_dim,
            "branch_weights": {
                "bow": self.bow_weight,
                "numeric": self.numeric_weight,
                "semantic": self.semantic_weight,
            },
            "numeric_scaling": "signed-log1p + median/IQR robust scaling (IQR fallback to std)",
            "semantic_aggregation": "sum of normalized field-conditioned embeddings, scaled by sqrt(field count)",
        }

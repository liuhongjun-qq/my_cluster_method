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


def _alias_key(value):
    """Normalize numeric category codes: 6, 6.0 and '6' share one key."""
    text = str(value).strip().lower()
    number = _to_number(text)
    if number is not None and number.is_integer():
        return str(int(number))
    return text


# IANA IP protocol numbers commonly found in network-flow datasets.
# Unknown codes are retained as explicit protocol-code categories; they are
# NOT guessed to be TCP/UDP or another named protocol.
_DEFAULT_SEMANTIC_ALIASES = {
    "proto": {
        "0": "IPv6 Hop-by-Hop Option", "1": "ICMP", "2": "IGMP",
        "4": "IPv4 encapsulation", "6": "TCP", "17": "UDP",
        "41": "IPv6 encapsulation", "43": "IPv6 routing header",
        "44": "IPv6 fragmentation header", "47": "GRE",
        "50": "ESP", "51": "AH",
        "58": "ICMPv6", "59": "IPv6 no next header",
        "60": "IPv6 destination options", "89": "OSPF", "132": "SCTP",
    },
    "protocol": {
        "0": "IPv6 Hop-by-Hop Option", "1": "ICMP", "2": "IGMP",
        "4": "IPv4 encapsulation", "6": "TCP", "17": "UDP",
        "41": "IPv6 encapsulation", "43": "IPv6 routing header",
        "44": "IPv6 fragmentation header", "47": "GRE", "50": "ESP",
        "51": "AH", "58": "ICMPv6", "59": "IPv6 no next header",
        "60": "IPv6 destination options", "89": "OSPF", "132": "SCTP",
    },
    "ipprotocol": {
        "0": "IPv6 Hop-by-Hop Option", "1": "ICMP", "2": "IGMP",
        "4": "IPv4 encapsulation", "6": "TCP", "17": "UDP",
        "41": "IPv6 encapsulation", "43": "IPv6 routing header",
        "44": "IPv6 fragmentation header", "47": "GRE", "50": "ESP",
        "51": "AH", "58": "ICMPv6", "59": "IPv6 no next header",
        "60": "IPv6 destination options", "89": "OSPF", "132": "SCTP",
    },
}

_PROTOCOL_FIELD_KEYS = {"proto", "protocol", "ipprotocol"}

# Port columns are categorical identifiers even though their values are numeric.
# Their exact values remain in BOW; the semantic branch uses service-aware
# descriptions for common ports and a small range category for other ports.
_PORT_FIELD_KEYS = {
    "srcport", "dstport", "sourceport", "destinationport", "sport", "dport",
    "idorigp", "idrespp",
}

# Conventional service associations. These are descriptive hints, not proof of
# the application protocol: actual service use depends on transport and setup.
_DEFAULT_PORT_SEMANTIC_ALIASES = {
    "20": "commonly associated with FTP data transfer",
    "21": "commonly associated with FTP control",
    "22": "commonly associated with SSH remote access",
    "23": "commonly associated with Telnet remote access",
    "25": "commonly associated with SMTP email transfer",
    "53": "commonly associated with DNS name resolution",
    "67": "commonly associated with DHCP server traffic",
    "68": "commonly associated with DHCP client traffic",
    "69": "commonly associated with TFTP file transfer",
    "80": "commonly associated with HTTP web traffic",
    "110": "commonly associated with POP3 email retrieval",
    "123": "commonly associated with NTP time synchronization",
    "143": "commonly associated with IMAP email access",
    "161": "commonly associated with SNMP network management",
    "389": "commonly associated with LDAP directory services",
    "443": "commonly associated with HTTPS encrypted web traffic",
    "445": "commonly associated with SMB file and printer sharing",
    "465": "commonly associated with SMTPS email transfer",
    "500": "commonly associated with IPsec key exchange",
    "514": "commonly associated with Syslog messaging",
    "587": "commonly associated with SMTP message submission",
    "636": "commonly associated with LDAPS directory services",
    "993": "commonly associated with IMAPS encrypted email access",
    "995": "commonly associated with POP3S encrypted email retrieval",
    "1433": "commonly associated with Microsoft SQL Server",
    "1521": "commonly associated with Oracle database service",
    "1883": "commonly associated with MQTT messaging",
    "2049": "commonly associated with NFS file sharing",
    "2375": "commonly associated with Docker API without TLS",
    "2376": "commonly associated with Docker API with TLS",
    "3306": "commonly associated with MySQL database service",
    "3389": "commonly associated with RDP remote desktop",
    "5432": "commonly associated with PostgreSQL database service",
    "5672": "commonly associated with AMQP messaging",
    "5900": "commonly associated with VNC remote desktop",
    "6379": "commonly associated with Redis database service",
    "8080": "commonly associated with alternate HTTP web service",
    "8443": "commonly associated with alternate HTTPS web service",
    "9200": "commonly associated with Elasticsearch HTTP API",
    "27017": "commonly associated with MongoDB database service",
}


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
    "icmptype", "icmpcode",
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
        self.semantic_aliases = {
            field: {_alias_key(value): canonical for value, canonical in mapping.items()}
            for field, mapping in _DEFAULT_SEMANTIC_ALIASES.items()
        }
        self.port_semantic_aliases = {
            _alias_key(value): canonical
            for value, canonical in _DEFAULT_PORT_SEMANTIC_ALIASES.items()
        }
        # User-provided aliases override defaults and may add dataset-specific
        # protocol/category codes. No unknown code is assigned a guessed name.
        for field_name, mapping in (semantic_aliases or {}).items():
            if isinstance(mapping, dict):
                key = _field_key(field_name)
                self.semantic_aliases.setdefault(key, {}).update({
                    _alias_key(value): str(canonical).strip()
                    for value, canonical in mapping.items()
                })
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
        key = _field_key(field_name)
        normalized_value = _alias_key(value)

        # Port numbers are categorical, not continuous numeric measurements.
        # Keep exact port tokens in BOW, while Semantic captures common service
        # associations or a coarse IANA-style port range for uncommon values.
        if key in _PORT_FIELD_KEYS:
            number = _to_number(value)
            if number is not None and number.is_integer() and 0 <= number <= 65535:
                port_key = str(int(number))
                if port_key in self.port_semantic_aliases:
                    return f"port {port_key}, {self.port_semantic_aliases[port_key]}"
                if number <= 1023:
                    return "well-known system or network service port (0-1023)"
                if number <= 49151:
                    return "registered service port (1024-49151)"
                return "dynamic or private client port (49152-65535)"
            if number is not None:
                return f"non-standard port value {str(value).strip()}"

        mapping = self.semantic_aliases.get(key, {})
        alias = mapping.get(normalized_value)
        if alias is not None:
            return alias
        # Retain an unfamiliar numeric protocol code as its own category.
        # This phrase identifies the field and code without inventing meaning.
        if key in _PROTOCOL_FIELD_KEYS and _to_number(value) is not None:
            return f"unmapped protocol number {normalized_value}"
        return str(value).strip()

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
        # Port values are canonicalized during the data pass. This avoids
        # treating hundreds/thousands of distinct port numbers as thousands
        # of separate language-model categories.
        semantic_category_values = defaultdict(set)
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
                if key in _PORT_FIELD_KEYS:
                    semantic_category_values[name].add(
                        self._canonical_semantic_value(name, text_value)
                    )

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
            raw_values = category_values.get(name, set())
            values = (
                semantic_category_values.get(name, set())
                if key in _PORT_FIELD_KEYS
                else raw_values
            )
            if (
                key in _IDENTIFIER_FIELDS
                or key in _NON_SEMANTIC_CATEGORICAL_FIELDS
                or len(values) < 2
            ):
                # A constant field or a machine code/identifier cannot provide
                # useful semantic variation. BOW still retains its exact tokens.
                continue
            all_numeric = all(_to_number(value) is not None for value in raw_values)
            if all_numeric and key not in _PROTOCOL_FIELD_KEYS and key not in _PORT_FIELD_KEYS:
                # Do not reinterpret arbitrary numeric category codes as
                # language semantics unless their field has a verified alias.
                alias_map = self.semantic_aliases.get(key, {})
                if not raw_values or not all(
                    _alias_key(value) in alias_map for value in raw_values
                ):
                    continue
            # Protocol and port fields are explicitly categorical by schema.
            # Ports use canonical service/range categories, so high raw port
            # cardinality does not create an unbounded embedding vocabulary.
            if len(values) > self.max_categories_per_field:
                self.skipped_high_cardinality_fields.append(name)
                continue
            self.semantic_fields.append(name)

        # The semantic model is only needed when there are categorical values
        # eligible for semantic encoding.
        phrases = []
        phrase_keys = []
        for name in self.semantic_fields:
            key = _field_key(name)
            values = (
                semantic_category_values.get(name, set())
                if key in _PORT_FIELD_KEYS
                else category_values[name]
            )
            for value in sorted(values):
                canonical_value = self._canonical_semantic_value(name, value)
                cache_key = (name, canonical_value)
                phrase_keys.append(cache_key)
                phrases.append(self._semantic_phrase(name, canonical_value))

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

        # Never silently report a BOW+Semantic run when the semantic branch
        # could not be constructed. This is essential for controlled ablations.
        if self.semantic_weight > 0 and self.semantic_dim == 0:
            raise RuntimeError(
                "Semantic weight is greater than zero, but no semantic field "
                "was encoded. The requested BOW+Semantic experiment cannot "
                "run. Check semantic_fields, category cardinalities, and "
                "semantic aliases. Numeric weight has NOT been substituted."
            )

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
        # Cache by canonical meaning, not raw value. For ports this means all
        # uncommon values in the same range share one bounded semantic vector;
        # the exact port number is still retained in BOW.
        canonical_value = self._canonical_semantic_value(field_name, value)
        key = (field_name, canonical_value)
        if key in self.semantic_cache:
            return self.semantic_cache[key]
        if self.semantic_model is None:
            return None
        vector = self.semantic_model.encode(
            [self._semantic_phrase(field_name, canonical_value)],
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
            "port_semantic_alias_mapping_count": len(self.port_semantic_aliases),
            "numeric_fields": list(self.numeric_fields),
            "semantic_fields": list(self.semantic_fields),
            "skipped_high_cardinality_fields": list(self.skipped_high_cardinality_fields),
            "semantic_enabled": bool(self.semantic_dim > 0 and self.semantic_weight > 0),
            "numeric_enabled": bool(self.numeric_fields and self.numeric_weight > 0),
            "semantic_unknown_protocol_values_policy": (
                "retained as 'unmapped protocol number N'; no protocol name is guessed"
            ),
            "semantic_port_values_policy": (
                "common ports use conventional service associations; other ports use "
                "well-known/registered/dynamic range categories; exact port tokens remain in BOW"
            ),
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

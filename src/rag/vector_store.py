"""
vector_store.py — ChromaDB Vector Store Manager for EarningsLens RAG module.

Manages persistent storage and retrieval of earnings call target embeddings
using ChromaDB and sentence-transformers. Supports semantic similarity queries
for target continuity analysis across fiscal quarters.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — fail gracefully so the rest of the codebase can
# import this module even when the dependencies are not installed.
# ---------------------------------------------------------------------------
try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed. Vector store functionality will be limited.")

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Embedding functionality will be limited.")

# Default persist directory relative to the project root
_DEFAULT_PERSIST_DIR = str(
    Path(__file__).resolve().parents[3] / "data" / "cache" / "chromadb"
)
_DEFAULT_COLLECTION = "earnings_targets"
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cosine similarity thresholds (paper-calibrated defaults)
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "maintained": 0.80,
    "rephrased": 0.55,
}

_ENCODER_CACHE: Dict[tuple, "SentenceTransformer"] = {}

def _get_or_create_encoder(model_name: str, device: str) -> "SentenceTransformer":
    key = (model_name, device)
    cached = _ENCODER_CACHE.get(key)
    if cached is not None:
        logger.info("Reusing cached SentenceTransformer: model=%s device=%s", model_name, device)
        return cached
    if not _ST_AVAILABLE:
        raise RuntimeError("sentence-transformers is required. Install it with: pip install sentence-transformers")
    encoder = SentenceTransformer(model_name, device=device)
    _ENCODER_CACHE[key] = encoder
    logger.info("Cached new SentenceTransformer: model=%s device=%s", model_name, device)
    return encoder

class TargetVectorStore:
    """
    ChromaDB-backed vector store for earnings call target embeddings.

    Each target document is stored with the following metadata:
        - company_id   : str  — unique company identifier (e.g. CIK or ticker)
        - fiscal_quarter: str — e.g. "2023Q2"
        - metric_name  : str  — short label of the target
        - context      : str  — sentence(s) of surrounding context
        - is_financial : bool — True if the target is a financial metric
        - target_type  : str  — "financial" | "non-financial"

    The collection document ID is formed as:
        ``{company_id}__{fiscal_quarter}__{metric_name_slug}__{md5_8}``
    where ``md5_8`` is the first 8 hex chars of MD5(metric_name) to keep IDs
    unique even when long metric names share a slug-truncated prefix.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        persist_dir: str = _DEFAULT_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize ChromaDB client and sentence-transformer embedding model.

        Parameters
        ----------
        persist_dir:
            Directory where ChromaDB stores its on-disk data.
        collection_name:
            Name of the ChromaDB collection to use / create.
        embedding_model:
            HuggingFace model identifier for sentence-transformers.
        device:
            PyTorch device string ("cuda", "cuda:0", "mps", "cpu"). If ``None``
            (default), auto-detect: prefer CUDA when available, otherwise CPU.
            Pass ``"cpu"`` to force CPU even when a GPU is present.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Ensure persist directory exists
        os.makedirs(persist_dir, exist_ok=True)

        # Initialise ChromaDB persistent client
        if not _CHROMA_AVAILABLE:
            raise RuntimeError(
                "chromadb is required for TargetVectorStore. "
                "Install it with: pip install chromadb"
            )

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (persist_dir=%s, docs=%d)",
            collection_name,
            persist_dir,
            self._collection.count(),
        )

        # Initialise sentence-transformer encoder
        if not _ST_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is required for TargetVectorStore. "
                "Install it with: pip install sentence-transformers"
            )

        model_name = embedding_model.replace("sentence-transformers/", "")

        # ── Device selection ────────────────────────────────────────────────
        # SentenceTransformer's default constructor *should* auto-pick CUDA
        # when available, but in practice on Colab the auto-detection is
        # unreliable (e.g. when torch was imported before the runtime attached
        # the GPU, or when a CPU-only torch wheel was pulled in transitively).
        # We therefore resolve the device explicitly and log it so the user
        # always knows whether the encoder is on GPU or CPU.
        resolved_device = device
        if resolved_device is None:
            try:
                import torch  # local import so vector_store stays importable
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved_device = "cpu"
        self.device = resolved_device

        self._encoder = _get_or_create_encoder(model_name, resolved_device)
        logger.info(
            "Loaded embedding model: %s on device=%s",
            embedding_model, resolved_device,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_doc_id(company_id: str, fiscal_quarter: str, metric_name: str) -> str:
        """
        Build a deterministic document ID from compound key fields.

        The slug is truncated to fit a reasonable ID length, so distinct metric
        names that share a long prefix (e.g. "newly enrolled people in
        individual exchange offerings selecting Bronze plans" vs "... Silver
        plans") would otherwise collide once truncated. To make IDs uniquely
        determined by metric_name while remaining readable, we append a short
        deterministic hash of the *full* metric_name as a disambiguator.

        Parameters
        ----------
        company_id:     Company identifier string.
        fiscal_quarter: Quarter string such as "2023Q2".
        metric_name:    Target metric name (will be slugified).

        Returns
        -------
        str: Composite document ID safe for use as a ChromaDB ID.
        """
        # Readable prefix — same character substitutions as before, but
        # truncated slightly shorter to leave room for the hash tail so the
        # full ID still fits comfortably under ChromaDB's per-id length limits.
        slug = metric_name.lower().replace(" ", "_").replace("/", "_")[:54]
        # 8 hex chars = 32 bits ≈ 1 collision per 65k items in a worst-case
        # birthday-bound; we never have more than a few hundred metrics per
        # (company, quarter) so this is effectively collision-free.
        digest = hashlib.md5(metric_name.encode("utf-8")).hexdigest()[:8]
        return f"{company_id}__{fiscal_quarter}__{slug}__{digest}"

    def _embed_target(self, target: Dict[str, Any]) -> List[float]:
        """
        Produce a single embedding vector for a target dict.

        The text used for embedding is the concatenation of
        ``metric_name`` and ``context`` (if present).

        Parameters
        ----------
        target: Dict containing at minimum ``metric_name`` and optionally ``context``.

        Returns
        -------
        List[float]: Embedding vector of shape (embedding_dim,).
        """
        metric_name = target.get("metric_name", "")
        context = target.get("context", "")
        text = f"{metric_name}. {context}".strip().rstrip(".")
        vector = self._encoder.encode(text, normalize_embeddings=True)
        return vector.tolist()

    # ------------------------------------------------------------------
    # Core indexing
    # ------------------------------------------------------------------

    def index_targets(
        self,
        company_id: str,
        fiscal_quarter: str,
        targets: List[Dict[str, Any]],
    ) -> int:
        """
        Embed each target's ``metric_name + context`` and store with metadata.

        Parameters
        ----------
        company_id:     Unique company identifier.
        fiscal_quarter: Fiscal quarter string (e.g. "2023Q2").
        targets:        List of target dicts. Expected keys:
                            metric_name (str, required)
                            context     (str, optional)
                            is_financial (bool, optional, default False)
                            target_type  (str, optional)

        Returns
        -------
        int: Number of targets successfully indexed.
        """
        if not targets:
            logger.warning("No targets to index for %s %s", company_id, fiscal_quarter)
            return 0

        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        for target in targets:
            metric_name = target.get("metric_name", "").strip()
            if not metric_name:
                logger.debug("Skipping target with empty metric_name")
                continue

            doc_id = self._make_doc_id(company_id, fiscal_quarter, metric_name)
            context = target.get("context", "")
            is_financial = bool(target.get("is_financial", False))
            target_type = target.get("target_type", "financial" if is_financial else "non-financial")
            document_text = f"{metric_name}. {context}".strip()

            embedding = self._embed_target(target)

            ids.append(doc_id)
            embeddings.append(embedding)
            documents.append(document_text)
            metadatas.append(
                {
                    "company_id": company_id,
                    "fiscal_quarter": fiscal_quarter,
                    "metric_name": metric_name,
                    "context": context[:512],  # ChromaDB metadata string limit
                    "is_financial": is_financial,
                    "target_type": target_type,
                }
            )

        if not ids:
            return 0

        # Upsert to handle re-indexing gracefully
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "Indexed %d targets for %s %s",
            len(ids),
            company_id,
            fiscal_quarter,
        )
        return len(ids)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query_historical_targets(
        self,
        company_id: str,
        current_quarter: str,
        n_quarters: int = 4,
        include_embeddings: bool = True,
        lag_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical targets for a company.

        Two modes (paper-strict t-k vs window):
        - ``lag_only=False`` (default, backward-compatible window mode):
          Return targets from the prior ``n_quarters`` quarters
          (i.e. t-1, t-2, ..., t-n_quarters as a union).
        - ``lag_only=True`` (paper-strict mode):
          Return targets from the SINGLE quarter at exact lag
          ``n_quarters`` (i.e. t-k where k = n_quarters). This matches
          the Moving Targets paper definition (default k=4).

        Parameters
        ----------
        company_id:        Company identifier.
        current_quarter:   The current quarter (excluded from results).
        n_quarters:        Window size (lag_only=False) OR exact lag
                           (lag_only=True). Default 4.
        include_embeddings: If True (default), return the stored vectors so
                           downstream similarity computation can reuse them.
                           Set False to skip the embedding payload (smaller
                           transfer when only metadata is needed).
        lag_only:          If True, fetch only the single quarter at exact
                           lag ``n_quarters`` (paper-strict t-k semantics).
                           Default False preserves prior behavior.

        Returns
        -------
        List[Dict]: Each dict contains all metadata fields plus ``document`` text
                    and (optionally) ``embedding`` vector.
        """
        if lag_only:
            single_q = self._get_quarter_at_lag(current_quarter, n_quarters)
            prior_quarters = [single_q] if single_q else []
        else:
            prior_quarters = self._get_prior_quarters(current_quarter, n_quarters)
        if not prior_quarters:
            return []

        # Build the chromadb `include` list dynamically
        include_fields: List[str] = ["metadatas", "documents"]
        if include_embeddings:
            include_fields.append("embeddings")

        # Query collection with a where filter
        try:
            results = self._collection.get(
                where={
                    "$and": [
                        {"company_id": {"$eq": company_id}},
                        {"fiscal_quarter": {"$in": prior_quarters}},
                    ]
                },
                include=include_fields,
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        historical: List[Dict[str, Any]] = []
        ids = results.get("ids", [])
        
        metadatas = results.get("metadatas")
        if metadatas is None:
            metadatas = []
            
        documents = results.get("documents")
        if documents is None:
            documents = []
            
        embeddings = results.get("embeddings",)
        if embeddings is None:
            embeddings = []

        for i, doc_id in enumerate(ids):
            record: Dict[str, Any] = {
                "id": doc_id,
                "document": documents[i] if i < len(documents) else "",
            }
            if include_embeddings:
                record["embedding"] = (
                    embeddings[i] if i < len(embeddings) else []
                )
            if i < len(metadatas) and metadatas[i]:
                record.update(metadatas[i])
            historical.append(record)

        logger.debug(
            "Retrieved %d historical targets for %s (quarters: %s, lag_only=%s)",
            len(historical),
            company_id,
            prior_quarters,
            lag_only,
        )
        return historical

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def compute_similarity(
        self,
        current_targets: List[Dict[str, Any]],
        historical_targets: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Compute a pairwise cosine similarity matrix between current and historical targets.

        Rows = current targets; Columns = historical targets.

        Parameters
        ----------
        current_targets:   List of current-quarter target dicts.
        historical_targets: List of historical target dicts (with embeddings).

        Returns
        -------
        pd.DataFrame: Shape (len(current_targets), len(historical_targets)).
                      Index = current metric_names, Columns = historical metric_names.
        """
        if not current_targets or not historical_targets:
            return pd.DataFrame()

        # Embed current targets — BATCHED single forward pass.
        # The earlier per-target loop (`[self._embed_target(t) for t in ...]`)
        # paid ~5-10ms of SentenceTransformer dispatch overhead per call, which
        # dominated the ~570ms-per-quarter cost in NB04's batch loop. One
        # batched encode is 30-50× faster on GPU and 5-10× on CPU.
        current_texts = [
            f"{t.get('metric_name', '')}. {t.get('context', '')}".strip().rstrip(".")
            for t in current_targets
        ]
        current_embeddings = np.asarray(
            self._encoder.encode(
                current_texts,
                batch_size=min(64, len(current_texts)),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

        # Use stored embeddings for historical targets where available,
        # otherwise re-embed.
        hist_embeddings_list = []
        for ht in historical_targets:
            emb = ht.get("embedding")
            if emb is not None and len(emb) > 0:
                hist_embeddings_list.append(np.array(emb, dtype=np.float32))
            else:
                hist_embeddings_list.append(
                    np.array(self._embed_target(ht), dtype=np.float32)
                )
        historical_embeddings = np.array(hist_embeddings_list, dtype=np.float32)

        # Normalise (should already be unit-norm but re-normalise for safety)
        def _l2_normalize(mat: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return mat / norms

        current_embeddings = _l2_normalize(current_embeddings)
        historical_embeddings = _l2_normalize(historical_embeddings)

        # Cosine similarity = dot product of unit vectors
        similarity_matrix = current_embeddings @ historical_embeddings.T

        # Build labels for the similarity DataFrame. We MUST guarantee uniqueness
        # because downstream classify_continuity() does `df[col]` which returns a
        # DataFrame (not a Series) on duplicate column labels — that breaks the
        # `float(col_scores.max())` reduction. Two historical targets within the
        # same prior quarter can legitimately share a metric_name (NB03 sometimes
        # emits near-duplicates), so we append a positional index ``[i]`` to keep
        # labels human-readable while ensuring uniqueness.
        current_labels = [
            f"{t.get('metric_name', f'current_{i}')} [{i}]"
            for i, t in enumerate(current_targets)
        ]
        historical_labels = [
            f"{t.get('metric_name', f'hist_{i}')} ({t.get('fiscal_quarter', '')}) [{i}]"
            for i, t in enumerate(historical_targets)
        ]

        return pd.DataFrame(
            similarity_matrix,
            index=current_labels,
            columns=historical_labels,
        )

    # ------------------------------------------------------------------
    # Continuity classification
    # ------------------------------------------------------------------

    def classify_continuity(
        self,
        similarity_matrix: pd.DataFrame,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Classify each historical target as maintained, rephrased, or dropped.

        Decision rule for each historical target column:
            - max similarity across all current targets > thresholds['maintained']
              → maintained
            - thresholds['rephrased'] < max ≤ thresholds['maintained']
              → rephrased
            - max ≤ thresholds['rephrased']
              → dropped

        Parameters
        ----------
        similarity_matrix:
            DataFrame produced by :meth:`compute_similarity`.
        thresholds:
            Dict with keys ``maintained`` and ``rephrased``. Defaults to
            ``DEFAULT_THRESHOLDS``.

        Returns
        -------
        Dict with keys:
            maintained  : List[str] — historical target labels
            rephrased   : List[str]
            dropped     : List[str]
            details     : Dict[str, Dict] — per-target detail (best_score, best_match)
        """
        if thresholds is None:
            thresholds = DEFAULT_THRESHOLDS

        maintained_thr = thresholds.get("maintained", DEFAULT_THRESHOLDS["maintained"])
        rephrased_thr = thresholds.get("rephrased", DEFAULT_THRESHOLDS["rephrased"])

        maintained: List[str] = []
        rephrased: List[str] = []
        dropped: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}

        if similarity_matrix.empty:
            return {
                "maintained": maintained,
                "rephrased": rephrased,
                "dropped": dropped,
                "details": details,
            }

        for hist_col in similarity_matrix.columns:
            col_scores = similarity_matrix[hist_col]
            best_score = float(col_scores.max())
            best_current = str(col_scores.idxmax())

            details[hist_col] = {
                "best_match_score": best_score,
                "best_match_current": best_current,
            }

            if best_score > maintained_thr:
                maintained.append(hist_col)
                details[hist_col]["classification"] = "maintained"
            elif best_score > rephrased_thr:
                rephrased.append(hist_col)
                details[hist_col]["classification"] = "rephrased"
            else:
                dropped.append(hist_col)
                details[hist_col]["classification"] = "dropped"

        logger.info(
            "Continuity classification: maintained=%d, rephrased=%d, dropped=%d",
            len(maintained),
            len(rephrased),
            len(dropped),
        )
        return {
            "maintained": maintained,
            "rephrased": rephrased,
            "dropped": dropped,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Bulk indexing
    # ------------------------------------------------------------------

    def build_full_index(
        self,
        all_targets: Dict[str, Dict[str, List[Dict]]],
        encode_batch_size: int = 64,
        upsert_chunk_size: int = 1000,
        show_progress: bool = False,
    ) -> int:
        """
        Bulk-index all targets from the corpus with batched encoding and upserts.

        Performance-oriented rewrite of the per-quarter loop:
          1. Walk the nested dict and collect every (id, text, metadata) row.
          2. Encode all texts in batches of ``encode_batch_size`` via a single
             SentenceTransformer call (orders of magnitude faster than the
             one-call-per-target path used by ``index_targets``).
          3. Upsert into ChromaDB in chunks of ``upsert_chunk_size``.

        Parameters
        ----------
        all_targets:
            Nested dict of the form::

                {
                    company_id: {
                        fiscal_quarter: [target_dict, ...]
                    }
                }
        encode_batch_size:
            Mini-batch size handed to ``SentenceTransformer.encode``. Larger
            batches better saturate the GPU/CPU; 128-256 is typical for
            MiniLM-class models on a Colab T4.
        upsert_chunk_size:
            Number of records sent to ``collection.upsert`` per call. Keeps
            ChromaDB request payloads bounded.
        show_progress:
            If True, pass through to ``SentenceTransformer.encode``'s tqdm bar.

        Returns
        -------
        int: Total number of documents indexed.
        """
        # 1) Flatten the nested input into parallel arrays
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        for company_id, quarters in all_targets.items():
            for fiscal_quarter, targets in quarters.items():
                if not targets:
                    continue
                for target in targets:
                    metric_name = target.get("metric_name", "").strip()
                    if not metric_name:
                        continue
                    context = target.get("context", "")
                    is_financial = bool(target.get("is_financial", False))
                    target_type = target.get(
                        "target_type",
                        "financial" if is_financial else "non-financial",
                    )
                    document_text = f"{metric_name}. {context}".strip()
                    embed_text = document_text.rstrip(".")

                    ids.append(self._make_doc_id(company_id, fiscal_quarter, metric_name))
                    texts.append(embed_text)
                    documents.append(document_text)
                    metadatas.append(
                        {
                            "company_id": company_id,
                            "fiscal_quarter": fiscal_quarter,
                            "metric_name": metric_name,
                            "context": context[:512],
                            "is_financial": is_financial,
                            "target_type": target_type,
                        }
                    )

        if not ids:
            logger.info("build_full_index: no targets to index")
            return 0

        # 2) Batched encoding — single forward pass over the SentenceTransformer
        logger.info(
            "build_full_index: encoding %d targets (batch_size=%d)",
            len(ids), encode_batch_size,
        )
        embedding_array = self._encoder.encode(
            texts,
            batch_size=encode_batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        # 3) Chunked upsert — keep ChromaDB request payload bounded.
        #
        # Performance note: previously we materialised ALL embeddings as a
        # giant Python list-of-lists via ``[vec.tolist() for vec in array]``.
        # That allocates ``n_records * dim`` Python float objects (~28 bytes
        # each) up-front — e.g. 500k × 384 ≈ 192M floats ≈ 5+ GB on top of the
        # already-materialised numpy array. On Colab's 12 GB instance this
        # frequently swaps and looks like a hang. Instead, slice the numpy
        # array per chunk and only call ``.tolist()`` on the chunk we hand to
        # ChromaDB. Memory peak drops from O(n_records) to O(upsert_chunk_size).
        total = 0
        n_records = len(ids)
        chunk_iter = range(0, n_records, upsert_chunk_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm as _tqdm
                chunk_iter = _tqdm(
                    list(chunk_iter),
                    desc="Upserting to ChromaDB",
                    unit="chunk",
                )
            except Exception:
                pass

        for start in chunk_iter:
            end = min(start + upsert_chunk_size, n_records)
            chunk_embeddings = embedding_array[start:end].tolist()
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=chunk_embeddings,
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            total += end - start

        logger.info("build_full_index complete: %d total documents indexed", total)
        return total

    # ------------------------------------------------------------------
    # Quarter utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_prior_quarters(current_quarter: str, n: int) -> List[str]:
        """
        Return the list of the ``n`` quarters immediately preceding ``current_quarter``.

        Parameters
        ----------
        current_quarter: Quarter string in format "YYYYQq" (e.g. "2023Q2").
        n:               Number of prior quarters to generate.

        Returns
        -------
        List[str]: Sorted list of prior quarter strings (earliest first).
        """
        try:
            year = int(current_quarter[:4])
            q = int(current_quarter[5])
        except (ValueError, IndexError):
            logger.warning("Invalid quarter format: '%s'", current_quarter)
            return []

        quarters = []
        for _ in range(n):
            q -= 1
            if q == 0:
                q = 4
                year -= 1
            quarters.append(f"{year}Q{q}")

        return list(reversed(quarters))

    @staticmethod
    def _get_quarter_at_lag(current_quarter: str, lag: int) -> Optional[str]:
        """
        Return the single quarter at exact lag ``lag`` before ``current_quarter``.

        Example: _get_quarter_at_lag("2016Q3", 4) -> "2015Q3".

        Parameters
        ----------
        current_quarter: Quarter string in format "YYYYQq".
        lag:             Number of quarters to step back (>=1).

        Returns
        -------
        str or None: Quarter string "YYYYQq" at exact lag, or None on bad input.
        """
        if lag < 1:
            logger.warning("_get_quarter_at_lag requires lag>=1, got %d", lag)
            return None
        try:
            year = int(current_quarter[:4])
            q = int(current_quarter[5])
        except (ValueError, IndexError):
            logger.warning("Invalid quarter format: '%s'", current_quarter)
            return None

        for _ in range(lag):
            q -= 1
            if q == 0:
                q = 4
                year -= 1
        return f"{year}Q{q}"

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of documents in the collection."""
        return self._collection.count()

    def delete_company_data(self, company_id: str) -> None:
        """
        Remove all documents for a given company from the collection.

        Parameters
        ----------
        company_id: Company identifier to remove.
        """
        self._collection.delete(where={"company_id": {"$eq": company_id}})
        logger.info("Deleted all documents for company_id='%s'", company_id)

    def reset_collection(self) -> None:
        """Drop and recreate the collection (destructive — use with caution)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection '%s' has been reset.", self.collection_name)

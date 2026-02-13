"""NVIDIA NIM API backend for Evo2 inference."""

import base64
import io
import json
import random
import time

import httpx
import numpy as np
from loguru import logger

from virosense.backends.base import EmbeddingRequest, EmbeddingResult, Evo2Backend
from virosense.utils.constants import (
    EVO2_MODELS,
    NIM_BASE_URL,
    NIM_FORWARD_ENDPOINT,
    NIM_MAX_SEQUENCE_LENGTH,
    NIM_REQUEST_DELAY,
    NIM_REQUEST_TIMEOUT,
    get_nvidia_api_key,
)


class NIMBackend(Evo2Backend):
    """Evo2 inference via NVIDIA NIM API.

    Default backend — works on any machine with internet access.
    Requires NVIDIA_API_KEY environment variable.

    The NIM cloud API serves the Evo2 40B model. Sequences are sent
    individually (one per request) and embeddings are returned as
    base64-encoded NPZ data. Per-position embeddings are mean-pooled
    to produce a single vector per sequence.
    """

    MAX_RETRIES = 5
    RETRY_BACKOFF = 2.0  # exponential backoff base (seconds)

    def __init__(self, api_key: str | None = None, model: str = "evo2_7b"):
        self.api_key = api_key or get_nvidia_api_key()
        self.model = model
        self._base_url = NIM_BASE_URL

    def extract_embeddings(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Extract embeddings via NIM API.

        Sends one request per sequence, decodes the base64 NPZ response,
        and mean-pools per-position embeddings into sequence-level vectors.

        Args:
            request: EmbeddingRequest with sequences and layer specification.

        Returns:
            EmbeddingResult with (N, embed_dim) mean-pooled embeddings.

        Raises:
            RuntimeError: If API key is missing or API returns an error.
            ValueError: If any sequence exceeds the 16,000 bp limit.
        """
        if not self.api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY not set. Get one at "
                "https://build.nvidia.com/settings/api-keys"
            )

        sanitized = self._sanitize_sequences(request.sequences)

        # NIM cloud API accepts native Evo2 layer names (blocks.N.mlp.l3)
        # directly — no translation to Megatron naming needed.
        layer = request.layer
        url = f"{self._base_url}{NIM_FORWARD_ENDPOINT}"
        sequence_ids = list(sanitized.keys())
        all_embeddings = []
        n_seqs = len(sequence_ids)

        logger.info(
            f"Extracting embeddings for {n_seqs} sequences "
            f"via NIM API (layer: {layer})"
        )

        with httpx.Client(timeout=NIM_REQUEST_TIMEOUT) as client:
            for i, (seq_id, sequence) in enumerate(sanitized.items()):
                embedding = self._extract_single(
                    client, url, seq_id, sequence, layer
                )
                all_embeddings.append(embedding)

                if (i + 1) % 50 == 0 or i + 1 == n_seqs:
                    logger.info(f"  Progress: {i + 1}/{n_seqs} sequences")

                if i < n_seqs - 1:
                    time.sleep(NIM_REQUEST_DELAY)

        embeddings_matrix = np.stack(all_embeddings).astype(np.float32)
        logger.info(
            f"Extracted embeddings: {embeddings_matrix.shape} "
            f"({len(sequence_ids)} sequences)"
        )

        return EmbeddingResult(
            sequence_ids=sequence_ids,
            embeddings=embeddings_matrix,
            layer=request.layer,
            model=request.model,
        )

    def _extract_single(
        self,
        client,
        url: str,
        seq_id: str,
        sequence: str,
        layer: str,
    ) -> np.ndarray:
        """Extract embedding for a single sequence with retry logic.

        The NIM API has two response modes:
        - Short sequences: 200 with JSON {"data": "<base64 NPZ>", "elapsed_ms": N}
        - Longer sequences: 302 redirect to S3 presigned URL containing an NPZ
          with a UUID-keyed entry whose value is the same JSON structure

        Returns:
            1D array of shape (embed_dim,) — mean-pooled embedding.
        """
        payload = {
            "sequence": sequence,
            "output_layers": [layer],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                response = client.post(url, json=payload, headers=headers)
            except httpx.TransportError as e:
                wait = self.RETRY_BACKOFF ** (attempt + 1)
                logger.warning(
                    f"Network error for {seq_id}: {e}. "
                    f"Retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                time.sleep(wait)
                continue

            if response.status_code == 200:
                return self._decode_response(response.json(), layer, seq_id)

            if response.status_code == 302:
                return self._fetch_s3_result(client, response, layer, seq_id)

            if response.status_code == 429:
                wait = self.RETRY_BACKOFF ** (attempt + 1)
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    wait = max(wait, float(retry_after))
                logger.warning(
                    f"Rate limited on {seq_id}, retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                time.sleep(wait)
                continue

            if response.status_code == 503:
                wait = self.RETRY_BACKOFF ** (attempt + 1)
                logger.warning(
                    f"Model not ready for {seq_id}, retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                time.sleep(wait)
                continue

            if response.status_code == 422:
                raise ValueError(
                    f"Sequence {seq_id} rejected by NIM API (422): "
                    f"{response.text}. Sequence length: {len(sequence)} bp "
                    f"(max: {NIM_MAX_SEQUENCE_LENGTH})."
                )

            raise RuntimeError(
                f"NIM API error for {seq_id}: "
                f"HTTP {response.status_code}: {response.text}"
            )

        raise RuntimeError(
            f"NIM API failed for {seq_id} after {self.MAX_RETRIES} retries"
        )

    def _fetch_s3_result(
        self, client, response, layer: str, seq_id: str
    ) -> np.ndarray:
        """Follow a 302 redirect to S3 and decode the NPZ result.

        The S3 response is an NPZ with a UUID-keyed entry containing
        JSON bytes with the same {"data": "<base64>", "elapsed_ms": N}
        structure as a direct 200 response. Retries on network errors.
        """
        s3_url = response.headers["location"]

        for attempt in range(self.MAX_RETRIES):
            try:
                # GET the S3 presigned URL without auth headers
                s3_response = client.get(s3_url, headers={})
                if s3_response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to fetch S3 result for {seq_id}: "
                        f"HTTP {s3_response.status_code}"
                    )

                outer_npz = np.load(
                    io.BytesIO(s3_response.content), allow_pickle=True
                )
                keys = list(outer_npz.keys())
                if not keys:
                    raise RuntimeError(
                        f"Empty S3 NPZ response for {seq_id}"
                    )

                inner_json = json.loads(outer_npz[keys[0]])
                return self._decode_response(inner_json, layer, seq_id)

            except httpx.TransportError as e:
                wait = self.RETRY_BACKOFF ** (attempt + 1)
                logger.warning(
                    f"S3 fetch failed for {seq_id}: {e}. "
                    f"Retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"S3 fetch failed for {seq_id} after {self.MAX_RETRIES} retries"
        )

    def _decode_response(
        self, data: dict, layer: str, seq_id: str
    ) -> np.ndarray:
        """Decode base64 NPZ response and mean-pool to sequence-level embedding.

        Args:
            data: JSON response dict with 'data' (base64 NPZ) and 'elapsed_ms'.
            layer: NIM layer name used in the request.
            seq_id: Sequence identifier for logging.

        Returns:
            1D array of shape (embed_dim,) — mean-pooled embedding.
        """
        elapsed = data.get("elapsed_ms", "?")
        raw = base64.b64decode(data["data"].encode("ascii"))
        npz = np.load(io.BytesIO(raw))

        key = f"{layer}.output"
        if key not in npz:
            available = list(npz.keys())
            raise RuntimeError(
                f"Expected key {key!r} in NPZ response for {seq_id}, "
                f"got: {available}"
            )

        per_position = npz[key]  # (1, seq_len, hidden_dim)
        seq_embedding = np.mean(per_position, axis=1).squeeze()  # (hidden_dim,)

        logger.debug(
            f"  {seq_id}: {per_position.shape[1]} positions -> "
            f"({seq_embedding.shape[0]},) embedding ({elapsed}ms)"
        )
        return seq_embedding

    @staticmethod
    def _sanitize_sequences(sequences: dict[str, str]) -> dict[str, str]:
        """Validate and sanitize sequences before sending to NIM API.

        Replaces ambiguous bases (N) with random ACGT — standard
        bioinformatics practice for tools that require unambiguous DNA.
        Rejects sequences with non-DNA characters.

        Returns:
            New dict with sanitized sequences.
        """
        valid_bases = set("ACGTN")
        sanitized = {}
        for seq_id, seq in sequences.items():
            seq = seq.upper()
            if len(seq) > NIM_MAX_SEQUENCE_LENGTH:
                raise ValueError(
                    f"Sequence {seq_id} is {len(seq)} bp, exceeds NIM "
                    f"max of {NIM_MAX_SEQUENCE_LENGTH} bp. Consider "
                    f"splitting or truncating."
                )
            if len(seq) == 0:
                raise ValueError(f"Sequence {seq_id} is empty.")
            invalid = set(seq) - valid_bases
            if invalid:
                raise ValueError(
                    f"Sequence {seq_id} contains invalid characters: "
                    f"{invalid}. Only A, C, G, T, N are allowed."
                )
            n_count = seq.count("N")
            if n_count > 0:
                logger.debug(
                    f"Replacing {n_count} N bases in {seq_id} "
                    f"({100 * n_count / len(seq):.1f}%)"
                )
                seq = "".join(
                    random.choice("ACGT") if c == "N" else c for c in seq
                )
            sanitized[seq_id] = seq
        return sanitized

    def is_available(self) -> bool:
        """Check if NIM API is accessible (API key is set)."""
        return self.api_key is not None

    def max_context_length(self) -> int:
        """Return max context length supported by NIM cloud API."""
        return NIM_MAX_SEQUENCE_LENGTH

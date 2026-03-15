"""Tests for the NIM API backend."""

import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from virosense.backends.base import EmbeddingRequest
from virosense.backends.nim import NIMBackend
from virosense.utils.constants import (
    NIM_MAX_CONCURRENT,
    NIM_MAX_SEQUENCE_LENGTH,
    NIM_SELF_HOSTED_MAX_CONCURRENT,
    translate_layer_to_nim,
    translate_layer_to_native,
)


# --- Helpers ---


def make_mock_npz_response(layer_name: str, seq_len: int = 100, hidden_dim: int = 4096):
    """Create a mock NIM API response with base64-encoded NPZ data."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((1, seq_len, hidden_dim)).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, **{f"{layer_name}.output": embeddings})
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return {"data": encoded, "elapsed_ms": 150}, embeddings


def make_httpx_response(status_code: int, json_data=None, text="", headers=None):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = headers or {}
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def make_async_client(responses):
    """Create a mock httpx.AsyncClient for async tests.

    Args:
        responses: A single mock response or list for sequential post() calls.
    """
    mock_client = AsyncMock()
    # Ensure `async with AsyncClient() as client:` yields mock_client itself
    mock_client.__aenter__.return_value = mock_client
    if isinstance(responses, list):
        mock_client.post = AsyncMock(side_effect=responses)
    else:
        mock_client.post = AsyncMock(return_value=responses)
    return mock_client


# --- Layer name translation tests ---
# These test the translation utilities (still useful for local backend)


class TestLayerTranslation:
    def test_native_mlp_l3_to_nim(self):
        assert translate_layer_to_nim("blocks.28.mlp.l3") == "decoder.layers.28.mlp.linear_fc2"

    def test_native_mlp_l1_to_nim(self):
        assert translate_layer_to_nim("blocks.20.mlp.l1") == "decoder.layers.20.mlp.linear_fc1"

    def test_native_mlp_l2_to_nim(self):
        assert translate_layer_to_nim("blocks.10.mlp.l2") == "decoder.layers.10.mlp.linear_fc1"

    def test_native_mixer_to_nim(self):
        assert translate_layer_to_nim("blocks.5.mixer") == "decoder.layers.5.mixer"

    def test_already_nim_format_passthrough(self):
        nim = "decoder.layers.20.mlp.linear_fc2"
        assert translate_layer_to_nim(nim) == nim

    def test_special_layers_passthrough(self):
        assert translate_layer_to_nim("embedding") == "embedding"
        assert translate_layer_to_nim("decoder.final_norm") == "decoder.final_norm"
        assert translate_layer_to_nim("output_layer") == "output_layer"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown layer name format"):
            translate_layer_to_nim("some.random.name")

    def test_nim_to_native_roundtrip(self):
        native = "blocks.28.mlp.l3"
        nim = translate_layer_to_nim(native)
        back = translate_layer_to_native(nim)
        assert back == native

    def test_native_to_nim_various_layers(self):
        assert translate_layer_to_nim("blocks.0.mlp.l3") == "decoder.layers.0.mlp.linear_fc2"
        assert translate_layer_to_nim("blocks.49.mlp.l3") == "decoder.layers.49.mlp.linear_fc2"


# --- Sequence sanitization tests ---


class TestSequenceSanitization:
    def test_valid_sequences_unchanged(self):
        result = NIMBackend._sanitize_sequences({"s1": "ATGC", "s2": "GCTAGCTA"})
        assert result == {"s1": "ATGC", "s2": "GCTAGCTA"}

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="empty"):
            NIMBackend._sanitize_sequences({"s1": ""})

    def test_too_long_sequence_raises(self):
        long_seq = "A" * (NIM_MAX_SEQUENCE_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds NIM max"):
            NIMBackend._sanitize_sequences({"s1": long_seq})

    def test_invalid_characters_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            NIMBackend._sanitize_sequences({"s1": "ATGCXYZ"})

    def test_max_length_sequence_ok(self):
        result = NIMBackend._sanitize_sequences({"s1": "A" * NIM_MAX_SEQUENCE_LENGTH})
        assert len(result["s1"]) == NIM_MAX_SEQUENCE_LENGTH

    def test_n_bases_replaced(self):
        result = NIMBackend._sanitize_sequences({"s1": "ATGNNNGC"})
        assert "N" not in result["s1"]
        assert len(result["s1"]) == 8
        assert result["s1"][:3] == "ATG"
        assert result["s1"][-2:] == "GC"
        assert all(c in "ACGT" for c in result["s1"])

    def test_lowercase_uppercased(self):
        result = NIMBackend._sanitize_sequences({"s1": "atgc"})
        assert result["s1"] == "ATGC"


# --- Response decoding tests ---


class TestResponseDecoding:
    def test_decode_response(self):
        backend = NIMBackend(api_key="test")
        layer = "blocks.20.mlp.l3"
        response_data, raw_embeddings = make_mock_npz_response(layer, seq_len=50, hidden_dim=4096)

        result = backend._decode_response(response_data, layer, "seq_1")

        assert result.shape == (4096,)
        expected = np.mean(raw_embeddings, axis=1).squeeze()
        np.testing.assert_array_almost_equal(result, expected)

    def test_decode_response_missing_key_raises(self):
        backend = NIMBackend(api_key="test")
        layer = "blocks.20.mlp.l3"
        # Two keys with neither matching expected — should raise
        response_data, _ = make_mock_npz_response("blocks.99.mlp.l3")
        buf = io.BytesIO()
        arr1 = np.random.randn(1, 100, 4096).astype(np.float32)
        arr2 = np.random.randn(1, 100, 4096).astype(np.float32)
        np.savez(buf, **{"blocks.99.mlp.l3.output": arr1, "blocks.88.mlp.l3.output": arr2})
        multi_key_data = {
            "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            "elapsed_ms": 150,
        }

        with pytest.raises(RuntimeError, match="Expected key"):
            backend._decode_response(multi_key_data, layer, "seq_1")


# --- Layer remapping tests ---


class TestLayerRemapping:
    """Test NIM backend layer remapping for 40B model."""

    def test_7b_default_remapped_to_40b(self):
        backend = NIMBackend(api_key="test")
        assert backend._resolve_layer("blocks.28.mlp.l3") == "blocks.20.mlp.l3"

    def test_7b_mlp_l1_remapped(self):
        backend = NIMBackend(api_key="test")
        assert backend._resolve_layer("blocks.28.mlp.l1") == "blocks.20.mlp.l1"

    def test_1b_default_remapped(self):
        backend = NIMBackend(api_key="test")
        assert backend._resolve_layer("blocks.14.mlp.l3") == "blocks.10.mlp.l3"

    def test_valid_40b_layer_unchanged(self):
        backend = NIMBackend(api_key="test")
        assert backend._resolve_layer("blocks.20.mlp.l3") == "blocks.20.mlp.l3"

    def test_low_block_unchanged(self):
        backend = NIMBackend(api_key="test")
        assert backend._resolve_layer("blocks.10.mlp.l3") == "blocks.10.mlp.l3"

    def test_high_block_falls_back(self):
        # Block 30 is in the dead zone (>= 25), falls back to block 20
        backend = NIMBackend(api_key="test")
        assert backend._resolve_layer("blocks.30.mlp.l3") == "blocks.20.mlp.l3"

    def test_block_25_falls_back(self):
        backend = NIMBackend(api_key="test")
        assert backend._resolve_layer("blocks.25.mlp.l3") == "blocks.20.mlp.l3"

    @patch("virosense.backends.nim.asyncio.sleep", new_callable=AsyncMock)
    def test_remapping_used_in_api_request(self, mock_sleep):
        """Verify remapped layer is sent to the API and used for response decoding."""
        backend = NIMBackend(api_key="test_key")
        remapped_layer = "blocks.20.mlp.l3"
        response_data, _ = make_mock_npz_response(remapped_layer, seq_len=100)
        mock_response = make_httpx_response(200, json_data=response_data)
        mock_client = make_async_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"seq_1": "ATGCATGC"},
                layer="blocks.28.mlp.l3",  # 7B default — should be remapped
            )
            result = backend.extract_embeddings(request)

        assert result.embeddings.shape == (1, 4096)
        # The original layer is preserved in the result
        assert result.layer == "blocks.28.mlp.l3"
        # But the API request used the remapped layer
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["output_layers"] == [remapped_layer]


# --- Full extract_embeddings tests (mocked async HTTP) ---


class TestExtractEmbeddings:
    @patch("virosense.backends.nim.asyncio.sleep", new_callable=AsyncMock)
    def test_single_sequence(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        layer = "blocks.20.mlp.l3"
        response_data, _ = make_mock_npz_response(layer, seq_len=100)
        mock_response = make_httpx_response(200, json_data=response_data)
        mock_client = make_async_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"seq_1": "ATGCATGC"},
                layer="blocks.20.mlp.l3",
            )
            result = backend.extract_embeddings(request)

        assert result.sequence_ids == ["seq_1"]
        assert result.embeddings.shape == (1, 4096)
        assert result.layer == "blocks.20.mlp.l3"

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["output_layers"] == [layer]

    @patch("virosense.backends.nim.asyncio.sleep", new_callable=AsyncMock)
    def test_multiple_sequences(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        layer = "blocks.20.mlp.l3"
        response_data, _ = make_mock_npz_response(layer, seq_len=50)
        mock_response = make_httpx_response(200, json_data=response_data)
        mock_client = make_async_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC", "s2": "GCTA", "s3": "TTTT"},
                layer="blocks.20.mlp.l3",
            )
            result = backend.extract_embeddings(request)

        assert result.sequence_ids == ["s1", "s2", "s3"]
        assert result.embeddings.shape == (3, 4096)
        assert mock_client.post.call_count == 3
        # Async: no sleeps between successful requests (concurrency via semaphore)
        mock_sleep.assert_not_called()

    def test_no_api_key_raises(self):
        backend = NIMBackend(api_key=None)
        request = EmbeddingRequest(sequences={"s1": "ATGC"})
        with pytest.raises(RuntimeError, match="NVIDIA_API_KEY not set"):
            backend.extract_embeddings(request)

    def test_too_long_sequence_raises(self):
        backend = NIMBackend(api_key="test_key")
        request = EmbeddingRequest(
            sequences={"s1": "A" * (NIM_MAX_SEQUENCE_LENGTH + 1)},
        )
        with pytest.raises(ValueError, match="exceeds NIM max"):
            backend.extract_embeddings(request)


# --- Retry and error handling tests ---


class TestRetryLogic:
    @patch("virosense.backends.nim.asyncio.sleep", new_callable=AsyncMock)
    def test_429_retry_then_success(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        layer = "blocks.20.mlp.l3"
        response_data, _ = make_mock_npz_response(layer, seq_len=50)

        rate_limited = make_httpx_response(429, text="Rate limited")
        success = make_httpx_response(200, json_data=response_data)
        mock_client = make_async_client([rate_limited, success])

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer="blocks.20.mlp.l3",
            )
            result = backend.extract_embeddings(request)

        assert result.sequence_ids == ["s1"]
        assert mock_client.post.call_count == 2

    @patch("virosense.backends.nim.asyncio.sleep", new_callable=AsyncMock)
    def test_503_retry_then_success(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        layer = "blocks.20.mlp.l3"
        response_data, _ = make_mock_npz_response(layer, seq_len=50)

        unavailable = make_httpx_response(503, text="Model not ready")
        success = make_httpx_response(200, json_data=response_data)
        mock_client = make_async_client([unavailable, success])

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer="blocks.20.mlp.l3",
            )
            result = backend.extract_embeddings(request)

        assert result.sequence_ids == ["s1"]

    @patch("virosense.backends.nim.asyncio.sleep", new_callable=AsyncMock)
    def test_429_exhausts_retries(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        rate_limited = make_httpx_response(429, text="Rate limited")
        mock_client = make_async_client(rate_limited)

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer="blocks.20.mlp.l3",
            )
            with pytest.raises(RuntimeError, match="All sequences failed"):
                backend.extract_embeddings(request)

    @patch("virosense.backends.nim.asyncio.sleep", new_callable=AsyncMock)
    def test_422_skips_sequence(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        rejected = make_httpx_response(422, text="Sequence too long")
        mock_client = make_async_client(rejected)

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer="blocks.20.mlp.l3",
            )
            # Single sequence fails -> all failed -> RuntimeError
            with pytest.raises(RuntimeError, match="All sequences failed"):
                backend.extract_embeddings(request)

        # 422 should not retry
        assert mock_client.post.call_count == 1

    @patch("virosense.backends.nim.asyncio.sleep", new_callable=AsyncMock)
    def test_422_too_busy_retries(self, mock_sleep):
        """Self-hosted NIM returns 422 'Too Busy' when overloaded — should retry."""
        backend = NIMBackend(
            api_key="test_key", nim_url="http://localhost:8000"
        )
        layer = "blocks.28.mlp.l3"
        response_data, _ = make_mock_npz_response(layer, seq_len=50)

        too_busy = make_httpx_response(422, text='{"error":"Too Busy"}')
        success = make_httpx_response(200, json_data=response_data)
        mock_client = make_async_client([too_busy, success])

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer=layer,
            )
            result = backend.extract_embeddings(request)

        assert result.sequence_ids == ["s1"]
        assert mock_client.post.call_count == 2

    @patch("virosense.backends.nim.asyncio.sleep", new_callable=AsyncMock)
    def test_unexpected_error_raises_immediately(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        error = make_httpx_response(500, text="Internal server error")
        mock_client = make_async_client(error)

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer="blocks.20.mlp.l3",
            )
            with pytest.raises(RuntimeError, match="All sequences failed"):
                backend.extract_embeddings(request)

        assert mock_client.post.call_count == 1


# --- Backend property and constructor tests ---


class TestBackendProperties:
    def test_max_context_length(self):
        backend = NIMBackend(api_key="test")
        assert backend.max_context_length() == NIM_MAX_SEQUENCE_LENGTH

    def test_available_with_key(self):
        backend = NIMBackend(api_key="test_key")
        assert backend.is_available() is True

    def test_unavailable_without_key(self):
        backend = NIMBackend(api_key=None)
        assert backend.is_available() is False


class TestConstructor:
    """Test NIMBackend constructor with new nim_url and max_concurrent params."""

    def test_default_cloud_concurrency(self):
        backend = NIMBackend(api_key="test")
        assert backend._max_concurrent == NIM_MAX_CONCURRENT

    def test_custom_url_default_concurrency(self):
        backend = NIMBackend(api_key="test", nim_url="http://localhost:8000")
        assert backend._max_concurrent == NIM_SELF_HOSTED_MAX_CONCURRENT
        assert backend._custom_url is True

    def test_custom_url_sets_base_url(self):
        backend = NIMBackend(api_key="test", nim_url="http://gpu-server:8000/v1/")
        assert backend._base_url == "http://gpu-server:8000/v1"

    def test_custom_max_concurrent(self):
        backend = NIMBackend(api_key="test", max_concurrent=20)
        assert backend._max_concurrent == 20

    def test_custom_url_with_explicit_concurrency(self):
        backend = NIMBackend(
            api_key="test", nim_url="http://localhost:8000", max_concurrent=5
        )
        assert backend._max_concurrent == 5
        assert backend._custom_url is True


class TestSelfHostedMode:
    """Test NIM backend in self-hosted mode (--nim-url)."""

    def test_self_hosted_layer_resolution_blocks_to_decoder(self):
        backend = NIMBackend(nim_url="http://localhost:8000")
        assert backend._resolve_layer("blocks.28.mlp.l3") == "decoder.layers.28"

    def test_self_hosted_layer_resolution_preserves_decoder_format(self):
        backend = NIMBackend(nim_url="http://localhost:8000")
        assert backend._resolve_layer("decoder.layers.28") == "decoder.layers.28"

    def test_self_hosted_layer_resolution_final_norm(self):
        backend = NIMBackend(nim_url="http://localhost:8000")
        assert backend._resolve_layer("decoder.final_norm") == "decoder.final_norm"

    def test_self_hosted_no_40b_remapping(self):
        """Self-hosted doesn't remap high blocks to 40B equivalents."""
        backend = NIMBackend(nim_url="http://localhost:8000")
        # blocks.28 stays as decoder.layers.28, NOT remapped to decoder.layers.20
        assert backend._resolve_layer("blocks.28.mlp.l3") == "decoder.layers.28"

    def test_self_hosted_available_without_api_key(self):
        backend = NIMBackend(nim_url="http://localhost:8000")
        assert backend.is_available() is True

    def test_self_hosted_no_api_key_doesnt_raise(self):
        """Self-hosted mode doesn't require API key."""
        backend = NIMBackend(nim_url="http://localhost:8000")
        layer = "decoder.layers.28"
        response_data, _ = make_mock_npz_response(layer, seq_len=100)
        mock_response = make_httpx_response(200, json_data=response_data)
        mock_client = make_async_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"seq_1": "ATGCATGC"},
                layer="blocks.28.mlp.l3",
            )
            result = backend.extract_embeddings(request)

        assert result.embeddings.shape == (1, 4096)
        # No Authorization header sent
        call_kwargs = mock_client.post.call_args
        assert "Authorization" not in call_kwargs[1]["headers"]

    def test_self_hosted_uses_correct_endpoint(self):
        """Self-hosted uses /biology/arc/evo2/forward not /evo2-40b/forward."""
        backend = NIMBackend(nim_url="http://localhost:8000")
        layer = "decoder.layers.28"
        response_data, _ = make_mock_npz_response(layer, seq_len=100)
        mock_response = make_httpx_response(200, json_data=response_data)
        mock_client = make_async_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"seq_1": "ATGCATGC"},
                layer="blocks.28.mlp.l3",
            )
            backend.extract_embeddings(request)

        call_args = mock_client.post.call_args
        url = call_args[0][0]
        assert "/biology/arc/evo2/forward" in url
        assert "evo2-40b" not in url

    def test_decode_response_single_key_fallback(self):
        """When NPZ has a single key that doesn't match, use it anyway."""
        backend = NIMBackend(api_key="test")
        # Response has key "decoder.layers.28.output" but we ask for "blocks.20.mlp.l3"
        response_data, _ = make_mock_npz_response("decoder.layers.28", seq_len=100)
        result = backend._decode_response(response_data, "blocks.20.mlp.l3", "seq_1")
        assert result.shape == (4096,)

    def test_decode_self_hosted_tensor_shape(self):
        """Self-hosted NIM returns (seq_len, 1, hidden_dim) instead of (1, seq_len, hidden_dim)."""
        backend = NIMBackend(nim_url="http://localhost:8000")
        rng = np.random.default_rng(42)
        # Self-hosted shape: (seq_len, 1, hidden_dim)
        embeddings = rng.standard_normal((100, 1, 4096)).astype(np.float64)
        buf = io.BytesIO()
        np.savez(buf, **{"decoder.layers.28.output": embeddings})
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        response_data = {"data": encoded, "elapsed_ms": 400}

        result = backend._decode_response(response_data, "decoder.layers.28", "seq_1")
        assert result.shape == (4096,)
        assert result.dtype == np.float32

    def test_self_hosted_max_context_length(self):
        """Self-hosted NIM has 10,000bp max context."""
        backend = NIMBackend(nim_url="http://localhost:8000")
        assert backend.max_context_length() == 10_000

    def test_cloud_max_context_length(self):
        """Cloud NIM has 16,000bp max context."""
        backend = NIMBackend(api_key="test")
        assert backend.max_context_length() == 16_000


class TestModelAutoCorrection:
    """Cloud NIM always serves 40B — model name should be corrected."""

    def test_cloud_corrects_7b_to_40b(self):
        backend = NIMBackend(api_key="test", model="evo2_7b")
        assert backend.model == "evo2_40b"

    def test_cloud_corrects_1b_to_40b(self):
        backend = NIMBackend(api_key="test", model="evo2_1b_base")
        assert backend.model == "evo2_40b"

    def test_cloud_keeps_40b(self):
        backend = NIMBackend(api_key="test", model="evo2_40b")
        assert backend.model == "evo2_40b"

    def test_self_hosted_preserves_7b(self):
        backend = NIMBackend(api_key="test", model="evo2_7b",
                             nim_url="http://localhost:8000")
        assert backend.model == "evo2_7b"

    def test_self_hosted_preserves_40b(self):
        backend = NIMBackend(api_key="test", model="evo2_40b",
                             nim_url="http://localhost:8000")
        assert backend.model == "evo2_40b"

    def test_default_model_corrected_for_cloud(self):
        """Default model (evo2_7b) auto-corrects to 40b on cloud."""
        backend = NIMBackend(api_key="test")
        assert backend.model == "evo2_40b"

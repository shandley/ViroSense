"""Tests for the NIM API backend."""

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from virosense.backends.base import EmbeddingRequest
from virosense.backends.nim import NIMBackend
from virosense.utils.constants import (
    NIM_MAX_SEQUENCE_LENGTH,
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
        layer = "blocks.28.mlp.l3"
        response_data, raw_embeddings = make_mock_npz_response(layer, seq_len=50, hidden_dim=4096)

        result = backend._decode_response(response_data, layer, "seq_1")

        assert result.shape == (4096,)
        expected = np.mean(raw_embeddings, axis=1).squeeze()
        np.testing.assert_array_almost_equal(result, expected)

    def test_decode_response_missing_key_raises(self):
        backend = NIMBackend(api_key="test")
        layer = "blocks.28.mlp.l3"
        response_data, _ = make_mock_npz_response("blocks.99.mlp.l3")

        with pytest.raises(RuntimeError, match="Expected key"):
            backend._decode_response(response_data, layer, "seq_1")


# --- Full extract_embeddings tests (mocked HTTP) ---


class TestExtractEmbeddings:
    @patch("virosense.backends.nim.time.sleep")
    def test_single_sequence(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        layer = "blocks.28.mlp.l3"
        response_data, _ = make_mock_npz_response(layer, seq_len=100)
        mock_response = make_httpx_response(200, json_data=response_data)

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"seq_1": "ATGCATGC"},
                layer="blocks.28.mlp.l3",
            )
            result = backend.extract_embeddings(request)

        assert result.sequence_ids == ["seq_1"]
        assert result.embeddings.shape == (1, 4096)
        assert result.layer == "blocks.28.mlp.l3"

        # Verify native layer name is sent directly (no translation)
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["output_layers"] == [layer]

    @patch("virosense.backends.nim.time.sleep")
    def test_multiple_sequences(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        layer = "blocks.28.mlp.l3"
        response_data, _ = make_mock_npz_response(layer, seq_len=50)
        mock_response = make_httpx_response(200, json_data=response_data)

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC", "s2": "GCTA", "s3": "TTTT"},
                layer="blocks.28.mlp.l3",
            )
            result = backend.extract_embeddings(request)

        assert result.sequence_ids == ["s1", "s2", "s3"]
        assert result.embeddings.shape == (3, 4096)
        assert mock_client.post.call_count == 3
        # Rate limiting sleep between requests (not after last)
        assert mock_sleep.call_count == 2

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
    @patch("virosense.backends.nim.time.sleep")
    def test_429_retry_then_success(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        layer = "blocks.28.mlp.l3"
        response_data, _ = make_mock_npz_response(layer, seq_len=50)

        rate_limited = make_httpx_response(429, text="Rate limited")
        success = make_httpx_response(200, json_data=response_data)

        mock_client = MagicMock()
        mock_client.post.side_effect = [rate_limited, success]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer="blocks.28.mlp.l3",
            )
            result = backend.extract_embeddings(request)

        assert result.sequence_ids == ["s1"]
        assert mock_client.post.call_count == 2

    @patch("virosense.backends.nim.time.sleep")
    def test_503_retry_then_success(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        layer = "blocks.28.mlp.l3"
        response_data, _ = make_mock_npz_response(layer, seq_len=50)

        unavailable = make_httpx_response(503, text="Model not ready")
        success = make_httpx_response(200, json_data=response_data)

        mock_client = MagicMock()
        mock_client.post.side_effect = [unavailable, success]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer="blocks.28.mlp.l3",
            )
            result = backend.extract_embeddings(request)

        assert result.sequence_ids == ["s1"]

    @patch("virosense.backends.nim.time.sleep")
    def test_429_exhausts_retries(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        rate_limited = make_httpx_response(429, text="Rate limited")

        mock_client = MagicMock()
        mock_client.post.return_value = rate_limited
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer="blocks.28.mlp.l3",
            )
            with pytest.raises(RuntimeError, match="after 5 retries"):
                backend.extract_embeddings(request)

    @patch("virosense.backends.nim.time.sleep")
    def test_422_raises_immediately(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        rejected = make_httpx_response(422, text="Sequence too long")

        mock_client = MagicMock()
        mock_client.post.return_value = rejected
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer="blocks.28.mlp.l3",
            )
            with pytest.raises(ValueError, match="rejected by NIM API"):
                backend.extract_embeddings(request)

        # 422 should not retry
        assert mock_client.post.call_count == 1

    @patch("virosense.backends.nim.time.sleep")
    def test_unexpected_error_raises_immediately(self, mock_sleep):
        backend = NIMBackend(api_key="test_key")
        error = make_httpx_response(500, text="Internal server error")

        mock_client = MagicMock()
        mock_client.post.return_value = error
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            request = EmbeddingRequest(
                sequences={"s1": "ATGC"},
                layer="blocks.28.mlp.l3",
            )
            with pytest.raises(RuntimeError, match="HTTP 500"):
                backend.extract_embeddings(request)

        assert mock_client.post.call_count == 1


# --- Backend property tests ---


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

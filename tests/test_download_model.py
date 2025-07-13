#!/usr/bin/env python3

import tempfile
from pathlib import Path

import pytest

from fetch.predict_onnx import MODEL_REGISTRY, calculate_md5, download_model


class TestDownloadModelIntegration:
    """Integration tests for actual model downloading"""

    def test_download_model_b_actual(self):
        """Test actual download of model 'b' (smallest model)"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Download model 'b'
            result_path = download_model("b", tmp_path)

            # Verify the file was created
            assert result_path.exists()
            assert result_path.name == "model_b.onnx"

            # Verify the file size is reasonable (should be around 87MB)
            file_size_mb = result_path.stat().st_size / (1024 * 1024)
            expected_size = MODEL_REGISTRY["b"]["size_mb"]

            # Allow some tolerance for size differences
            assert abs(file_size_mb - expected_size) < 5.0, (
                f"File size {file_size_mb:.2f}MB differs significantly from expected {expected_size}MB"
            )

            # Verify the hash matches
            actual_hash = calculate_md5(result_path)
            expected_hash = MODEL_REGISTRY["b"]["md5"]
            assert actual_hash == expected_hash, (
                f"Hash mismatch: expected {expected_hash}, got {actual_hash}"
            )

            print(
                f"Successfully downloaded model 'b': {file_size_mb:.2f}MB, hash verified"
            )

    def test_download_model_b_cached(self):
        """Test that cached model is used when available and valid"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # First download
            result_path1 = download_model("b", tmp_path)
            original_mtime = result_path1.stat().st_mtime

            # Second download (should use cached version)
            result_path2 = download_model("b", tmp_path)
            cached_mtime = result_path2.stat().st_mtime

            # File should be the same (not re-downloaded)
            assert result_path1 == result_path2
            assert original_mtime == cached_mtime

            print("Cached model was used correctly")

    def test_download_model_b_corrupted_cache(self):
        """Test re-download when cached model is corrupted"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_path = tmp_path / "model_b.onnx"

            # Create a corrupted cached file
            model_path.write_bytes(b"corrupted content")
            original_mtime = model_path.stat().st_mtime

            # Download should detect corruption and re-download
            result_path = download_model("b", tmp_path)
            new_mtime = result_path.stat().st_mtime

            # File should have been replaced
            assert result_path == model_path
            assert new_mtime > original_mtime

            # Verify the new file is valid
            actual_hash = calculate_md5(result_path)
            expected_hash = MODEL_REGISTRY["b"]["md5"]
            assert actual_hash == expected_hash

            print("Corrupted cache was detected and model re-downloaded")

    @pytest.mark.parametrize("model_idx", ["a", "c", "d"])
    def test_download_model_registry_other_models(self, model_idx):
        """Test that other models in registry are downloadable (metadata check only)"""
        # Just verify the registry entry is well-formed for other models
        # without actually downloading them (to save time/bandwidth)

        model_info = MODEL_REGISTRY[model_idx]

        # Verify URL is accessible (just check format, don't download)
        assert model_info["url"].startswith("https://zenodo.org/")
        assert f"model_{model_idx}.onnx" in model_info["url"]

        # Verify hash format
        assert len(model_info["md5"]) == 32
        assert all(c in "0123456789abcdef" for c in model_info["md5"])

        # Verify size is reasonable
        assert 0 < model_info["size_mb"] < 500  # All models should be under 500MB

        print(f"Model {model_idx} registry entry is valid")

    def test_download_model_network_timeout(self):
        """Test behavior with network issues (mocked)"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # This test would require mocking to simulate network issues
            # For now, just verify that the function handles the case properly
            # by testing with an invalid model that should fail quickly

            # Temporarily modify registry to test error handling
            original_url = MODEL_REGISTRY["b"]["url"]
            MODEL_REGISTRY["b"]["url"] = (
                "https://invalid-url-that-does-not-exist.com/model.onnx"
            )

            try:
                with pytest.raises(RuntimeError, match="Failed to download model b"):
                    download_model("b", tmp_path)
            finally:
                # Restore original URL
                MODEL_REGISTRY["b"]["url"] = original_url

            print("Network error handling works correctly")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])

from typing import Never

import pytest

from dpt import gpu_check


def test_gpu_command_fails_clearly_when_optional_dependency_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing_dependency() -> Never:
        raise gpu_check.PreflightError(
            "optional dependency 'torch' is not installed; run `uv sync --group gpu` first"
        )

    monkeypatch.setattr(gpu_check, "run_preflight", missing_dependency)

    assert gpu_check.main([]) == 1
    captured = capsys.readouterr()
    assert "GPU preflight FAILED" in captured.err
    assert "uv sync --group gpu" in captured.err
    assert "PASSED" not in captured.out


def test_gpu_command_help_describes_real_cuda_check(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        gpu_check.main(["--help"])

    assert "real tiny CUDA operations" in capsys.readouterr().out

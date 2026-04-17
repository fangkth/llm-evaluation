"""deployment_context：本机 URL 识别与环境说明字段。"""

from __future__ import annotations

from utils.deployment_context import (
    build_environment_assumptions,
    service_url_looks_local,
)


def test_localhost_urls() -> None:
    assert service_url_looks_local("http://127.0.0.1:8000/v1")
    assert service_url_looks_local("http://localhost:8000")
    assert service_url_looks_local("http://[::1]:8000/")


def test_environment_assumptions_local() -> None:
    env = build_environment_assumptions("http://127.0.0.1:8000")
    assert env["resource_sampling_scope"] == "local_machine"
    assert env["deployment_scope"] == "single_node_only"
    assert env["remote_service_warning"] is False
    assert env["warning_message"] == ""


def test_environment_assumptions_remote_placeholder() -> None:
    env = build_environment_assumptions("http://example.invalid:8000")
    assert env["remote_service_warning"] is True
    assert env["warning_message"]

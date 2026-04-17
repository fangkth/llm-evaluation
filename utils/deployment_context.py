"""单机部署与资源采样范围：判断 base_url 是否指向本机，并生成 summary 用的环境说明字段。"""

from __future__ import annotations

import socket
from typing import Any
from urllib.parse import urlparse


def _machine_ipv4_addresses() -> set[str]:
    """本机 IPv4（仅用 stdlib，避免在受限环境下调用底层网卡枚举）。"""
    ips: set[str] = set()
    try:
        hn = socket.gethostname()
        for res in socket.getaddrinfo(hn, None, family=socket.AF_INET):
            ips.add(res[4][0])
    except OSError:
        pass
    try:
        ips.add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    return ips


def service_url_looks_local(base_url: str) -> bool:
    """判断服务 URL 的主机是否为本机常见地址（loopback 或本机网卡 IPv4）。"""
    try:
        p = urlparse(base_url.strip())
        host_raw = p.hostname
    except Exception:
        return False
    if not host_raw:
        return False
    host = host_raw.strip().lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    if host.startswith("127."):
        return True
    if host in ("::1", "[::1]"):
        return True
    if host.startswith("[") and host.endswith("]"):
        inner = host[1:-1].lower()
        if inner == "::1":
            return True

    local_ips = _machine_ipv4_addresses()
    try:
        for fam, _, _, _, sockaddr in socket.getaddrinfo(
            host_raw, None, type=socket.SOCK_STREAM
        ):
            ip = sockaddr[0]
            if isinstance(ip, str):
                if ip.startswith("127.") or ip == "::1":
                    return True
                if fam == socket.AF_INET and ip in local_ips:
                    return True
    except OSError:
        pass
    return False


def build_environment_assumptions(base_url: str) -> dict[str, Any]:
    """写入 summary.json 的固定字段 + 远端服务警告。"""
    remote = not service_url_looks_local(base_url)
    warn_msg = ""
    if remote:
        warn_msg = (
            "base_url 未识别为本机地址：资源采样仅在工具运行机上进行，"
            "与远端推理服务实际占用不一定一致；资源瓶颈类结论需谨慎解读。"
        )
    return {
        "resource_sampling_scope": "local_machine",
        "deployment_scope": "single_node_only",
        "remote_service_warning": remote,
        "warning_message": warn_msg,
    }

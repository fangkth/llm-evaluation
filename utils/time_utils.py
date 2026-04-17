"""时间工具函数。"""

from datetime import datetime


def run_tag_from_now() -> str:
    """生成基于当前时间的运行标签，格式：run_20260417_143000。"""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """将秒数格式化为可读字符串，如 1h23m45s。"""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"

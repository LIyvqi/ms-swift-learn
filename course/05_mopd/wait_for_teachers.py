#!/usr/bin/env python3

import json
import os
import sys
import time
import urllib.request


def ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            json.load(response)
        return True
    except Exception:
        return False


deadline = time.monotonic() + 600
pending = set(sys.argv[1:])
pids = [int(pid) for pid in os.environ.get("TEACHER_PIDS", "").split(",") if pid]
while pending and time.monotonic() < deadline:
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise SystemExit(f"教师进程 {pid} 已提前退出，请查看 course/05_mopd/logs")
    pending = {url for url in pending if not ready(url)}
    if pending:
        time.sleep(2)
if pending:
    raise SystemExit(f"教师服务启动超时: {sorted(pending)}")
print("两个教师服务均已就绪")

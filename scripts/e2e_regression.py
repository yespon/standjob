#!/usr/bin/env python3
"""Minimal E2E regression checks for standjob backend API.

Covers:
1) session bootstrap
2) upload endpoint
3) review flow progression
4) stuck escalation fields (hint_level / stuck_counter)

Run:
  python3 scripts/e2e_regression.py
"""
from __future__ import annotations

import io
import json
import os
import uuid
import mimetypes
import urllib.request

from openpyxl import Workbook

BASE_URL = os.environ.get("STANDJOB_BASE_URL", "http://localhost:3000")


def _request(method: str, path: str, payload: dict | None = None, headers: dict | None = None) -> dict:
    body = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        url=f"{BASE_URL}{path}",
        data=body,
        headers=req_headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def _multipart_upload(path: str, filename: str, data: bytes) -> dict:
    boundary = f"----standjob-{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    buffer = io.BytesIO()
    buffer.write(f"--{boundary}\r\n".encode("utf-8"))
    buffer.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8")
    )
    buffer.write(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    buffer.write(data)
    buffer.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url=f"{BASE_URL}{path}",
        data=buffer.getvalue(),
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def _build_sample_xlsx() -> tuple[str, bytes]:
    wb = Workbook()
    ws = wb.active
    ws.title = "岗位价值与岗位任务"
    ws.append(["岗位价值", "岗位效能", "核心任务", None, None, "资源投入", "辅助任务", None, None, "资源投入"])
    ws.append([None, None, "任务名称", "任务目的", "成果标准", None, "任务名称", "任务目的", "成果标准", None])
    ws.append([
        "负责软件开发工作",
        "无",
        "写代码",
        "完成功能开发",
        "完成任务",
        "60%",
        "开会",
        "信息同步",
        "参加会议",
        "40%",
    ])

    output = io.BytesIO()
    wb.save(output)
    return "regression.xlsx", output.getvalue()


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    print("[1/5] start session")
    session = _request("POST", "/api/session/start")
    thread_id = session["thread_id"]
    assert_true(session["phase"] == "loaded", "phase should be loaded after start")

    print("[2/5] upload xlsx")
    filename, filedata = _build_sample_xlsx()
    upload = _multipart_upload("/api/upload", filename, filedata)
    assert_true(upload.get("file_path"), "upload should return file_path")

    print("[3/5] trigger parse+review")
    chat1 = _request(
        "POST",
        "/api/chat",
        {
            "thread_id": thread_id,
            "message": "我已上传文件，请开始",
            "file_path": upload["file_path"],
        },
    )
    assert_true(chat1["phase"] in {"guiding", "reviewing"}, "phase should enter review flow")

    print("[4/5] send unresolved replies to trigger escalation")
    for _ in range(4):
        _request(
            "POST",
            "/api/chat",
            {
                "thread_id": thread_id,
                "message": "我再想想",
                "file_path": None,
            },
        )

    print("[5/5] verify state fields")
    state = _request("GET", f"/api/session/{thread_id}/state")
    assert_true("current_focus_id" in state, "state must expose current_focus_id")
    assert_true("hint_level" in state, "state must expose hint_level")
    assert_true("stuck_counter" in state, "state must expose stuck_counter")
    assert_true("rubric_eval_summary" in state, "state must expose rubric_eval_summary")
    assert_true(state["hint_level"] >= 1, "hint_level should increase after repeated unresolved replies")

    print("PASS: e2e regression checks completed")


if __name__ == "__main__":
    main()

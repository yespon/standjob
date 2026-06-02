#!/usr/bin/env python3
"""管理岗标教练的评审进度文件。

用法:
    python progress.py init <xlsx_path> [items_json]   初始化进度文件
    python progress.py show                            显示当前进度
    python progress.py update <item> <status> [note]   更新某条状态
    python progress.py next                            推进到下一个待处理项
    python progress.py reset                           清除进度文件

item 编号: 1-14 对应评分标准条目
status: pending / fail / pass / skip
"""
import sys
import json
from pathlib import Path

PROGRESS_FILE = Path("/tmp/gangbiao-coach-progress.json")


def load():
    if not PROGRESS_FILE.exists():
        return None
    with open(PROGRESS_FILE) as f:
        return json.load(f)


def save(data):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cmd_init(args):
    if len(args) < 1:
        print("用法: progress.py init <xlsx_path> [items_json]", file=sys.stderr)
        sys.exit(1)
    xlsx_path = args[0]
    # 可选：直接传入 items dict 的 JSON
    items = {}
    if len(args) > 1:
        try:
            items = json.loads(args[1])
        except json.JSONDecodeError:
            print("items_json 格式错误", file=sys.stderr)
            sys.exit(1)
    else:
        for i in range(1, 15):
            items[str(i)] = {"status": "pending", "evidence": ""}

    data = {
        "file": xlsx_path,
        "current_item": 1,
        "items": items,
        "history": []
    }
    save(data)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_show(_args):
    data = load()
    if not data:
        print("无进度文件")
        return
    summary = {"file": data["file"], "current_item": data["current_item"]}
    by_status = {"fail": [], "pass": [], "pending": [], "skip": []}
    for k, v in data["items"].items():
        by_status.get(v["status"], []).append(k)
    summary["counts"] = {s: sorted(ks) for s, ks in by_status.items() if ks}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_update(args):
    if len(args) < 2:
        print("用法: progress.py update <item> <status> [note]", file=sys.stderr)
        sys.exit(1)
    item, status = args[0], args[1]
    note = " ".join(args[2:]) if len(args) > 2 else ""
    if status not in ("pending", "fail", "pass", "skip"):
        print(f"无效状态: {status}，可选: pending/fail/pass/skip", file=sys.stderr)
        sys.exit(1)
    data = load()
    if not data:
        print("无进度文件，请先 init", file=sys.stderr)
        sys.exit(1)
    if item not in data["items"]:
        print(f"无效条目: {item}，可选: 1-14", file=sys.stderr)
        sys.exit(1)
    data["items"][item]["status"] = status
    if note:
        data["items"][item]["evidence"] = note
    data["history"].append({"item": item, "action": f"→{status}", "note": note})
    save(data)
    print(f"条目 {item} → {status}" + (f" ({note})" if note else ""))


def cmd_next(_args):
    data = load()
    if not data:
        print("无进度文件", file=sys.stderr)
        sys.exit(1)
    # 找下一个 fail 或 pending 的条目（按 1-14 顺序）
    for i in range(1, 15):
        k = str(i)
        if k in data["items"] and data["items"][k]["status"] in ("fail", "pending"):
            data["current_item"] = i
            save(data)
            print(f"当前条目: {k} ({data['items'][k]['status']})")
            return
    print("所有条目已处理完毕")


def cmd_reset(_args):
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("进度文件已清除")
    else:
        print("无进度文件")


COMMANDS = {
    "init": cmd_init,
    "show": cmd_show,
    "update": cmd_update,
    "next": cmd_next,
    "reset": cmd_reset,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"未知命令: {cmd}，可选: {', '.join(COMMANDS)}", file=sys.stderr)
        sys.exit(1)
    COMMANDS[cmd](sys.argv[2:])

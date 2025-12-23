"""
一次性批量写入用户 user101-user150 的种子脚本。

默认密码：pbl123；可通过命令行参数覆盖。
运行示例：
    python backend/seed_users.py
    python backend/seed_users.py --start 101 --end 120 --password mypass
    python backend/seed_users.py --db backend/data/pbl.db
"""
from __future__ import annotations

import argparse
import logging
from typing import List, Tuple

import sys
from pathlib import Path

# 允许脚本在直接调用时找到 backend 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import get_connection


def build_users(start: int, end: int, password: str) -> List[Tuple[str, str]]:
    if start > end:
        start, end = end, start
    return [(f"user{i}", password) for i in range(start, end + 1)]


def seed_users(start: int, end: int, password: str, db_path: str | None = None) -> int:
    users = build_users(start, end, password)
    conn = get_connection(db_path)
    cur = conn.cursor()
    inserted = 0
    for username, pwd in users:
        cur.execute(
            "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
            (username, pwd),
        )
        if cur.rowcount > 0:
            inserted += 1
    conn.commit()
    conn.close()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed exam users into the database.")
    parser.add_argument("--start", type=int, default=101, help="起始编号，默认 101")
    parser.add_argument("--end", type=int, default=150, help="结束编号，默认 150（包含）")
    parser.add_argument("--password", type=str, default="pbl123", help="默认密码")
    parser.add_argument("--db", type=str, default=None, help="自定义数据库路径，可选")
    args = parser.parse_args()

    inserted = seed_users(args.start, args.end, args.password, args.db)
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "已完成导入，范围 user%s-user%s，新增 %s 条记录（已存在的会忽略）。",
        args.start,
        args.end,
        inserted,
    )


if __name__ == "__main__":
    main()

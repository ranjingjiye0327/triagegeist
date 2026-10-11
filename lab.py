# -*- coding: utf-8 -*-
"""
SQL 练习环境
============
第一次运行：把 data/ 里的四个 CSV 导入 SQLite，生成 triagegeist.db
之后每次运行：执行 query.sql 里的 SQL，把结果打印出来

用法：
    1. 把你要写的 SQL 存进 query.sql
    2. 运行  python lab.py
"""

import os
import sqlite3

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
DB_PATH = os.path.join(BASE, "triagegeist.db")
QUERY_PATH = os.path.join(BASE, "query.sql")

TABLES = [
    ("train", "train.csv"),
    ("test", "test.csv"),
    ("patient_history", "patient_history.csv"),
    ("chief_complaints", "chief_complaints.csv"),
]


def build_db():
    """把 CSV 导入 SQLite。只在第一次运行时执行。"""
    print("首次运行，正在导入数据（大概要十几秒）...")
    conn = sqlite3.connect(DB_PATH)
    for table, csv_name in TABLES:
        df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"  {table:<20} {len(df):>7} 行  {len(df.columns):>3} 列")
    conn.commit()
    conn.close()
    print(f"已生成数据库：{DB_PATH}\n")


def strip_comments(text):
    """去掉以 -- 开头的整行注释。"""
    return "\n".join(line for line in text.splitlines()
                     if not line.strip().startswith("--"))


def run_query():
    """执行 query.sql，可以一次写多条（用分号隔开），逐条打印结果。"""
    if not os.path.exists(QUERY_PATH):
        print(f"没找到 {QUERY_PATH}")
        return

    sql_text = open(QUERY_PATH, encoding="utf-8").read()
    statements = [s.strip() for s in strip_comments(sql_text).split(";") if s.strip()]

    if not statements:
        print("query.sql 里没有可执行的语句，先写点 SQL 进去。")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        for i, stmt in enumerate(statements, 1):
            print("─" * 70)
            if len(statements) > 1:
                print(f"[第 {i} 条 / 共 {len(statements)} 条]")
            print(stmt)
            print("─" * 70)
            kind = stmt.lstrip().upper()
            if kind.startswith(("SELECT", "WITH", "PRAGMA", "EXPLAIN")):
                # 查询语句：把结果读成表打印出来
                try:
                    df = pd.read_sql_query(stmt, conn)
                except Exception as e:
                    print(f"\n[报错] {type(e).__name__}: {e}\n")
                    continue
                with pd.option_context("display.max_columns", 60,
                                       "display.width", 250,
                                       "display.max_colwidth", 24):
                    print(df.head(20).to_string(index=False))
                print(f"\n返回 {len(df)} 行 × {len(df.columns)} 列\n")
            else:
                # 建表/删表/插入这类语句没有结果集，直接执行
                try:
                    conn.execute(stmt)
                    conn.commit()
                except Exception as e:
                    print(f"\n[报错] {type(e).__name__}: {e}\n")
                    continue
                print("执行成功（这条语句不返回结果集）\n")
    finally:
        conn.close()


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        build_db()
    run_query()

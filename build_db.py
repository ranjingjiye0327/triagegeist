# -*- coding: utf-8 -*-
"""
建库脚本
========
把 data/ 里的四个 CSV 导入 SQLite，再执行 sql/ 目录下的脚本，
生成分析宽表 wide_train / wide_test。

数据变了、SQL 改了，重跑一遍就行：

    python build_db.py

之后建模用的 preprocessing.py 会直接从这两张宽表读数据 ——
也就是说，「把四张表接成一张」这件事，从此由 SQL 负责，不再由 pandas 负责。
"""

import glob
import os
import sqlite3

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
SQL_DIR = os.path.join(BASE, "sql")
DB_PATH = os.path.join(BASE, "triagegeist.db")

# 原始表和源文件
TABLES = [
    ("train", "train.csv"),
    ("test", "test.csv"),
    ("patient_history", "patient_history.csv"),
    ("chief_complaints", "chief_complaints.csv"),
]


def import_csvs(conn):
    """把四个 CSV 原样导进数据库。"""
    print("① 导入原始 CSV")
    for table, csv_name in TABLES:
        df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"   {table:<18} {len(df):>7} 行 × {len(df.columns):>3} 列")


def run_sql_files(conn):
    """按文件名顺序执行 sql/ 下的所有 .sql。"""
    print("② 执行 sql/ 下的脚本")
    files = sorted(glob.glob(os.path.join(SQL_DIR, "*.sql")))
    if not files:
        print("   （sql/ 里没有 .sql 文件）")
        return
    for path in files:
        with open(path, encoding="utf-8") as f:
            conn.executescript(f.read())
        print(f"   {os.path.basename(path)}  已执行")


def report(conn):
    """验收：行数、列数、有没有重复的 patient_id。"""
    print("③ 验收")
    for table in ["wide_train", "wide_test"]:
        n, = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        ncols = len(conn.execute(f"SELECT * FROM {table} LIMIT 1").description)
        dup, = conn.execute(
            f"SELECT COUNT(*) - COUNT(DISTINCT patient_id) FROM {table}"
        ).fetchone()
        flag = "OK" if dup == 0 else "有重复！"
        print(f"   {table:<11} {n:>7} 行 × {ncols:>3} 列   重复 patient_id: {dup}  {flag}")


def build():
    conn = sqlite3.connect(DB_PATH)
    try:
        import_csvs(conn)
        run_sql_files(conn)
        conn.commit()
        report(conn)
    finally:
        conn.close()
    print(f"\n数据库已生成：{DB_PATH}")


if __name__ == "__main__":
    build()

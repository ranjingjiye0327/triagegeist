# -*- coding: utf-8 -*-
"""
主诉文本实验：文本里到底藏了多少信息？
======================================
回答一个问题：chief_complaint_raw 对预测分诊等级有没有用？

用法（在 triagegeist 根目录下执行）：

    python src/text_experiment.py

结论先写在这（跑完可对照）：

    只用结构化特征        QWK ≈ 0.899
    只用主诉文本          QWK ≈ 0.999
    结构化 + 主诉文本     QWK ≈ 0.999   （和只用文本差距在噪声范围内）

    原因不在模型，在数据：4,949 条不同的主诉文本里，
    4,934 条唯一对应一个分诊等级。数据生成的过程把标签
    写进了文本 —— 所以文本一加进来，模型等于在查表。

这也是为什么本项目的主模型**只用结构化特征**：
把文本加进去得到的高分不是建模能力，而是数据集的性质。
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "triagegeist.db")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_text_and_structured():
    """读宽表里的主诉文本，并取结构化特征矩阵。两者按行对齐。"""
    from preprocessing import get_processed_dataframes

    X, _X_test, y, _ids = get_processed_dataframes()

    conn = sqlite3.connect(DB_PATH)
    try:
        raw = pd.read_sql_query(
            "SELECT chief_complaint_raw, triage_acuity FROM wide_train", conn)
    finally:
        conn.close()

    if not (raw["triage_acuity"].values == y.values).all():
        raise RuntimeError("文本和特征矩阵的行顺序没有对齐")

    text = raw["chief_complaint_raw"].fillna("").values
    return X, y, text, raw["chief_complaint_raw"]


def check_determinism(raw_text, y):
    """每条主诉文本对应几个分诊等级？"""
    g = pd.DataFrame({"t": raw_text, "y": y.values}).groupby("t")["y"] \
          .agg(["nunique", "size"])
    single = int((g["nunique"] == 1).sum())
    multi = g[g["nunique"] > 1]

    print("\n① 文本与标签的对应关系")
    print(f"   不同的主诉文本            {len(g):>6,} 条")
    print(f"   唯一对应一个分诊等级的    {single:>6,} 条  ({single / len(g) * 100:.2f}%)")
    print(f"   对应两个等级的            {len(multi):>6,} 条  "
          f"（覆盖 {int(multi['size'].sum()):,} 行，占 {multi['size'].sum() / len(raw_text) * 100:.2f}%）")
    return single, len(g)


def evaluate(X, y, text, use_structured, use_text, n_splits=3):
    """分层交叉验证。文本向量化放在折内做，避免信息泄漏。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    qwk, acc, n_feat = [], [], []

    for tr, va in skf.split(X, y):
        parts_tr, parts_va = [], []
        if use_structured:
            parts_tr.append(sparse.csr_matrix(X.values[tr]))
            parts_va.append(sparse.csr_matrix(X.values[va]))
        if use_text:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                  max_features=20000, sublinear_tf=True)
            parts_tr.append(vec.fit_transform(text[tr]))
            parts_va.append(vec.transform(text[va]))

        # 只有一块特征时直接用它，两块以上才需要横向拼接
        A = sparse.hstack(parts_tr).tocsr() if len(parts_tr) > 1 else parts_tr[0]
        B = sparse.hstack(parts_va).tocsr() if len(parts_va) > 1 else parts_va[0]

        model = LogisticRegression(max_iter=1000, C=1.0,
                                   class_weight="balanced", solver="lbfgs")
        model.fit(A, y[tr])
        pred = model.predict(B)
        qwk.append(cohen_kappa_score(y[va], pred, weights="quadratic"))
        acc.append(accuracy_score(y[va], pred))
        n_feat.append(A.shape[1])

    return float(np.mean(qwk)), float(np.mean(acc)), int(np.mean(n_feat))


def main():
    X, y, text, raw_text = load_text_and_structured()
    print(f"样本 {len(y):,} 条，结构化特征 {X.shape[1]} 列")

    single, total = check_determinism(raw_text, y)

    print("\n② 三组对照实验（逻辑回归，3 折交叉验证）")
    configs = [
        ("只用结构化特征",    True,  False),
        ("只用主诉文本",      False, True),
        ("结构化 + 主诉文本", True,  True),
    ]
    results = {}
    for label, s, t in configs:
        q, a, n = evaluate(X, y, text, s, t)
        results[label] = (q, a, n)
        print(f"   {label:<18} QWK={q:.4f}   Accuracy={a:.4f}   特征数={n:,}")

    base = results["只用结构化特征"][0]
    both = results["结构化 + 主诉文本"][0]
    print("\n③ 结论")
    print(f"   加上主诉文本，QWK 从 {base:.4f} 变成 {both:.4f}（{both - base:+.4f}）")
    print(f"   原因：{single:,}/{total:,} 条文本唯一对应一个等级，文本里已经写着标签。")
    print("   所以本项目的主模型只用结构化特征 —— 0.93 是模型的成绩，")
    print("   0.999 是数据集的性质，两者不能混为一谈。")


if __name__ == "__main__":
    main()

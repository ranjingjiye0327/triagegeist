# -*- coding: utf-8 -*-
"""
生成报告用的全部图表
====================
用法（在 triagegeist 根目录下执行）：

    python src/figures.py            # 数据侧图表（图 1–7），几秒钟
    python src/figures.py --model    # 额外训练随机森林，生成图 8、图 9（几分钟）

数据来源：triagegeist.db 里的宽表（由 build_db.py 用 SQL 生成）。
图表输出到 figures/。
"""

import argparse
import os
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "triagegeist.db")
OUT_DIR = os.path.join(BASE, "figures")
TARGET = "triage_acuity"

# 中文标签依赖 Windows 自带的字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110


def save(fig, name):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  已生成 {name}")


def load_wide():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"没找到数据库 {DB_PATH}，先运行：python build_db.py")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM wide_train", conn)
    finally:
        conn.close()
    return df


# ── 图 1：目标变量分布 ───────────────────────────────────────
def fig01_target_distribution(df):
    counts = df[TARGET].value_counts().sort_index()
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    for b, c, p in zip(bars, counts.values, pct.values):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("分诊等级（1 = 最危重，5 = 最不紧急）")
    ax.set_ylabel("人次")
    ax.set_title("图 1  分诊等级分布（训练集 80,000 人次）")
    ax.set_ylim(0, counts.max() * 1.20)
    sns.despine()
    save(fig, "01_target_distribution.png")


# ── 图 2：关键指标随等级的变化 ───────────────────────────────
def fig02_vitals_by_triage(df):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    sns.boxplot(data=df, x=TARGET, y="news2_score", ax=axes[0], color="#DD8452")
    axes[0].set_title("NEWS2 早期预警评分")
    sns.boxplot(data=df, x=TARGET, y="shock_index", ax=axes[1], color="#55A868")
    axes[1].set_title("休克指数")
    for ax in axes:
        ax.set_xlabel("分诊等级")
        ax.set_ylabel("")
    fig.suptitle("图 2  关键指标随分诊等级的变化", fontsize=13)
    save(fig, "02_news2_shock_by_triage.png")


# ── 图 3：数值特征相关性 ─────────────────────────────────────
def fig03_correlation(df):
    num = df.select_dtypes(include="number")
    # 只保留真正连续的列：0/1 的病史列和取值很少的列放进去会看不清
    keep = [c for c in num.columns if c not in ("arrival_hour", "arrival_month")
            and num[c].nunique() > 3]
    corr = num[keep].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10.5))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=.4, cbar_kws={"shrink": .6}, ax=ax)
    ax.set_title(f"图 3  数值特征相关性热力图（{len(keep)} 列）")
    save(fig, "03_correlation_heatmap.png")


# ── 图 4：缺失情况 ───────────────────────────────────────────
def fig04_missing(df):
    na = df.isnull().sum()
    na = na[na > 0].sort_values(ascending=False)
    if na.empty:
        print("  没有缺失值，跳过图 4")
        return

    rate = (na / len(df) * 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(rate.index[::-1], rate.values[::-1], color="#C44E52")
    for b, v in zip(bars, rate.values[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.2f}%", va="center", fontsize=9)
    ax.set_xlabel("缺失率 (%)")
    ax.set_title("图 4a  各字段缺失率")
    ax.set_xlim(0, rate.max() * 1.25)
    sns.despine()
    save(fig, "04_missing_summary.png")

    cols = list(na.index)
    sample = df[cols].isnull().sample(n=min(4000, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(10, 2.0 + 0.45 * len(cols)))
    ax.imshow(sample.values.T, aspect="auto", cmap="gray_r", interpolation="nearest")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xlabel(f"随机抽取的 {len(sample):,} 条记录")
    ax.set_title("图 4b  缺失分布（黑色 = 缺失）")
    save(fig, "04_missing_pattern.png")


# ── 图 5：主要分类字段分布 ───────────────────────────────────
def fig05_categorical(df):
    cols = ["sex", "age_group", "arrival_mode", "shift",
            "insurance_type", "mental_status_triage"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, c in zip(axes.ravel(), cols):
        vc = df[c].value_counts()
        ax.bar(vc.index.astype(str), vc.values, color="#4C72B0")
        ax.set_title(c, fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.set_ylabel("人次", fontsize=9)
    fig.suptitle("图 5  主要分类字段的分布", fontsize=13)
    save(fig, "05_categorical_dist.png")


# ── 图 6：既往病史患病率 ─────────────────────────────────────
def fig06_history(df):
    hx = [c for c in df.columns if c.startswith("hx_")]
    rate = (df[hx].mean() * 100).sort_values()

    fig, ax = plt.subplots(figsize=(8.5, 7))
    labels = [c[3:].replace("_", " ") for c in rate.index]
    ax.barh(labels, rate.values, color="#55A868")
    for i, v in enumerate(rate.values):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("患病比例 (%)")
    ax.set_title("图 6  既往病史患病率（25 项）")
    ax.set_xlim(0, rate.max() * 1.25)
    sns.despine()
    save(fig, "06_history_prevalence.png")


# ── 图 7：模型对比 ───────────────────────────────────────────
def fig07_model_comparison():
    path = os.path.join(BASE, "submissions", "model_comparison.csv")
    if not os.path.exists(path):
        print("  没找到 submissions/model_comparison.csv，跳过图 7")
        return

    df = pd.read_csv(path)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - 0.2, df["QWK"], width=0.4, label="QWK", color="#4C72B0")
    ax.bar(x + 0.2, df["Accuracy"], width=0.4, label="Accuracy", color="#DD8452")
    for i, (q, a) in enumerate(zip(df["QWK"], df["Accuracy"])):
        ax.text(i - 0.2, q, f"{q:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, a, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["模型"], rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.75, 0.97)
    ax.set_ylabel("交叉验证得分")
    ax.set_title("图 7  五个模型的五折交叉验证表现")
    ax.legend(loc="lower right")
    sns.despine()
    save(fig, "07_model_comparison.png")


# ── 图 8、9：需要训练一个模型 ────────────────────────────────
def fig08_09_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import get_processed_dataframes

    X, _X_test, y, _ids = get_processed_dataframes()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("  正在训练随机森林（这一步比较慢）...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=10,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)

    # 图 8：特征重要性
    imp = pd.Series(model.feature_importances_, index=X.columns) \
            .sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#4C72B0")
    ax.set_xlabel("重要性")
    ax.set_title("图 8  特征重要性 Top 20（随机森林）")
    sns.despine()
    save(fig, "08_feature_importance.png")

    # 图 9：混淆矩阵
    labels = sorted(y.unique())
    cm = confusion_matrix(y_val, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("预测等级")
    ax.set_ylabel("真实等级")
    ax.set_title("图 9  混淆矩阵（随机森林，20% 留出集）")
    save(fig, "09_confusion_matrix.png")


def main():
    ap = argparse.ArgumentParser(description="生成报告图表")
    ap.add_argument("--model", action="store_true",
                    help="额外训练随机森林，生成图 8、图 9（几分钟）")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_wide()
    print(f"宽表：{df.shape[0]:,} 行 × {df.shape[1]} 列\n开始画图：")

    fig01_target_distribution(df)
    fig02_vitals_by_triage(df)
    fig03_correlation(df)
    fig04_missing(df)
    fig05_categorical(df)
    fig06_history(df)
    fig07_model_comparison()

    if args.model:
        fig08_09_model()
    else:
        print("  提示：加 --model 可生成图 8（特征重要性）和图 9（混淆矩阵）")

    print(f"\n图表已输出到 {OUT_DIR}")


if __name__ == "__main__":
    main()

"""
Triagegeist — 主运行脚本
=======================
运行全部5个模型，保存提交文件，生成对比报告
"""

import numpy as np
import pandas as pd
import os
from preprocessing import get_processed_data
from models import ALL_MODELS

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_submission(test_ids, predictions, model_name):
    """保存Kaggle提交格式的CSV"""
    sub = pd.DataFrame({
        'patient_id': test_ids,
        'triage_acuity': predictions.astype(int)
    })
    path = f"{OUTPUT_DIR}/submission_{model_name}.csv"
    sub.to_csv(path, index=False)
    print(f"  提交文件已保存: {path}")
    return path


def main():
    # ── Step 1: 数据预处理 ──
    X_train, y_train, X_test, test_ids = get_processed_data()

    # ── Step 2: 运行所有模型 ──
    all_predictions = {}
    all_scores = {}

    for name, model_fn in ALL_MODELS.items():
        print(f"\n{'=' * 60}")
        print(f">>> {name}")
        print(f"{'=' * 60}")

        try:
            preds, scores = model_fn(X_train, y_train, X_test)
            all_predictions[name] = preds
            all_scores[name] = scores
            save_submission(test_ids, preds, name)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Step 3: 汇总报告 ──
    print("\n\n" + "=" * 70)
    print("模型对比总结")
    print("=" * 70)
    print(f"{'模型':<32} {'Accuracy':>10} {'QWK':>10}")
    print("-" * 60)

    for name, scores in all_scores.items():
        model_name = scores['name']
        print(f"{model_name:<32} {scores['accuracy']:>10.4f} {scores['qwk']:>10.4f}")

    # 最佳模型
    if all_scores:
        best = max(all_scores, key=lambda k: all_scores[k]['qwk'])
        print(f"\n{'=' * 70}")
        print(f"最佳模型: {all_scores[best]['name']}")
        print(f"Accuracy: {all_scores[best]['accuracy']:.4f}")
        print(f"QWK:      {all_scores[best]['qwk']:.4f}")
        print(f"\n推荐使用提交文件: submission_{best}.csv")

    # ── 保存汇总表 ──
    if all_scores:
        summary = pd.DataFrame([
            {'模型': v['name'], 'Accuracy': f"{v['accuracy']:.4f}", 'QWK': f"{v['qwk']:.4f}"}
            for k, v in all_scores.items()
        ])
        summary.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False, encoding='utf-8-sig')
        print(f"\n对比表已保存: {OUTPUT_DIR}/model_comparison.csv")


if __name__ == "__main__":
    main()

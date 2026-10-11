"""
Triagegeist — 5个不同模型实现
===============================
每个模型封装为独立函数 f(X_train, y_train, X_test)，便于统一对比。

基于文献:
- Raita et al. (2019) Critical Care: DNN, RF, GBT, Lasso 全优于ESI基线
- Wicha et al. (2025) JCMR: XGBoost > RF > GBM for ESI 1-3
- Mejri (2026): 有序Logistic回归作为主要模型, ML作为对比
- Tougma et al. (2025): XGBoost & LightGBM 82.3% accuracy on ESI 1-5
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

N_FOLDS = 5


def quadratic_weighted_kappa(y_true, y_pred):
    """二次加权Kappa —— 适用于有序分类的评估指标"""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def cv_evaluate(model, X, y, n_folds=N_FOLDS):
    """5折交叉验证，返回 accuracy 和 QWK"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    acc_scores, qwk_scores = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_tr, y_tr)
        y_pred = model_copy.predict(X_val)

        acc_scores.append(accuracy_score(y_val, y_pred))
        qwk_scores.append(quadratic_weighted_kappa(y_val, y_pred))

    return np.mean(acc_scores), np.std(acc_scores), np.mean(qwk_scores), np.std(qwk_scores)


# ═══════════════════════════════════════════════════════════════
# 模型 1: 有序 Logistic 回归
# 文献: Mejri (2026) 推荐为 ordinal ESI 的主要模型
# 特点: 统计经典，可解释性强，处理有序分类
# ═══════════════════════════════════════════════════════════════
def model_ordinal_logistic(X_train, y_train, X_test):
    """
    有序Logistic回归 —— 经典统计方法
    使用 multinomial logistic + 更强的正则化来近似有序分类
    """
    print("\n" + "=" * 60)
    print("模型1: 有序Logistic回归")
    print("=" * 60)

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        C=0.5,                   # L2正则化强度
        class_weight='balanced',
        random_state=42
    )

    # 交叉验证
    acc, acc_std, qwk, qwk_std = cv_evaluate(model, X_train, y_train)
    print(f"  CV Accuracy: {acc:.4f} ± {acc_std:.4f}")
    print(f"  CV QWK:      {qwk:.4f} ± {qwk_std:.4f}")

    # 全量训练 & 预测
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return preds, {'accuracy': acc, 'qwk': qwk, 'name': 'Ordinal Logistic Regression'}


# ═══════════════════════════════════════════════════════════════
# 模型 2: 随机森林
# 文献: Raita 2019, 台湾2022研究, Tougma 2025
# 特点: Bagging集成，可解释性最强，不易过拟合
# ═══════════════════════════════════════════════════════════════
def model_random_forest(X_train, y_train, X_test):
    """
    随机森林 —— Bagging集成方法
    适合处理高维特征，可解释性好(SHAP重要性)
    """
    print("\n" + "=" * 60)
    print("模型2: 随机森林")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )

    acc, acc_std, qwk, qwk_std = cv_evaluate(model, X_train, y_train)
    print(f"  CV Accuracy: {acc:.4f} ± {acc_std:.4f}")
    print(f"  CV QWK:      {qwk:.4f} ± {qwk_std:.4f}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # 特征重要性
    importances = model.feature_importances_
    top5_idx = np.argsort(importances)[-5:][::-1]
    print(f"  Top5特征: (需要feature_names来显示名称)")

    return preds, {'accuracy': acc, 'qwk': qwk, 'name': 'Random Forest'}


# ═══════════════════════════════════════════════════════════════
# 模型 3: XGBoost
# 文献: Wicha 2025的最佳模型, 几乎所有研究的前两名
# 特点: 梯度提升，性能通常最优，正则化强
# ═══════════════════════════════════════════════════════════════
def model_xgboost(X_train, y_train, X_test):
    """
    XGBoost —— 梯度提升树
    几乎所有ESI预测文献中表现最好的模型
    """
    print("\n" + "=" * 60)
    print("模型3: XGBoost")
    print("=" * 60)

    # 计算类别权重
    unique, counts = np.unique(y_train, return_counts=True)
    scale_pos_weight = counts.max() / counts  # 多数类/各类

    # XGBoost 要求标签 0-based，将1-5映射为0-4
    y_train_xgb = y_train - 1

    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softmax',
        num_class=5,
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )

    acc, acc_std, qwk, qwk_std = cv_evaluate(model, X_train, y_train_xgb)
    print(f"  CV Accuracy: {acc:.4f} ± {acc_std:.4f}")
    print(f"  CV QWK:      {qwk:.4f} ± {qwk_std:.4f}")

    model.fit(X_train, y_train_xgb)
    preds = model.predict(X_test) + 1  # 映射回1-5

    return preds, {'accuracy': acc, 'qwk': qwk, 'name': 'XGBoost'}


# ═══════════════════════════════════════════════════════════════
# 模型 4: LightGBM
# 文献: Tougma 2025 并列最佳, leaf-wise生长策略不同于XGBoost
# 特点: leaf-wise树生长，训练快，处理分类特征强
# ═══════════════════════════════════════════════════════════════
def model_lightgbm(X_train, y_train, X_test):
    """
    LightGBM —— 轻量梯度提升
    leaf-wise生长 vs XGBoost的level-wise，方法有本质区别
    """
    print("\n" + "=" * 60)
    print("模型4: LightGBM")
    print("=" * 60)

    model = LGBMClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        objective='multiclass',
        num_class=5,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )

    acc, acc_std, qwk, qwk_std = cv_evaluate(model, X_train, y_train)
    print(f"  CV Accuracy: {acc:.4f} ± {acc_std:.4f}")
    print(f"  CV QWK:      {qwk:.4f} ± {qwk_std:.4f}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return preds, {'accuracy': acc, 'qwk': qwk, 'name': 'LightGBM'}


# ═══════════════════════════════════════════════════════════════
# 模型 5: 神经网络 MLP
# 文献: Raita 2019 DNN AUC=0.86 vs ESI 0.74, Ketseridou 2021 CNN
# 特点: 深度学习，捕捉非线性交互，完全不同范式
# ═══════════════════════════════════════════════════════════════
def model_neural_network(X_train, y_train, X_test):
    """
    多层感知机 —— 深度学习方法
    与树模型完全不同的学习范式，捕捉复杂特征交互
    """
    print("\n" + "=" * 60)
    print("模型5: 神经网络 MLP")
    print("=" * 60)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,        # L2正则化
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )

    acc, acc_std, qwk, qwk_std = cv_evaluate(model, X_train, y_train)
    print(f"  CV Accuracy: {acc:.4f} ± {acc_std:.4f}")
    print(f"  CV QWK:      {qwk:.4f} ± {qwk_std:.4f}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return preds, {'accuracy': acc, 'qwk': qwk, 'name': 'Neural Network (MLP)'}


# ═══════════════════════════════════════════════════════════════
# 统一接口：跑所有模型
# ═══════════════════════════════════════════════════════════════
ALL_MODELS = {
    'OrdinalLogistic': model_ordinal_logistic,
    'RandomForest':    model_random_forest,
    'XGBoost':         model_xgboost,
    'LightGBM':        model_lightgbm,
    'NeuralNetwork':   model_neural_network,
}


def run_all_models(X_train, y_train, X_test):
    """运行全部5个模型，返回预测结果和评分"""
    results = {}
    predictions = {}

    for name, model_fn in ALL_MODELS.items():
        print(f"\n{'#' * 60}")
        print(f"# {name}")
        print(f"{'#' * 60}")

        preds, scores = model_fn(X_train, y_train, X_test)
        predictions[name] = preds
        results[name] = scores

    # 汇总
    print("\n" + "=" * 60)
    print("模型对比汇总")
    print("=" * 60)
    print(f"{'模型':<30} {'Accuracy':>10} {'QWK':>10}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{r['name']:<30} {r['accuracy']:>10.4f} {r['qwk']:>10.4f}")

    best_model = max(results, key=lambda k: results[k]['qwk'])
    print(f"\n最佳模型: {results[best_model]['name']} (QWK={results[best_model]['qwk']:.4f})")

    return predictions, results


if __name__ == "__main__":
    from preprocessing import get_processed_data
    X_train, y_train, X_test, test_ids = get_processed_data()
    predictions, results = run_all_models(X_train, y_train, X_test)

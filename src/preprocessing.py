"""
Triagegeist 数据预处理流水线
=====================================
数据读入、清洗、特征工程、编码。

主入口 get_processed_data() 返回处理好的特征矩阵与标签，
模型侧只需要 f(X_train, y_train, X_test) 这一个接口。
"""

import pandas as pd
import numpy as np
import os
import sqlite3
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 宽表（wide_train / wide_test）由 build_db.py 用 SQL 生成，
# 定义在 triagegeist/sql/01_wide_tables.sql 里。
# 本文件在 src/ 子目录下，往上一级就是项目根目录。
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "triagegeist.db")


def load_raw_data():
    """读取分析宽表。

    四张原始表的关联（train + patient_history + chief_complaints）
    已经在 SQL 里做完了，这里只负责把宽表读进来。
    宽表怎么建的 → sql/01_wide_tables.sql
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"没找到数据库：{DB_PATH}\n"
            "先在这个文件夹的上一级目录运行一次：python build_db.py"
        )

    conn = sqlite3.connect(DB_PATH)
    try:
        train = pd.read_sql_query("SELECT * FROM wide_train", conn)
        test = pd.read_sql_query("SELECT * FROM wide_test", conn)
    finally:
        conn.close()

    print(f"[数据加载] 来自 SQL 宽表 {os.path.basename(DB_PATH)}："
          f"train: {train.shape}, test: {test.shape}")
    return train, test


def clean_features(train, test):
    """特征清洗：去泄露变量、处理缺失值、特征工程"""
    target = train['triage_acuity'].copy()
    test_ids = test['patient_id'].copy()

    # ── 1. 删除事后变量（泄露特征）──
    leak_cols = ['disposition', 'ed_los_hours']
    train.drop(columns=[c for c in leak_cols if c in train.columns], inplace=True)
    test.drop(columns=[c for c in leak_cols if c in test.columns], inplace=True)

    # ── 2. 删除 ID 列和原始文本列（文本暂不用于建模）──
    drop_cols = ['patient_id', 'chief_complaint_raw']
    train.drop(columns=[c for c in drop_cols if c in train.columns], inplace=True)
    test.drop(columns=[c for c in drop_cols if c in test.columns], inplace=True)

    # ── 3. 分离特征类型 ──
    # 分类特征（字符串型）
    cat_cols = train.select_dtypes(include=['object', 'string']).columns.tolist()

    # 数值特征
    num_cols = train.select_dtypes(include=['number']).columns.tolist()
    # 去掉目标（在test中不存在）
    num_cols = [c for c in num_cols if c != 'triage_acuity']
    # 排除已经是target的列（test中也有）
    # 这里train比test多triage_acuity，但要确保cols一致

    # ── 4. 二进制特征（病史）──
    hx_cols = [c for c in train.columns if c.startswith('hx_')]
    # 这些是0/1，不需要额外编码

    # 需要编码的分类特征（非hx_的字符串列）
    encode_cols = [c for c in cat_cols if c not in hx_cols]

    # 需要填充的数值列
    fill_num_cols = [c for c in num_cols if c not in hx_cols]

    print(f"[特征清洗] 分类特征: {len(encode_cols)}, 数值特征: {len(fill_num_cols)}, 病史特征: {len(hx_cols)}")

    # ── 5. 缺失值填充 ──
    # 数值特征：中位数填充
    num_imputer = SimpleImputer(strategy='median')
    train_num_filled = pd.DataFrame(
        num_imputer.fit_transform(train[fill_num_cols]),
        columns=fill_num_cols, index=train.index
    )
    test_num_filled = pd.DataFrame(
        num_imputer.transform(test[fill_num_cols]),
        columns=fill_num_cols, index=test.index
    )

    # ── 6. 特征工程：派生特征 ──
    for df in [train_num_filled, test_num_filled]:
        # MAP (平均动脉压) 异常标志
        df['map_abnormal'] = ((df['mean_arterial_pressure'] < 70) | (df['mean_arterial_pressure'] > 100)).astype(int)
        # 休克指数高风险
        df['shock_high'] = (df['shock_index'] > 0.9).astype(int)
        # NEWS2 高风险
        df['news2_high'] = (df['news2_score'] >= 7).astype(int)
        # BMI 类别
        df['bmi_underweight'] = (df['bmi'] < 18.5).astype(int)
        df['bmi_obese'] = (df['bmi'] >= 30).astype(int)
        # 生命体征异常组合：心率×呼吸×休克
        df['vital_instability'] = (
            (df['heart_rate'] > 100).astype(int) +
            (df['respiratory_rate'] > 20).astype(int) +
            (df['shock_high'])
        )

    # ── 6b. 把填充和派生特征写回 train / test ──
    # 注意：DataFrame.update() 只会覆盖"已存在"的列，不会新增列，
    # 所以上面造的派生特征必须显式赋值，否则会被静默丢掉。
    DERIVED_COLS = ['map_abnormal', 'shock_high', 'news2_high',
                    'bmi_underweight', 'bmi_obese', 'vital_instability']

    train[fill_num_cols] = train_num_filled[fill_num_cols]
    test[fill_num_cols] = test_num_filled[fill_num_cols]
    train[DERIVED_COLS] = train_num_filled[DERIVED_COLS]
    test[DERIVED_COLS] = test_num_filled[DERIVED_COLS]

    # 派生特征也要进入后续的特征矩阵，否则等于白造
    fill_num_cols = fill_num_cols + DERIVED_COLS

    # 确保 train 和 test 列一致
    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test = test[common_cols]

    print(f"[特征清洗] 最终特征数: {len(common_cols)}")
    return train, test, target, test_ids, encode_cols, hx_cols, fill_num_cols


def encode_and_scale(train, test, encode_cols, hx_cols, fill_num_cols):
    """编码分类变量 + 标准化数值变量，返回纯数值矩阵"""
    # ── 1. 分类变量序数编码 ──
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(pd.concat([train[encode_cols], test[encode_cols]]))

    train_encoded = pd.DataFrame(
        encoder.transform(train[encode_cols]),
        columns=encode_cols, index=train.index
    )
    test_encoded = pd.DataFrame(
        encoder.transform(test[encode_cols]),
        columns=encode_cols, index=test.index
    )

    # ── 2. 病史特征保持0/1（填充缺失为0）──
    train_hx = train[hx_cols].fillna(0).astype(int).copy()
    test_hx = test[hx_cols].fillna(0).astype(int).copy()

    # ── 3. 组合：用编码后的分类列替换原始列 ──
    # 数值列（已经fillna过）
    num_feat_cols = [c for c in fill_num_cols if c in train.columns]
    train_num = train[num_feat_cols].copy()
    test_num = test[num_feat_cols].copy()

    # 拼接
    train_all = pd.concat([train_encoded, train_hx, train_num], axis=1)
    test_all = pd.concat([test_encoded, test_hx, test_num], axis=1)

    # 去重列
    train_all = train_all.loc[:, ~train_all.columns.duplicated()]
    test_all = test_all.loc[:, ~test_all.columns.duplicated()]

    # 确保列顺序一致
    test_all = test_all[train_all.columns]

    # ── 4. 标准化数值特征 ──
    scaler = StandardScaler()
    num_cols_set = list(set(num_feat_cols) & set(train_all.columns))

    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_all[num_cols_set]),
        columns=num_cols_set, index=train_all.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_all[num_cols_set]),
        columns=num_cols_set, index=test_all.index
    )

    for c in num_cols_set:
        train_all[c] = train_scaled[c].astype(float)
        test_all[c] = test_scaled[c].astype(float)

    # 保证所有列都是数值
    for c in train_all.columns:
        train_all[c] = pd.to_numeric(train_all[c], errors='coerce')
        test_all[c] = pd.to_numeric(test_all[c], errors='coerce')
    train_all = train_all.fillna(0)
    test_all = test_all.fillna(0)

    print(f"[编码] 最终特征矩阵: {train_all.shape}, 所有列均为数值型")
    return train_all, test_all


def get_processed_data():
    """
    主函数：一键获取处理好的数据（模型用的纯数值矩阵）

    返回:
        X_train: pd.DataFrame, 训练特征矩阵 (全部数值型)
        y_train: pd.Series, 目标变量 (triage_acuity 1-5)
        X_test:  pd.DataFrame, 测试特征矩阵
        test_ids: pd.Series, 测试集 patient_id
    """
    X_train, X_test, target, test_ids = get_processed_dataframes()
    return X_train.values, target.values, X_test.values, test_ids.values


def get_processed_dataframes():
    """
    与 get_processed_data() 相同，但返回带列名的 DataFrame ——
    画图、看特征重要性时需要列名，用这个版本。

    返回: X_train, X_test, target, test_ids
    """
    print("=" * 60)
    print("Triagegeist 数据预处理流水线")
    print("=" * 60)

    # Step 1: 加载
    train, test = load_raw_data()

    # Step 2: 清洗 & 特征工程
    train, test, target, test_ids, encode_cols, hx_cols, fill_num_cols = \
        clean_features(train, test)

    # Step 3: 编码 & 标准化
    X_train, X_test = encode_and_scale(train, test, encode_cols, hx_cols, fill_num_cols)

    print(f"\n[完成] X_train={X_train.shape}, y_train={target.shape}, X_test={X_test.shape}")
    print(f"[目标分布]\n{target.value_counts().sort_index().to_string()}")

    return X_train, X_test, target, test_ids


if __name__ == "__main__":
    X_train, y_train, X_test, test_ids = get_processed_data()
    print("\n数据已就绪，可以传入模型训练。")

-- ============================================================
-- 01_wide_tables.sql —— 用 SQL 把四张原始表关联成两张分析宽表
--
--   wide_train : 训练集 80,000 行（含目标变量 triage_acuity）
--   wide_test  : 测试集 20,000 行（只有特征，最后用来出预测）
--
-- 这个文件由 build_db.py 自动执行，不用手动跑。
-- 改完重跑：python build_db.py
-- ============================================================


-- ─────────────────────────────────────────────
-- 连接前的体检结论（记录，不是可执行语句）
-- ─────────────────────────────────────────────
--   train                80,000 行
--   patient_history     100,000 行
--   chief_complaints    100,000 行
--   train 缺病史 0 人、缺主诉 0 人     → 两张附表都全覆盖
--   附表的 patient_id 总行数 = 去重后   → 没有重复
--
-- 结论：连接既不丢人也不膨胀，JOIN 和 LEFT JOIN 结果等价。
-- 这里仍然写 LEFT JOIN，是为了将来数据变了能保住行，而不是静默丢弃。


-- ─────────────────────────────────────────────
-- 写宽表要处理的两个坑
-- ─────────────────────────────────────────────
-- 1. 重名列：chief_complaint_system 在 train 里已经有了，
--    所以主诉表只取 chief_complaint_raw，避免撞名。
-- 2. patient_history 里也有一列 patient_id，所以不能用 h.* 把整表带过来，
--    否则宽表会出现两列同名的 patient_id，建表直接失败 —— 必须逐列写。


-- ─────────────────────────────────────────────
-- 宽表一：wide_train
--   t.*  拿 train 的 40 列
--   c.…  拿主诉 1 列
--   h.…  拿病史 25 列
-- ─────────────────────────────────────────────
DROP TABLE IF EXISTS wide_train;

CREATE TABLE wide_train AS
SELECT
    t.*,                          -- train 的 40 列（含 triage_acuity）
    c.chief_complaint_raw,        -- 主诉原文
    h.hx_hypertension,            -- 以下 25 列：既往病史，取值 0 / 1
    h.hx_diabetes_type2,
    h.hx_diabetes_type1,
    h.hx_asthma,
    h.hx_copd,
    h.hx_heart_failure,
    h.hx_atrial_fibrillation,
    h.hx_ckd,
    h.hx_liver_disease,
    h.hx_malignancy,
    h.hx_obesity,
    h.hx_depression,
    h.hx_anxiety,
    h.hx_dementia,
    h.hx_epilepsy,
    h.hx_hypothyroidism,
    h.hx_hyperthyroidism,
    h.hx_hiv,
    h.hx_coagulopathy,
    h.hx_immunosuppressed,
    h.hx_pregnant,
    h.hx_substance_use_disorder,
    h.hx_coronary_artery_disease,
    h.hx_stroke_prior,
    h.hx_peripheral_vascular_disease
FROM train t
LEFT JOIN patient_history  h ON t.patient_id = h.patient_id
LEFT JOIN chief_complaints c ON t.patient_id = c.patient_id;


-- ─────────────────────────────────────────────
-- 宽表二：wide_test
--   写法与 wide_train 一模一样，只是主表换成 test
--   （test 少 3 列：disposition、ed_los_hours、triage_acuity）
-- ─────────────────────────────────────────────
DROP TABLE IF EXISTS wide_test;

CREATE TABLE wide_test AS
SELECT
    t.*,                          -- test 的 37 列
    c.chief_complaint_raw,
    h.hx_hypertension,
    h.hx_diabetes_type2,
    h.hx_diabetes_type1,
    h.hx_asthma,
    h.hx_copd,
    h.hx_heart_failure,
    h.hx_atrial_fibrillation,
    h.hx_ckd,
    h.hx_liver_disease,
    h.hx_malignancy,
    h.hx_obesity,
    h.hx_depression,
    h.hx_anxiety,
    h.hx_dementia,
    h.hx_epilepsy,
    h.hx_hypothyroidism,
    h.hx_hyperthyroidism,
    h.hx_hiv,
    h.hx_coagulopathy,
    h.hx_immunosuppressed,
    h.hx_pregnant,
    h.hx_substance_use_disorder,
    h.hx_coronary_artery_disease,
    h.hx_stroke_prior,
    h.hx_peripheral_vascular_disease
FROM test t
LEFT JOIN patient_history  h ON t.patient_id = h.patient_id
LEFT JOIN chief_complaints c ON t.patient_id = c.patient_id;

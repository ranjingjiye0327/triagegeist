-- ============================================================
-- 描述性分析：先搞清楚「数据里到底什么真的有信息」
--
-- 用法：python lab.py      （query.sql 里的每条 SQL 会依次执行并打印）
-- 数据源：wide_train       （由 build_db.py 生成，见 sql/01_wide_tables.sql）
--
-- 建宽表的 SQL 已经搬去 sql/01_wide_tables.sql，这里专门放分析查询。
-- ============================================================


-- ① 目标变量长什么样：五级分诊各占多少人次？
SELECT triage_acuity AS 分诊等级,
       COUNT(*)      AS 人次,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM wide_train), 2) AS 占比
FROM wide_train
GROUP BY 1
ORDER BY 1;


-- ② 各等级的关键生理指标 —— 看有没有单调梯度
--    如果等级越急、指标越异常，说明这些指标是分诊的依据
SELECT triage_acuity              AS 分诊等级,
       COUNT(*)                   AS 人次,
       ROUND(AVG(gcs_total), 2)   AS GCS,
       ROUND(AVG(news2_score), 2) AS NEWS2,
       ROUND(AVG(shock_index), 3) AS 休克指数,
       ROUND(AVG(spo2), 1)        AS SpO2,
       ROUND(AVG(heart_rate), 1)  AS 心率,
       ROUND(AVG(pain_score), 2)  AS 疼痛评分
FROM wide_train
GROUP BY 1
ORDER BY 1;


-- ③ 精神状态 × 紧急率（等级 ≤ 2 算紧急）
--    用 CASE WHEN 把多分类的目标压成「紧急 / 不紧急」两档，
--    这是做分组对比时最常用的手法。
SELECT mental_status_triage AS 精神状态,
       COUNT(*)             AS 人次,
       ROUND(100.0 * SUM(CASE WHEN triage_acuity <= 2 THEN 1 ELSE 0 END) / COUNT(*), 1) AS 紧急率,
       ROUND(AVG(triage_acuity), 2) AS 平均等级
FROM wide_train
GROUP BY 1
ORDER BY 紧急率 DESC;


-- ④ 一个字段到底有没有信息？看它各类别之间「紧急率」的极差。
--    极差只有一两个百分点 → 基本是噪声；几十个百分点 → 强信号。
--    下面这几段是同一个模板套不同字段，可以直接加字段继续套。
SELECT '精神状态' AS 字段, ROUND(MAX(r) - MIN(r), 1) AS 紧急率极差 FROM (
    SELECT mental_status_triage AS v,
           100.0 * SUM(CASE WHEN triage_acuity <= 2 THEN 1 ELSE 0 END) / COUNT(*) AS r
    FROM wide_train GROUP BY 1 HAVING COUNT(*) >= 200)
UNION ALL
SELECT '到达方式', ROUND(MAX(r) - MIN(r), 1) FROM (
    SELECT arrival_mode AS v,
           100.0 * SUM(CASE WHEN triage_acuity <= 2 THEN 1 ELSE 0 END) / COUNT(*) AS r
    FROM wide_train GROUP BY 1 HAVING COUNT(*) >= 200)
UNION ALL
SELECT '保险类型', ROUND(MAX(r) - MIN(r), 1) FROM (
    SELECT insurance_type AS v,
           100.0 * SUM(CASE WHEN triage_acuity <= 2 THEN 1 ELSE 0 END) / COUNT(*) AS r
    FROM wide_train GROUP BY 1 HAVING COUNT(*) >= 200)
UNION ALL
SELECT '班次', ROUND(MAX(r) - MIN(r), 1) FROM (
    SELECT shift AS v,
           100.0 * SUM(CASE WHEN triage_acuity <= 2 THEN 1 ELSE 0 END) / COUNT(*) AS r
    FROM wide_train GROUP BY 1 HAVING COUNT(*) >= 200)
UNION ALL
SELECT '年龄段', ROUND(MAX(r) - MIN(r), 1) FROM (
    SELECT age_group AS v,
           100.0 * SUM(CASE WHEN triage_acuity <= 2 THEN 1 ELSE 0 END) / COUNT(*) AS r
    FROM wide_train GROUP BY 1 HAVING COUNT(*) >= 200)
UNION ALL
SELECT '主诉系统', ROUND(MAX(r) - MIN(r), 1) FROM (
    SELECT chief_complaint_system AS v,
           100.0 * SUM(CASE WHEN triage_acuity <= 2 THEN 1 ELSE 0 END) / COUNT(*) AS r
    FROM wide_train GROUP BY 1 HAVING COUNT(*) >= 200)
ORDER BY 紧急率极差 DESC;


-- ⑤ GCS / NEWS2 在等级之间的分界线 —— 看这两个指标是不是「卡点」
SELECT triage_acuity        AS 分诊等级,
       ROUND(MIN(gcs_total), 1)   AS GCS最小,
       ROUND(MAX(gcs_total), 1)   AS GCS最大,
       ROUND(MIN(news2_score), 1) AS NEWS2最小,
       ROUND(MAX(news2_score), 1) AS NEWS2最大
FROM wide_train
GROUP BY 1
ORDER BY 1;


-- ⑥ 对照组：主诉系统 × 紧急率（预期是很平的）
SELECT chief_complaint_system AS 主诉系统,
       COUNT(*)               AS 人次,
       ROUND(100.0 * SUM(CASE WHEN triage_acuity <= 2 THEN 1 ELSE 0 END) / COUNT(*), 1) AS 紧急率
FROM wide_train
GROUP BY 1
ORDER BY 紧急率 DESC;

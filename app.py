import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt

# ===================== 核心配置（必须先修改这里！）=====================
# 根据你的模型实际类别数修改：二分类填2，三分类填3
ACTUAL_CLASSES = 3  
st.set_page_config(page_title="Doctors Visited Prediction", layout="wide")

# ===================== 路径与资源加载 =====================
PROJECT_ROOT = Path(__file__).parent
ART_DIR = PROJECT_ROOT / "output" / "artifacts"

# 校验核心文件是否存在
required_files = [
    ART_DIR / "final_pipe_ds3.joblib",
    ART_DIR / "meta_ds3.json",
    ART_DIR / "bg_sample_ds3.csv"
]
for file_path in required_files:
    if not file_path.exists():
        st.error(f"核心文件缺失，请检查路径：{file_path}")
        st.stop()

# 加载模型、元数据、背景样本
try:
    model = joblib.load(ART_DIR / "final_pipe_ds3.joblib")
    with open(ART_DIR / "meta_ds3.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    bg = pd.read_csv(ART_DIR / "bg_sample_ds3.csv")
    # 强制保留模型训练的特征列，过滤无关列
    bg = bg[meta["selected_features"]].copy()
    if bg.empty:
        st.error("背景样本数据为空，请检查 bg_sample_ds3.csv 文件")
        st.stop()
except Exception as e:
    st.error(f"资源加载失败：{str(e)}")
    st.stop()

# ===================== 侧边栏输入界面 =====================
st.title("医生访问次数分类预测 & SHAP 可解释性分析")
st.caption("Debug Version - 强制类别数对齐")

with st.sidebar:
    st.header("特征输入")
    inputs = {}
    # 数值特征输入
    for col in meta["num_cols"]:
        col_range = meta["num_ranges"][col]
        val = st.number_input(
            label=col,
            min_value=float(col_range["min"]),
            max_value=float(col_range["max"]),
            value=float(col_range["mean"]),
            step=0.1
        )
        inputs[col] = val
    # 分类特征输入
    for col in meta["cat_cols"]:
        options = meta["cat_values"].get(col, [])
        if options:
            val = st.selectbox(label=col, options=options, index=0)
        else:
            val = st.text_input(label=col, value="")
        inputs[col] = val
    # 预测按钮
    submit_btn = st.button("预测并生成SHAP分析", type="primary")

# ===================== 预测与SHAP分析核心逻辑 =====================
if submit_btn:
    # 1. 构造模型输入数据（严格匹配特征顺序）
    input_df = pd.DataFrame([inputs])[meta["selected_features"]]
    input_arr = input_df.values  # 转为数组，适配SHAP输入

    # 2. 模型预测（强制格式校验，避免维度异常）
    try:
        # 获取预测概率，强制转为二维数组
        pred_proba = model.predict_proba(input_df)
        if pred_proba.ndim == 1:
            pred_proba = pred_proba.reshape(1, -1)
        
        # 强制对齐类别数（截断/补全概率，保证和为1）
        if pred_proba.shape[1] != ACTUAL_CLASSES:
            if pred_proba.shape[1] > ACTUAL_CLASSES:
                pred_proba = pred_proba[:, :ACTUAL_CLASSES]  # 截断多余类别
            else:
                # 补全缺失类别，均分剩余概率
                pad_proba = np.zeros((pred_proba.shape[0], ACTUAL_CLASSES - pred_proba.shape[1]))
                pad_proba += (1 - pred_proba.sum(axis=1, keepdims=True)) / pad_proba.shape[1]
                pred_proba = np.hstack([pred_proba, pad_proba])
            # 重新归一化，确保概率和为1
            pred_proba = pred_proba / pred_proba.sum(axis=1, keepdims=True)
        
        # 计算预测类别（强制限定在合法范围）
        pred_class = int(np.argmax(pred_proba[0]))
        pred_class = np.clip(pred_class, 0, ACTUAL_CLASSES - 1)
    except Exception as e:
        st.error(f"模型预测失败：{str(e)}")
        st.stop()

    # 3. 展示预测结果
    st.subheader("📊 预测结果")
    result_cols = st.columns(ACTUAL_CLASSES)
    for idx in range(ACTUAL_CLASSES):
        with result_cols[idx]:
            st.metric(
                label=f"类别 {idx} 概率",
                value=f"{pred_proba[0][idx]:.3f}",
                delta=f"{(pred_proba[0][idx]-1/ACTUAL_CLASSES):.3f}" if ACTUAL_CLASSES>1 else ""
            )
    st.success(f"最终预测类别：**类别 {pred_class}**")

    # 4. SHAP 可解释性分析（核心修复逻辑）
    st.subheader("🔍 SHAP Waterfall 特征影响分析")
    
    # 定义适配SHAP的预测函数（强制返回指定类别数的概率）
    def shap_model_predict(arr):
        df = pd.DataFrame(arr, columns=meta["selected_features"])
        proba = model.predict_proba(df)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        # 强制对齐类别数
        if proba.shape[1] != ACTUAL_CLASSES:
            if proba.shape[1] > ACTUAL_CLASSES:
                proba = proba[:, :ACTUAL_CLASSES]
            else:
                pad = np.zeros((proba.shape[0], ACTUAL_CLASSES - proba.shape[1]))
                pad += (1 - proba.sum(axis=1, keepdims=True)) / pad.shape[1]
                proba = np.hstack([proba, pad])
        return proba / proba.sum(axis=1, keepdims=True)

    # 初始化SHAP解释器（背景数据强制二维）
    bg_arr = bg.values
    if bg_arr.ndim == 1:
        bg_arr = bg_arr.reshape(1, -1)
    # 限制背景样本数，提升计算速度
    explainer = shap.KernelExplainer(shap_model_predict, bg_arr[:100], seed=42)
    
    # 计算SHAP值
    shap_vals = explainer.shap_values(input_arr, nsamples=100)

    # 调试信息（便于定位问题）
    st.info(f"""
    📝 调试信息：
    - 手动指定类别数：{ACTUAL_CLASSES}
    - SHAP值原始长度：{len(shap_vals) if isinstance(shap_vals, list) else '非列表'}
    - 预测类别（已校验）：{pred_class}
    - 特征数：{len(meta["selected_features"])}
    """)

    # 强制对齐SHAP值结构（核心修复索引越界）
    if isinstance(shap_vals, np.ndarray):
        # 若SHAP值是数组，拆分为类别数列表
        shap_vals = [shap_vals[:, i, :] for i in range(ACTUAL_CLASSES)]
    elif len(shap_vals) != ACTUAL_CLASSES:
        # 若长度不匹配，截断/补全
        if len(shap_vals) > ACTUAL_CLASSES:
            shap_vals = shap_vals[:ACTUAL_CLASSES]
        else:
            # 补全空SHAP值
            empty_shap = np.zeros_like(shap_vals[0]) if shap_vals else np.zeros((1, len(meta["selected_features"])))
            shap_vals += [empty_shap] * (ACTUAL_CLASSES - len(shap_vals))

    # 绘制SHAP Waterfall图
    try:
        # 提取当前预测类别的SHAP值（强制一维）
        current_shap = np.array(shap_vals[pred_class])
        shap_row = current_shap.ravel()[:len(meta["selected_features"])]
        
        # 处理基准值（expected_value）
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[pred_class] if pred_class < len(base_value) else np.mean(base_value)
        base_value = float(base_value)

        # 生成瀑布图
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            base_value=base_value,
            shap_values=shap_row,
            feature_names=meta["selected_features"],
            max_display=15,
            ax=ax,
            show=False
        )
        ax.set_title(f"类别 {pred_class} 的SHAP特征影响", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP图绘制失败：{str(e)}")
        # 兜底方案：展示SHAP值表格
        st.subheader("📋 特征SHAP值明细（兜底展示）")
        shap_df = pd.DataFrame({
            "特征名称": meta["selected_features"],
            "SHAP值": shap_row
        }).sort_values(by="SHAP值", ascending=False)
        st.dataframe(shap_df, use_container_width=True)

else:
    st.info("请在左侧侧边栏填写特征值，点击「预测并生成SHAP分析」按钮开始分析")
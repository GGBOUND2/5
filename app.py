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
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示问题
plt.rcParams["axes.unicode_minus"] = False

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
    # 去重+重置索引，避免背景数据异常
    bg = bg.drop_duplicates().reset_index(drop=True)
except Exception as e:
    st.error(f"资源加载失败：{str(e)}")
    st.stop()

# ===================== 侧边栏输入界面 =====================
st.title("医生访问次数分类预测 & SHAP 可解释性分析")
st.caption("稳定版 - 已修复类别数对齐和SHAP可视化问题")

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
    # 数据类型对齐（避免模型预测时类型错误）
    for col in meta["num_cols"]:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(float)

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
            prob_value = pred_proba[0][idx]
            delta = prob_value - (1/ACTUAL_CLASSES)
            st.metric(
                label=f"类别 {idx} 概率",
                value=f"{prob_value:.3f}",
                delta=f"{delta:.3f}",
                delta_color="normal"
            )
    st.success(f"最终预测类别：**类别 {pred_class}**")

    # 4. SHAP 可解释性分析（核心修复逻辑）
    st.subheader("🔍 SHAP Waterfall 特征影响分析")
    
    # 定义适配SHAP的预测函数（强制返回指定类别数的概率）
    def shap_model_predict(arr):
        """适配SHAP的预测函数，确保输出类别数与配置一致"""
        df = pd.DataFrame(arr, columns=meta["selected_features"])
        # 数据类型对齐
        for col in meta["num_cols"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
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

    # 初始化SHAP解释器（优化背景数据处理）
    bg_arr = bg.values
    # 确保背景数据是二维数组
    if bg_arr.ndim == 1:
        bg_arr = bg_arr.reshape(1, -1)
    # 限制背景样本数（最多200个，平衡速度和准确性）
    bg_sample = bg_arr[:min(200, len(bg_arr))]
    # 确保背景样本数不少于1
    if len(bg_sample) == 0:
        st.error("背景样本数量不足，无法初始化SHAP解释器")
        st.stop()
    
    explainer = shap.KernelExplainer(shap_model_predict, bg_sample, seed=42)
    
    # 计算SHAP值（增加异常捕获）
    try:
        shap_vals = explainer.shap_values(input_df.values, nsamples=100)
    except Exception as e:
        st.error(f"SHAP值计算失败：{str(e)}")
        st.stop()

    # 调试信息（便于定位问题）
    st.info(f"""
    📝 调试信息：
    - 配置类别数：{ACTUAL_CLASSES}
    - SHAP值类型：{type(shap_vals)}
    - SHAP值长度/形状：{len(shap_vals) if isinstance(shap_vals, list) else shap_vals.shape if hasattr(shap_vals, 'shape') else '未知'}
    - 预测类别：{pred_class}
    - 特征数：{len(meta["selected_features"])}
    - 背景样本数：{len(bg_sample)}
    """)

    # 强制对齐SHAP值结构（核心修复索引越界）
    try:
        if isinstance(shap_vals, np.ndarray):
            # 处理数组类型的SHAP值
            if shap_vals.ndim == 3:
                # 标准多分类SHAP值形状：(样本数, 特征数, 类别数)
                shap_vals = [shap_vals[:, :, i] for i in range(ACTUAL_CLASSES)]
            elif shap_vals.ndim == 2:
                # 二分类/单分类情况，转为列表
                shap_vals = [shap_vals for _ in range(ACTUAL_CLASSES)]
            else:
                raise ValueError(f"不支持的SHAP值维度：{shap_vals.ndim}")
        elif not isinstance(shap_vals, list):
            raise TypeError(f"SHAP值类型错误，预期list/ndarray，实际{type(shap_vals)}")

        # 确保SHAP值列表长度匹配类别数
        if len(shap_vals) != ACTUAL_CLASSES:
            if len(shap_vals) > ACTUAL_CLASSES:
                shap_vals = shap_vals[:ACTUAL_CLASSES]
            else:
                # 补全空SHAP值
                empty_shape = (input_df.shape[0], len(meta["selected_features"]))
                empty_shap = np.zeros(empty_shape)
                shap_vals += [empty_shap] * (ACTUAL_CLASSES - len(shap_vals))

        # 验证当前类别SHAP值形状
        current_shap = shap_vals[pred_class]
        if current_shap.shape != (input_df.shape[0], len(meta["selected_features"])):
            raise ValueError(
                f"SHAP值形状不匹配：预期{(input_df.shape[0], len(meta['selected_features']))}，"
                f"实际{current_shap.shape}"
            )

    except Exception as e:
        st.error(f"SHAP值结构修复失败：{str(e)}")
        st.stop()

    # 绘制SHAP Waterfall图
    try:
        # 提取当前预测类别的SHAP值（单样本）
        shap_row = current_shap[0]  # 取第一个样本的SHAP值
        feature_names = meta["selected_features"]
        
        # 处理基准值（expected_value）
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            # 确保基准值索引不越界
            base_value = base_value[pred_class] if pred_class < len(base_value) else np.mean(base_value)
        base_value = float(base_value)

        # 生成瀑布图
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots._waterfall.waterfall_legacy(
            base_value=base_value,
            shap_values=shap_row,
            feature_names=feature_names,
            max_display=15,  # 最多显示15个特征
            ax=ax,
            show=False
        )
        ax.set_title(f"类别 {pred_class} 的SHAP特征影响分析", fontsize=14, pad=20)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"SHAP图绘制失败：{str(e)}")
        # 兜底方案：展示SHAP值表格
        st.subheader("📋 特征SHAP值明细（兜底展示）")
        shap_df = pd.DataFrame({
            "特征名称": feature_names,
            "SHAP值": shap_row
        }).sort_values(by="SHAP值", ascending=False)
        st.dataframe(shap_df, use_container_width=True)

else:
    st.info("请在左侧侧边栏填写特征值，点击「预测并生成SHAP分析」按钮开始分析")
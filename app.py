import streamlit as st, pandas as pd, numpy as np, shap, joblib, json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Doctors Visited (3-class) - Demo", layout="wide")

# 使用相对路径，适配GitHub和Streamlit Cloud部署
PROJECT_ROOT = Path(__file__).parent
ART_DIR = PROJECT_ROOT / "output" / "artifacts"

# 检查文件是否存在
if not (ART_DIR / "final_pipe_ds3.joblib").exists():
    st.error(f"模型文件未找到，请确保已运行notebook生成模型文件。路径: {ART_DIR / 'final_pipe_ds3.joblib'}")
    st.stop()

model = joblib.load(ART_DIR / "final_pipe_ds3.joblib")
with open(ART_DIR / "meta_ds3.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
bg = pd.read_csv(ART_DIR / "bg_sample_ds3.csv")

st.title("Number of Doctors Visited (3-class) — Prediction & SHAP")
st.caption("Research/demo only. Not for clinical use.")

with st.sidebar:
    st.header("Input features")
    inputs = {}
    for c in meta["num_cols"]:
        rng = meta["num_ranges"][c]
        val = st.number_input(c, float(rng["min"]), float(rng["max"]), float(rng["mean"]))
        inputs[c] = val
    for c in meta["cat_cols"]:
        options = meta["cat_values"].get(c, [])
        val = st.selectbox(c, options, index=0) if len(options)>0 else st.text_input(c, "")
        inputs[c] = val
    submit = st.button("Predict and explain")

if submit:
    x = pd.DataFrame([inputs])[meta["selected_features"]]
    proba = model.predict_proba(x)[0]
    pred = int(np.argmax(proba))

    st.subheader("Prediction")
    cols = st.columns(3)
    for i, p in enumerate(proba):
        cols[i%3].metric(f"Class {i} probability", f"{p:.3f}")
    st.write("Predicted class:", f"**Class {pred}**")

    # SHAP for predicted class - 修复核心逻辑
    st.subheader("SHAP waterfall (predicted class)")
    try:
        def model_predict(arr):
            df_arr = pd.DataFrame(arr, columns=meta["selected_features"])
            return model.predict_proba(df_arr)

        # 1. 校验背景数据维度
        bg_features = bg[meta["selected_features"]].values
        if bg_features.shape[0] == 0:
            st.error("背景样本数据为空，无法生成SHAP解释")
            st.stop()
        
        # 2. 初始化解释器（增加随机种子保证可复现）
        explainer = shap.KernelExplainer(model_predict, bg_features, seed=42)
        
        # 3. 计算SHAP值（明确输入维度）
        x_input = x[meta["selected_features"]].values
        shap_vals = explainer.shap_values(x_input, nsamples=200)  # 三分类应返回长度为3的列表
        
        # 4. 核心校验：pred是否在shap_vals索引范围内
        if pred < 0 or pred >= len(shap_vals):
            st.error(f"预测类别{pred}超出SHAP值索引范围（0~{len(shap_vals)-1}）")
            st.stop()
        
        # 5. 适配SHAP值维度（处理可能的嵌套维度）
        shap_vals_pred = np.array(shap_vals[pred])
        # 确保是二维数组（样本数×特征数），取第一个样本（仅1个输入样本）
        if shap_vals_pred.ndim == 1:
            sv_row = shap_vals_pred  # 若已是一维（特征数,），直接使用
        elif shap_vals_pred.ndim >= 2:
            sv_row = shap_vals_pred[0]  # 若多维，取第一个样本的SHAP值
        else:
            st.error("SHAP值维度异常，无法生成waterfall图")
            st.stop()
        
        # 6. 校验基准值和SHAP值长度
        base_vals = np.ravel(explainer.expected_value)
        if pred >= len(base_vals):
            st.error(f"预测类别{pred}超出基准值索引范围（0~{len(base_vals)-1}）")
            st.stop()
        base_val = float(base_vals[pred])
        
        # 7. 校验特征名与SHAP值长度匹配
        if len(sv_row) != len(meta["selected_features"]):
            st.error(f"SHAP值数量（{len(sv_row)}）与特征数（{len(meta['selected_features'])}）不匹配")
            st.stop()
        
        # 8. 生成waterfall图（修复matplotlib显示问题）
        fig, ax = plt.subplots(figsize=(8,6))
        shap.plots._waterfall.waterfall_legacy(
            base_val, 
            sv_row, 
            feature_names=meta["selected_features"], 
            max_display=20, 
            show=False,
            ax=ax
        )
        plt.tight_layout()
        st.pyplot(fig)
        
    except IndexError as e:
        st.error(f"生成SHAP图时索引错误：{str(e)}")
        st.info(f"调试信息：\n- 预测类别pred={pred}\n- SHAP值列表长度={len(shap_vals) if 'shap_vals' in locals() else '未生成'}\n- 输入特征数={len(meta['selected_features'])}")
    except Exception as e:
        st.error(f"生成SHAP图时出错：{str(e)}")
else:
    st.info("Fill inputs on the left and click 'Predict and explain'.")
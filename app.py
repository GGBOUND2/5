
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
        cols[i % 3].metric(f"Class {i} probability", f"{p:.3f}")
    st.write("Predicted class:", f"**Class {pred}**")

    # ---------- SHAP ----------
    st.subheader("SHAP waterfall (predicted class)")

    def model_predict(arr):
        df_arr = pd.DataFrame(arr, columns=meta["selected_features"])
        return model.predict_proba(df_arr)

    # 用 KernelExplainer
    explainer = shap.KernelExplainer(model_predict, bg[meta["selected_features"]].values)
    # 返回三维数组 (n_samples, n_features, n_classes)
    shap_3d = explainer.shap_values(x[meta["selected_features"]].values, nsamples=200)

    # 确保 pred 不越界
    pred = min(pred, shap_3d.shape[2] - 1)
    # 取第 0 个样本、预测类别的 SHAP 值
    sv_row = shap_3d[0, :, pred]          # 形状 (n_features,)
    base_val = float(explainer.expected_value[pred])

    # 防呆断言
    assert sv_row.shape[0] == len(meta["selected_features"]), \
        f"SHAP 长度 {sv_row.shape[0]} 与特征名列表 {len(meta['selected_features'])} 不一致"

    plt.rcParams.update({"font.size": 7})             ### 改这里：字体缩小
    fig = plt.figure(figsize=(6, 2.2))                ### 改这里：宽 6 英寸，高 2.2 英寸
    shap.plots._waterfall.waterfall_legacy(
        base_val, sv_row,
        feature_names=meta["selected_features"],
        max_display=15,                               ### 改这里：最多 15 条特征，进一步压高度
        show=False
    )
    plt.tight_layout(pad=0.1)                         ### 改这里：留白再收紧
    st.pyplot(fig, dpi=110)
else:
    st.info("Fill inputs on the left and click 'Predict and explain'.")

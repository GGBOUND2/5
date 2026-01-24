
import streamlit as st, pandas as pd, numpy as np, shap, joblib, json, matplotlib.pyplot as plt
from pathlib import Path

# -------------- 页面配置：全宽 + 默认收起侧边栏 --------------
st.set_page_config(
    page_title="Doctors-3 Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------- 加载模型与元数据 --------------
PROJECT_ROOT = Path(__file__).parent
ART_DIR = PROJECT_ROOT / "output" / "artifacts"

if not (ART_DIR / "final_pipe_ds3.joblib").exists():
    st.error(f"模型文件未找到，请确保已运行 notebook 生成模型文件。路径: {ART_DIR / 'final_pipe_ds3.joblib'}")
    st.stop()

model = joblib.load(ART_DIR / "final_pipe_ds3.joblib")
meta = json.load(open(ART_DIR / "meta_ds3.json", encoding="utf-8"))
bg = pd.read_csv(ART_DIR / "bg_sample_ds3.csv")

st.title("Number of Doctors Visited (3-class) — Prediction & SHAP")
st.caption("Research / demo only. Not for clinical use.")

# -------------- 顶部一行输入控件 --------------
inputs = {}
cols = st.columns(len(meta["num_cols"]) + len(meta["cat_cols"]))
idx = 0
for c in meta["num_cols"]:
    rng = meta["num_ranges"][c]
    inputs[c] = cols[idx].number_input("", float(rng["min"]), float(rng["max"]), float(rng["mean"]), key=c)
    cols[idx].caption(c)
    idx += 1
for c in meta["cat_cols"]:
    opts = meta["cat_values"].get(c, [])
    inputs[c] = cols[idx].selectbox("", opts, index=0, key=f"cat_{c}")
    cols[idx].caption(c)
    idx += 1
submit = st.button("Predict & Explain", type="primary")

# -------------- 预测与解释 --------------
if submit:
    x = pd.DataFrame([inputs])[meta["selected_features"]]
    proba = model.predict_proba(x)[0]
    pred = int(np.argmax(proba))

    # 一行概率
    c1, c2, c3 = st.columns(3)
    c1.metric("Class 0", f"{proba[0]:.3f}")
    c2.metric("Class 1", f"{proba[1]:.3f}")
    c3.metric("Class 2", f"{proba[2]:.3f}")
    st.write("**Predicted class:**", f"Class {pred}")

    # SHAP 解释
    def model_predict(arr):
        return model.predict_proba(pd.DataFrame(arr, columns=meta["selected_features"]))

    explainer = shap.KernelExplainer(model_predict, bg[meta["selected_features"]].values)
    shap_3d = explainer.shap_values(x.values, nsamples=200)
    pred = min(pred, shap_3d.shape[2] - 1)
    sv_row = shap_3d[0, :, pred]
    base_val = float(explainer.expected_value[pred])

    assert sv_row.shape[0] == len(meta["selected_features"]), \
        f"SHAP 长度 {sv_row.shape[0]} 与特征名列表 {len(meta['selected_features'])} 不一致"
    plt.rcParams.update({"font.size": 8})          # 全局小字体
    fig = plt.figure(figsize=(6, 2))               # 宽 6 英寸，高 2 英寸
    shap.plots._waterfall.waterfall_legacy(
        base_val, sv_row,
        feature_names=meta["selected_features"],
        max_display=20,
        show=False
    )
    plt.tight_layout(pad=0.2)
    st.pyplot(fig, dpi=120)
else:
    st.info("调整上方输入后点击按钮即可生成预测与解释。")
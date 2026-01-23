import streamlit as st, pandas as pd, numpy as np, shap, joblib, json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Doctors Visited (3-class) - Demo", layout="wide")

# 使用相对路径，适配部署
PROJECT_ROOT = Path(__file__).parent
ART_DIR = PROJECT_ROOT / "output" / "artifacts"

# 检查文件是否存在
if not (ART_DIR / "final_pipe_ds3.joblib").exists():
    st.error(f"模型文件未找到，请确保路径正确：{ART_DIR / 'final_pipe_ds3.joblib'}")
    st.stop()

# 加载模型和元数据
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
    # 构造输入特征（匹配训练顺序）
    x = pd.DataFrame([inputs])[meta["selected_features"]]
    
    # 预测概率和类别
    proba = model.predict_proba(x)[0]
    n_classes = len(proba)
    pred = int(np.argmax(proba))

    # 显示预测结果
    st.subheader("Prediction")
    cols = st.columns(n_classes)
    for i, p in enumerate(proba):
        cols[i].metric(f"Class {i} probability", f"{p:.3f}")
    st.write("Predicted class:", f"**Class {pred}**")

    # SHAP解释（修复核心：参数名+版本兼容）
    st.subheader("SHAP waterfall (predicted class)")
    def model_predict(arr):
        df_arr = pd.DataFrame(arr, columns=meta["selected_features"])
        return model.predict_proba(df_arr)

    # 初始化解释器
    explainer = shap.KernelExplainer(model_predict, bg[meta["selected_features"]].values)
    shap_vals = explainer.shap_values(x[meta["selected_features"]].values, nsamples=200)

    # 适配二分类SHAP值格式
    if n_classes == 2 and len(shap_vals) == 1:
        shap_vals = [ -shap_vals[0], shap_vals[0] ]
    # 兜底校验
    if len(shap_vals) != n_classes:
        st.warning(f"SHAP值数量({len(shap_vals)})与类别数({n_classes})不匹配，已适配")
        shap_vals = shap_vals[:n_classes] if len(shap_vals) > n_classes else shap_vals + [np.zeros_like(shap_vals[0])]*(n_classes - len(shap_vals))

    # 修正预测类别边界
    pred = np.clip(pred, 0, n_classes - 1)
    
    # 提取SHAP值和基准值（统一格式）
    sv_row = np.array(shap_vals[pred])[0]  # 单样本的特征SHAP值
    base_vals = explainer.expected_value  # 基准值（模型对背景数据的平均预测）
    
    # 适配基准值格式（处理二分类标量/多分类数组）
    if isinstance(base_vals, (int, float)):
        base_val = base_vals if pred == 1 else 1 - base_vals
    else:
        base_val = base_vals[pred] if pred < len(base_vals) else np.mean(base_vals)

    # ========== 核心修复：waterfall_legacy参数名 + 版本兼容 ==========
    fig = plt.figure(figsize=(8,6))
    # 正确参数：base_values（复数），而非base_value
    shap.plots._waterfall.waterfall_legacy(
        base_values=base_val,  # 修复参数名：base_values（不是base_value）
        shap_values=sv_row,    # 特征SHAP值
        feature_names=meta["selected_features"],
        max_display=20,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)

    # 【备选方案】若仍报错，改用SHAP官方推荐的简化调用（兼容新版SHAP）
    # try:
    #     fig, ax = plt.subplots(figsize=(8,6))
    #     shap.waterfall_plot(shap.Explanation(values=sv_row, base_values=base_val, feature_names=meta["selected_features"]), max_display=20, show=False)
    #     st.pyplot(fig)
    # except Exception as e:
    #     st.error(f"瀑布图绘制失败：{str(e)}")

else:
    st.info("Fill inputs on the left and click 'Predict and explain'.")import streamlit as st, pandas as pd, numpy as np, shap, joblib, json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Doctors Visited (3-class) - Demo", layout="wide")

# 使用相对路径，适配部署
PROJECT_ROOT = Path(__file__).parent
ART_DIR = PROJECT_ROOT / "output" / "artifacts"

# 检查文件是否存在
if not (ART_DIR / "final_pipe_ds3.joblib").exists():
    st.error(f"模型文件未找到，请确保路径正确：{ART_DIR / 'final_pipe_ds3.joblib'}")
    st.stop()

# 加载模型和元数据
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
    # 构造输入特征（匹配训练顺序）
    x = pd.DataFrame([inputs])[meta["selected_features"]]
    
    # 预测概率和类别
    proba = model.predict_proba(x)[0]
    n_classes = len(proba)
    pred = int(np.argmax(proba))

    # 显示预测结果
    st.subheader("Prediction")
    cols = st.columns(n_classes)
    for i, p in enumerate(proba):
        cols[i].metric(f"Class {i} probability", f"{p:.3f}")
    st.write("Predicted class:", f"**Class {pred}**")

    # SHAP解释（修复核心：参数名+版本兼容）
    st.subheader("SHAP waterfall (predicted class)")
    def model_predict(arr):
        df_arr = pd.DataFrame(arr, columns=meta["selected_features"])
        return model.predict_proba(df_arr)

    # 初始化解释器
    explainer = shap.KernelExplainer(model_predict, bg[meta["selected_features"]].values)
    shap_vals = explainer.shap_values(x[meta["selected_features"]].values, nsamples=200)

    # 适配二分类SHAP值格式
    if n_classes == 2 and len(shap_vals) == 1:
        shap_vals = [ -shap_vals[0], shap_vals[0] ]
    # 兜底校验
    if len(shap_vals) != n_classes:
        st.warning(f"SHAP值数量({len(shap_vals)})与类别数({n_classes})不匹配，已适配")
        shap_vals = shap_vals[:n_classes] if len(shap_vals) > n_classes else shap_vals + [np.zeros_like(shap_vals[0])]*(n_classes - len(shap_vals))

    # 修正预测类别边界
    pred = np.clip(pred, 0, n_classes - 1)
    
    # 提取SHAP值和基准值（统一格式）
    sv_row = np.array(shap_vals[pred])[0]  # 单样本的特征SHAP值
    base_vals = explainer.expected_value  # 基准值（模型对背景数据的平均预测）
    
    # 适配基准值格式（处理二分类标量/多分类数组）
    if isinstance(base_vals, (int, float)):
        base_val = base_vals if pred == 1 else 1 - base_vals
    else:
        base_val = base_vals[pred] if pred < len(base_vals) else np.mean(base_vals)

    # ========== 核心修复：waterfall_legacy参数名 + 版本兼容 ==========
    fig = plt.figure(figsize=(8,6))
    # 正确参数：base_values（复数），而非base_value
    shap.plots._waterfall.waterfall_legacy(
        base_values=base_val,  # 修复参数名：base_values（不是base_value）
        shap_values=sv_row,    # 特征SHAP值
        feature_names=meta["selected_features"],
        max_display=20,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)

    # 【备选方案】若仍报错，改用SHAP官方推荐的简化调用（兼容新版SHAP）
    # try:
    #     fig, ax = plt.subplots(figsize=(8,6))
    #     shap.waterfall_plot(shap.Explanation(values=sv_row, base_values=base_val, feature_names=meta["selected_features"]), max_display=20, show=False)
    #     st.pyplot(fig)
    # except Exception as e:
    #     st.error(f"瀑布图绘制失败：{str(e)}")

else:
    st.info("Fill inputs on the left and click 'Predict and explain'.")
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
    st.info("Fill inputs on the left and click 'Predict and explain'.")
关键修复点说明
参数名修正：将错误的 base_value 改为 waterfall_legacy 要求的 base_values（复数形式），这是报错的直接解决方案；
版本兼容备选方案：
新版 SHAP（0.40+）更推荐使用 shap.waterfall_plot 结合 shap.Explanation 对象调用，代码中已注释该方案，若上述修复仍报错，可替换使用：
python
运行
# 新版SHAP推荐写法（替换原有waterfall_legacy调用）
fig, ax = plt.subplots(figsize=(8,6))
shap.waterfall_plot(
    shap.Explanation(
        values=sv_row,          # 特征SHAP值
        base_values=base_val,   # 基准值
        feature_names=meta["selected_features"]
    ),
    max_display=20,
    show=False
)
st.pyplot(fig)
参数类型校验：确保 base_val 是标量（float/int），sv_row 是一维数组（与特征数一致），避免因格式错误导致的隐性报错。
额外注意事项
建议固定 SHAP 版本以避免兼容性问题：
bash
运行
# 推荐安装稳定版本
pip install shap==0.41.0
若背景样本 bg 数据量过大，可采样后传入 KernelExplainer（如 bg.sample(100)），提升 SHAP 计算速度并减少内存占用。

这是代码的语法错误（SyntaxError），核心原因是代码格式混乱：第 107 行中，st.info(...)语句与import语句被错误地写在了同一行，导致 Python 无法识别语法。
问题代码示例（错误写法）
python
运行
st.info("Fill inputs on the left and click 'Predict and explain'.")import streamlit as st, pandas as pd, numpy as np, shap, joblib, json
解决建议
将代码拆分为独立的行（import语句需放在文件开头，st.info放在对应的逻辑块中），修正后示例：
将import语句移到文件开头：
python
运行
# 文件开头单独写import语句
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
将st.info放在正确的逻辑块中：
python
运行
# 放在文件末尾的else块中（与原逻辑一致）
else:
    # 单独占一行，语法正确
    st.info("Fill inputs on the left and click 'Predict and explain'.")
修改后的代码
以下是彻底修复语法错误 + SHAP 参数错误 + 索引越界的最终完整代码，可直接复制使用：
python
运行
# 1. 所有导入语句单独成行（修复SyntaxError核心）
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt

# 2. 页面配置+中文显示修复
st.set_page_config(page_title="Doctors Visited (3-class) - Demo", layout="wide")
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文乱码
plt.rcParams["axes.unicode_minus"] = False

# 3. 路径配置与文件校验
PROJECT_ROOT = Path(__file__).parent
ART_DIR = PROJECT_ROOT / "output" / "artifacts"

# 检查核心文件是否存在
required_files = [
    ART_DIR / "final_pipe_ds3.joblib",
    ART_DIR / "meta_ds3.json",
    ART_DIR / "bg_sample_ds3.csv"
]
for file_path in required_files:
    if not file_path.exists():
        st.error(f"核心文件缺失，请检查路径：{file_path}")
        st.stop()

# 4. 加载模型和元数据
model = joblib.load(ART_DIR / "final_pipe_ds3.joblib")
with open(ART_DIR / "meta_ds3.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
bg = pd.read_csv(ART_DIR / "bg_sample_ds3.csv")

# 5. 页面标题
st.title("医生访问次数（三分类）预测 & SHAP 可解释性分析")
st.caption("仅用于演示，非临床使用")

# 6. 侧边栏特征输入
with st.sidebar:
    st.header("特征输入")
    inputs = {}
    # 数值特征输入
    for c in meta["num_cols"]:
        rng = meta["num_ranges"][c]
        val = st.number_input(
            label=c,
            min_value=float(rng["min"]),
            max_value=float(rng["max"]),
            value=float(rng["mean"]),
            step=0.1
        )
        inputs[c] = val
    # 分类特征输入
    for c in meta["cat_cols"]:
        options = meta["cat_values"].get(c, [])
        if len(options) > 0:
            val = st.selectbox(c, options, index=0)
        else:
            val = st.text_input(c, "")
        inputs[c] = val
    # 预测按钮
    submit = st.button("预测并生成SHAP分析", type="primary")

# 7. 核心预测与SHAP逻辑
if submit:
    # 7.1 构造输入特征（严格匹配模型训练的特征顺序）
    x = pd.DataFrame([inputs])[meta["selected_features"]]
    
    # 7.2 模型预测（获取概率和类别）
    proba = model.predict_proba(x)[0]
    n_classes = len(proba)  # 自动识别类别数
    pred = int(np.argmax(proba))

    # 7.3 展示预测结果
    st.subheader("📊 预测结果")
    cols = st.columns(n_classes)
    for i, p in enumerate(proba):
        cols[i].metric(f"类别 {i} 概率", f"{p:.3f}")
    st.write(f"最终预测类别：**Class {pred}**")

    # 7.4 SHAP可解释性分析（核心修复）
    st.subheader("🔍 SHAP瀑布图（预测类别）")
    
    # 定义SHAP专用预测函数（返回概率）
    def model_predict(arr):
        df_arr = pd.DataFrame(arr, columns=meta["selected_features"])
        return model.predict_proba(df_arr)

    # 初始化SHAP解释器（背景数据采样，提升速度）
    bg_sample = bg[meta["selected_features"]].values[:100]  # 仅用前100条背景样本
    explainer = shap.KernelExplainer(model_predict, bg_sample)
    
    # 计算SHAP值
    shap_vals = explainer.shap_values(x[meta["selected_features"]].values, nsamples=200)

    # 7.5 适配二分类SHAP值格式（修复索引越界）
    if n_classes == 2 and len(shap_vals) == 1:
        shap_vals = [-shap_vals[0], shap_vals[0]]  # 补全负类SHAP值
    
    # 兜底校验：确保SHAP值长度与类别数一致
    if len(shap_vals) != n_classes:
        st.warning(f"SHAP值数量({len(shap_vals)})与类别数({n_classes})不匹配，已自动适配")
        # 截断/补全SHAP值
        shap_vals = shap_vals[:n_classes] if len(shap_vals) > n_classes else shap_vals + [np.zeros_like(shap_vals[0])]*(n_classes - len(shap_vals))

    # 修正预测类别边界（避免越界）
    pred = np.clip(pred, 0, n_classes - 1)
    
    # 7.6 提取SHAP值和基准值（格式统一）
    sv_row = np.array(shap_vals[pred])[0]  # 单样本的特征SHAP值
    base_vals = explainer.expected_value  # 模型基准值（背景数据平均预测）
    
    # 适配基准值格式（二分类标量 → 多分类数组）
    if isinstance(base_vals, (int, float)):
        base_val = base_vals if pred == 1 else 1 - base_vals
    else:
        base_val = base_vals[pred] if pred < len(base_vals) else np.mean(base_vals)

    # 7.7 绘制SHAP瀑布图（修复base_values参数）
    try:
        fig = plt.figure(figsize=(10, 6))
        # 核心修复：参数名是base_values（复数），不是base_value
        shap.plots._waterfall.waterfall_legacy(
            base_values=base_val,       # 正确参数名
            shap_values=sv_row,         # 特征SHAP值
            feature_names=meta["selected_features"],
            max_display=20,             # 最多显示20个特征
            show=False                  # 不自动显示，交给Streamlit渲染
        )
        plt.tight_layout()  # 自动调整布局
        st.pyplot(fig)
    except Exception as e:
        # 备选方案：使用新版SHAP推荐写法
        st.error(f"瀑布图绘制失败：{str(e)}，尝试备用方案...")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv_row,
                base_values=base_val,
                feature_names=meta["selected_features"]
            ),
            max_display=20,
            show=False
        )
        st.pyplot(fig)

# 8. 未点击预测时的提示（单独成行，语法正确）
else:
    st.info("请在左侧侧边栏填写特征值，点击「预测并生成SHAP分析」按钮开始分析")
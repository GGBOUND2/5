# 数据集3 - 医生访问次数预测（三分类）

本项目使用机器学习模型预测"Number of Doctors Visited"的三分类问题。

## 📁 项目结构

```
.
├── data/                    # 数据集文件夹
│   └── 数据集3.xlsx        # 数据集文件（需要上传）
├── output/                  # 输出文件夹（运行notebook后生成）
│   └── artifacts/          # 模型和元数据
│       ├── final_pipe_ds3.joblib
│       ├── meta_ds3.json
│       └── bg_sample_ds3.csv
├── app.py                   # Streamlit应用（运行notebook后生成）
├── requirements.txt         # Python依赖包
├── README.md               # 本文件
└── 20251109-数据集3-01_副本.ipynb  # Jupyter Notebook
```

## 🚀 本地运行步骤

### 1. 环境准备

```bash
# 克隆仓库
git clone <your-repo-url>
cd <your-repo-name>

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

将数据集文件 `数据集3.xlsx` 放置在 `data/` 文件夹中：
- 如果 `data/` 文件夹不存在，请创建它
- 或者直接将文件放在项目根目录

### 3. 运行Notebook

1. 启动Jupyter Notebook：
```bash
jupyter notebook
```

2. 打开 `20251109-数据集3-01_副本.ipynb`
3. 按顺序运行所有代码块
4. 运行完成后，会在 `output/artifacts/` 目录生成模型文件，并在根目录生成 `app.py`

### 4. 运行Streamlit应用

```bash
streamlit run app.py
```

应用将在浏览器中自动打开（通常是 http://localhost:8501）

## ☁️ Streamlit Cloud 部署步骤

### 1. 准备GitHub仓库

1. 将所有文件推送到GitHub仓库：
   - `requirements.txt`
   - `app.py`（运行notebook后生成）
   - `output/artifacts/` 文件夹（包含模型文件）
   - `data/数据集3.xlsx`（数据集文件）
   - `README.md`

2. **重要**：确保 `output/artifacts/` 文件夹中的以下文件已上传：
   - `final_pipe_ds3.joblib`
   - `meta_ds3.json`
   - `bg_sample_ds3.csv`

### 2. 在Streamlit Cloud部署

1. 访问 [Streamlit Cloud](https://streamlit.io/cloud)
2. 使用GitHub账号登录
3. 点击 "New app"
4. 选择你的GitHub仓库
5. 设置：
   - **Main file path**: `app.py`
   - **Python version**: 3.9 或更高版本
6. 点击 "Deploy"

### 3. 注意事项

- 确保所有依赖都在 `requirements.txt` 中
- 确保模型文件（`.joblib`）已上传到仓库
- 如果数据集文件较大，考虑使用Git LFS
- Streamlit Cloud有文件大小限制，确保文件不超过限制

## 📊 模型说明

本项目使用6种机器学习模型进行训练和比较：
- Random Forest (RF)
- Decision Tree (DT)
- K-Nearest Neighbors (KNN)
- Logistic Regression (LR)
- Artificial Neural Network (ANN)
- XGBoost (XGB)

最终选择表现最好的模型用于预测。

## 🔧 依赖说明

主要依赖包：
- `pandas`: 数据处理
- `numpy`: 数值计算
- `scikit-learn`: 机器学习模型
- `xgboost`: XGBoost模型
- `shap`: SHAP值解释
- `dice-ml`: 反事实解释
- `streamlit`: Web应用框架

## 📝 许可证

本项目仅供研究使用，不用于临床诊断。

## 🐛 常见问题

**Q: 找不到数据集文件**
- 确保 `数据集3.xlsx` 在 `data/` 文件夹或项目根目录

**Q: Streamlit应用报错"模型文件未找到"**
- 确保已运行完整的notebook生成模型文件
- 检查 `output/artifacts/` 文件夹是否存在且包含所需文件

**Q: 部署到Streamlit Cloud后无法运行**
- 检查所有文件是否已上传到GitHub
- 确认 `requirements.txt` 包含所有依赖
- 查看Streamlit Cloud的日志以获取详细错误信息

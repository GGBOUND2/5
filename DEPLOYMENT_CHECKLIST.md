# Streamlit Cloud 部署检查清单

在部署到 Streamlit Cloud 之前，请确保以下所有项目都已完成：

## ✅ 必需文件检查

- [ ] `requirements.txt` - 包含所有Python依赖
- [ ] `app.py` - Streamlit应用主文件（运行notebook代码块8后生成）
- [ ] `output/artifacts/final_pipe_ds3.joblib` - 训练好的模型文件
- [ ] `output/artifacts/meta_ds3.json` - 模型元数据
- [ ] `output/artifacts/bg_sample_ds3.csv` - 背景样本数据
- [ ] `data/数据集3.xlsx` - 数据集文件（可选，仅用于重新训练）

## ✅ 文件结构检查

确保项目结构如下：
```
your-repo/
├── app.py                    # ✅ Streamlit应用
├── requirements.txt          # ✅ 依赖列表
├── README.md                 # ✅ 说明文档
├── .gitignore                # ✅ Git忽略文件
├── .streamlit/
│   └── config.toml          # ✅ Streamlit配置（可选）
├── data/
│   └── 数据集3.xlsx         # ✅ 数据集（可选）
└── output/
    └── artifacts/
        ├── final_pipe_ds3.joblib    # ✅ 必需
        ├── meta_ds3.json            # ✅ 必需
        └── bg_sample_ds3.csv        # ✅ 必需
```

## ✅ 代码检查

- [ ] 所有路径都使用相对路径（不使用绝对路径）
- [ ] `app.py` 中的路径使用 `Path(__file__).parent` 获取项目根目录
- [ ] 所有文件路径都相对于项目根目录

## ✅ GitHub 上传检查

1. **初始化Git仓库**（如果还没有）：
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **推送到GitHub**：
   ```bash
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

3. **确认所有必需文件已上传**：
   - 检查 `output/artifacts/` 文件夹中的所有文件
   - 确认 `.joblib` 文件已上传（可能较大，需要时间）

## ✅ Streamlit Cloud 部署步骤

1. 访问 [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. 使用GitHub账号登录
3. 点击 "New app"
4. 选择你的GitHub仓库
5. 配置设置：
   - **Main file path**: `app.py`
   - **Python version**: 3.9 或更高（推荐 3.9 或 3.10）
6. 点击 "Deploy"

## ⚠️ 常见问题

### 问题1: 模型文件太大无法上传
**解决方案**: 
- 使用 Git LFS (Large File Storage)
  ```bash
  git lfs install
  git lfs track "*.joblib"
  git add .gitattributes
  git add output/artifacts/*.joblib
  git commit -m "Add model files with LFS"
  git push
  ```

### 问题2: 部署后显示"模型文件未找到"
**解决方案**:
- 检查 `output/artifacts/` 文件夹是否已上传
- 确认文件路径在 `app.py` 中正确
- 查看 Streamlit Cloud 的日志获取详细错误信息

### 问题3: 依赖安装失败
**解决方案**:
- 检查 `requirements.txt` 中的版本号是否兼容
- 尝试固定版本号，例如：`pandas==1.5.3`
- 查看 Streamlit Cloud 的构建日志

### 问题4: 应用运行缓慢
**解决方案**:
- SHAP计算可能较慢，考虑减少 `nsamples` 参数
- 在 `app.py` 中，将 `nsamples=200` 改为 `nsamples=50` 或更少

## 📝 部署后验证

部署成功后，请验证：
- [ ] 应用能够正常加载
- [ ] 侧边栏输入框正常显示
- [ ] 点击"Predict and explain"按钮后能正常预测
- [ ] SHAP图表能正常显示

## 🔗 有用的链接

- [Streamlit Cloud 文档](https://docs.streamlit.io/streamlit-community-cloud)
- [Git LFS 文档](https://git-lfs.github.com/)
- [GitHub 文件大小限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

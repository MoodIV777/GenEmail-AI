# Email Agent on Colab

Start date: 07/02/2025
End date: 07/10/2025
Progress: 0
Status: Done

### About project

## **GenEmail AI - 个性化邮件生成模型 PoC (v1.0)**

| **文档状态** | **草稿 / DRAFT** |
| --- | --- |
| **项目代号** | GenEmail (Generative Email) |
| **创建日期** | 2025-07-02 |
| **最后更新** | 2024-07-11 |
| **负责人 (Owner)** | Alex Nie |
| **核心干系人** | [研发Leader], [算法Leader], [项目经理] |
| **目标上线** | PoC v1 完成：2025-07-10 |

---

### Action items

### **1. 背景与价值 (Background & Value)**

### 1.1 项目背景

在当前运营环境中，我们通过邮件与用户进行沟通，但普遍采用模板化、非个性化的内容，导致邮件打开率、点击率和用户转化率等关键指标未达到理想水平。用户的多样化背景（职业、兴趣、地域等）未被有效利用，造成了潜在的互动与商业机会流失。

### 1.2 项目愿景

我们计划利用前沿的大语言模型（LLM）技术，构建一个能深度理解用户画像、并自动生成高度个性化邮件内容的AI智能体（Agent）。该智能体旨在提升用户沟通体验，增强用户粘性，并最终驱动业务增长。

### 1.3 商业价值

- **提升用户参与度：** 通过个性化内容吸引用户，预期可提升邮件打开率与点击率。
- **提高转化效率：** 根据用户画像定制营销话术或产品推荐，有望提高核心转化指标。
- **验证技术路径：** 本项目作为概念验证（PoC），旨在探索“LLM+用户数据”在企业场景下的技术可行性与成本效益，为未来更大规模、更深层次的AI应用（如企业私有化部署、实时智能客服等）奠定基础。

---

### **2. 项目目标与衡量指标 (Objectives & Key Results)**

### 2.1 一期目标 (Phase 1 Objectives)

本项目的首要目标是完成一个功能可用的概念验证（PoC），验证核心技术路径和产品假设。

1. **O1：技术可行性验证** - 成功在Colab Pro环境中，使用LoRA技术对Qwen3-8B模型进行微调，使其能够根据用户画像生成指定风格的邮件。
2. **O2：核心能力构建** - 产出一个微调后的模型，该模型能够接收结构化的用户画像输入，并生成内容相关、逻辑通顺、符合预设风格（如商务、亲切、营销）的个性化邮件。

### 2.2 核心结果 (Key Results)

- **KR1 (针对O1):** 完成LoRA训练流程，并将训练好的Adapter权重成功保存至Google Drive，实现模型的持久化存储与可复现训练。
- **KR2 (针对O2):** 针对 ≥3 种预设邮件风格（例如：商务、亲切、营销），模型生成内容的**人工评估通过率达到85%**（评估维度：相关性、流畅度、风格一致性）。

---

### **3. 项目范围 (Scope)**

### 3.1 范围内 (In Scope)

- **模型选型：** 采用 Qwen/Qwen3-8B-Instruct (非量化版本)作为基础模型。
- **训练环境：** 使用 Google Colab Pro 作为本次PoC的训练与推理平台。
- **训练技术：** 核心采用PEFT (Parameter-Efficient Fine-Tuning)中的LoRA技术，以在有限显存下完成微调。
- **数据集：**
    - 初期使用**模拟数据集**跑通流程（约2000条）。
    - 数据集包含 `instruction` (指令/风格)、`input` (用户画像)、`output` (目标邮件) 三个核心字段。
    - 支持 `.csv` 和 `.json` 格式的数据源。
- **功能实现：**
    - 模型能根据用户画像字段（如年龄、职业、家庭、收入等）生成内容。
    - 模型能根据 `instruction` prompt 控制生成邮件的风格。
    - 提供基础的Web Agent（Gradio）进行交互式演示。

### 3.2 范围外 (Out of Scope)

- **全量微调：** 本次不进行全量微调（Full Fine-tuning）。
- **企业级部署：** 不涉及生产环境的私有化部署、高并发API服务等。
- **复杂数据处理：** 不处理非结构化数据源或需要复杂ETL流程的异构数据。
- **实时反馈循环：** 不包含用户真实邮件回复数据的实时收集与模型自动迭代（Reinforcement Learning from Human Feedback）。
- **A/B测试框架：** 不开发用于线上评估模型效果的A/B测试系统。

---

### **4. 功能需求 (Functional Requirements)**

| 模块 | 需求ID | 需求描述 | 优先级 | 备注 |
| --- | --- | --- | --- | --- |
| **数据准备** | FR-1 | **数据集构建与格式化** | **P0** | 系统需支持从 `.csv` 或 `.json` 文件加载结构化数据。数据需包含用户画像、指令和目标输出三个字段。 |
|  | FR-2 | **数据预处理** | **P0** | 预先清洗数据集，筛选所需的属性并对缺失值进行处理。将数据集转化为`.json` 或者`.jsonl` 格式。 |
|  |  |  |  |  |
| **模型训练** | FR-3 | **LoRA微调** | **P0** | 实现基于`peft`库的LoRA微调流程。LoRA配置（如`r`, `lora_alpha`, `target_modules`）需可配置。训练过程应使用`bfloat16`或`fp16`以优化显存占用。 |
|  | FR-4 | **模型断点续训与保存** | **P1** | 训练过程中需定期（如 `save_steps=50`）将LoRA adapter权重保存至Google Drive，以防止Colab会话中断导致进度丢失，并支持从检查点恢复训练。 |
|  | FR-5 | **训练监控** | **P1** | 训练过程应输出关键日志（如 `loss`, `learning_rate`），便于监控训练状态。 |
|  |  |  |  |  |
| **模型推理** | FR-6 | **个性化邮件生成** | **P0** | 开发一个推理函数，能够加载基础模型和微调后的LoRA权重，根据输入的用户画像和风格指令，生成一封完整的邮件。 |
|  | FR-7 | **风格可控性** | **P1** | 推理函数需支持通过Prompt调整生成内容的风格/语气，例如：“请生成一封**商务严谨**的欢迎邮件”。 |

---

### **5. 技术可行性分析与方案 (Technical Feasibility & Design)**

### 5.1 核心技术栈

- **基础模型:** `Qwen/Qwen3-8B-Instruct`
- **训练框架:** `PyTorch`, `Transformers`
- **微调技术:** `PEFT (LoRA)`
- **开发环境:** `Google Colab Pro` (提供T4/A100 GPU)
- **数据存储:** `Google Drive` (用于挂载数据集和保存模型权重)

### 5.2 显存占用分析 (VRAM Feasibility)

根据初步评估，Colab Pro提供的GPU资源（约15-16GB VRAM）足以支持本项目的技术方案。

| **操作模式** | **预估VRAM需求** | **Colab Pro (T4/A100) 支持度** | **备注** |
| --- | --- | --- | --- |
| **全量微调 (FP16)** | > 70GB | ❌ **不支持** | 显存不足，不在考虑范围内。 |
| **LoRA 微调 (FP16, r=64)** | ~7-25GB | ✅ **强烈推荐** | **核心方案。** 显存占用可控，是平衡效果与资源的最优选。 |
| **模型推理** | ~9-12GB | ✅ **支持** | 可通过CPU offloading或不保存KV缓存进一步优化。 |

### 5.3 实施路线图 (Roadmap)

**Phase 1: 环境搭建与流程验证 (预计1周)**

1. **任务：** 配置Colab Pro环境，安装所有依赖库。
2. **任务：** 编写脚本，成功加载`Qwen3-8B-Instruct`模型到Colab实例。
3. **产出：** 一个可运行的环境配置Notebook。

**Phase 2: 数据集构建与首次微调 (预计1-2周)**

1. **任务：** 定义最终的数据集Schema，并生成一个包含约300条样本的模拟数据集（包含3种风格）。
2. **任务：** 实现LoRA微调的完整代码，并在模拟数据集上跑通训练流程。
3. **产出：**
    - 第一版模拟数据集 (`customer_segmentation.csv`)。
    - 可工作的LoRA微调Notebook (`finetune_notebook_v1.ipynb`)。
    - 第一个微调后的LoRA模型权重，保存在Google Drive。

---

### **6. 风险与对策 (Risks & Mitigation)**

| 风险类别 | 风险描述 | 可能性 | 影响 | 应对措施 |
| --- | --- | --- | --- | --- |
| **技术风险** | Colab Pro资源不稳定（如GPU被回收、运行时长限制）导致训练中断。 | 中 | 中 | **对策:** 采用断点续训机制，定期将模型权重和训练状态保存到Google Drive。 |
| **数据风险** | 模拟数据与真实场景存在偏差，导致模型在真实数据上表现不佳（Domain Shift）。 | 高 | 中 | **对策:** 在设计模拟数据时，确保用户画像的字段、分布与真实数据保持一致。本项目PoC阶段接受此风险，后续阶段需引入真实数据。 |
| **效果风险** | 模型生成内容出现事实性错误（Hallucination）或不符合预期的、有害的内容。 | 中 | 高 | **对策:** <br>1. 在Prompt中加入严格的指令约束。<br>2. 对生成结果进行人工审核和后处理。<br>3. 本阶段不直接对客，风险可控。 |

---

### **7. 附录 (Appendix)**

### 7.1 数据集格式示例 (`.cvs/.jsonl`)

```bash
<bound method NDFrame.head of          ID  Gender  Age   Education Marital_Status   Income  KidAge  \
0      5524  Female   57  Graduation         Single  58138.0    -1.0   
1      2174    Male   60  Graduation         Single  46344.0    35.0   
2      4141    Male   49  Graduation       Together  71613.0    -1.0   
3      6182    Male   30  Graduation       Together  26646.0     5.0   
4      5324  Female   33         PhD        Married  58293.0     8.0   
...     ...     ...  ...         ...            ...      ...     ...   
2235  10870    Male   47  Graduation        Married  61223.0    22.0   
2236   4001    Male   68         PhD       Together  64014.0    43.0   
2237   7270  Female   33  Graduation       Divorced  56981.0    -1.0   
2238   8235  Female   58      Master       Together  69245.0    33.0   
2239   9405  Female   60         PhD        Married  52869.0    35.0   

      Familysize  Recency  
0              1       58  
1              3       38  
2              2       26  
3              3       26  
4              3       94  
...          ...      ...  
2235           3       46  
2236           5       56  
2237           1       91  
2238           3        8  
2239           4       40  

[2240 rows x 9 columns]
```

```json
//中文字符已被转化为 Unicode 编码
{"input":"[Segment: Need_Reengagement] \u8be5\u7528\u6237\u4fe1\u606f\u5982\u4e0b\uff1a\n\u6027\u522b\u4e3a Female\uff0c\u4eca\u5e74\u7ea6 57 \u5c81\uff0c\u5a5a\u59fb\u72b6\u51b5\u4e3a Single\uff0c\u5bb6\u5ead\u6210\u5458\u603b\u5171 1 \u4eba\uff0c\u5e74\u5ea6\u6536\u5165\u4f30\u8ba1\u4e3a 58138 \u5143\uff0c\u81ea\u4e0a\u6b21\u4e92\u52a8\u8d77 58 \u5929","output":"\u6211\u4eec\u60f3\u5ff5\u60a8\uff0c\n\n\u6211\u4eec\u51c6\u5907\u4e86\u4e3a\u8001\u5ba2\u6237\u5b9a\u5236\u7684\u4ea7\u54c1\u63a8\u8350\uff0c\u7a0d\u65e9\u7684\u53cd\u9988\u5df2\u88ab\u6211\u4eec\u7eb3\u5165\u91cd\u65b0\u8bbe\u8ba1\n\n\u4e3a\u60a8\u5b9a\u5236\uff0c\u70b9\u51fb\u9886\u53d6 \u2192 [\u8bbf\u95ee\u94fe\u63a5] \n\n\u795d\u597d\uff0c\n\u60a8\u7684\u54c1\u724c\u56e2\u961f","segment":"Need_Reengagement"}
```

### 7.2 核心代码片段参考

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# 读取数据
data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/...')

# 处理缺失值
data['KidAge'] = data['KidAge'].fillna(-1)  # -1 表示没有小孩
data['Income'] = data['Income'].fillna(data['Income'].median())
data[['Gender', 'Education', 'Marital_Status']] = data[['Gender', 'Education', 'Marital_Status']].fillna('Unknown')

print(data.head)
```

---


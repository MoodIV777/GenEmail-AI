# Email Agent 用户分层分类

Start date: 07/14/2025
End date: 07/21/2025
Progress: 0
Status: Done

### About project

---

## **GenEmail AI - 用户分层与个性化策略模块 (v1.1)**

| **文档状态** | **草稿 / DRAFT** |
| --- | --- |
| **项目代号** | GenEmail (Generative Email) |
| **关联文档** | `GenMail AI - 个性化邮件生成模型 PoC (v1.0)` |
| **创建日期** | 2025-07-14 |
| **最后更新** | 2025-07-22 |
| **负责人 (Owner)** | Alex Nie |
| **核心干系人** | [研发Leader], [算法Leader], [数据科学Leader], [运营负责人] |
| **目标上线** | Phase 1 (分层模型) 完成：2025-07-21 |

---

### **1. 背景与价值 (Background & Value)**

### 1.1 项目背景

在 `GenMail AI v1.0 PoC` 中，我们成功验证了利用LoRA技术对大语言模型（如Qwen3-8B）进行微调，以生成个性化邮件的技术可行性。然而，该方案采用“一刀切”的模型策略，即用一个统一模型应对所有用户，这在扩展性和精细化运营上面临挑战。

为了将AI能力真正转化为业务增长动能，我们必须从“千人一面”走向更智能的“**分层分类，千人千面**”。我们发现，直接实现完全的“千人千面”成本高、可解释性差且难以与现有运营策略结合。因此，本项目提出一个更务实、更高效的演进路径：**先对用户进行精细化分层，再为不同层级的用户匹配最优的个性化沟通策略。**

### 1.2 核心策略对比与选择

| 维度 | **一步到位“千人千面” (A)** | **本次方案“分层分类” (B)** | **选择理由** |
| --- | --- | --- | --- |
| **实现成本** | 高（数据量、算力、模型复杂度均高） | **中低（可控）** | 方案B资源消耗可控，更适合当前阶段快速迭代。 |
| **可解释性** | 低（LLM生成过程为黑盒） | **高** | 方案B的用户分层标签（如“高价值-价格敏感型”）清晰，便于业务理解和策略调整。 |
| **策略可控性** | 低（难以对单一用户生成结果进行策略干预） | **高** | 可针对特定用户群体，统一调整邮件的风格、主题和营销重点，便于A/B测试。 |
| **迭代效率** | 慢（模型调整牵一发而动全身） | **快** | 用户分类模型与邮件生成模型可解耦迭代，灵活性高。 |

**结论：** 本项目选择**分层分类 (B)** 作为核心实现策略，因为它在成本、可控性和业务结合度上具有明显优势，是通往最终“千人千面”愿景的坚实桥梁。

---

### **2. 项目目标与衡量指标 (Objectives & Key Results)**

### 2.1 一期目标 (Phase 1 Objectives)

1. **O1：实现用户自动分层能力** - 基于用户的静态画像和动态行为数据，构建一个能够自动对用户进行多层次分类的模型。
2. **O2：构建分层-风格映射与生成机制** - 为每个细分用户群体，定义并实现一套与之匹配的个性化邮件生成策略（通过Prompt Engineering或轻量级微调）。
3. **O3：验证分层策略的业务有效性** - 通过原型系统和模拟数据，证明分层策略相比单一策略，在内容相关性和多样性上有显著提升。

### 2.2 核心结果 (Key Results)

- **KR1 (针对O1):** 成功构建一个层次化分类模型，能将用户库划分为至少 **3** 个一级类别（如：高/中/低价值）后续提升泛化能力和鲁棒性以达到更多的类别如 **5** 个二级子类别（如：高价值-新品尝鲜型），在手动标注的测试集上，分类**准确率 > 85%**。
- **KR2 (针对O2):** 为每个二级子类别定义并实现至少1种独特的生成风格（通过Prefix-Tuning或LoRA Adapter），模型输出与指定风格的**匹配度人工评估 > 90%**。
- **KR3 (针对O3):** 上线一个内部演示版本，输入一个用户ID，系统能自动展示其所属分类、匹配的生成策略，并生成一封个性化邮件。

---

### **3. 系统架构与功能需求 (Architecture & Functional Requirements)**

### 3.1 系统架构设计

本项目将构建一个模块化的邮件生成流水线（Pipeline）：

*(这是一个示意图，实际PRD中可由设计工具绘制)*

1. **用户特征工程 (User Feature Engineering):** 整合用户画像、行为数据，并生成用户嵌入向量。
2. **用户分层模块 (User Segmentation Module):** 接收用户特征，输出用户的类别标签。
3. **策略/风格映射模块 (Strategy/Style Mapping Module):** 根据用户标签，选择对应的邮件生成策略（如：特定的Prefix、LoRA Adapter、Prompt模板）。
4. **个性化生成模块 (Personalized Generation Module):** 加载基础LLM和选定的策略模块，生成最终邮件内容。

### 3.2 功能需求 (Functional Requirements)

| 模块 | 需求ID | 需求描述 | 优先级 |
| --- | --- | --- | --- |
| **用户分层** | FR-9 | **层次化分类模型训练** | **P0** |
|  | FR-10 | **用户聚类分析** | **P1** |
|  |  |  |  |
| **策略映射** | FR-11 | **风格模块化管理** | **P0** |
|  | FR-12 | **动态策略加载** | **P0** |
|  |  |  |  |
| **生成增强** | FR-13 | **用户嵌入注入** | **P1** |
|  | FR-14 | **轻量级微调 (PEFT)** | **P0** |

---

### **4. 实施路线图 (Roadmap)**

**Phase 1: 核心分层与粗粒度生成 (快速验证)**

1. **数据与标注：** 基于业务规则，手动标注一个小规模数据集（约2000条），定义3-5个核心用户类别。
2. **模型构建：** 使用`SK-Learn`或`PyTorch`构建一个基础分类模型，验证分层能力。
3. **生成实现：** 采用 **Prompt Engineering + 风格指令**的方式，为每个类别设计不同的邮件生成模板。
4. **目标：** 跑通整个Pipeline，验证分层策略优于单一策略。

**Phase 2: 细粒度微调与效果提升**

1. **层次化扩展：** 在大类下引入2-3级分类/聚类，细化用户群体。
2. **技术升级：** 引入 **LoRA Adapter Tuning**，为每个关键的子类别训练一个专属的Adapter，实现更精细的风格控制。
3. **数据增强：** 引入用户行为序列数据，提升分层模型的准确性。
4. **目标：** 生成内容的个性化程度和多样性显著提升。

**Phase 3: 向动态个性化演进**

1. **技术探索：** 探索将实时用户行为向量作为LLM的输入嵌入，实现更动态的风格调整。
2. **闭环反馈：** 设计初步的反馈机制，利用邮件打开/点击率等指标，作为未来模型迭代（如RLAIF）的信号。
3. **目标：** 验证从“分层”到“准千人千面”的技术路径。

---

### **5. 技术栈选型 (Tech Stack)**

- **模型框架:** `Hugging Face Transformers`, `PyTorch`
- **PEFT库:** `peft` (用于LoRA, Prefix-Tuning等)
- **分类/聚类工具:** `scikit-learn` (用于快速基线), `KMeans`/`HDBSCAN` (用于可视化和高级聚类)
- **实验平台:** `Google Colab Pro`
- **目标部署环境:** `AWS SageMaker` / `阿里云PAI` (支持多Adapter/多模型部署)

---

### **6. 风险与对策 (Risks & Mitigation)**

| 风险类别 | 风险描述 | 可能性 | 影响 | 应对措施 |
| --- | --- | --- | --- | --- |
| **数据风险** | **“冷启动”问题**：新用户数据不足，难以准确分类。 | 高 | 中 | **对策:** 为新用户或特征稀疏的用户设计一个通用的、高质量的默认沟通策略。 |
| **模型风险** | **类别固化**：分层模型可能导致用户被长期锁定在某个类别，无法反映其成长和变化。 | 中 | 中 | **对策:** 定期（如每季度）重新训练分层模型；在特征工程中加入时间衰减因子，赋予近期行为更高权重。 |
| **业务风险** | **策略与业务脱节**：定义的分类和风格与实际业务目标不符，导致“为了AI而AI”。 | 中 | 高 | **对策:** 项目初期即让**运营团队**深度参与，共同定义用户分类标准和各类别沟通目标，确保技术服务于业务。 |

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

# 现在构造聚类模型（用作 segment 1 粗标签）
X = data[['Age', 'KidAge', 'Familysize', 'Income', 'Recency']]
ct = ColumnTransformer([
    ('num', StandardScaler(), ['Age', 'KidAge', 'Familysize', 'Income']),
], remainder='passthrough')

# 假设划分为4类用户：活跃/价值/召回/休眠
kmeans = KMeans(n_clusters=4, random_state=42)
pipeline = Pipeline([
    ('preprocessor', ct),
    ('cluster', kmeans)
])

# 训练聚类模型并标注
data['L1_Label'] = pipeline.fit_predict(X)
```

```python
# 基于 Recency/Income 构造标签策略（模拟 L2 细分）

def assign_segment_label(row):
    if row['Recency'] <= 30:
        if row['Income'] > 80000:
            return "HighValue_Active"
        elif row['KidAge'] > 0:
            return "Parent_Active"
        else:
            return "Regular_Active"
    elif row['Recency'] <= 90:
        return "Need_Reengagement"
    else:
        return "Dormant"
# 添加 L2 粒度标签
data['L2_Label'] = data.apply(assign_segment_label, axis=1)
```

```python
# LoRA 微调模型与客户标签 Adapter 构建（训练阶段）

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import torch

# 使用 Qwen3 - 8B
# 模型从 Drive 的直接调用方式

# Model & Tokenizer Path
# 指定模型名称和 Google Drive 保存路径
model_path = "/content/drive/My Drive/Colab Notebooks/..."

# Load Tokenizer (from your local Drive)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load Model using accelerate (automatic device management + GPU offload)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,   # 减少显存占用
    device_map="auto",       # 自动 offload（根据显存负荷将层数划分到 GPU / CPU）
    trust_remote_code=True
)

# 添加 LoRA Adapter 配置
# 注意：Qwen3 属于因果语言模型（CAUSAL_LM）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # LoRA 低秩矩阵的秩
    lora_alpha=16,          # 用于缩放 LoRA 权重，默认与 r 搭配使用
    target_modules=["q_proj", "v_proj"],  # Qwen3 的注意力模块投影层
    lora_dropout=0.1,       # Dropout 概率
    bias="none"             # 不使用 bias 的 LoRA
)

# 构建 LoRA 微调模型 - 冻结了98%以上主模型的参数
peft_model = get_peft_model(model, lora_config)

# 可选：查看模型结构和哪些模块被 LoRA 适配
peft_model.print_trainable_parameters()
```

```python
# 模拟用户-邮件训练数据 construction

# 数据列中必须存在的字段：
# - L2_Label（用户 segment）
# - Age, KidAge, Marital_Status, Income, Familysize, Recency

import pandas as pd

def generate_pseudo_email(label):
    greeting = {
        'HighValue_Active': '亲爱的尊贵客户',
        'Parent_Active': '您好，家长朋友',
        'Dormant': '亲爱的顾客',
        'Need_Reengagement': '我们想念您',
        'Regular_Active': '您好，尊贵用户',
    }.get(label, "您好")

    body = {
        'HighValue_Active': '我们已经为您预留了最新的精品产品，近期我们也有为 VIP 用户准备的试点活动，诚邀您优先体验',
        'Parent_Active': '我们的暑期亲子商品正式上新，适合家庭备货的优惠套餐请查收',
        'Dormant': '您有一段时间没有购物了，我们诚邀您回来享受全新改版的专属优惠',
        'Need_Reengagement': '我们准备了为老客户定制的产品推荐，稍早的反馈已被我们纳入重新设计',
        'Regular_Active': '我们也为您精选了一些新品，有兴趣的话欢迎查阅，如需专属推荐请联系我们！',
    }.get(label, "我们为您准备了一些新品推荐，祝您生活愉快")

    call_to_action = {
        'HighValue_Active': '优先查看 →',
        'Parent_Active': '立即查看亲子专区 →',
        'Dormant': '限时优惠，立即激活账户 →',
        'Need_Reengagement': '为您定制，点击领取 →',
        'Regular_Active': '详细了解 →',
    }.get(label, "了解更多 →")

    template = f"{greeting}，\n\n{body}\n\n{call_to_action} [访问链接] \n\n祝好，\n您的品牌团队"
    return template

def user_profile_to_prompt(row):
    # 函数：将用户画像转为“能让模型理解的语义描述”
    prompt = f"[Segment: {row['L2_Label']}] 该用户信息如下：\n"

    if row['Gender'] != 'Unknown':
        prompt += f"性别为 {row['Gender']}，"
    prompt += f"今年约 {row['Age']} 岁，"
    if row['KidAge'] > 0:
        prompt += f"家中孩子平均年龄为 {int(row['KidAge'])} 岁，"
    prompt += f"婚姻状况为 {row['Marital_Status']}，"
    # prompt += f"教育水平为 {row['Education']}，"
    prompt += f"家庭成员总共 {int(row['Familysize'])} 人，"
    prompt += f"年度收入估计为 {'%.0f'%row['Income']} 元，"
    prompt += f"自上次互动起 {int(row['Recency'])} 天"

    return prompt

# 假设 data 是当前的 Pandas DataFrame，包含 L2_Label 等字段
# 向量化构造 input / output
data['input'] = data.apply(user_profile_to_prompt, axis=1)
data['output'] = data['L2_Label'].apply(generate_pseudo_email)
data['segment'] = data['L2_Label']

# 构建 Dataset 数据集对象，准备供 Model 读入训练
from datasets import Dataset
train_dataset = Dataset.from_pandas(data[['input', 'output', 'segment']])
# train_dataset.to_json("/content/drive/My Drive/Colab Notebooks/...", lines=True, orient='records', index=False)

# 可以输出样本看看
print("示例训练样本：")
print(train_dataset[0]['input'])
print("\n模型应输出的邮件内容为：")
print(train_dataset[0]['output'])
```

```python
--- 当前环境加载的库版本 ---
TRL Version: 0.19.1
Transformers Version: 4.53.2
PEFT Version: 0.16.0
------------------------------
```


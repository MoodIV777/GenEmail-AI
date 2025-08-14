# GenEmail-AI

📨 Personalized Email Agent with LoRA-finetuned Qwen3-8B
·生成一封精准且风格定制的 Email，用于激活/召回/个性化推送

该项目会基于客户画像和用户类型（Segmentation）自动为每一位用户生成高性能个性化邮件及内容推送文案，训练方案使用 LoRA（Low-Rank Adaptation） 对 Qwen3-8B 分类微调，模型推理支持注入强化 prompt+内联适配体，构建更加针对性的文案建议。

---

## 🧠 核心技术栈

- **LLM**: Alibaba Qwen3-8B（本地或 Colab 运行）
- **LoRA/CLL** 继续支持 PEFT 适配（self-gross unified strategy per-segment）
- **SFT Learning (TRL)** - 使用 `trl` 框架来对 Prompt-Response pairs 微调
- **数据驱动 pipeline**：人物画像策略数据 → NaturalPrompt → Email Generation
- **Transformers + AutoTokenizer** 包装适配推理语言逻辑
- **可扩展性强**：新的 segment 和 prompt template 可快速集成进系统
- **推理时进行 Row2Email 转换，支持 API/数据流程推送邮件生成**

---

## 📁 项目目录结构

```bash
Email-LLM-Agent/
│
├── data/
│   ├── train_emails.jsonl            # 已生成的原始训练数据（input + output + segment）
│   ├── Customer Segmentation Classification            # 人物画像数据（含 ID, Age, KidAge, Profession ...）
│   └── Customer Segmentation Clustering
│
├── adapters/
│   ├── Dormant/                     # per-segment 的 LoRA weights
│   ├── HighValue_Active/
│   ├── Parent_Active/
│   └── ...
│
├── README.md                        # 当前文件
└── requirements.txt                 # 所需库清单
```

---

## 📊 数据集结构说明

### `profiles.csv`: 来自 customer data 的人物特征

| Field | Description |
|-------|-------------|
| ID | 唯一+特征 |
| Gender | 性别 |
| Marital_Status | 婚育状况 |
| Age | 年龄 |
| KidAge | 最小最大或中位年龄 |
| Education | 教育经历/学位 |
| Profession | 职业 |
| Work_Experience | 工作年数 |
| FamilySize | 家庭成员的数量 |
| Income | 个人年度收入 |
|Level | 内部客户评级 |
+输出

### `enhanced_emails.jsonl`

```json
{
  "input": "客户档案信息...",
  "output": "完整的邮件推荐文案...",
  "segment": "Parent_Active"
}
```

---

## 📦 安装准备

```bash
pip install trl peft transformers datasets accelerate torch
```

如果你使用的是 HuggingFace model hub，可改用：

```bash
pip install transformers[ray]
```

---

## 🔧 使用方法指导（Mac/Colab/本地 Python）

### 🧵 数据准备：从 profile 生成 `jsonl`

```bash
python scripts/profile2prompt.py \
    --input profiles.csv \
    --output enhanced_emails.jsonl
```

✅ 输出提示：根据每个用户特征生成 instruction（prompt），格式为 `.jsonl`

### 🔧 注入训练：LoRA + per Segment

```bash
python scripts/train_per_segment.py \
    --data enhanced_emails.jsonl \
    --base-model-path ./Models/Qwen3-8B \
    --output-base-path ./adapters
```

✅ 输出每个 segment 的 adapter 模块，位于各自目录下

### 🤖 生成邮件（推理阶段）

```bash
python scripts/generate_with_profile.py \
    --id 12345  # ID 测试
```

✅ 输出：根据该用户 profile 生成的完整 Email 内容

---

## 🎯 技术目标（面向企业应用）

| 模块 | 目标 |
|------|--------|
| 🎯 Segmentation Tool | 根据用户数据正确分群 |
| 🧠 Prompt Generator | 拓展用户描述为自然 prompt 输入模型 |
| 🟨 LoRA Fine-tuning | 已有 base 不变，训练每个 customer type 能力 |
| 🧾 Language Generation | 精准输出客户风格适配的邮件内容 |
| ⚙️ Build-in Metrics | BLEU / PPL 用于评估语言表现 |
| 🚀 多模型推理能力 | 后期支持 FastAPI email generator service |
| 🧩 Adapter Switching | 支持根据不同 segment 生成个性化风格逻辑 |

---

## 🔬 推荐模型策略进阶（可扩展）

- ✅ prompt 中可加入 `role: 你是一封 personalized email 机器`
- ✅ 现有 LoRA 模型支持多版本加载（adapter + chat template／Roadmap）
- ✅ `RAG + Persona Prompting`，模型生成风格更贴近真实 customer interaction 历史
- ✅ 更进一步可输出`邮件内容情感向量`，用于 email 打分系统
  - 使用 BERT + emotion-smash，判断 tonal → positive / persuasive / warmer 等维度

---

## 💌 适配的 Customer types 为以下几种

请你替换成你自己的名单（分类模型可支持这些群体）👇

```bash
Parent_Active, Regular_Active, Dormant, Need_Reengagement, HighValue_Active
```

这意味着你有一个可精准控制、可进一步扩展 **细分客户层邮件定制文本生成系统**（后续将会精细优化该客户细分功能，支持千人千面）

---

## 💡 示例邮件（可替换成你训练出的内容）

- **Dormant 用户 + profile prompt**

```prompt
[客户已离线 137 天，为她构造 Prompt]

这是一位女性用户 (35岁)，持本科教育背景，过去年购买量锐减
近期社交媒体互动较少，电子邮件点击率低

为她生成一封柔性邮件正文内容
```

- **模型输出**

```output
尊敬的 Julia，我们想念您！很高兴再次为您准备了独特的明细，我们挑选的必备主题和季节推荐基于您的往期喜爱...
```


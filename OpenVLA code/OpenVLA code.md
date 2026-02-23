# OpenVLA 整体代码架构

### 目录结构

```
openvla/
├── vla-scripts/          # 入口脚本
│   ├── train.py          # 训练主入口
│   ├── finetune.py       # LoRA 微调
│   └── deploy.py         # 部署
├── prismatic/            # 核心框架
│   ├── conf/             # 配置
│   ├── models/           # 模型
│   ├── vla/              # VLA专用组件
│   ├── training/         # 训练基础设施
│   ├── util/             # 工具函数
│   └── extern/hf/        # HuggingFace集成
└── experiments/          # 机器人实验
```

整个代码库的运转可以划分为三个紧密相连的阶段，由 6 个核心文件完美分工：

#### 1. 数据炼金炉 (Data Pipeline)

这个阶段的目标是把海量的、装不进内存的物理世界视频数据，洗成 AI 能懂的张量。

- **`datasets.py` (搬运工)**：搭建流式读取（IterableDataset）的传送带，源源不断地从硬盘抽取 Open-X 机器人轨迹数据，并进行图像增强。
    
- **`action_tokenizer.py` (翻译官)**：霸占 Llama 2 词表的最后 256 个位置，把连续的物理浮点数动作，强制离散化为这 256 个特殊 Token。
    
- **`materialize.py` (包装厂)**：工厂函数，将搬运工和翻译官组装起来，把数据包装成标准化对话（"What action should the robot take..."），并巧妙地给非动作部分的答案打上 `-100` 的 Loss 掩码。
    

#### 2. 大脑组装车间 (Model Architecture)

- **`prismatic.py` (核心胶水层)**：在这里，模型长出了眼睛和大脑。它实例化了视觉编码器（SigLIP+DinoV2）和语言模型（Llama 2），并通过一个 Projector 将视觉特征翻译并拼接到 LLM 的词嵌入空间中。它不写 Loss 计算，全权委托给 LLM 处理。
    

#### 3. 训练指挥部 (Training Orchestration)

- **`train.py` (总司令)**：这是一个纯粹的编排脚本，不含计算逻辑。它负责读取超参数，决定冻结哪些网络层（如 `vla-train` 模式下冻结视觉），并拉起所需的数据集和训练策略。
    
- **`base_strategy.py` (执行官)**：负责具体的反向传播和梯度更新。特别是在 `run_vla_training` 中，它针对无限流数据集取消了梯度累积，并且在训练时将预测的 Token 还原回连续动作来计算 L1 Loss，专门用于监控机器人的学习进度。

---
# train.py 分析
**定位**：训练主入口，264 行，纯编排脚本，自身不含任何模型计算逻辑。

### 整体结构

```
train.py
├── TrainConfig (L46-103)     配置定义
└── train(cfg) (L107-259)     训练函数
    ├── 初始化环境             L111-131
    ├── 加载模型               L140-157
    ├── 冻结参数               L159-180
    ├── 构建数据集             L189-204
    ├── 构建训练策略           L206-227
    ├── 构建 Metrics           L229-240
    └── 启动训练               L242-259
```

---
### TrainConfig (L46-103)

用 `draccus` 定义所有超参，关键字段：

```python
vla: VLAConfig          # 指向 VLARegistry 预设配置
                        # 默认: DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS
data_root_dir: Path     # Open-X 数据集路径
pretrained_checkpoint   # 有值→续训，无值→从base VLM开始
save_interval: int = 2500
image_aug: bool = False
seed: int = 7
```

`__post_init__` 从 `vla` 里提升超参到顶层（epochs/lr/batch_size 等），并校验 `expected_world_size == 实际GPU数`。

---
### train() 五个阶段

**① 模型加载 (L140-157)**

```python
vlm = load_vla(pretrained_checkpoint, ...)   # 续训
# 或
vlm = load(cfg.vla.base_vlm, ...)            # 从头
# 断言所有参数必须是 FP32
```

**② 冻结策略 (L159-180)**

根据配置决定 stage，共 4 种：

|stage|冻结情况|
|---|---|
|`vla-full-train`|全部解冻|
|`vla-train`|冻结视觉编码器（默认）|
|`vla-sandwich-train`|冻结LLM主体，训练视觉+projector+最后一层|
|`vla-last-layer-train`|只训LLM最后一层|

**③ 数据集构建 (L189-204)**

```python
vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
    image_transform=vlm.vision_backbone.get_image_transform(),  # 从模型取
    tokenizer=vlm.llm_backbone.get_tokenizer(),                 # 从模型取
    ...
)
```

`image_transform` 和 `tokenizer` 直接从模型取，保证数据预处理和模型完全匹配。

**④ 训练策略 (L206-227)**

```python
train_strategy = get_train_strategy(train_strategy=cfg.train_strategy, ...)
# cfg.train_strategy 来自 VLAConfig，通常是 "fsdp-full-shard"
train_strategy.run_setup(run_dir, n_train_examples)
```

**⑤ 启动训练 (L242-259)**

```python
train_strategy.run_vla_training(vla_dataset, collator, action_tokenizer, metrics, ...)
metrics.finalize()
dist.barrier()
dist.destroy_process_group()
```

---
### 启动方式

```bash
# 单卡调试
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py

# 多卡
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py

# 覆盖配置
torchrun ... vla-scripts/train.py --vla.type dinosiglip-224px+mx-oxe-magic-soup-plus
```
# materialize.py
是一个纯粹的**工厂函数**，没有任何计算逻辑，只负责把各个组件组装起来。
```python
def get_vla_dataset_and_collator(
    data_root_dir,           # 数据集根目录
    data_mix,                # 数据集混合名，如 "oxe_magic_soup_plus"
    image_transform,         # 从 vision_backbone 取来的图像预处理函数
    tokenizer,               # 从 llm_backbone 取来的分词器
    prompt_builder_fn,       # 对话格式构造函数（LLaMA2Chat/Vicuna等）
    default_image_resolution,# 如 (3, 224, 224)
    padding_side="right",
    predict_stop_token=True,
    shuffle_buffer_size=100_000,
    train=True,
    episodic=False,          # True→返回完整episode，False→单步transition
    image_aug=False,
) -> (Dataset, ActionTokenizer, PaddedCollatorForActionPrediction)
```

---

## 三步组装

**Step 1：创建 ActionTokenizer (L36)**

```python
action_tokenizer = ActionTokenizer(tokenizer)
# 用 LLaMA tokenizer 初始化，建立 256 bins 和词表末尾 token 的映射
```

**Step 2：创建 RLDSBatchTransform (L37-39)**

```python
batch_transform = RLDSBatchTransform(
    action_tokenizer,    # 动作离散化
    tokenizer,           # 文本tokenize
    image_transform,     # 图像预处理
    prompt_builder_fn,   # 构造chat prompt
    predict_stop_token,
)
# 这是每条数据从RLDS格式→模型输入格式的转换器
```

**Step 3：创建 Collator (L40-42)**

```python
collator = PaddedCollatorForActionPrediction(
    tokenizer.model_max_length,  # 最大序列长度（截断用）
    tokenizer.pad_token_id,      # padding token
    padding_side="right",
)
# 把一个batch里长度不同的样本pad到相同长度
```

**Step 4：创建 Dataset (L45-54)**

```python
cls = RLDSDataset if not episodic else EpisodicRLDSDataset
dataset = cls(
    data_root_dir, data_mix, batch_transform,
    resize_resolution=default_image_resolution[1:],  # (224,224)，去掉channel维
    ...
)
```

---

## 在 train.py 里的调用

```python
# train.py L191-200
vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
    cfg.data_root_dir,
    cfg.vla.data_mix,
    image_transform=vlm.vision_backbone.get_image_transform(),  # 模型决定预处理
    tokenizer=vlm.llm_backbone.get_tokenizer(),                 # 模型决定tokenizer
    prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
    default_image_resolution=vlm.vision_backbone.default_image_resolution,
    ...
)
```

这个文件的核心价值是：**把模型的配置（image_transform、tokenizer）和数据管道绑定在一起**，确保数据预处理和模型期望的输入格式完全一致。
# datasets.py
## 一、`RLDSBatchTransform` (L30-67) — 核心转换器

`@dataclass`，持有 4 个组件：

```python
action_tokenizer: ActionTokenizer
base_tokenizer:   PreTrainedTokenizerBase
image_transform:  ImageTransform
prompt_builder_fn: Type[PromptBuilder]
```

`__call__` 是整个数据管道最关键的一步，把一条 RLDS 原始数据转成模型输入：

```
RLDS batch
    ↓ 提取 image / action / language
    ↓ action_tokenizer(action)     连续动作 → token字符串
    ↓ prompt_builder               构造 chat 格式
    ↓ base_tokenizer               文本 → input_ids
    ↓ image_transform              PIL → pixel_values tensor
    ↓ label mask                   只保留最后 8 个位置（7动作+EOS）
    → {pixel_values, input_ids, labels, dataset_name}
```

---

## 二、`RLDSDataset` (L70-154) — 流式数据集

继承 `IterableDataset`，包装 RLDS TensorFlow 数据管道。

**`__init__`** 做两件事：

1. 根据 `data_mix` 查 `OXE_NAMED_MIXTURES` 得到数据集列表和采样权重
2. 配置 RLDS pipeline 参数：

```python
traj_transform_kwargs = {
    "window_size": 1,              # 单步预测，不做 action chunking
    "future_action_window_size": 0,
    "skip_unlabeled": True,        # 跳过没有语言标注的轨迹
}
frame_transform_kwargs = {
    "resize_size": (224, 224),
    "num_parallel_calls": 16,      # TF侧并行处理
}
```

图像增强（`image_aug=True` 时）：

```python
random_resized_crop(scale=[0.9, 0.9])
random_brightness/contrast/saturation/hue
```

**`__iter__`**：

```python
for rlds_batch in self.dataset.as_numpy_iterator():
    yield self.batch_transform(rlds_batch)
```

TF Dataset → numpy → `RLDSBatchTransform` → PyTorch tensor，这是 TF 和 PyTorch 的边界。

**`__getitem__` 故意抛异常**，强制使用流式迭代，不支持随机访问。

---

## 三、`EpisodicRLDSDataset` (L157-177) — 完整 episode 版本

继承 `RLDSDataset`，区别只有两点：

- `make_dataset` 用 `make_single_dataset`（只支持单数据集）
- `__iter__` 返回整条轨迹的所有步骤列表，而不是单步

```python
# 普通版：每次 yield 一个 step
yield self.batch_transform(rlds_batch)

# Episodic版：每次 yield 一整条轨迹
out = [self.batch_transform(tree_map(lambda x: x[i], rlds_batch))
       for i in range(rlds_batch["action"].shape[0])]
yield out
```

用途：可视化、评估时需要看完整轨迹。

---

## 四、`DummyDataset` (L180-232) — 调试用数据集

继承普通 `Dataset`（非 Iterable），用随机数据模拟真实数据集，逻辑和 `RLDSBatchTransform.__call__` 完全一致，方便在没有真实数据时验证训练流程。

```python
image = Image.fromarray(np.random.rand(224, 224, 3) * 255)
action = np.random.rand(7).astype(np.float32)
instruction = "do something spectacular"
```

---

## 整体关系

```
RLDSDataset
    └── make_dataset()  → make_interleaved_dataset (RLDS管道)
    └── __iter__()      → batch_transform(rlds_batch)
                              └── RLDSBatchTransform.__call__()

EpisodicRLDSDataset     (可视化/评估用)
    └── 同上，但yield整条轨迹

DummyDataset            (调试用，随机数据)
```

# base_strategy.py
## 一、类结构

```
TrainingStrategy (ABC)
├── __init__()              存储所有超参，计算 grad_accumulation_steps
├── save_checkpoint()       抽象方法，子类实现（DDP/FSDP不同）
├── run_setup()             抽象方法，子类实现（优化器/调度器初始化）
├── clip_grad_norm()        抽象方法，子类实现（DDP/FSDP梯度裁剪不同）
├── run_training()          VLM预训练循环（支持梯度累积）
└── run_vla_training()      VLA训练循环（不支持梯度累积）
```

三个抽象方法由 [ddp.py](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/prismatic/training/strategies/ddp.py) 和 [fsdp.py](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/prismatic/training/strategies/fsdp.py) 各自实现。

---

## 二、`__init__` (L35-88)

核心计算：

```python
self.grad_accumulation_steps = global_batch_size // per_device_batch_size // world_size
```

例如：`global_batch_size=256, per_device_batch_size=8, world_size=8` → `grad_accumulation_steps=4`

两个硬约束：

- `global_batch_size % per_device_batch_size == 0`
- 混合精度只支持 BF16，不支持 FP16

---

## 三、`run_training` (L106-241) — VLM 预训练循环

用于 `align` / `finetune` 阶段，支持梯度累积，有外层 epoch 循环。

**Sampler 选择 (L116-138)**：

```python
if "finetune" in stage and batch_construction_strategy == "split-modality":
    sampler = SplitModalitySampler(...)   # 有图/无图样本分开采样，提升效率
else:
    sampler = DistributedSampler(...)     # 标准分布式采样
```

**梯度累积 (L208-212)**：

```python
normalized_loss = loss / self.grad_accumulation_steps
normalized_loss.backward()

if (train_idx + 1) % self.grad_accumulation_steps == 0:
    self.clip_grad_norm()
    self.optimizer.step()
    ...
```

注释里坦承这个实现"技术上不正确"——每个 batch mask 掉的 token 数不同，直接除以 steps 不等于正确的加权平均。但实验证明 7B 规模下效果一样，13B 规模下梯度累积本身就有问题（BF16 精度不够）。

**checkpoint 时机**：每个 epoch 结束时保存（`max_steps=None`），或达到 `max_steps` 时保存。

---

## 四、`run_vla_training` (L245-389) — VLA 训练循环

专为 OpenVLA 设计，与 `run_training` 的关键区别：

|             | run_training    | run_vla_training          |     |     |
| ----------- | --------------- | ------------------------- | --- | --- |
| 数据集类型       | `Dataset`（有限）   | `IterableDataset`（无限流）    |     |     |
| epoch 循环    | 有外层 `for epoch` | 无，靠 `max_steps` 终止        |     |     |
| 梯度累积        | 支持              | 强制禁止（assert == 1）         |     |     |
| num_workers | 2               | 0（RLDS自己管并行）              |     |     |
| 额外指标        | 无               | action_accuracy + L1 loss |     |     |
| 保存时机        | epoch结束         | 每 `save_interval` 步       |     |     |

**每个 batch 的完整流程**：

```python
# 1. 前向 + loss
output = self.vlm(input_ids, attention_mask, pixel_values, labels)
loss = output.loss          # CrossEntropy，只在动作token位置

# 2. 反向
loss.backward()

# 3. 计算监控指标（不影响梯度）
action_preds = output.logits[:, num_patches:-1].argmax(dim=2)
action_gt    = batch["labels"][:, 1:]
mask         = action_gt > action_token_begin_idx
action_accuracy = (action_preds == action_gt)[mask].mean()
action_l1_loss  = F.l1_loss(decode(action_preds[mask]), decode(action_gt[mask]))

# 4. 按数据集分别统计（仅 rank 0）
for ds in set(batch["dataset_names"]):
    ...

# 5. 梯度更新
clip_grad_norm() → optimizer.step() → lr_scheduler.step()

# 6. 保存 checkpoint
if step % save_interval == 0 or step >= max_steps:
    save_checkpoint(...)
    dist.barrier()          # 所有GPU同步
```

**logits 偏移的处理 (L314)**：

```python
action_preds = output.logits[:, num_patches:-1].argmax(dim=2)
#                             ↑ 跳过图像patch位置  ↑ 去掉最后一个（shift对齐）
action_gt = batch["labels"][:, 1:]
#                            ↑ 同样shift一位
```

因为 `forward()` 里把图像 patch 插入序列，logits 的位置和原始 labels 不对齐，必须手动跳过 `num_patches` 个位置。

**epoch 的计算方式 (L369)**：

```python
epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)
```

不是真实 epoch，是用 step 数反推的估算值，因为 RLDS 是无限流没有真正的 epoch 边界。

# prismatic.py
定义 `PrismaticVLM` 类，是整个模型的**核心胶水层**，把视觉编码器、projector、LLM 三个组件组装在一起。
## 类结构总览

```
PrismaticVLM(VLM)
├── __init__()                    组装三组件，初始化projector
├── from_pretrained()             从.pt文件加载权重（推理用）
├── get_prompt_builder()          获取对话格式构造器
├── freeze_backbones(stage)       按训练阶段控制参数冻结
├── load_from_checkpoint()        align→finetune阶段权重迁移
├── get_fsdp_wrapping_policy()    FSDP分片策略
├── forward()                     核心前向传播（训练+推理共用）
├── prepare_inputs_for_generation() KV cache管理
├── generate_batch()              批量推理
└── generate()                    单样本推理
```

---

## 一、`__init__` (L39-83)

**参数**：接收已实例化的 `vision_backbone` 和 `llm_backbone`，自身只创建 `projector`。

```python
# 用 embed_dim 作为随机种子，保证不同视觉骨干对应不同初始化
torch.manual_seed(vision_backbone.embed_dim)

# 根据 arch_specifier 选择 projector 类型
"linear"         → LinearProjector(vision_dim, llm_dim)   # 单层线性
"gelu-mlp"       → MLPProjector(vision_dim, llm_dim)      # Linear→GELU→Linear
"fused-gelu-mlp" → FusedMLPProjector(vision_dim, llm_dim) # 融合加速版
```

`string2idx`：预存 True/False/Yes/No/A-Z 的 token id，推理时计算概率用，避免重复查 tokenizer。

---

## 二、`from_pretrained` (L85-123)

专为**推理**设计的类方法：

```python
vlm = cls(...)                          # 先建空模型
model_state_dict = torch.load(ckpt)["model"]  # 加载checkpoint
vlm.projector.load_state_dict(...)      # 必须有
vlm.llm_backbone.load_state_dict(...)   # 必须有
vlm.vision_backbone.load_state_dict()   # 可选（通常冻结未变）
vlm.requires_grad_(False)               # 默认冻结
vlm.eval()
```

训练时不走这个方法，走 `load.py` 里的 `load()`。

---

## 三、`freeze_backbones` (L129-241)

5 种训练 stage：

|stage|视觉|Projector|LLM主体|LLM最后层|
|---|---|---|---|---|
|`align`|冻|训|冻|冻|
|`vla-train`|冻|训|训|训|
|`vla-full-train`|训(FP32)|训|训|训|
|`vla-last-layer-train`|冻|冻|冻|训|
|`vla-sandwich-train`|训(FP32)|训|冻|训|

注意 `vla-full-train` 和 `vla-sandwich-train` 会强制把视觉骨干切到 FP32，因为训练时梯度精度要求更高。

`vision_backbone_requires_grad` 这个 tracker 在 `forward()` 里用来控制是否为视觉前向保留计算图，节省显存。

---

## 四、`forward` (L312-481) — 最核心

处理三种情况：

**情况1：推理 KV cache 模式 (L329)**

```python
if input_ids.shape[1] == 1 and past_key_values is not None:
    # 自回归第2+步，直接走LLM，跳过图像处理
    return self.llm_backbone(input_ids=input_ids, past_key_values=past_key_values)
```

**情况2：纯文本 batch (L353)**

```python
elif len(multimodal_indices) == 0:
    return self.llm_backbone(input_ids=input_ids, ...)
```

**情况3：正常多模态前向 (L367-481)**

```python
# Step 1: 图像 → patch特征
with torch.set_grad_enabled(self.vision_backbone_requires_grad):
    patch_features = self.vision_backbone(pixel_values[multimodal_indices])
# shape: [bsz, 257, vision_dim]  (256 patches + 1 CLS token)

# Step 2: 投影到LLM维度
projected_patch_embeddings = self.projector(patch_features)
# shape: [bsz, 257, llm_dim]

# Step 3: 文本token → embedding
input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
# shape: [bsz, seq_len, llm_dim]

# Step 4: 拼接序列
multimodal_embeddings = torch.cat([
    input_embeddings[:, :1, :],        # BOS token
    projected_patch_embeddings,         # 257个图像patch
    input_embeddings[:, 1:, :],         # 文本tokens（指令+动作）
], dim=1)
# shape: [bsz, 1+257+seq_len, llm_dim]

# Step 5: labels同步扩展，图像位置全设为 IGNORE_INDEX=-100
projected_patch_labels = torch.full((...), IGNORE_INDEX)
multimodal_labels = torch.cat([
    labels[:, :1],           # BOS对应的label（也是-100）
    projected_patch_labels,  # 257个-100
    labels[:, 1:],           # 原始labels（只有动作token非-100）
])

# Step 6: 处理混合batch（有图/无图样本）
# 无图样本用零padding补齐图像位置，attention_mask=False屏蔽
unimodal_embeddings_pad = torch.zeros(...)
fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])

# Step 7: 送入LLM
return self.llm_backbone(inputs_embeds=fused_embeddings, labels=fused_labels)
# 返回 CausalLMOutputWithPast，包含 .loss 和 .logits
```

---

## 五、推理相关方法 (L488-621)

**`prepare_inputs_for_generation` (L488)**：HuggingFace GenerationMixin 要求的接口，处理 KV cache：

- 有 cache：只传最后一个 token `input_ids[:, -1:]`
- 第一步：用 `inputs_embeds`，后续步用 `input_ids`

**`generate_batch` (L521)**：批量推理，支持两种模式：

- 普通生成：返回文本字符串列表
- `return_string_probabilities`：返回指定词（True/False 等）的归一化概率，用于 VQA 评测

**`generate` (L594)**：单样本推理，接受 PIL Image + prompt 字符串：

```python
input_ids = tokenizer(prompt_text).input_ids
pixel_values = image_transform(image)
generated_ids = super().generate(input_ids, pixel_values)  # GenerationMixin
return tokenizer.decode(generated_ids[0, input_ids.shape[1]:])
```

---
## 关键设计点

**不继承 `PreTrainedModel`**，只继承 `GenerationMixin`，避免 HuggingFace 的冗余逻辑，但 `forward()` 签名必须手动对齐 `LlamaForCausalLM`。

**loss 完全委托给 LLM**：传入 `labels` 后 HuggingFace 内部自动做 token shift + CrossEntropy，`prismatic.py` 不写任何 loss 计算代码。

**`multimodal_indices`**：允许一个 batch 里混合有图/无图样本，无图样本用零 padding 补齐，保证 tensor shape 一致，这是支持混合数据集训练的关键。

#  action_tokenizer.py
### 核心设计思想

LLaMA 只能处理离散 token，但机器人动作是连续值（如关节角度 0.73）。ActionTokenizer 的职责就是在两者之间做双向转换：

```
连续动作 [-1, 1]  ←→  离散 bin 索引  ←→  词表末尾 token ID
```

---

### `__init__`

```python
self.bins = np.linspace(min_action, max_action, self.n_bins)
# bins = [-1.0, -0.992, -0.984, ..., 0.992, 1.0]  共 256 个边界点

self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
# bin_centers = [-0.996, -0.988, ..., 0.996]  共 255 个中心点

self.action_token_begin_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))
# 例如 vocab_size=32000, n_bins=256 → begin_idx = 31743
```

关键设计：**复用词表末尾 256 个 token** 来表示动作，不新增 token，不改变模型结构。

---

### `__call__`（编码：连续 → token 字符串）

```python
def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
    action = np.clip(action, a_min=-1.0, a_max=1.0)
    discretized_action = np.digitize(action, self.bins)
    # np.digitize 返回 action 落在 bins 的哪个区间，结果范围 [1, 256]

    # 单步动作（shape=[7]）
    return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
    # vocab_size - [1..256] → [vocab_size-1 .. vocab_size-256]
    # 即词表最后 256 个 token ID
```

数值示例（7维动作）：

```
action = [0.5, -0.3, 0.0, 1.0, -1.0, 0.7, -0.8]
↓ clip（无变化）
↓ digitize → [192, 109, 128, 256, 1, 218, 26]
↓ vocab_size(32000) - above → [31808, 31891, 31872, 31744, 31999, 31782, 31974]
↓ tokenizer.decode → "<tok_31808><tok_31891>..."  （一个字符串）
```

这个字符串就是 `RLDSBatchTransform` 里 GPT 回复的内容，会被 tokenizer 再次 encode 进 `input_ids`。

---

### `decode_token_ids_to_actions`（解码：token ID → 连续值）

```python
def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    discretized_actions = self.tokenizer.vocab_size - action_token_ids
    # 逆操作：token ID → bin 索引，范围 [1, 256]

    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
    # -1 后范围变 [0, 255]，但 bin_centers 只有 255 个（索引 0~254）
    # clip 把越界的 255 压到 254（最后一个 bin center）

    return self.bin_centers[discretized_actions]
    # 用 bin 中心值作为连续动作的近似
```

为什么要 `-1` 再 `clip`？

- `np.digitize` 返回 `[1, n_bins]`，是 1-indexed
- `bin_centers` 是 0-indexed，共 `n_bins - 1 = 255` 个
- 当 digitize 返回 256（最大值，即动作恰好等于 1.0），`256 - 1 = 255` 会越界
- `clip(..., 0, 254)` 把 255 压到 254，映射到最后一个 bin center

---

### 训练时如何使用

[base_strategy.py:316](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/prismatic/training/strategies/base_strategy.py#L316)：

```python
mask = action_gt > action_tokenizer.action_token_begin_idx
# 只有动作 token（词表末尾）才参与 accuracy 计算
# 普通文字 token 被过滤掉

action_preds = output.logits[:, num_patches:-1].argmax(dim=2)
# logits 取动作位置，argmax 得到预测 token ID

continuous_actions_pred = torch.tensor(
    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
)
# 预测 token → 连续动作，用于计算 L1 loss（仅用于监控，不反传）
```

---

### 整体数据流总结

```
训练：
  连续动作 [0.5, -0.3, ...]
    → __call__() → token 字符串 "<tok_31808>..."
    → base_tokenizer.encode() → input_ids [31808, 31891, ...]
    → LLaMA forward(labels=input_ids) → cross-entropy loss（在 token 空间）

推理：
  LLaMA.generate() → 输出 token IDs [31808, 31891, ...]
    → decode_token_ids_to_actions() → 连续动作 [0.496, -0.302, ...]
    → 发送给机器人执行
```








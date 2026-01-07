
# Day 4：工程化 RAG

**把 RAG 从“一段脚本”升级为“一个可调用、可扩展、可兜底的工程能力”。**

对应可跑代码在 [src/RAG/day4/index.ts](./index.ts)。

---

## 一、从“能跑”到“能用”

Day 3 的 RAG 已经像流程了：切分 → 写入向量库 → 检索 → 拼上下文 → 问模型。

但一旦你想把它放进真实系统，会立刻遇到工程问题：

- 文档来源不可能永远写死在代码里
- 向量库初始化（切分/embedding/写入）不该每次都做
- 需要一个统一入口（比如 `answer(question)`），才能被 HTTP / Job / Agent 复用
- 检索不到时必须有“系统级兜底”，不能只靠 prompt 让模型自觉

该章节就是专门补这一层：**流程控制 + 可靠性兜底 + 可替换边界**。

---

## 二、RAG 的最小工程抽象

把一个工程化的 RAG 想成一个函数调用：

```mermaid
flowchart TB
	Q[answer(question)] --> I[初始化知识库（只做一次）]
	I --> R[检索相关文档]
	R --> G{是否检索到？}
	G -- 否 --> F[系统兜底：返回“我不知道”】【确定性】]
	G -- 是 --> C[构造上下文 Context]
	C --> L[调用大模型生成回答]
```

一句话：**RAG 的工程本质不是“模型多聪明”，而是“流程是否可控、可复用、可兜底”。**

---

## 三、把 RAG 封装成 `answer(question)`

Day 4 的核心产物就是这个函数签名：

```ts
export async function answer(question: string): Promise<string>
```

- 复用：任何地方都能调用（HTTP handler / 定时任务 / Agent）
- 可替换：底层向量库/检索器/模型都可以换，但入口不变
- 可测试：单测只测 `answer()` 行为即可（在真实项目里很关键）

---

## 四、文档来源抽象：RAG 核心逻辑不关心“资料从哪来”

在工程里，最重要的边界之一是“文档来源”。

在 [src/RAG/day4/index.ts](./index.ts) 中先用最简单的 seed 文档演示：

```ts
function loadDocuments(): Document[] {
	return [
		new Document({ pageContent: "...", metadata: { source: "seed", id: 1 } }),
		new Document({ pageContent: "...", metadata: { source: "seed", id: 2 } }),
	];
}
```

关键点是：**RAG 核心流程只要求拿到 `Document[]`。**

未来你想换成：

- 本地文件（md/pdf）
- 数据库
- CMS
- 外部接口

只要还能产出 `Document[]`，`answer()` 这条主流程就不用改。

---

## 五、初始化知识库只做一次（避免重复 embedding）

工程上非常关键的一点：**初始化不要每次都做。**

Day 4 用一个最简单但很实用的方式：

```ts
let initialized = false;

async function initKnowledgeBase() {
	if (initialized) return;

	const rawDocs = loadDocuments();
	const splitDocs = await splitter.splitDocuments(rawDocs);
	await vectorStore.addDocuments(splitDocs);

	initialized = true;
}
```

它解决的不是“正确性”，而是“成本/性能/稳定性”：

- 避免重复 embedding（省钱、也更快）
- 减少外部接口调用次数（更稳）
- 把“建库”与“问答”职责分清

---

## 六、检索失败兜底：这是系统可靠性的底线

新手很容易把“说不知道”这件事交给 prompt：

> “如果资料里没有答案，请回答我不知道。”

但工程上更可靠的是：**系统先兜底**。

Day 4 的兜底非常直接：

```ts
if (relevantDocs.length === 0) {
	return "我不知道（知识库中没有相关信息）";
}
```

为什么这很重要？

- 这是确定性的逻辑，不会受模型“听不听话”影响
- 这是可观测、可统计、可告警的行为（后续可做 metrics）
- 这是生产级 RAG 的底线能力

---

## 七、哪些是“可替换层”，哪些是“核心逻辑”？

在 [src/RAG/day4/index.ts](./index.ts) 里，你可以把代码分成两类：

**可替换层（将来一定会换）：**

- 文档来源：`loadDocuments()`
- 向量库实现：`MemoryVectorStore`（未来可能换 Milvus / PGVector）
- 模型/embedding：`ChatOpenAI` / `OpenAIEmbeddings` 的具体配置
- 切分策略：`RecursiveCharacterTextSplitter` 的参数

**核心逻辑（工程形状不应变）：**

- `answer(question)` 作为统一入口
- “先 init（只做一次）再检索再生成”的流程控制
- “检索失败系统兜底”的可靠性策略

---

## 八、Day 4 自检清单

1) 为什么向量库初始化要“只做一次”？

2) 为什么兜底逻辑不能只靠 prompt？

3) 文档来源换了，RAG 核心逻辑要不要改？

4) 如果未来把 `MemoryVectorStore` 换成 Milvus / PGVector，哪些代码需要动？

---

## 九、答案

> 1. 因为切分 + embedding + 写入向量库通常是成本最高的部分，重复做会增加耗时和外部请求成本，也更不稳定。
> 2. prompt 是“建议模型怎么做”，但模型不一定严格遵守；系统级兜底是确定性逻辑，更可靠。
> 3. 不用改主流程；只要仍然能产出 `Document[]`，`answer()` 的核心步骤可以保持不变。
> 4. 主要动向量库初始化/构造与写入/检索相关的那一层；`answer(question)` 的入口与兜底逻辑应尽量不变。


# Day 2：跑通一个“真正能回答问题”的最小 RAG

**RAG = 先找资料，再按资料回答。**

对应可跑代码在 [src/RAG/day2/index.ts](./index.ts)。

## 一、目标

用 3 段固定文本，做一个“能回答问题”的 RAG。

---
流程思路：

```mermaid
flowchart TB
  start((开始)) --> q[用户输入问题]
  q --> s[向量数据库检索]
  s --> docs[拿到相关文档]
  docs --> prompt[增强数据：拼 Prompt（资料+问题）]
  prompt --> llm[传给 LLM]
  llm --> out[输出结果]
  out --> end((结束))

  subgraph prep[准备阶段（建库）]
    d0[准备文档] --> e0[初始化 embeddings]
    e0 --> vs0[存入向量数据库]
  end

  vs0 -. 为检索提供数据 .-> s
```

---

## 二、最小 RAG 示例代码

### Embedding

在这句：

```ts
await vectorStore.addDocuments(documents);
```

- `addDocuments()` 内部会调用传进去的 `OpenAIEmbeddings`
- 发网络请求去做向量化

### 检索

在这两句：

```ts
const retriever = vectorStore.asRetriever({ k: 1 });
const relevantDocs = await retriever.invoke(question);
```

你可以把 Retriever 理解成：**专门负责找资料的那个人**。

`k` 是你能控制的旋钮：

- `k=1`：更省 token、更干净，但可能漏
- `k=3`：更全一点，但可能更吵

### Prompt

把“资料 + 问题”喂给模型：

```ts
const response = await chatModel.invoke([
  { role: "system", content: "你是一个基于 RAG 的助手..." },
  { role: "user", content: `已检索到的上下文如下:\n\n${context}\n\n请基于以上内容回答...` },
]);
```

注意：这句里最关键的是 `${context}`。

**没有 `${context}`，就是裸问；有了 `${context}`，才是 RAG。**

---

## 三、自检回顾

1) 如果把“检索”删掉（也就是不查向量库）会发生什么？

2) 模型“看起来知道资料内容”，是因为哪一段代码？

3) 如果检索返回了错误文档，模型能不能纠正？

---

## 四、答案
> 1. 就变成“裸问模型”。要么不喂资料，要么把所有资料一股脑塞进去（贵+吵）。
> 2. 不是它自己知道，把检索到的内容拼进了 `context`，并在 `chatModel.invoke([...])` 里发给了它。
> 3. 不能（一般也不应该指望它纠正）。模型会尽量根据你给的上下文回答，错上下文就容易错答案。

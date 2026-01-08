# Day 7：生产化复盘与进阶路线（评测 / 调参 / 故障模式）

对应可跑代码在 [src/RAG/day7/index.ts](./index.ts)。

---

## 一、Day 7 到底解决什么问题？

上线后你会遇到最真实的场景：

> 这次答得很好；下次又很差。你要怎么优化？

如果你不能回答这些问题：

- 这次候选召回有哪些？分数分布是什么？
- 阈值过滤掉了什么？保留了什么？
- 最终上下文有哪些 chunk？
- 为什么兜底？是“无检索”，还是“术语不覆盖”？

那么你只会“反复改 prompt”，但没有稳定收益。

该章节的核心是：把 RAG 的迭代变成一个工程闭环。

---

## 二、最小闭环：基准集 + 日志分布 + 调参

### 2.1 基准集（Dataset）

我们用一组小问题集（带预期）来做回归：

- 这些问题不是为了“难”，而是为了覆盖故障模式：
  - 精确 token（`user_id` / `user_123`）
  - 概念解释（RAG / 阈值 / 混合检索）
  - 无关问题（应该兜底）

你不需要一上来就做严肃的自动化评测；先把“可重复对比”做出来。

### 2.2 日志分布（Distribution）

真正能调参的是“分布”，不是某一次结果：

- 候选分数分布（`candidates`）
- 通过阈值的数量（`passedCount`）
- 术语覆盖情况（`terms` / `covered` / `coverageRatio`）
- 最终上下文的 docId#chunkIndex 列表

当你把这些都打印出来，你就能回答：

- 阈值该提高还是降低？
- top-k 是否过大导致噪声？
- 关键词检索是不是误命中？

---

## 三、Day 7 的可运行评测器做了什么？

见 [src/RAG/day7/index.ts](./index.ts)，默认只跑：

- Indexing（切分 + 写入向量库）
- Query（语义检索 + 阈值 + 关键词检索 + Hybrid 合并）
- 可回答性判定（关键术语覆盖）

默认不调用 LLM（省钱、可快速迭代）。你也可以用 `--with-llm` 开启生成回答，做人工验收。

---

## 四、如何运行（建议从 help 开始）

- 查看参数：

```bash
pnpm tsx ./src/RAG/day7 -- --help
```

- 跑内置评测集（只检索，不调用 LLM）：

```bash
pnpm tsx ./src/RAG/day7
```

- 指定阈值（观察 passed 与 fallback 的变化）：

```bash
pnpm tsx ./src/RAG/day7 -- --threshold 0.45
```

- 单题调试（看候选、分数、过滤、最终上下文）：

```bash
pnpm tsx ./src/RAG/day7 -- --case "user_id 字段名是什么？"
```

- 开启 LLM（成本更高，适合少量 case 人工抽查）：

```bash
pnpm tsx ./src/RAG/day7 -- --with-llm
```

> 说明：仓库目前 `package.json` 没有 `rag:day7` 脚本；如果想对齐 Day2–Day6，可加一行：`"rag:day7": "tsx ./src/RAG/day7"`。

---

## 五、常见故障模式

1) **弱相关命中**：
- 表现：命中了，但答案胡说/不完整
- 看：候选分数分布、阈值过滤后是否只剩“可疑 chunk”

2) **主题相关但不含答案**：
- 表现：上下文在讲同一主题，但没有明确结论/字段/步骤
- 看：最终上下文具体 chunk 文本；是否缺“关键细节词”（字段名、路径、规则）

3) **数据污染/噪声召回**：
- 表现：召回了旅行/做饭等噪声内容
- 看：top-k 是否过大、阈值是否过低、chunk 是否太长导致语义糊

4) **关键词误命中**：
- 表现：出现“字符串碰巧包含”的错误召回
- 看：关键词列表、命中来源（keyword/semantic）、关键词分数贡献

5) **术语覆盖误判**：
- 表现：上下文包含关键词，但仍不可回答（或相反）
- 看：`coverageRatio` 与命中 chunk 的内容质量；考虑升级可回答性判定

---

## 六、下一步演进清单

- 返回 sources：对每个回答返回 docId/chunkIndex/preview，便于 UI 展示与溯源
- 结构化输出：JSON Schema 输出（answer / confidence / citations / fallbackReason）
- 缓存：文档向量持久化、问题→检索结果缓存
- 向量库替换：MemoryVectorStore → 持久化向量库（PGVector/Milvus 等）
- 可回答性判定升级：从“术语覆盖”升级为“回答性分类器/规则 + 证据充足性”

---

## 七、自检清单

1) 把阈值从 0.35 调到 0.5，你的 passed 数量如何变化？fallback 是否更频繁？
2) 把 `HYBRID_TOP_K` 从 6 调到 10，噪声是否变多？token 成本是否上升？
3) 对 `user_id`、`user_123` 这类问题，关键词通道是否确实提升命中？
4) 你能不能用日志解释一次“答得不好”的原因？

---

## 八、答案

1) 阈值从 0.35 → 0.5 的变化规律
- 一般现象：`passed` 数量会下降（更严格），`fallback` 更可能发生（因为过滤后没文档，或上下文更短导致术语覆盖失败）。
- 你该看什么：单题用 `--case` 跑一次，对比两次输出里的 `semanticCandidates / semanticPassed / finalDocs` 与 `coverageRatio`。
- 工程含义：阈值越高，上下文“更干净但更少”；如果 `fallbackRate` 明显上升，说明阈值可能过高，或语料/切分导致相似度整体偏低。

2) `HYBRID_TOP_K` 从 6 → 10 的变化规律
- 一般现象：`finalDocs` 增加，上下文更长；噪声更容易混入；token 成本通常上升（即使 Day7 默认不调 LLM，上线后也会体现在 LLM 输入 token 上）。
- 你该看什么：`FINAL_DOCS` 列表里是否开始出现 `noise_*` 或“主题相关但不含答案”的 chunk；同时看 `coverageRatio` 是否真的提升。
- 调参方向：如果 `coverageRatio` 不变但噪声变多，说明 top-k 过大；优先回调 `HYBRID_TOP_K`，或提高阈值/改权重。

3) 关键词通道是否提升命中（以 `user_id` / `user_123` 为例）
- 观察方式：用 `--case` 跑问题，看 `KEYWORDS` 是否包含目标 token（如 `user_id`、`user_123`），以及 `KEYWORD_HITS` 里是否命中对应业务文档（例如 `api_login`、`user_id_rule`）。
- 判断标准：如果语义候选里没有把业务规则/字段 chunk 排到前面，但关键词能把它们顶上去，并且最终 `FINAL_DOCS` 包含这些 chunk，说明关键词通道有效。
- 反例：如果 `KEYWORD_HITS` 常命中“碰巧包含字符串”的无关 chunk，就需要收紧关键词提取（停用词/最短长度/正则规则）或降低关键词权重。

4) 用日志解释一次“答得不好”的方法（最小排障路径）
- 第一步：看 `RESULT` 的 `fallback` 与 `reason`（`no_docs` vs `term_not_covered`），先判断是“没召回”还是“上下文不对齐”。
- 第二步：看 `RETRIEVAL`（`semanticPassed` 是否为 0、`keywordHits` 是否为 0、`finalDocs` 是否过少/过多）。
- 第三步：看 `FINAL_DOCS`（是否混入 `noise_*`、是否缺关键业务 chunk）。
- 第四步（可选）：开启 `--with-llm` 抽查少量 case，把“错误答案”对应回“错误上下文”，再决定调哪一个旋钮（阈值 / top-k / chunk / 关键词）。

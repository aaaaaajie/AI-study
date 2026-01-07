import dotenv from "dotenv";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";

dotenv.config();

/**
 * ======================
 * 基础配置
 * ======================
 */
const apiKey = process.env.VOLCENGINE_API_KEY;
if (!apiKey) throw new Error("缺少 VOLCENGINE_API_KEY");

const VOLC_BASE_URL = process.env.VOLC_BASE_URL ?? "https://ark.cn-beijing.volces.com/api/v3";
const CHAT_MODEL = process.env.VOLC_CHAT_MODEL ?? "";
const EMBED_MODEL = process.env.VOLC_EMBED_MODEL ?? "";

if (!CHAT_MODEL || !EMBED_MODEL) {
  throw new Error("请在 .env 中设置 VOLC_CHAT_MODEL 和 VOLC_EMBED_MODEL（ep-xxx）");
}

/**
 * ======================
 * 初始化模型 & 向量库
 * ======================
 */
const embeddings = new OpenAIEmbeddings({
  apiKey,
  model: EMBED_MODEL,
  configuration: { baseURL: VOLC_BASE_URL },
});

const chatModel = new ChatOpenAI({
  apiKey,
  modelName: CHAT_MODEL,
  configuration: { baseURL: VOLC_BASE_URL },
});

const vectorStore = new MemoryVectorStore(embeddings);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 220,
  chunkOverlap: 60,
});

/**
 * ======================
 * 文档来源（模拟真实知识库）
 * 目标：多主题 + 噪声 + “语义相似但不含答案” + “精确字符串可命中”
 * ======================
 */
function loadRawDocuments(): Document[] {
  return [
    new Document({
      pageContent: `
RAG（Retrieval Augmented Generation，检索增强生成）是一种“先检索、再生成”的架构：
1) 把用户问题转成向量并在向量库里检索相关 chunk
2) 把检索到的 chunk 拼成上下文
3) 让大模型只能基于上下文回答

关键点：检索到 ≠ 可用。需要做质量控制（例如相似度阈值），否则模型会被噪声上下文带偏。
      `.trim(),
      metadata: { source: "kb", docId: "rag_intro", topic: "rag" },
    }),

    new Document({
      pageContent: `
向量检索擅长语义相似（同义表达、不同说法），但对“精确匹配”不敏感：
- 专有名词（接口名、字段名）
- ID（例如 user_123）
- 代码片段（例如 getUserById）
这类问题通常需要关键词检索或规则检索兜底。
      `.trim(),
      metadata: { source: "kb", docId: "vector_vs_keyword", topic: "retrieval" },
    }),

    new Document({
      pageContent: `
相似度阈值（quality gate）的目标是：宁可没有上下文，也不要把低质量上下文喂给模型。
Retriever 往往会“无论如何返回 k 条”，所以工程上通常要做：
- 取 top-k
- 再做阈值过滤（低于阈值的直接丢弃）
- 如果过滤后为空，触发兜底（例如“我不知道”）
      `.trim(),
      metadata: { source: "kb", docId: "similarity_threshold", topic: "retrieval" },
    }),

    new Document({
      pageContent: `
关键词检索（极简版思路）：
- 从问题里抽取关键词（中文短语、英文 token、user_\\d+ 这类模式）
- 在文本里做 contains / 计数
- 以命中数/权重作为分数

生产里常用 BM25/全文索引（Elastic/Lucene/PG FTS），这里用“可解释的小实现”演示它的价值。
      `.trim(),
      metadata: { source: "kb", docId: "keyword_retrieval", topic: "retrieval" },
    }),

    new Document({
      pageContent: `
Hybrid 检索（混合检索）常见做法：向量检索召回语义相关内容，关键词检索召回精确命中内容，然后合并、去重、排序。
一个直觉：如果一个 chunk 同时被语义检索和关键词检索命中，它往往更可靠（可以给一个小的加分）。
      `.trim(),
      metadata: { source: "kb", docId: "hybrid", topic: "retrieval" },
    }),

    // —— 用来体现“关键词检索的意义”：精确字符串 / ID / 规则 —— //
    new Document({
      pageContent: `
账号体系约定：
- 用户 ID 格式：user_xxx，其中 xxx 是数字（例如 user_1、user_42、user_123）。
- 任何不符合该格式的字符串都不是合法用户 ID（例如 user_abc 不合法）。
      `.trim(),
      metadata: { source: "kb", docId: "user_id_rule", topic: "business" },
    }),

    new Document({
      pageContent: `
接口字段说明（节选）：
POST /login
Request JSON:
- user_id: string，例如 "user_123"
- password: string

备注：字段名是 user_id（下划线），不是 userid / userId。
      `.trim(),
      metadata: { source: "kb", docId: "api_login", topic: "business" },
    }),

    // —— “语义相关但不含答案”的噪声：会被语义召回，但不一定能回答具体问题 —— //
    new Document({
      pageContent: `
我们在 Demo 中使用 MemoryVectorStore 做向量库：
- 优点：无需外部依赖，适合本地学习与快速验证
- 缺点：不持久化、不适合生产、数据量大时性能有限
      `.trim(),
      metadata: { source: "kb", docId: "memory_vectorstore", topic: "langchain" },
    }),

    new Document({
      pageContent: `
Embedding 是把文本映射到向量空间的过程：语义相近的文本向量更接近。
注意：Embedding 模型不负责“回答问题”，它只负责把文本变成向量用于检索。
      `.trim(),
      metadata: { source: "kb", docId: "embedding_basics", topic: "embedding" },
    }),

    // —— 明显无关的噪声（帮助你观察阈值过滤的效果）—— //
    new Document({
      pageContent: `
厨房小贴士：煎牛排之前让肉回温 20 分钟更容易受热均匀；盐最好在煎后撒，避免出水影响上色。
      `.trim(),
      metadata: { source: "kb", docId: "noise_cooking", topic: "noise" },
    }),

    new Document({
      pageContent: `
旅行备忘：冬季去北海道注意防滑鞋；在札幌雪天步行建议走地下通道连接的商业区。
      `.trim(),
      metadata: { source: "kb", docId: "noise_travel", topic: "noise" },
    }),
  ];
}

/**
 * ======================
 * 初始化知识库（只做一次）
 * ======================
 */
let initialized = false;
let knowledgeChunks: Document[] = [];

async function initKnowledgeBase() {
  if (initialized) return;

  const rawDocs = loadRawDocuments();
  const splitDocs = await splitter.splitDocuments(rawDocs);

  knowledgeChunks = splitDocs.map((doc, chunkIndex) => {
    return new Document({
      pageContent: doc.pageContent,
      metadata: {
        ...(doc.metadata ?? {}),
        chunkIndex,
        chunkLen: doc.pageContent.length,
      },
    });
  });

  await vectorStore.addDocuments(knowledgeChunks);
  initialized = true;
}

/**
 * ======================
 * Day 5 核心一：相似度阈值过滤
 * ======================
 */
const SIMILARITY_THRESHOLD = 0.35;
const SEMANTIC_TOP_K = 8;

type ScoredDoc = { doc: Document; score: number; source: string };

async function semanticRetrieveWithScores(question: string) {
  const results = await vectorStore.similaritySearchWithScore(question, SEMANTIC_TOP_K);
  const candidates: ScoredDoc[] = results
    .map(([doc, score]) => ({ doc, score, source: "semantic" }))
    .sort((a, b) => b.score - a.score);

  const passed = candidates.filter((x) => x.score >= SIMILARITY_THRESHOLD);
  return { candidates, passed };
}

/**
 * ======================
 * Day 5 核心二：关键词检索（可解释的小实现）
 * ======================
 */
const STOPWORDS = new Set([
  "什么",
  "如何",
  "为什么",
  "怎么",
  "怎样",
  "是否",
  "可以",
  "能否",
  "请问",
  "一下",
  "介绍",
  "解释",
  "概念",
  "含义",
  "的",
  "了",
  "吗",
  "呢",
  "啊",
]);

function extractKeywords(question: string): string[] {
  const tokens = new Set<string>();
  const lower = question.toLowerCase();

  // 1) 精确模式：user_123
  (lower.match(/user_\d+/g) ?? []).forEach((t) => tokens.add(t));

  // 2) 英文/数字 token：user_id、rag、bm25、getUserById
  (lower.match(/[a-z0-9_]{2,}/g) ?? []).forEach((t) => tokens.add(t));

  // 3) 中文片段：用连续中文串 + 2 字切片，增强可命中率（非常简化）
  const zhSeqs = question.match(/[\u4e00-\u9fff]{2,}/g) ?? [];
  for (const seq of zhSeqs) {
    if (!STOPWORDS.has(seq)) tokens.add(seq);
    if (seq.length >= 4) {
      for (let i = 0; i < seq.length - 1; i++) {
        const bi = seq.slice(i, i + 2);
        if (!STOPWORDS.has(bi)) tokens.add(bi);
      }
    }
  }

  return Array.from(tokens)
    .map((t) => t.trim())
    .filter((t) => t.length >= 2)
    .filter((t) => !STOPWORDS.has(t));
}

function keywordScore(text: string, keywords: string[]) {
  const lower = text.toLowerCase();
  let score = 0;

  for (const kw of keywords) {
    if (!kw) continue;
    if (lower.includes(kw.toLowerCase())) {
      // 长词更重要；user_123 这类会天然更长
      if (kw.length >= 8) score += 4;
      else if (kw.length >= 5) score += 3;
      else if (kw.length >= 3) score += 2;
      else score += 1;
    }
  }
  return score;
}

const KEYWORD_TOP_K = 8;

function keywordRetrieveWithScores(question: string): ScoredDoc[] {
  const keywords = extractKeywords(question);

  const scored = knowledgeChunks
    .map((doc) => ({
      doc,
      score: keywordScore(doc.pageContent, keywords),
      source: "keyword" as const,
    }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, KEYWORD_TOP_K);
  return scored;
}

/**
 * ======================
 * Day 5 核心三：Hybrid Retriever（更能体现意义的最小实现）
 * ======================
 */
const HYBRID_TOP_K = 6;

function safeMax(values: number[]) {
  return values.length === 0 ? 0 : Math.max(...values);
}

async function hybridRetrieve(question: string) {
  const { passed: semanticPassed } = await semanticRetrieveWithScores(question);
  const keywordHits = keywordRetrieveWithScores(question);

  const semanticMax = safeMax(semanticPassed.map((x) => x.score));
  const keywordMax = safeMax(keywordHits.map((x) => x.score));

  function norm(score: number, max: number) {
    return max > 0 ? score / max : 0;
  }

  const merged = new Map<
    string,
    {
      doc: Document;
      semanticScore?: number;
      keywordScore?: number;
      hybridScore: number;
    }
  >();

  function keyOf(doc: Document) {
    // chunk 粒度，避免同一 doc 不同 chunk 被强行合并
    const m = (doc.metadata ?? {}) as any;
    const id = m.docId ?? m.id ?? "unknown";
    const chunkIndex = m.chunkIndex ?? "?";
    return `${String(id)}#${String(chunkIndex)}`;
  }

  for (const s of semanticPassed) {
    const k = keyOf(s.doc);
    merged.set(k, { doc: s.doc, semanticScore: s.score, hybridScore: 0 });
  }

  for (const kdoc of keywordHits) {
    const k = keyOf(kdoc.doc);
    const existed = merged.get(k);
    if (existed) existed.keywordScore = kdoc.score;
    else merged.set(k, { doc: kdoc.doc, keywordScore: kdoc.score, hybridScore: 0 });
  }

  const mergedList = Array.from(merged.values()).map((x) => {
    const s = x.semanticScore ?? 0;
    const k = x.keywordScore ?? 0;

    // 归一化后加权：语义为主，关键词为辅；同时命中给一点点 boost
    const sn = norm(s, semanticMax);
    const kn = norm(k, keywordMax);
    const bothBoost = s > 0 && k > 0 ? 0.15 : 0;

    const hybridScore = 0.75 * sn + 0.25 * kn + bothBoost;
    return { ...x, hybridScore };
  });

  mergedList.sort((a, b) => b.hybridScore - a.hybridScore);
  const selected = mergedList.slice(0, HYBRID_TOP_K).map((x) => x.doc);

  return selected;
}

/**
 * ======================
 * Day 5：最终 RAG 函数
 * ======================
 */
async function answer(question: string): Promise<string> {
  await initKnowledgeBase();

  const docs = await hybridRetrieve(question);

  if (docs.length === 0) {
    return "我不知道（未检索到足够相关的上下文）";
  }

  const context = docs
    .map((d, i) => {
      const m = (d.metadata ?? {}) as any;
      const tag = `${String(m.docId ?? "unknown")}#${String(m.chunkIndex ?? "?")}`;
      return `【${i + 1} ${tag}】\n${d.pageContent}`;
    })
    .join("\n\n");

  const res = await chatModel.invoke([
    {
      role: "system",
      content: `
你是一个 RAG 助手。
只能根据上下文回答问题。
如果上下文不足以回答，请直接说“我不知道”。
      `.trim(),
    },
    {
      role: "user",
      content: `
上下文：
${context}

问题：
${question}
      `.trim(),
    },
  ]);

  return res.content as string;
}

/**
 * ======================
 * 程序入口
 * ======================
 */
async function main() {
  const question = process.argv.slice(2).join(" ").trim() || "什么是 RAG？";
  console.log("问题：", question);

  const result = await answer(question);

  console.log("\n=== 回答 ===\n");
  console.log(result);
}

main().catch((err) => {
  console.error("运行出错:", err);
  process.exit(1);
});
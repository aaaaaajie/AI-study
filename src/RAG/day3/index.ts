import "dotenv/config";

import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const apiKey = process.env.VOLCENGINE_API_KEY;
if (!apiKey) {
    console.error("请先设置环境变量 VOLCENGINE_API_KEY");
    process.exit(1);
}

// ===== 1. 火山引擎配置 =====
const VOLC_BASE_URL = process.env.VOLC_BASE_URL ?? "https://ark.cn-beijing.volces.com/api/v3";
const CHAT_MODEL = process.env.VOLC_CHAT_MODEL ?? "";
const EMBED_MODEL = process.env.VOLC_EMBED_MODEL ?? "";
const EMBED_TIMEOUT_MS = Number(process.env.VOLC_EMBED_TIMEOUT_MS ?? "10000");
const CHAT_TIMEOUT_MS = Number(process.env.VOLC_CHAT_TIMEOUT_MS ?? "15000");

if (!CHAT_MODEL || !EMBED_MODEL) {
    console.error("请在 .env 中设置 VOLC_CHAT_MODEL 和 VOLC_EMBED_MODEL（ep-xxx）");
    process.exit(1);
}

// ===== 2. Embeddings =====
const embeddings = new OpenAIEmbeddings({
    apiKey,
    model: EMBED_MODEL,
    configuration: {
        baseURL: VOLC_BASE_URL,
    },
    timeout: EMBED_TIMEOUT_MS,
});

// ===== 3. Vector Store =====
const vectorStore = new MemoryVectorStore(embeddings);

// ===== 4. 原始文档 =====
const rawDocuments = [
    {
        pageContent: `
火山引擎是字节跳动旗下的云服务平台，
提供大模型、向量计算、存储、算力等云服务能力。
Ark 是火山引擎提供的大模型服务平台。
`,
        metadata: { id: 1 },
    },
    {
        pageContent: `
RAG（Retrieval Augmented Generation，检索增强生成）
是一种先从知识库中检索相关文档，
再将这些文档作为上下文交给大模型生成回答的技术。

RAG 的关键价值在于：
1) 把“外部知识”从模型参数里解耦出来，知识更新更快。
2) 让回答有可追溯的来源（命中的文档片段）。
3) 在上下文窗口有限时，只把最相关的内容喂给模型。

一个典型的 RAG 流程包括：
- 文档加载（PDF/网页/数据库）
- 文档切分（chunking）
- 向量化（embedding）并写入向量库
- 检索（retrieval）召回 top-k chunks
- 组合上下文 + 问题，交给 LLM 生成回答

切分（chunking）在这里非常关键：
如果文档很长，不切分就会导致：
- embedding 把多个主题混在一起，检索不够精确
- prompt 上下文太长、噪声太多、成本更高
如果切分为更小的 chunk，则检索能更“聚焦”，回答更稳定。
`,
        metadata: { id: 2 },
    },
    {
        pageContent: `
MemoryVectorStore 是 LangChain 提供的内存向量数据库实现，
适合本地学习、Demo 和小规模实验，
不依赖外部向量数据库服务。
`,
        metadata: { id: 3 },
    },
];

// ===== 5. 文档切分 =====
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 40,
});

async function main() {
    const question = process.argv[2] ?? "什么是 RAG？";
    console.log("用户问题:", question);

    console.log("\n开始切分文档...");
    const splitDocs = await splitter.splitDocuments(
        rawDocuments.map((d) => new Document({ pageContent: d.pageContent, metadata: d.metadata })),
    );
    const chunks = splitDocs.map(
        (doc, chunkIndex) =>
            new Document({
                pageContent: doc.pageContent,
                metadata: {
                    ...(doc.metadata ?? {}),
                    chunkIndex,
                    chunkLen: doc.pageContent.length,
                },
            }),
    );
    console.log(`原始文档数: ${rawDocuments.length}`);
    console.log(`切分后文档块数: ${chunks.length}`);

    console.log("\n=== 切分后的 chunks（展示 chunk 的意义） ===\n");
    chunks.forEach((doc) => {
        const preview = doc.pageContent.replace(/\s+/g, " ").trim().slice(0, 80);
        console.log(
            `sourceId=${String((doc.metadata as any)?.id)} chunkIndex=${String(
                (doc.metadata as any)?.chunkIndex,
            )} len=${String((doc.metadata as any)?.chunkLen)} preview=${preview}`,
        );
    });

    console.log("\n写入向量库（调用 embedding 接口）...");
    await vectorStore.addDocuments(chunks);
    console.log("向量库写入完成");

    // ===== 6. Retriever =====
    const retriever = vectorStore.asRetriever({ k: 2 });
    const relevantDocs = await retriever.invoke(question);

    const context = relevantDocs
        .map((doc, i) => `【文档${i + 1}】\n${doc.pageContent}`)
        .join("\n\n");

    // ===== 7. Chat Model =====
    const chatModel = new ChatOpenAI({
        apiKey,
        modelName: CHAT_MODEL,
        configuration: {
            baseURL: VOLC_BASE_URL,
        },
        timeout: CHAT_TIMEOUT_MS,
    });

    // ===== 8. Day 3 标准 RAG Prompt =====
    const response = await chatModel.invoke([
        {
            role: "system",
            content: `
你是一个基于 RAG 的问答助手。
你必须严格根据给定的上下文回答问题。
如果上下文中没有明确答案，请直接回答「我不知道」。
不要进行任何推测或补充。
      `.trim(),
        },
        {
            role: "user",
            content: `
已检索到的上下文如下：

${context}

请基于以上内容回答用户问题：
${question}
      `.trim(),
        },
    ]);

    // ===== 9. 输出 =====
    console.log("\n=== RAG 回答 ===\n");
    console.log(response.content);

    console.log("\n=== 命中的文档片段 ===\n");
    relevantDocs.forEach((doc, i) => {
        const sourceId = String((doc.metadata)?.id);
        const chunkIndex = String((doc.metadata)?.chunkIndex);
        const chunkLen = String((doc.metadata)?.chunkLen ?? doc.pageContent.length);
        console.log(`文档${i + 1}: sourceId=${sourceId} chunkIndex=${chunkIndex} len=${chunkLen}`);
        console.log(doc.pageContent);
        console.log("----");
    });
}

main().catch((err) => {
    console.error("运行出错:", err);
    process.exit(1);
});
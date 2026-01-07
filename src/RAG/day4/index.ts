import "dotenv/config";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";

const apiKey = process.env.VOLCENGINE_API_KEY;
if (!apiKey) {
    console.error("请先设置环境变量 VOLCENGINE_API_KEY");
    process.exit(1);
}

// ===== 火山引擎配置 =====
const VOLC_BASE_URL = process.env.VOLC_BASE_URL ?? "https://ark.cn-beijing.volces.com/api/v3";
const CHAT_MODEL = process.env.VOLC_CHAT_MODEL ?? "";
const EMBED_MODEL = process.env.VOLC_EMBED_MODEL ?? "";

if (!CHAT_MODEL || !EMBED_MODEL) {
    console.error("请在 .env 中设置 VOLC_CHAT_MODEL 和 VOLC_EMBED_MODEL（ep-xxx）");
    process.exit(1);
}

// ===== 初始化一次 =====
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
    chunkSize: 200,
    chunkOverlap: 40,
});

// ===== 文档来源 =====
function loadDocuments(): Document[] {
    return [
        new Document({
            pageContent: "RAG 是一种通过检索外部知识库，再结合大模型生成回答的技术。",
            metadata: { source: "seed", id: 1 },
        }),
        new Document({
            pageContent: "MemoryVectorStore 是 LangChain 提供的内存向量数据库实现。",
            metadata: { source: "seed", id: 2 },
        }),
    ];
}

// ===== 初始化知识库 =====
let initialized = false;

async function initKnowledgeBase() {
    if (initialized) return;

    const rawDocs = loadDocuments();
    const splitDocs = await splitter.splitDocuments(rawDocs);
    await vectorStore.addDocuments(splitDocs);

    initialized = true;
}

// ===== 核心函数 =====
export async function answer(question: string): Promise<string> {
    await initKnowledgeBase();

    const retriever = vectorStore.asRetriever({ k: 2 });
    const relevantDocs = await retriever.invoke(question);

    // —— 兜底逻辑 ——
    if (relevantDocs.length === 0) {
        return "我不知道（知识库中没有相关信息）";
    }

    const context = relevantDocs
        .map((d) => d.pageContent)
        .join("\n\n");

    const res = await chatModel.invoke([
        {
            role: "system",
            content:
                "你是一个 RAG 助手，只能根据上下文回答，不能编造。",
        },
        {
            role: "user",
            content: `上下文：\n${context}\n\n问题：${question}`,
        },
    ]);

    return res.content as string;
}

async function main() {
    const question = process.argv[2] ?? "什么是 RAG？";
    const result = await answer(question);
    console.log(result);
}

main();
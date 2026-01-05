import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";

const apiKey = process.env.VOLCENGINE_API_KEY;
if (!apiKey) {
    console.error("请先设置环境变量 VOLCENGINE_API_KEY");
    console.error("示例: VOLCENGINE_API_KEY=xxx npm run dev -- \"今天天气如何？\"");
    process.exit(1);
}

const VOLC_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/";
const CHAT_MODEL = "ep-20260106015138-zk265"; // 聊天/生成模型
const EMBED_MODEL = "ep-20260106020301-wwzgb"; // 向量模型

const embeddings = new OpenAIEmbeddings({
    apiKey,
    configuration: { baseURL: VOLC_BASE_URL },
    model: EMBED_MODEL,
    timeout: 10_000,
});

const vectorStore = new MemoryVectorStore(embeddings);

const documents = [
    {
        pageContent:
            "火山引擎是字节跳动旗下的云服务平台，提供大模型、存储、计算等云服务。",
        metadata: { id: 1 },
    },
    {
        pageContent:
            "RAG（检索增强生成）通过先从知识库检索相关文档，再交给大模型生成回答。",
        metadata: { id: 2 },
    },
    {
        pageContent:
            "本示例使用 LangChain 的 MemoryVectorStore 作为内存向量数据库，不依赖外部向量服务。",
        metadata: { id: 3 },
    },
];

const chatModel = new ChatOpenAI({
    apiKey,
    configuration: {
        baseURL: VOLC_BASE_URL
    },
    modelName: CHAT_MODEL,
});

async function main() {
    const question = process.argv[2] ?? "什么是 RAG？";

    console.log("用户问题:", question);

    console.log("开始写入向量库（会调用火山 embedding 接口）...");
    await vectorStore.addDocuments(documents);
    console.log("向量库写入完成");

    const retriever = vectorStore.asRetriever({ k: 1 });
    const relevantDocs = await retriever.invoke(question);

    const context = relevantDocs
        .map((doc, i) => `【文档${i + 1}】` + doc.pageContent)
        .join("\n\n");

    const response = await chatModel.invoke([
        {
            role: "system",
            content:
                "你是一个基于 RAG 的助手，只能根据给定的上下文回答问题，不要编造与上下文无关的内容。",
        },
        {
            role: "user",
            content: `已检索到的上下文如下:\n\n${context}\n\n请基于以上内容回答用户问题: ${question}`,
        },
    ]);

    console.log("\n=== RAG 回答 ===\n");
    console.log(response.content);

    console.log("\n=== 命中的文档片段 ===\n");
    relevantDocs.forEach((doc, i) => {
        console.log(`文档${i + 1}:`, doc.pageContent);
    });
}

main().catch((err) => {
    console.error("运行出错:", err);
    process.exit(1);
});

import dotenv from "dotenv";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";

dotenv.config();

type ScoredDoc = { doc: Document; score: number; source: string };

type CliOptions = {
	help: boolean;
	withLlm: boolean;
	caseQuestion?: string;
	threshold?: number;
	semanticTopK?: number;
	keywordTopK?: number;
	hybridTopK?: number;
	chunkSize?: number;
	chunkOverlap?: number;
};

function printHelp() {
	console.log(`\nDay 7 · RAG 评测 / 调参 / 复盘工具\n\n用法:\n  pnpm tsx ./src/RAG/day7\n\n参数:\n  --help                 查看帮助\n  --case <question>       只跑单个问题（用于调试日志）\n  --with-llm              额外调用 LLM 生成答案（更贵）\n\n调参参数（覆盖默认值）:\n  --threshold <number>    相似度阈值，例如 0.35\n  --semantic-top-k <n>    语义检索 top-k，例如 8\n  --keyword-top-k <n>     关键词检索 top-k，例如 8\n  --hybrid-top-k <n>      混合检索最终 top-k，例如 6\n  --chunk-size <n>        切分 chunkSize，例如 220\n  --chunk-overlap <n>     切分 chunkOverlap，例如 60\n\n示例:\n  pnpm tsx ./src/RAG/day7 -- --threshold 0.45\n  pnpm tsx ./src/RAG/day7 -- --case "user_id 字段名是什么？"\n  pnpm tsx ./src/RAG/day7 -- --with-llm\n`);
}

function parseArgs(argv: string[]): CliOptions {
	const opts: CliOptions = { help: false, withLlm: false };

	function takeValue(i: number) {
		const v = argv[i + 1];
		if (!v || v.startsWith("--")) throw new Error(`参数 ${argv[i]} 需要一个值`);
		return v;
	}

	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		// 兼容 `pnpm ... -- --help` 这类透传，会把 `--` 作为一个独立参数传进来
		if (a === "--") continue;
		if (a === "--help" || a === "-h") opts.help = true;
		else if (a === "--with-llm") opts.withLlm = true;
		else if (a === "--case") opts.caseQuestion = takeValue(i++);
		else if (a === "--threshold") opts.threshold = Number(takeValue(i++));
		else if (a === "--semantic-top-k") opts.semanticTopK = Number(takeValue(i++));
		else if (a === "--keyword-top-k") opts.keywordTopK = Number(takeValue(i++));
		else if (a === "--hybrid-top-k") opts.hybridTopK = Number(takeValue(i++));
		else if (a === "--chunk-size") opts.chunkSize = Number(takeValue(i++));
		else if (a === "--chunk-overlap") opts.chunkOverlap = Number(takeValue(i++));
		else throw new Error(`未知参数: ${a}`);
	}

	return opts;
}

function safeMax(values: number[]) {
	return values.length === 0 ? 0 : Math.max(...values);
}

function padRight(s: string, len: number) {
	return s.length >= len ? s : s + " ".repeat(len - s.length);
}

function loadRawDocuments(): Document[] {
	// 与 Day5/Day6 保持一致：多主题 + 噪声 + 精确匹配案例
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

	(lower.match(/user_\d+/g) ?? []).forEach((t) => tokens.add(t));
	(lower.match(/[a-z0-9_]{2,}/g) ?? []).forEach((t) => tokens.add(t));

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
			if (kw.length >= 8) score += 4;
			else if (kw.length >= 5) score += 3;
			else if (kw.length >= 3) score += 2;
			else score += 1;
		}
	}
	return score;
}

function keyOf(doc: Document) {
	const m = (doc.metadata ?? {}) as any;
	const id = m.docId ?? m.id ?? "unknown";
	const chunkIndex = m.chunkIndex ?? "?";
	return `${String(id)}#${String(chunkIndex)}`;
}

async function main() {
	const opts = parseArgs(process.argv.slice(2));
	if (opts.help) {
		printHelp();
		return;
	}

	const apiKey = process.env.VOLCENGINE_API_KEY;
	if (!apiKey) throw new Error("缺少 VOLCENGINE_API_KEY");

	const VOLC_BASE_URL = process.env.VOLC_BASE_URL ?? "https://ark.cn-beijing.volces.com/api/v3";
	const CHAT_MODEL = process.env.VOLC_CHAT_MODEL ?? "";
	const EMBED_MODEL = process.env.VOLC_EMBED_MODEL ?? "";
	if (!CHAT_MODEL || !EMBED_MODEL) {
		throw new Error("请在 .env 中设置 VOLC_CHAT_MODEL 和 VOLC_EMBED_MODEL（ep-xxx）");
	}

	const SIMILARITY_THRESHOLD = Number(opts.threshold ?? process.env.SIMILARITY_THRESHOLD ?? 0.35);
	const SEMANTIC_TOP_K = Number(opts.semanticTopK ?? process.env.SEMANTIC_TOP_K ?? 8);
	const KEYWORD_TOP_K = Number(opts.keywordTopK ?? process.env.KEYWORD_TOP_K ?? 8);
	const HYBRID_TOP_K = Number(opts.hybridTopK ?? process.env.HYBRID_TOP_K ?? 6);
	const chunkSize = Number(opts.chunkSize ?? process.env.CHUNK_SIZE ?? 220);
	const chunkOverlap = Number(opts.chunkOverlap ?? process.env.CHUNK_OVERLAP ?? 60);

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
	const splitter = new RecursiveCharacterTextSplitter({ chunkSize, chunkOverlap });

	let initialized = false;
	let knowledgeChunks: Document[] = [];

	async function initKnowledgeBase() {
		if (initialized) return;
		const rawDocs = loadRawDocuments();
		const splitDocs = await splitter.splitDocuments(rawDocs);
		knowledgeChunks = splitDocs.map((doc, chunkIndex) =>
			new Document({
				pageContent: doc.pageContent,
				metadata: { ...(doc.metadata ?? {}), chunkIndex, chunkLen: doc.pageContent.length },
			}),
		);
		await vectorStore.addDocuments(knowledgeChunks);
		initialized = true;
	}

	async function semanticRetrieveWithScores(question: string) {
		const started = Date.now();
		const results = await vectorStore.similaritySearchWithScore(question, SEMANTIC_TOP_K);
		const candidates: ScoredDoc[] = results
			.map(([doc, score]) => ({ doc, score, source: "semantic" }))
			.sort((a, b) => b.score - a.score);
		const passed = candidates.filter((x) => x.score >= SIMILARITY_THRESHOLD);
		return { candidates, passed, ms: Date.now() - started };
	}

	function keywordRetrieveWithScores(question: string) {
		const started = Date.now();
		const keywords = extractKeywords(question);

		const hits: ScoredDoc[] = knowledgeChunks
			.map((doc) => ({ doc, score: keywordScore(doc.pageContent, keywords), source: "keyword" }))
			.filter((x) => x.score > 0)
			.sort((a, b) => b.score - a.score)
			.slice(0, KEYWORD_TOP_K);

		return { keywords, hits, ms: Date.now() - started };
	}

	async function hybridRetrieve(question: string) {
		const semantic = await semanticRetrieveWithScores(question);
		const keyword = keywordRetrieveWithScores(question);

		const semanticMax = safeMax(semantic.passed.map((x) => x.score));
		const keywordMax = safeMax(keyword.hits.map((x) => x.score));

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

		for (const s of semantic.passed) {
			const k = keyOf(s.doc);
			merged.set(k, { doc: s.doc, semanticScore: s.score, hybridScore: 0 });
		}

		for (const kdoc of keyword.hits) {
			const k = keyOf(kdoc.doc);
			const existed = merged.get(k);
			if (existed) existed.keywordScore = kdoc.score;
			else merged.set(k, { doc: kdoc.doc, keywordScore: kdoc.score, hybridScore: 0 });
		}

		const mergedList = Array.from(merged.values()).map((x) => {
			const s = x.semanticScore ?? 0;
			const k = x.keywordScore ?? 0;
			const sn = norm(s, semanticMax);
			const kn = norm(k, keywordMax);
			const bothBoost = s > 0 && k > 0 ? 0.15 : 0;
			const hybridScore = 0.75 * sn + 0.25 * kn + bothBoost;
			return { ...x, hybridScore };
		});

		mergedList.sort((a, b) => b.hybridScore - a.hybridScore);
		const selected = mergedList.slice(0, HYBRID_TOP_K).map((x) => x.doc);

		return { selected, semantic, keyword, mergedList };
	}

	function termCoverage(question: string, context: string) {
		const terms = extractKeywords(question).slice(0, 8);
		const lowerContext = context.toLowerCase();
		const covered = terms.filter((t) => lowerContext.includes(t.toLowerCase()));
		const ratio = terms.length === 0 ? 0 : covered.length / terms.length;
		return { terms, covered, ratio };
	}

	type EvalCase = { name: string; question: string; expectHit: boolean };
	const dataset: EvalCase[] = [
		{ name: "概念：RAG", question: "什么是 RAG？", expectHit: true },
		{ name: "质量闸门", question: "为什么需要相似度阈值？", expectHit: true },
		{ name: "混合检索", question: "什么是 Hybrid 检索？", expectHit: true },
		{ name: "精确字段", question: "login 接口的 user_id 字段名是什么？", expectHit: true },
		{ name: "精确 ID 规则", question: "user_abc 是合法的吗？user_123 呢？", expectHit: true },
		{ name: "无关问题应兜底", question: "北海道冬天怎么玩？", expectHit: false },
	];

	await initKnowledgeBase();

	console.log("\n=== Day7 配置 ===");
	console.log({ chunkSize, chunkOverlap, SIMILARITY_THRESHOLD, SEMANTIC_TOP_K, KEYWORD_TOP_K, HYBRID_TOP_K });

	const casesToRun = opts.caseQuestion
		? [{ name: "单题", question: opts.caseQuestion, expectHit: true }]
		: dataset;

	let hitCount = 0;
	let fallbackCount = 0;
	let coverageSum = 0;

	for (const c of casesToRun) {
		const started = Date.now();
		const retrieved = await hybridRetrieve(c.question);
		const docs = retrieved.selected;

		const context = docs
			.map((d, i) => {
				const m = (d.metadata ?? {}) as any;
				return `【${i + 1} ${String(m.docId ?? "unknown")}#${String(m.chunkIndex ?? "?")}】\n${d.pageContent}`;
			})
			.join("\n\n");

		const cov = termCoverage(c.question, context);
		const fallback = docs.length === 0 || cov.covered.length === 0;
		const hit = !fallback;
		const reason = !fallback ? "ok" : docs.length === 0 ? "no_docs" : "term_not_covered";

		const ms = Date.now() - started;
		coverageSum += cov.ratio;
		if (hit) hitCount++;
		else fallbackCount++;

		console.log("\n========================================");
		console.log(`[CASE] ${c.name}`);
		console.log("[Q]", c.question);
		console.log("[RESULT]", { hit, fallback, reason, ms });
		console.log("[TERM]", { terms: cov.terms, covered: cov.covered, coverageRatio: Number(cov.ratio.toFixed(2)) });

		console.log("[RETRIEVAL]", {
			semanticCandidates: retrieved.semantic.candidates.length,
			semanticPassed: retrieved.semantic.passed.length,
			keywordHits: retrieved.keyword.hits.length,
			finalDocs: docs.length,
		});

		const finalTags = docs.map((d) => keyOf(d));
		console.log("[FINAL_DOCS]", finalTags);

		if (opts.caseQuestion) {
			console.log("\n[SEMANTIC_CANDIDATES]");
			retrieved.semantic.candidates.forEach((x) => {
				const tag = keyOf(x.doc);
				console.log(`- ${padRight(tag, 24)} score=${x.score.toFixed(4)}`);
			});

			console.log("\n[KEYWORDS]", retrieved.keyword.keywords);
			console.log("[KEYWORD_HITS]");
			retrieved.keyword.hits.forEach((x) => {
				const tag = keyOf(x.doc);
				console.log(`- ${padRight(tag, 24)} score=${String(x.score)}`);
			});

			console.log("\n[MERGED_TOP]");
			retrieved.mergedList.slice(0, 12).forEach((x) => {
				const tag = keyOf(x.doc);
				console.log(
					`- ${padRight(tag, 24)} hybrid=${x.hybridScore.toFixed(3)} semantic=${String(
						x.semanticScore?.toFixed?.(4) ?? "-",
					)} keyword=${String(x.keywordScore ?? "-")}`,
				);
			});
		}

		if (opts.withLlm) {
			if (fallback) {
				console.log("\n[LLM] 跳过（已兜底）");
			} else {
				const res = await chatModel.invoke([
					{
						role: "system",
						content: "你是一个 RAG 助手，只能根据上下文回答问题；如果上下文不足以回答，请说‘我不知道’。",
					},
					{
						role: "user",
						content: `上下文：\n${context}\n\n问题：\n${c.question}`,
					},
				]);
				console.log("\n[ANSWER]\n");
				console.log(res.content);
			}
		}

		if (opts.caseQuestion) break;
		if (c.expectHit !== hit) {
			console.log("[WARN] 与预期不一致：", { expectHit: c.expectHit, gotHit: hit });
		}
	}

	if (!opts.caseQuestion) {
		const total = casesToRun.length;
		console.log("\n=== 汇总 ===");
		console.log({
			total,
			hitRate: Number((hitCount / total).toFixed(2)),
			fallbackRate: Number((fallbackCount / total).toFixed(2)),
			avgCoverageRatio: Number((coverageSum / total).toFixed(2)),
		});
		console.log("\n提示：调参建议先改 --threshold，再看 passed / fallback 的变化。\n");
	}
}

main().catch((err) => {
	console.error("运行出错:", err);
	process.exit(1);
});

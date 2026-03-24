import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// RAG PIPELINE ENGINE
// All logic runs client-side using Claude API for embeddings + generation
// ============================================================

// --- Simple TF-IDF Vector Store (no external deps needed) ---
class VectorStore {
  constructor() {
    this.documents = [];
    this.tfidf = [];
    this.vocabulary = {};
    this.idf = {};
  }

  tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2);
  }

  buildVocabulary(docs) {
    const allWords = new Set();
    docs.forEach((doc) => {
      this.tokenize(doc.content).forEach((w) => allWords.add(w));
    });
    let i = 0;
    allWords.forEach((w) => (this.vocabulary[w] = i++));
  }

  computeIDF(docs) {
    const N = docs.length;
    const df = {};
    docs.forEach((doc) => {
      const words = new Set(this.tokenize(doc.content));
      words.forEach((w) => (df[w] = (df[w] || 0) + 1));
    });
    Object.keys(this.vocabulary).forEach((w) => {
      this.idf[w] = Math.log((N + 1) / ((df[w] || 0) + 1)) + 1;
    });
  }

  vectorize(text) {
    const tokens = this.tokenize(text);
    const tf = {};
    tokens.forEach((t) => (tf[t] = (tf[t] || 0) + 1));
    const vec = {};
    Object.keys(tf).forEach((w) => {
      if (this.vocabulary[w] !== undefined) {
        vec[w] = (tf[w] / tokens.length) * (this.idf[w] || 1);
      }
    });
    return vec;
  }

  cosineSimilarity(vecA, vecB) {
    let dot = 0,
      magA = 0,
      magB = 0;
    const allKeys = new Set([...Object.keys(vecA), ...Object.keys(vecB)]);
    allKeys.forEach((k) => {
      const a = vecA[k] || 0;
      const b = vecB[k] || 0;
      dot += a * b;
      magA += a * a;
      magB += b * b;
    });
    if (magA === 0 || magB === 0) return 0;
    return dot / (Math.sqrt(magA) * Math.sqrt(magB));
  }

  ingest(docs) {
    this.documents = docs;
    this.buildVocabulary(docs);
    this.computeIDF(docs);
    this.tfidf = docs.map((doc) => ({
      ...doc,
      vec: this.vectorize(doc.content),
    }));
  }

  search(query, topK = 5) {
    const qVec = this.vectorize(query);
    const scored = this.tfidf.map((doc) => ({
      ...doc,
      score: this.cosineSimilarity(qVec, doc.vec),
    }));
    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .filter((d) => d.score > 0.01);
  }
}

// --- Web Crawler / Content Ingestion ---
async function crawlUrl(url) {
  // In a real deployment, this would call a backend crawler.
  // Here we use a CORS proxy approach and parse the HTML client-side.
  try {
    const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`;
    const res = await fetch(proxyUrl);
    const data = await res.json();
    if (!data.contents) throw new Error("No content");
    const parser = new DOMParser();
    const doc = parser.parseFromString(data.contents, "text/html");

    // Remove noise
    ["script", "style", "nav", "footer", "header", "noscript", "iframe"].forEach((tag) => {
      doc.querySelectorAll(tag).forEach((el) => el.remove());
    });

    const title = doc.querySelector("title")?.textContent?.trim() || url;
    const body = doc.querySelector("main, article, .content, #content, body");
    const text = body?.innerText || doc.body?.textContent || "";

    // Chunk the content
    const chunks = chunkText(text, 600, 100);
    return chunks.map((chunk, i) => ({
      id: `${url}#chunk${i}`,
      url,
      title,
      content: chunk,
      chunkIndex: i,
    }));
  } catch (e) {
    console.warn("Crawl failed for", url, e.message);
    return [];
  }
}

function chunkText(text, chunkSize = 600, overlap = 100) {
  const words = text.replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    if (chunk.trim().length > 50) chunks.push(chunk);
    i += chunkSize - overlap;
  }
  return chunks;
}

// --- Claude API Integration ---
async function callClaude(messages, systemPrompt, onStream) {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { 
      "Content-Type": "application/json" 
      "x-api-key": import.meta.env.VITE_ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      system: systemPrompt,
      messages,
      stream: true,
    }),
  });

  if (!response.ok) throw new Error(`API error: ${response.status}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    const lines = chunk.split("\n").filter((l) => l.startsWith("data: "));
    for (const line of lines) {
      try {
        const json = JSON.parse(line.slice(6));
        if (json.type === "content_block_delta" && json.delta?.text) {
          fullText += json.delta.text;
          onStream?.(fullText);
        }
      } catch {}
    }
  }
  return fullText;
}

// ============================================================
// DEFAULT DEMO KNOWLEDGE BASE
// ============================================================
const DEMO_KNOWLEDGE = [
  {
    id: "demo1",
    url: "internal://about",
    title: "About Us",
    content:
      "We are a technology company founded in 2018 specializing in AI-powered business solutions. Our mission is to democratize artificial intelligence for businesses of all sizes. We have offices in San Francisco, New York, and London. Our team consists of over 200 engineers, data scientists, and product specialists.",
    chunkIndex: 0,
  },
  {
    id: "demo2",
    url: "internal://products",
    title: "Products & Services",
    content:
      "Our flagship product is SmartAssist Pro, an AI platform that automates customer service workflows. It includes natural language processing, sentiment analysis, and multi-channel support across email, chat, and voice. Pricing starts at $299/month for the Starter plan, $799/month for the Professional plan, and custom Enterprise pricing.",
    chunkIndex: 0,
  },
  {
    id: "demo3",
    url: "internal://products-features",
    title: "Features",
    content:
      "Key features include: Real-time AI responses with under 200ms latency, Integration with Salesforce, HubSpot, Zendesk, and 50+ other tools, GDPR and SOC 2 Type II compliance, 99.9% uptime SLA, Custom AI model fine-tuning, Multi-language support for 40+ languages, Advanced analytics dashboard, White-label options for agencies.",
    chunkIndex: 0,
  },
  {
    id: "demo4",
    url: "internal://support",
    title: "Support & Contact",
    content:
      "Support is available 24/7 for Enterprise customers and 9am-6pm PST for other plans. You can reach us via email at support@company.com, live chat on our website, or phone at +1-800-555-0199. Our average response time is under 2 hours. We also have an extensive help center with 500+ articles.",
    chunkIndex: 0,
  },
  {
    id: "demo5",
    url: "internal://pricing",
    title: "Pricing Details",
    content:
      "Starter Plan: $299/month — up to 1,000 conversations/month, 2 team seats, basic analytics, email support. Professional Plan: $799/month — up to 10,000 conversations/month, 10 team seats, advanced analytics, priority support, API access. Enterprise: Custom pricing — unlimited conversations, unlimited seats, dedicated account manager, custom integrations, on-premise deployment option.",
    chunkIndex: 0,
  },
  {
    id: "demo6",
    url: "internal://security",
    title: "Security & Compliance",
    content:
      "We take security seriously. All data is encrypted at rest using AES-256 and in transit using TLS 1.3. We are SOC 2 Type II certified, GDPR compliant, HIPAA compliant for healthcare customers, and ISO 27001 certified. Data is stored in AWS data centers in the US and EU. We perform regular third-party security audits.",
    chunkIndex: 0,
  },
  {
    id: "demo7",
    url: "internal://integrations",
    title: "Integrations",
    content:
      "SmartAssist Pro integrates with: CRM systems (Salesforce, HubSpot, Pipedrive), Help desks (Zendesk, Freshdesk, Intercom), E-commerce (Shopify, WooCommerce, Magento), Communication (Slack, Microsoft Teams, WhatsApp), and custom integrations via our REST API and webhooks. We also offer a Zapier integration connecting to 5,000+ apps.",
    chunkIndex: 0,
  },
  {
    id: "demo8",
    url: "internal://careers",
    title: "Careers",
    content:
      "We are currently hiring for Software Engineers (Backend, Frontend, ML), Product Managers, Sales Engineers, and Customer Success Managers. We offer competitive salaries, equity, remote-first culture, health insurance, 401k matching, $2,000 annual learning budget, and 4 weeks paid vacation. Apply at our careers page.",
    chunkIndex: 0,
  },
];

// ============================================================
// MAIN APP
// ============================================================
export default function RAGChatbot() {
  const [phase, setPhase] = useState("setup"); // setup | ingesting | ready | chatting
  const [urls, setUrls] = useState("");
  const [urlInput, setUrlInput] = useState("");
  const [store] = useState(() => new VectorStore());
  const [ingestProgress, setIngestProgress] = useState([]);
  const [docCount, setDocCount] = useState(0);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [activeTab, setActiveTab] = useState("chat");
  const [retrievedChunks, setRetrievedChunks] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [knowledgeBase, setKnowledgeBase] = useState([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamText]);

  // Load demo knowledge base immediately
  const loadDemo = useCallback(() => {
    store.ingest(DEMO_KNOWLEDGE);
    setDocCount(DEMO_KNOWLEDGE.length);
    setKnowledgeBase(DEMO_KNOWLEDGE);
    setPhase("ready");
    setMessages([
      {
        role: "assistant",
        content:
          "Hello! I'm your website assistant, powered by a RAG (Retrieval-Augmented Generation) pipeline. I've loaded a demo knowledge base with company information.\n\nYou can ask me about:\n• **Products & Pricing** — SmartAssist Pro plans\n• **Features & Integrations** — Capabilities and connected tools\n• **Security & Compliance** — Data protection details\n• **Support & Contact** — How to get help\n• **Careers** — Open positions\n\nOr add your own URLs in the **Sources** tab to chat with your real website content!",
        timestamp: new Date(),
      },
    ]);
  }, [store]);

  useEffect(() => {
    loadDemo();
  }, [loadDemo]);

  const handleIngestUrls = async () => {
    const urlList = urlInput
      .split("\n")
      .map((u) => u.trim())
      .filter((u) => u.startsWith("http"));
    if (urlList.length === 0) return;

    setPhase("ingesting");
    setIngestProgress([]);
    const allDocs = [...DEMO_KNOWLEDGE];

    for (const url of urlList) {
      setIngestProgress((p) => [...p, { url, status: "crawling" }]);
      const docs = await crawlUrl(url);
      if (docs.length > 0) {
        allDocs.push(...docs);
        setIngestProgress((p) =>
          p.map((item) =>
            item.url === url
              ? { ...item, status: "done", chunks: docs.length }
              : item
          )
        );
      } else {
        setIngestProgress((p) =>
          p.map((item) =>
            item.url === url ? { ...item, status: "failed" } : item
          )
        );
      }
    }

    store.ingest(allDocs);
    setDocCount(allDocs.length);
    setKnowledgeBase(allDocs);
    setPhase("ready");
    setUrlInput("");
  };

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;
    const userMsg = input.trim();
    setInput("");
    setRetrievedChunks([]);

    const newMessages = [...messages, { role: "user", content: userMsg, timestamp: new Date() }];
    setMessages(newMessages);
    setIsTyping(true);
    setStreamText("");

    // --- RAG: Retrieve ---
    const retrieved = store.search(userMsg, 5);
    setRetrievedChunks(retrieved);

    const context =
      retrieved.length > 0
        ? retrieved
            .map((d, i) => `[Source ${i + 1}: ${d.title}]\n${d.content}`)
            .join("\n\n---\n\n")
        : "No relevant content found in the knowledge base.";

    const systemPrompt = `You are a helpful website assistant. Your job is to answer questions based ONLY on the provided website content below.

RULES:
- Answer based solely on the provided context. Do not use external knowledge about other companies, tools, or websites.
- If the answer is not in the context, say "I don't have information about that on our website" and offer to help with something else.
- Never mention, link to, or recommend competitor products or external websites.
- Be concise, friendly, and helpful. Format responses with markdown when helpful.
- If the user asks something off-topic (not related to the website content), politely redirect them to topics you can help with.
- Always speak as if you represent the company whose website content you have.

WEBSITE CONTENT:
${context}`;

    const apiMessages = newMessages
      .slice(-10)
      .map((m) => ({ role: m.role, content: m.content }));

    try {
      const fullResponse = await callClaude(
        apiMessages,
        systemPrompt,
        (partial) => setStreamText(partial)
      );
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: fullResponse, timestamp: new Date() },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `I encountered an error: ${e.message}. Please check your API configuration.`,
          timestamp: new Date(),
          error: true,
        },
      ]);
    }
    setIsTyping(false);
    setStreamText("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatTime = (d) =>
    d?.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const renderMarkdown = (text) => {
    if (!text) return "";
    return text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`(.*?)`/g, "<code>$1</code>")
      .replace(/^### (.*$)/gm, "<h3>$1</h3>")
      .replace(/^## (.*$)/gm, "<h2>$1</h2>")
      .replace(/^• (.*$)/gm, "<li>$1</li>")
      .replace(/^- (.*$)/gm, "<li>$1</li>")
      .replace(/\n\n/g, "</p><p>")
      .replace(/\n/g, "<br/>");
  };

  return (
    <div style={styles.root}>
      {/* Background */}
      <div style={styles.bg}>
        <div style={styles.bgOrb1} />
        <div style={styles.bgOrb2} />
        <div style={styles.bgGrid} />
      </div>

      <div style={styles.layout}>
        {/* ── LEFT PANEL ── */}
        <aside style={styles.sidebar}>
          <div style={styles.sidebarHeader}>
            <div style={styles.logo}>
              <div style={styles.logoIcon}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#fff" strokeWidth="2" strokeLinejoin="round" />
                  <path d="M2 17l10 5 10-5" stroke="#fff" strokeWidth="2" strokeLinejoin="round" />
                  <path d="M2 12l10 5 10-5" stroke="#fff" strokeWidth="2" strokeLinejoin="round" />
                </svg>
              </div>
              <div>
                <div style={styles.logoName}>RAG Chat</div>
                <div style={styles.logoSub}>Powered by Claude</div>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div style={styles.tabs}>
            {["chat", "sources", "debug"].map((tab) => (
              <button
                key={tab}
                style={{ ...styles.tab, ...(activeTab === tab ? styles.tabActive : {}) }}
                onClick={() => setActiveTab(tab)}
              >
                {tab === "chat" && "💬"}
                {tab === "sources" && "📂"}
                {tab === "debug" && "🔍"}
                <span style={{ marginLeft: 6, textTransform: "capitalize" }}>{tab}</span>
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div style={styles.sidebarContent}>
            {activeTab === "chat" && (
              <div>
                <div style={styles.statCard}>
                  <div style={styles.statNum}>{docCount}</div>
                  <div style={styles.statLabel}>Knowledge chunks</div>
                </div>
                <div style={styles.statCard}>
                  <div style={styles.statNum}>{messages.filter((m) => m.role === "user").length}</div>
                  <div style={styles.statLabel}>Questions asked</div>
                </div>
                <div style={{ marginTop: 16 }}>
                  <div style={styles.sectionTitle}>Suggested questions</div>
                  {[
                    "What are your pricing plans?",
                    "What integrations do you support?",
                    "Is your platform GDPR compliant?",
                    "How can I contact support?",
                  ].map((q) => (
                    <button
                      key={q}
                      style={styles.suggestion}
                      onClick={() => {
                        setInput(q);
                        inputRef.current?.focus();
                      }}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {activeTab === "sources" && (
              <div>
                <div style={styles.sectionTitle}>Add Website URLs</div>
                <textarea
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder={"https://yoursite.com/about\nhttps://yoursite.com/pricing\nhttps://yoursite.com/faq"}
                  style={styles.urlInput}
                  rows={5}
                />
                <button
                  style={styles.ingestBtn}
                  onClick={handleIngestUrls}
                  disabled={phase === "ingesting" || !urlInput.trim()}
                >
                  {phase === "ingesting" ? "Crawling..." : "🕸️ Crawl & Ingest"}
                </button>

                {ingestProgress.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    {ingestProgress.map((p) => (
                      <div key={p.url} style={styles.progressItem}>
                        <span style={{ fontSize: 12, color: "#94a3b8", wordBreak: "break-all" }}>
                          {p.url.slice(0, 35)}…
                        </span>
                        <span
                          style={{
                            fontSize: 11,
                            color:
                              p.status === "done"
                                ? "#4ade80"
                                : p.status === "failed"
                                ? "#f87171"
                                : "#fbbf24",
                          }}
                        >
                          {p.status === "done" ? `✓ ${p.chunks} chunks` : p.status === "failed" ? "✗ failed" : "⟳ crawling"}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                <div style={{ marginTop: 20 }}>
                  <div style={styles.sectionTitle}>Loaded Sources ({knowledgeBase.length})</div>
                  {[...new Set(knowledgeBase.map((d) => d.title))].map((title) => (
                    <div key={title} style={styles.sourceItem}>
                      <div style={styles.sourceIcon}>📄</div>
                      <span style={{ fontSize: 12, color: "#cbd5e1" }}>{title}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === "debug" && (
              <div>
                <div style={styles.sectionTitle}>Last Retrieved Chunks</div>
                {retrievedChunks.length === 0 ? (
                  <div style={styles.emptyDebug}>Ask a question to see retrieved context here</div>
                ) : (
                  retrievedChunks.map((chunk, i) => (
                    <div key={chunk.id} style={styles.chunkCard}>
                      <div style={styles.chunkHeader}>
                        <span style={styles.chunkBadge}>#{i + 1}</span>
                        <span style={styles.chunkScore}>
                          score: {(chunk.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={styles.chunkTitle}>{chunk.title}</div>
                      <div style={styles.chunkText}>{chunk.content.slice(0, 180)}…</div>
                    </div>
                  ))
                )}

                <div style={{ marginTop: 20 }}>
                  <div style={styles.sectionTitle}>Pipeline Architecture</div>
                  {[
                    ["1. Crawl", "Fetch & parse HTML from URLs"],
                    ["2. Chunk", "Split into 600-word overlapping chunks"],
                    ["3. Index", "TF-IDF vector store built in-memory"],
                    ["4. Retrieve", "Cosine similarity top-5 search"],
                    ["5. Augment", "Inject context into system prompt"],
                    ["6. Generate", "Claude streams grounded response"],
                  ].map(([step, desc]) => (
                    <div key={step} style={styles.pipelineStep}>
                      <span style={styles.pipelineLabel}>{step}</span>
                      <span style={styles.pipelineDesc}>{desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* ── MAIN CHAT ── */}
        <main style={styles.chatMain}>
          {/* Header */}
          <div style={styles.chatHeader}>
            <div style={styles.chatHeaderLeft}>
              <div style={styles.avatarSmall}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="#a78bfa" strokeWidth="2" />
                  <path d="M8 12h8M12 8v8" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </div>
              <div>
                <div style={styles.chatTitle}>Website Assistant</div>
                <div style={styles.chatSubtitle}>
                  <span style={styles.onlineDot} />
                  RAG-powered · {docCount} chunks indexed
                </div>
              </div>
            </div>
            <button
              style={styles.clearBtn}
              onClick={() => {
                setMessages([]);
                setRetrievedChunks([]);
              }}
            >
              Clear chat
            </button>
          </div>

          {/* Messages */}
          <div style={styles.messages}>
            {messages.map((msg, i) => (
              <div
                key={i}
                style={{
                  ...styles.messageRow,
                  justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                {msg.role === "assistant" && (
                  <div style={styles.botAvatar}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                      <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#a78bfa" strokeWidth="2" strokeLinejoin="round" />
                      <path d="M2 17l10 5 10-5M2 12l10 5 10-5" stroke="#a78bfa" strokeWidth="2" strokeLinejoin="round" />
                    </svg>
                  </div>
                )}
                <div
                  style={{
                    ...styles.bubble,
                    ...(msg.role === "user" ? styles.userBubble : styles.botBubble),
                  }}
                >
                  {msg.role === "assistant" ? (
                    <div
                      style={styles.markdownContent}
                      dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                    />
                  ) : (
                    <div style={styles.userContent}>{msg.content}</div>
                  )}
                  <div style={styles.timestamp}>{formatTime(msg.timestamp)}</div>
                </div>
              </div>
            ))}

            {/* Streaming response */}
            {isTyping && (
              <div style={{ ...styles.messageRow, justifyContent: "flex-start" }}>
                <div style={styles.botAvatar}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#a78bfa" strokeWidth="2" strokeLinejoin="round" />
                    <path d="M2 17l10 5 10-5M2 12l10 5 10-5" stroke="#a78bfa" strokeWidth="2" strokeLinejoin="round" />
                  </svg>
                </div>
                <div style={{ ...styles.bubble, ...styles.botBubble }}>
                  {streamText ? (
                    <div
                      style={styles.markdownContent}
                      dangerouslySetInnerHTML={{ __html: renderMarkdown(streamText) }}
                    />
                  ) : (
                    <div style={styles.typingDots}>
                      <span style={{ ...styles.dot, animationDelay: "0ms" }} />
                      <span style={{ ...styles.dot, animationDelay: "160ms" }} />
                      <span style={{ ...styles.dot, animationDelay: "320ms" }} />
                    </div>
                  )}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div style={styles.inputArea}>
            {retrievedChunks.length > 0 && (
              <div style={styles.contextBar}>
                <span style={styles.contextIcon}>🔍</span>
                <span style={styles.contextText}>
                  Retrieved {retrievedChunks.length} relevant chunks from:{" "}
                  {retrievedChunks.map((c) => c.title).join(", ")}
                </span>
              </div>
            )}
            <div style={styles.inputRow}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything about our website content…"
                style={styles.input}
                rows={1}
                disabled={isTyping}
              />
              <button
                style={{
                  ...styles.sendBtn,
                  opacity: !input.trim() || isTyping ? 0.4 : 1,
                }}
                onClick={handleSend}
                disabled={!input.trim() || isTyping}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <path d="M22 2L11 13" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M22 2L15 22l-4-9-9-4 20-7z" stroke="white" strokeWidth="2" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
            <div style={styles.inputHint}>
              Press Enter to send · Shift+Enter for new line · Answers grounded in your website content only
            </div>
          </div>
        </main>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Space Grotesk', sans-serif; }
        @keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .4; transform: scale(.8); } }
        @keyframes orb1 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(40px,-30px) scale(1.1); } }
        @keyframes orb2 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-30px,40px) scale(.9); } }
        .dot { animation: pulse 1.2s ease-in-out infinite; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: rgba(148,163,184,.3); border-radius: 2px; }
        p { margin: .4em 0; } li { margin-left: 1.2em; list-style: disc; margin-bottom: .25em; } code { background: rgba(167,139,250,.15); color: #a78bfa; padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: .85em; } h2,h3 { margin: .6em 0 .3em; color: #e2e8f0; } strong { color: #f1f5f9; }
      `}</style>
    </div>
  );
}

// ============================================================
// STYLES
// ============================================================
const styles = {
  root: {
    width: "100vw",
    height: "100vh",
    background: "#0a0f1e",
    overflow: "hidden",
    position: "relative",
    fontFamily: "'Space Grotesk', sans-serif",
    color: "#e2e8f0",
    display: "flex",
    flexDirection: "column",
  },
  bg: {
    position: "absolute",
    inset: 0,
    overflow: "hidden",
    pointerEvents: "none",
  },
  bgOrb1: {
    position: "absolute",
    width: 600,
    height: 600,
    borderRadius: "50%",
    background: "radial-gradient(circle, rgba(124,58,237,.15) 0%, transparent 70%)",
    top: -200,
    left: -100,
    animation: "orb1 12s ease-in-out infinite",
  },
  bgOrb2: {
    position: "absolute",
    width: 500,
    height: 500,
    borderRadius: "50%",
    background: "radial-gradient(circle, rgba(59,130,246,.12) 0%, transparent 70%)",
    bottom: -150,
    right: -100,
    animation: "orb2 15s ease-in-out infinite",
  },
  bgGrid: {
    position: "absolute",
    inset: 0,
    backgroundImage:
      "linear-gradient(rgba(148,163,184,.03) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,.03) 1px, transparent 1px)",
    backgroundSize: "40px 40px",
  },
  layout: {
    position: "relative",
    zIndex: 1,
    display: "flex",
    height: "100vh",
    gap: 0,
  },
  // Sidebar
  sidebar: {
    width: 280,
    minWidth: 280,
    background: "rgba(15,23,42,.8)",
    backdropFilter: "blur(20px)",
    borderRight: "1px solid rgba(148,163,184,.08)",
    display: "flex",
    flexDirection: "column",
    height: "100vh",
  },
  sidebarHeader: {
    padding: "20px 16px 12px",
    borderBottom: "1px solid rgba(148,163,184,.08)",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: 10,
  },
  logoIcon: {
    width: 36,
    height: 36,
    borderRadius: 10,
    background: "linear-gradient(135deg, #7c3aed, #3b82f6)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  logoName: {
    fontSize: 15,
    fontWeight: 700,
    color: "#f1f5f9",
    letterSpacing: "-.02em",
  },
  logoSub: {
    fontSize: 11,
    color: "#64748b",
    fontWeight: 400,
  },
  tabs: {
    display: "flex",
    padding: "8px 8px 0",
    gap: 2,
  },
  tab: {
    flex: 1,
    padding: "7px 4px",
    background: "transparent",
    border: "none",
    borderRadius: 8,
    color: "#64748b",
    fontSize: 12,
    fontWeight: 500,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "all .15s",
    fontFamily: "'Space Grotesk', sans-serif",
  },
  tabActive: {
    background: "rgba(124,58,237,.15)",
    color: "#a78bfa",
  },
  sidebarContent: {
    flex: 1,
    overflowY: "auto",
    padding: "12px 12px",
  },
  statCard: {
    background: "rgba(30,41,59,.6)",
    border: "1px solid rgba(148,163,184,.08)",
    borderRadius: 10,
    padding: "12px 14px",
    marginBottom: 8,
  },
  statNum: {
    fontSize: 26,
    fontWeight: 700,
    color: "#a78bfa",
    lineHeight: 1,
  },
  statLabel: {
    fontSize: 11,
    color: "#64748b",
    marginTop: 3,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: 600,
    color: "#475569",
    textTransform: "uppercase",
    letterSpacing: ".06em",
    marginBottom: 8,
    marginTop: 4,
  },
  suggestion: {
    display: "block",
    width: "100%",
    textAlign: "left",
    background: "rgba(30,41,59,.5)",
    border: "1px solid rgba(148,163,184,.07)",
    borderRadius: 8,
    padding: "8px 10px",
    color: "#94a3b8",
    fontSize: 12,
    cursor: "pointer",
    marginBottom: 5,
    transition: "all .15s",
    fontFamily: "'Space Grotesk', sans-serif",
  },
  urlInput: {
    width: "100%",
    background: "rgba(15,23,42,.8)",
    border: "1px solid rgba(148,163,184,.12)",
    borderRadius: 8,
    color: "#e2e8f0",
    fontSize: 12,
    padding: "8px 10px",
    fontFamily: "'JetBrains Mono', monospace",
    resize: "vertical",
    outline: "none",
  },
  ingestBtn: {
    width: "100%",
    marginTop: 8,
    padding: "9px",
    background: "linear-gradient(135deg, #7c3aed, #3b82f6)",
    border: "none",
    borderRadius: 8,
    color: "#fff",
    fontWeight: 600,
    fontSize: 13,
    cursor: "pointer",
    fontFamily: "'Space Grotesk', sans-serif",
    transition: "opacity .15s",
  },
  progressItem: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "5px 0",
    borderBottom: "1px solid rgba(148,163,184,.05)",
  },
  sourceItem: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "5px 4px",
    borderRadius: 6,
  },
  sourceIcon: { fontSize: 12 },
  // Debug
  emptyDebug: {
    color: "#475569",
    fontSize: 12,
    fontStyle: "italic",
    padding: "12px 0",
  },
  chunkCard: {
    background: "rgba(30,41,59,.5)",
    border: "1px solid rgba(148,163,184,.08)",
    borderRadius: 8,
    padding: "10px",
    marginBottom: 8,
  },
  chunkHeader: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: 4,
  },
  chunkBadge: {
    fontSize: 10,
    fontWeight: 700,
    background: "rgba(124,58,237,.2)",
    color: "#a78bfa",
    padding: "2px 6px",
    borderRadius: 4,
  },
  chunkScore: {
    fontSize: 10,
    color: "#4ade80",
    fontFamily: "'JetBrains Mono', monospace",
  },
  chunkTitle: { fontSize: 11, fontWeight: 600, color: "#94a3b8", marginBottom: 4 },
  chunkText: { fontSize: 11, color: "#475569", lineHeight: 1.5 },
  pipelineStep: {
    display: "flex",
    flexDirection: "column",
    padding: "6px 0",
    borderBottom: "1px solid rgba(148,163,184,.05)",
  },
  pipelineLabel: { fontSize: 11, fontWeight: 700, color: "#a78bfa" },
  pipelineDesc: { fontSize: 11, color: "#64748b", marginTop: 1 },
  // Main chat
  chatMain: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    minWidth: 0,
  },
  chatHeader: {
    padding: "16px 24px",
    borderBottom: "1px solid rgba(148,163,184,.08)",
    background: "rgba(10,15,30,.6)",
    backdropFilter: "blur(12px)",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  chatHeaderLeft: { display: "flex", alignItems: "center", gap: 12 },
  avatarSmall: {
    width: 36,
    height: 36,
    borderRadius: 10,
    background: "rgba(124,58,237,.15)",
    border: "1px solid rgba(167,139,250,.2)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  chatTitle: { fontSize: 15, fontWeight: 700, color: "#f1f5f9" },
  chatSubtitle: {
    fontSize: 12,
    color: "#64748b",
    display: "flex",
    alignItems: "center",
    gap: 5,
    marginTop: 1,
  },
  onlineDot: {
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: "#4ade80",
    display: "inline-block",
  },
  clearBtn: {
    padding: "6px 12px",
    background: "rgba(30,41,59,.8)",
    border: "1px solid rgba(148,163,184,.12)",
    borderRadius: 8,
    color: "#64748b",
    fontSize: 12,
    cursor: "pointer",
    fontFamily: "'Space Grotesk', sans-serif",
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    padding: "20px 24px",
    display: "flex",
    flexDirection: "column",
    gap: 16,
  },
  messageRow: {
    display: "flex",
    alignItems: "flex-end",
    gap: 10,
  },
  botAvatar: {
    width: 30,
    height: 30,
    minWidth: 30,
    borderRadius: 8,
    background: "rgba(124,58,237,.12)",
    border: "1px solid rgba(167,139,250,.15)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  bubble: {
    maxWidth: "72%",
    borderRadius: 16,
    padding: "12px 16px",
    lineHeight: 1.6,
    fontSize: 14,
  },
  userBubble: {
    background: "linear-gradient(135deg, #7c3aed, #4f46e5)",
    borderBottomRightRadius: 4,
    color: "#f1f5f9",
  },
  botBubble: {
    background: "rgba(30,41,59,.8)",
    border: "1px solid rgba(148,163,184,.1)",
    borderBottomLeftRadius: 4,
    color: "#cbd5e1",
  },
  markdownContent: { color: "#cbd5e1", fontSize: 14, lineHeight: 1.7 },
  userContent: { color: "#f1f5f9", fontSize: 14 },
  timestamp: { fontSize: 10, color: "rgba(148,163,184,.4)", marginTop: 6 },
  typingDots: { display: "flex", gap: 5, padding: "4px 0", alignItems: "center" },
  dot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: "#a78bfa",
    display: "inline-block",
  },
  // Input
  inputArea: {
    padding: "12px 24px 16px",
    background: "rgba(10,15,30,.8)",
    backdropFilter: "blur(12px)",
    borderTop: "1px solid rgba(148,163,184,.08)",
  },
  contextBar: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 10px",
    background: "rgba(124,58,237,.08)",
    border: "1px solid rgba(167,139,250,.12)",
    borderRadius: 8,
    marginBottom: 8,
  },
  contextIcon: { fontSize: 12 },
  contextText: { fontSize: 11, color: "#a78bfa", flex: 1 },
  inputRow: { display: "flex", gap: 10, alignItems: "flex-end" },
  input: {
    flex: 1,
    background: "rgba(30,41,59,.8)",
    border: "1px solid rgba(148,163,184,.15)",
    borderRadius: 12,
    color: "#e2e8f0",
    fontSize: 14,
    padding: "12px 16px",
    fontFamily: "'Space Grotesk', sans-serif",
    outline: "none",
    resize: "none",
    lineHeight: 1.5,
    minHeight: 48,
    maxHeight: 120,
  },
  sendBtn: {
    width: 48,
    height: 48,
    borderRadius: 12,
    background: "linear-gradient(135deg, #7c3aed, #3b82f6)",
    border: "none",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "opacity .15s",
    flexShrink: 0,
  },
  inputHint: {
    fontSize: 11,
    color: "#334155",
    marginTop: 8,
    textAlign: "center",
  },
};

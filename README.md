# 🤖 RAG Chatbot — Powered by Claude

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions grounded exclusively in your website content. Built as a self-contained React component using the Anthropic Claude API with a TF-IDF vector store, real-time streaming responses, and a built-in web crawler.

---

## ✨ Features

- **Full RAG Pipeline** — Crawl → Chunk → Index → Retrieve → Augment → Generate, all in one component
- **Web Crawler** — Fetches any public URL via CORS proxy, strips nav/footer noise, and extracts clean body text
- **TF-IDF Vector Store** — In-memory index with IDF weighting and cosine similarity search (no external vector DB required)
- **Semantic Retrieval** — Top-5 chunk retrieval per query, injected into Claude's system prompt as grounded context
- **Claude Streaming** — Real-time token-by-token streaming responses via Anthropic's SSE API
- **Grounded-Only Answers** — Strict system prompt prevents hallucination and blocks competitor mentions
- **3-Tab Sidebar** — Chat stats, Sources (URL ingestion), and Debug (retrieved chunks + pipeline diagram)
- **Demo Knowledge Base** — Pre-loaded with 8 company content chunks so it works instantly out of the box

---
## Live demo
Here is an interavtive UI (https://ragchatbot-lake.vercel.app)
## 🚀 Quick Start

### Option 1 — Run in Claude.ai (Instant)

Paste the `.jsx` file content into a Claude.ai conversation and ask:

> "Render this as a live artifact"

No setup required. The chatbot loads immediately with demo data.

### Option 2 — Local Development with Vite

```bash
npm create vite@latest my-chatbot -- --template react
cd my-chatbot
npm install
```

Replace `src/App.jsx` with the contents of `rag-chatbot.jsx`, then:

```bash
npm run dev
# Opens at http://localhost:5173
```

### Option 3 — Create React App

```bash
npx create-react-app my-chatbot
cd my-chatbot
```

Replace `src/App.js` with the file contents, then:

```bash
npm start
# Opens at http://localhost:3000
```

### Option 4 — Online Sandbox

Paste the code directly into [stackblitz.com](https://stackblitz.com) or [codesandbox.io](https://codesandbox.io) — both support React out of the box.

---

## 🧠 How the RAG Pipeline Works

```
User Query
    │
    ▼
┌─────────────┐
│  1. Crawl   │  Fetch page HTML via CORS proxy → strip nav/footer/scripts → extract body text
└──────┬──────┘
       │
    ▼
┌─────────────┐
│  2. Chunk   │  Split text into 600-word windows with 100-word overlap
└──────┬──────┘
       │
    ▼
┌─────────────┐
│  3. Index   │  Build TF-IDF sparse vectors in-memory (VectorStore class)
└──────┬──────┘
       │
    ▼
┌─────────────┐
│  4. Retrieve│  Cosine similarity search → return top-5 matching chunks
└──────┬──────┘
       │
    ▼
┌─────────────┐
│  5. Augment │  Inject retrieved chunks into Claude's system prompt as context
└──────┬──────┘
       │
    ▼
┌─────────────┐
│  6. Generate│  Claude streams a grounded, context-only response
└─────────────┘
```

### Chunking Strategy

| Parameter    | Value | Purpose                                      |
|--------------|-------|----------------------------------------------|
| `chunkSize`  | 600 words | Balances context richness vs. retrieval precision |
| `overlap`    | 100 words | Prevents information loss at chunk boundaries |
| `minLength`  | 50 chars  | Filters out noise and nearly-empty chunks     |
| `topK`       | 5 chunks  | Retrieves the 5 most semantically relevant chunks |

---

## 🗂️ Component Architecture

```
RAGChatbot (default export)
│
├── VectorStore (class)
│   ├── tokenize()           Lowercase + strip punctuation + split
│   ├── buildVocabulary()    Build word index across all docs
│   ├── computeIDF()         Inverse document frequency weights
│   ├── vectorize()          TF-IDF sparse vector for any text
│   ├── cosineSimilarity()   Compare two sparse vectors
│   ├── ingest()             Index a document set
│   └── search()             Return top-K scored results
│
├── crawlUrl()               Fetch + parse + chunk a URL (async)
├── chunkText()              Split text into overlapping windows
├── callClaude()             SSE streaming call to Anthropic API
│
└── UI Components (inline)
    ├── Sidebar              Tabs: Chat | Sources | Debug
    │   ├── Chat Tab         Stats + suggested questions
    │   ├── Sources Tab      URL textarea + crawl progress + source list
    │   └── Debug Tab        Retrieved chunks + similarity scores + pipeline
    └── Chat Main
        ├── Header           Title + chunk count + clear button
        ├── Message List     User + assistant bubbles with markdown
        ├── Streaming View   Live token output + typing dots
        └── Input Bar        Textarea + send button + context bar
```

---

## 🎛️ State Management

All state is managed locally with React hooks:

| State Variable     | Type       | Description                                     |
|--------------------|------------|-------------------------------------------------|
| `phase`            | string     | `setup` / `ingesting` / `ready`                |
| `store`            | VectorStore| The TF-IDF index instance (stable ref)         |
| `knowledgeBase`    | Document[] | All ingested document chunks                   |
| `docCount`         | number     | Total number of indexed chunks                 |
| `messages`         | Message[]  | Full conversation history                      |
| `input`            | string     | Current textarea value                         |
| `isTyping`         | boolean    | Whether Claude is streaming a response         |
| `streamText`       | string     | Partial streamed text during generation        |
| `retrievedChunks`  | Chunk[]    | Last retrieval result (shown in Debug tab)     |
| `ingestProgress`   | Progress[] | Per-URL crawl status                           |
| `urlInput`         | string     | URL textarea content in Sources tab            |

---

## 📂 Adding Website URLs

1. Click the **📂 Sources** tab in the left sidebar
2. Paste one or more URLs into the text area (one per line):

```
https://yoursite.com/about
https://yoursite.com/pricing
https://yoursite.com/faq
https://yoursite.com/contact
```

3. Click **🕸️ Crawl & Ingest**
4. Each URL will show a status indicator:
   - `⟳ crawling` — fetch in progress
   - `✓ N chunks` — successfully indexed N text chunks
   - `✗ failed` — URL blocked by CORS or unreachable

> **Note:** Most commercial websites block cross-origin browser requests. Wikipedia and open documentation sites work well for testing. For production use with your own site, replace `crawlUrl()` with a call to a backend crawler endpoint.

---

## 🔍 Debug Panel

The **🔍 Debug** tab shows exactly what happens for every query:

- **Retrieved chunks** — the top-5 passages pulled from the index, ranked by cosine similarity score (0–100%)
- **Source titles** — which page each chunk came from
- **Pipeline diagram** — the 6 steps from Crawl to Generate, explained

This is useful for tuning retrieval quality and understanding why the chatbot gives certain answers.

---

## 🛡️ Content Grounding Rules

The system prompt enforces strict content grounding:

```
RULES:
- Answer based solely on the provided context.
- If the answer is not in the context, say:
  "I don't have information about that on our website."
- Never mention, link to, or recommend competitor products or external websites.
- Be concise, friendly, and use markdown formatting.
```

The chatbot will **never**:
- Make up information not present in indexed pages
- Reference external websites or tools
- Recommend competitors

---

## 🏗️ Production Upgrade Path

This component is designed as a starting point. For production deployment:

### Replace the CORS Proxy Crawler

```js
// Current (client-side, CORS-limited):
const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`

// Production (server-side, no CORS limits):
const res = await fetch('/api/crawl', {
  method: 'POST',
  body: JSON.stringify({ url })
})
```

### Replace the TF-IDF Store with a Real Vector DB

```js
// Current: in-memory TF-IDF (no embeddings)
store.ingest(docs)
const results = store.search(query, 5)

// Production: Pinecone with semantic embeddings
import { Pinecone } from '@pinecone-database/pinecone'
const index = pinecone.index('rag-chatbot')
const embedding = await getEmbedding(query)  // OpenAI / Voyage / Cohere
const results = await index.query({ vector: embedding, topK: 5 })
```

### Move API Key to Backend

```js
// Current: direct browser call (exposes API key)
fetch('https://api.anthropic.com/v1/messages', { ... })

// Production: proxy through your backend
fetch('/api/chat', { method: 'POST', body: JSON.stringify({ messages, context }) })
```

---

## 🗃️ Demo Knowledge Base

The chatbot ships with 8 pre-loaded demo content chunks covering:

| Topic               | Content Summary                                      |
|---------------------|------------------------------------------------------|
| About Us            | Company background, founding year, team size, offices |
| Products & Services | SmartAssist Pro, NLP features, pricing overview      |
| Features            | Integrations, compliance, SLA, language support      |
| Support & Contact   | Hours, email, phone, help center                     |
| Pricing Details     | Starter / Professional / Enterprise plan breakdown   |
| Security & Compliance | AES-256, TLS 1.3, SOC 2, GDPR, HIPAA, ISO 27001   |
| Integrations        | Salesforce, HubSpot, Zendesk, Shopify, Slack, Zapier|
| Careers             | Open roles, benefits, culture                        |

Replace this array with your own content or clear it after ingesting real URLs.

---

## 💡 Suggested Questions to Try

Once loaded, ask things like:

- *"What is artificial intelligence?"* (after ingesting a Wikipedia page)
- *"What are your pricing plans?"*
- *"Is your platform GDPR compliant?"*
- *"What integrations do you support?"*
- *"How can I contact support?"*
- *"What roles are you currently hiring for?"*

---

## 🧩 Key Dependencies

| Dependency         | Version  | Purpose                              |
|--------------------|----------|--------------------------------------|
| `react`            | ^18.2    | UI framework                         |
| `react-dom`        | ^18.2    | DOM rendering                        |
| Anthropic API      | —        | Claude Sonnet streaming responses    |
| allorigins.win     | —        | CORS proxy for client-side crawling  |
| Space Grotesk      | Google Fonts | UI typography                   |
| JetBrains Mono     | Google Fonts | Code/monospace elements         |

No other npm packages are required. The TF-IDF engine, chunker, and crawler are all implemented from scratch in the component.

---

## 📄 License

MIT — free to use, modify, and deploy.

---

## 🙏 Built With

- [Anthropic Claude](https://www.anthropic.com) — `claude-sonnet-4-20250514` for streaming generation
- [React](https://react.dev) — UI framework
- [allorigins.win](https://allorigins.win) — Open CORS proxy for client-side crawling

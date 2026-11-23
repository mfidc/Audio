# Agent-building guidance from AI labs and agent-product builders other than Anthropic

Verification legend used on every finding (read this first):
- **[PAGE]** = I fetched the page itself this session and the claim/quote comes from it. Only github.com fetches succeeded.
- **[SNIPPET]** = from web-search result summaries of the primary page (primary URL cited, wording seen only as search excerpts; quotes are short and likely accurate but not checked against the full page).
- **[SECONDARY]** = from a third-party summary of the source; treat as paraphrase.
- **[RECALL]** = from my prior knowledge of a well-known primary source, not re-verified this session. Use for orientation only; do not quote verbatim in the final report without checking.

Environment limits: direct fetches failed (DNS/allowlist) for openai.com, cdn.openai.com, cognition.ai, manus.im, rlancemartin.github.io. Per instructions I did not retry these or try further blog hosts. Those hosts would have to be allowed in the session network settings to upgrade [SNIPPET] items to [PAGE].

Date context: notes written 2026-11-23. Items from before mid-2025 are flagged as older where it matters.

## OpenAI: practical guide, Codex / AGENTS.md, harness engineering, Agents SDK

### Takeaway
OpenAI's guidance moved from 2025 "how to assemble an agent" advice (model + tools + instructions, start with one agent, layer guardrails) to a 2026 "harness engineering" stance: the engineer's job is designing the environment, repo-local knowledge, and feedback loops, with a short AGENTS.md acting as a map rather than a manual. The Agents SDK docs are neutral on LLM-driven vs code-driven orchestration and give a clear rule for agents-as-tools vs handoffs.

### Cited Findings

**Source: OpenAI, "A practical guide to building agents" (PDF business guide, published 2025; I recall April 2025 but the date was not confirmed by search) — https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf**
- Agent = three components: model, tools, instructions. [SNIPPET] — [OpenAI PDF](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- Model selection: "Not every task requires the smartest model"; small/fast models for retrieval or intent classification, stronger models for hard decisions. [SNIPPET] — [OpenAI PDF](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- Scale-out rule: "as the number of required tools increases, consider splitting tasks across multiple agents"; third-party summaries describe an incremental path (single agent in a loop first, multi-agent later). [SNIPPET for the quote; SECONDARY for "incremental path"] — [OpenAI PDF](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf); [summary](https://agentic-ai.readthedocs.io/en/latest/DesignPatterns/openai-patterns/)
- Two multi-agent patterns: manager (central agent calls specialist agents as tools) and decentralized handoffs (agents pass control peer to peer). [SECONDARY] — [summary](https://agentic-ai.readthedocs.io/en/latest/DesignPatterns/openai-patterns/)
- Guardrails are layered: no single guardrail is sufficient; combine input checks, output checks and tool-level safeguards, plus human intervention for high-risk actions. [SECONDARY] — [ibl.ai summary](https://ibl.ai/blog/openai-a-practical-guide-to-building-agents)
- The exact phrase "maximize a single agent's capabilities first" could NOT be confirmed by search; do not quote it. — [OpenAI PDF](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- Flag: this guide predates GPT-5-era Codex and the 2026 harness posts; its component model still holds but its multi-agent section is the older framing.

**Source: OpenAI Agents SDK docs, "Orchestrating multiple agents" (official docs, living document; read from the GitHub copy) — https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md**
- Definition: "An agent is an LLM equipped with instructions, tools and handoffs." [PAGE] — [Agents SDK docs](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)
- Two orchestration modes: let the LLM plan and choose steps, or orchestrate in code; "orchestrating via code makes tasks more deterministic and predictable" (speed, cost, performance). [PAGE] — [Agents SDK docs](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)
- Agents as tools (`Agent.as_tool()`): use when a specialist "should help with a bounded subtask but should not take over the user-facing conversation." Handoffs: use when "you want the chosen specialist to own the remainder of the current turn." [PAGE] — [Agents SDK docs](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)
- Tactics for LLM-driven orchestration: invest in good prompts (spell out tools, usage, constraints); "Monitor your app and iterate on it"; prefer specialized agents that excel at one task; "Invest in evals." [PAGE] — [Agents SDK docs](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)

**Source: Ryan Lopopolo (OpenAI), "Harness engineering: leveraging Codex in an agent-first world" (engineering blog, 2026-02-11) — https://openai.com/index/harness-engineering/**
- Framing: what changes when a team's primary job is "no longer to write code, but to design environments, specify intent, and build feedback loops." [SNIPPET] — [OpenAI](https://openai.com/index/harness-engineering/)
- One big AGENTS.md was tried first and "failed in predictable ways." Fix: a short AGENTS.md (about 100 lines) injected into context that "serves primarily as a map" to a structured `docs/` directory that is the system of record; a table of contents, not an encyclopedia; "a map rather than a 1,000-page manual" (the last phrase seen via translated versions). [SNIPPET] — [OpenAI](https://openai.com/index/harness-engineering/)
- Design docs are catalogued and indexed with verification status and a set of core beliefs (seen via Swedish-language version). [SNIPPET] — [OpenAI](https://openai.com/index/harness-engineering/)
- The initial AGENTS.md was itself written by Codex. [SNIPPET] — [OpenAI](https://openai.com/index/harness-engineering/)
- [RECALL, not verified this session] The post also describes: a product built with essentially no hand-written code over several months; architecture enforced mechanically by custom linters and structural tests whose error messages contain remediation instructions for the agent; making the app, logs and metrics legible to the agent (per-worktree app instances, browser automation, observability stack); recurring "garbage collection" agent runs to counter entropy/drift; and "golden principles" encoded in the repo. — [OpenAI](https://openai.com/index/harness-engineering/)

**Source: OpenAI Codex docs, "Best practices" (official docs, living; 2026) — https://developers.openai.com/codex/learn/best-practices**
- AGENTS.md is a README for agents, auto-loaded; scaffold with `/init`; layer global (`~/.codex`), repo-level and subdirectory files; keep the main file concise and reference task-specific markdown files. [SNIPPET] — [Codex best practices](https://developers.openai.com/codex/learn/best-practices)
- When Codex repeats a mistake, "ask for a retrospective and update AGENTS.md." [SNIPPET] — [Codex best practices](https://developers.openai.com/codex/learn/best-practices)
- Once a prompting pattern works, stop repeating it: package recurring instructions, context and supporting logic as a Skill (`SKILL.md`), usable across CLI, IDE extension and app. [SNIPPET] — [Codex best practices](https://developers.openai.com/codex/learn/best-practices)
- The in-repo Codex doc `docs/agents_md.md` is only a pointer to https://developers.openai.com/codex/guides/agents-md (so discovery order / size limits were not verified). [PAGE] — [openai/codex](https://github.com/openai/codex/blob/main/docs/agents_md.md)

**Source: Michael Bolin (OpenAI), "Unrolling the Codex agent loop" (engineering blog, 2026-01-23) — https://openai.com/index/unrolling-the-codex-agent-loop/**
- Explains the Codex CLI loop as repeated Responses API inference requests with tool calls. Only title, author, date and this one-line description were confirmed. [SNIPPET] — [OpenAI](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [RECALL] The post stresses prompt-prefix stability for prompt caching (append-only context, careful ordering of tools/instructions) and automatic compaction when the window fills, which lines up with Manus's KV-cache advice. Not verified this session. — [OpenAI](https://openai.com/index/unrolling-the-codex-agent-loop/)

### Inferences
- OpenAI's centre of gravity shifted from "orchestration patterns" (2025) to "environment and repo legibility" (2026). The 2026 advice is about what surrounds the model, not how to wire agents together.
- OpenAI's AGENTS.md advice (short map, progressive disclosure into docs and skills, update after repeated mistakes) is the same shape as the progressive-disclosure pattern other builders describe (Amp skills, LangChain deep agents offloading to files).
- The SDK's "specialized agents" tactic sits in mild tension with Cognition's single-thread default; the SDK resolves it by offering agents-as-tools, which keeps one agent in control.

### Gaps
- Could not read the practical guide, harness-engineering post, or Codex AGENTS.md guide directly (hosts blocked). Exact wording on guardrail taxonomy, AGENTS.md discovery order and size limits, and the harness post's throughput numbers is unverified.
- No verified material gathered on specific talks by OpenAI staff (e.g. DevDay or AI Engineer sessions) or on the GPT-5/Codex prompting guides; search did not surface them and fetches were unavailable.

## Google: agents whitepapers, ADK context architecture, Gemini CLI / Jules

### Takeaway
Google's strongest guidance is architectural: treat context as "a compiled view over a richer stateful system" (sessions, memory, artifacts as sources; a processor pipeline builds a minimal per-call working context), and treat sessions (short-term) and memory (long-term, declarative vs procedural) as separate engineered systems. The November 2025 Kaggle whitepaper series adds an "Agent Ops" operational discipline. I found little credible, specific practice guidance for Gemini CLI or Jules.

### Cited Findings

**Source: Google Developers Blog, "Architecting efficient context-aware multi-agent framework for production" (ADK team engineering blog, December 2025 per secondary dating) — https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/**
- Thesis: "context is a compiled view over a richer stateful system." [SNIPPET] — [Google Developers Blog](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)
- Separate durable state from per-call views: durable state lives in Sessions; a per-call "working context" is recomputed for each model call and discarded, so storage schemas and prompt formats evolve independently. [SNIPPET] — [Google Developers Blog](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)
- Artifacts: large files/logs are addressed by name and version rather than pasted into the prompt. [SNIPPET] — [Google Developers Blog](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)
- Sessions, memory and artifacts act as source material; an ordered pipeline of processors compiles a minimal task-specific context. [SECONDARY] — [commentary](https://raphaelmansuy.github.io/adk_training/blog/2025/12/08/context-engineering-google-adk-architecture)
- ADK docs: memory = searchable long-term cross-session recall; artifacts = versioned file storage; session state is read/written through the Context object. [SNIPPET] — [ADK docs](https://adk.dev/context/); [ADK sessions docs](https://adk.dev/sessions/)
- Production caveat: ADK's default in-process session and memory stores lose state on restart; swap in persistent services. [SECONDARY, vendor blog] — [Redis](https://redis.io/blog/build-google-adk-agents-with-persistent-real-time-memory-on-redis/)

**Source: Google / Kaggle, 5-Day AI Agents Intensive whitepapers (course ran 2025-11-10 to 2025-11-14; whitepapers include "Introduction to Agents" and "Context Engineering: Sessions, Memory") — overview at https://cloud.google.com/blog/products/ai-machine-learning/a-devs-guide-to-production-ready-ai-agents**
- Course ran live November 10-14, 2025 and is now self-paced; the Day 3 paper covers sessions as short-term conversation history and memory as long-term stored information. [SECONDARY] — [KDnuggets](https://www.kdnuggets.com/what-i-learned-from-kaggle-googles-free-agentic-ai-course); [Substack summary](https://intellibytes.substack.com/p/context-engineering-explained-how)
- Context engineering defined as dynamically assembling all needed inputs per turn to overcome LLM statelessness. [SECONDARY] — [summary](https://intellibytes.substack.com/p/context-engineering-explained-how)
- Memory taxonomy: declarative ("knowing what": facts and events about the user) vs procedural ("knowing how": processes and tool-use sequences). [SECONDARY] — [summary](https://intellibytes.substack.com/p/context-engineering-explained-how)
- "Introduction to Agents" is described as a 54-page guide (November 2025) that introduces "Agent Ops" as the operational discipline for managing unpredictability in agentic systems. [SECONDARY] — [ppc.land](https://ppc.land/google-cloud-releases-comprehensive-agentic-ai-framework-guideline/)
- Google Cloud's own post says the series covers foundational concepts plus separate material on tools/MCP and context engineering. [SNIPPET] — [Google Cloud blog](https://cloud.google.com/blog/products/ai-machine-learning/a-devs-guide-to-production-ready-ai-agents)
- An agent-capability taxonomy (from connected problem-solver up to collaborative multi-agent and self-evolving systems) is reported only in LinkedIn summaries; level definitions unverified. [SECONDARY, weak] — [ppc.land](https://ppc.land/google-cloud-releases-comprehensive-agentic-ai-framework-guideline/)
- The 72-page length of the sessions/memory paper is reported by one Substack only; unverified. [SECONDARY] — [Substack summary](https://intellibytes.substack.com/p/context-engineering-explained-how)

**Gemini CLI / Jules**
- Gemini CLI reads a `GEMINI.md` file in the project root for persistent memory/context and can connect to external sources via MCP. [SNIPPET, translated page] — [Google Cloud](https://cloud.google.com/discover/what-is-agentic-coding?hl=pt-BR)

### Inferences
- Google's "compiled view" model is the most explicit statement of a design other builders converge on implicitly: store everything durably, send the model only a rebuilt minimal view (compare Manus "restorable compression", LangChain "offload to files", Factory rolling summaries).
- Google documents multi-agent as a first-class ADK feature, but its context post is really about limiting what each agent sees, which is closer to Cognition's concern than the "multi-agent framework" title suggests.

### Gaps
- No primary-page verification of any Google whitepaper; all whitepaper content here is secondary. The original PDFs live on Kaggle and were not surfaced by search.
- No credible, specific best-practice source found for Jules, and only a one-line fact for Gemini CLI. Nothing found from Google DeepMind specifically on agent-building practice.
- The earlier (2024/early 2025) Kaggle "Agents" and "Agents Companion" whitepapers were not searched; they are older and likely superseded by the November 2025 series.

## Microsoft

### Takeaway
Microsoft's architecture guidance catalogues five multi-agent orchestration patterns but its reference architecture and training material default to a single agent and tell teams to justify multi-agent complexity.

### Cited Findings
**Source: Microsoft, Azure Architecture Center, "AI agent orchestration patterns" (official architecture guide, living document) — https://learn.microsoft.com/azure/architecture/ai-ml/guide/ai-agent-design-patterns**
- Lists sequential, concurrent, group chat, handoff, and magentic orchestration; positions multi-agent as for work that exceeds "a single agent that has access to many tools and knowledge sources." [SNIPPET] — [Azure Architecture Center](https://learn.microsoft.com/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- The baseline Foundry chat reference architecture "uses a single agent that has access to all necessary knowledge sources and tools" and points to the orchestration guide only if a workload meets multi-agent criteria. [SNIPPET] — [Architecture Center commit](https://github.com/MicrosoftDocs/architecture-center/commit/04ff7691f5)
- Microsoft Learn training says that if a system does not need multi-agent, consolidate to a single strong agent. [SNIPPET] — [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/aaai-implement-multi-agent-orchestration-azure-ai-foundry/2-differentiate-agentic-multi-agent-architectures)
- A Microsoft tech community post advises checking whether a single agent can do the job before adding more (2024; older). [SNIPPET] — [Tech Community](https://techcommunity.microsoft.com/blog/aiplatformblog/the-future-of-ai-single-agent-or-multi-agent---how-should-i-choose/4257104)

### Inferences
- Microsoft is the most pattern-catalogue-oriented of the labs, yet lands on the same "single agent first" default as OpenAI's guide and Cognition.

### Gaps
- Nothing verified on Microsoft Agent Framework (the AutoGen / Semantic Kernel successor) specific guidance, or on Copilot coding agent practices.

## Cognition (Devin): "Don't Build Multi-Agents" and how the position evolved

### Takeaway
In June 2025 Walden Yan argued for a single-threaded linear agent by default, because subagents without full shared context make conflicting implicit decisions. In April 2026 he narrowed rather than reversed this: multi-agent systems work when extra agents add intelligence (research, review) but "writes stay single-threaded"; parallel-writer swarms still have not seen meaningful adoption.

### Cited Findings

**Source: Walden Yan (Cognition), "Don't Build Multi-Agents" (company blog essay, 2025-06-12) — https://cognition.ai/blog/dont-build-multi-agents**
- Two principles: "Share context" (share full agent traces, not just individual messages) and "Actions carry implicit decisions" (conflicting decisions carry bad results). [SNIPPET] — [Cognition](https://cognition.ai/blog/dont-build-multi-agents)
- Principles 1 and 2 are "so critical, and so rarely worth violating, that you should by default rule out any agent architectures that don't abide by them." [SNIPPET] — [Cognition](https://cognition.ai/blog/dont-build-multi-agents)
- Default: "the simplest way to follow the principles is to just use a single-threaded linear agent." [SNIPPET] — [Cognition](https://cognition.ai/blog/dont-build-multi-agents)
- Failure example: a Flappy Bird clone split into subtasks yields a background and a bird in mismatched styles; in multi-agent setups "decision-making ends up being too dispersed and context isn't able to be shared thoroughly enough." [SNIPPET] — [Cognition](https://cognition.ai/blog/dont-build-multi-agents)
- [RECALL] For very long tasks the essay suggests a dedicated context-compression model that distils history into key decisions, and notes Claude Code subagents at the time only answered questions rather than writing code in parallel. Not verified this session. — [Cognition](https://cognition.ai/blog/dont-build-multi-agents)
- Harrison Chase's response notes that Cognition's essay and Anthropic's "How we built our multi-agent research system" have "seemingly opposite titles" but share lessons; commentary attributes the gap to problem shape (densely coupled coding writes vs separable research look-ups). [SNIPPET for Chase; SECONDARY for problem-shape reading] — [LangChain blog](https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems); [commentary](https://agentic-ai.readthedocs.io/en/latest/ContextEngineering/devin/)

**Source: Walden Yan (Cognition), "Multi-Agents: What's Actually Working" (company blog, 2026-04-22) — https://cognition.ai/blog/multi-agents-working**
- A narrower class of multi-agent systems now works: agents add intelligence but "writes stay single-threaded." [SNIPPET] — [Cognition](https://cognition.ai/blog/multi-agents-working)
- On parallel-writer swarms the original objections stand; most of the ambitious ideas in that space "have not seen meaningful adoption." [SNIPPET] — [Cognition](https://cognition.ai/blog/multi-agents-working)
- Working pattern: a code-review loop where a second model reviews the coder's output; the reviewer works best from a clean context rather than inheriting the coder's accumulated reasoning; with dedicated prompting, models can make reasonable judgment calls on review feedback. [SNIPPET] — [Cognition](https://cognition.ai/blog/multi-agents-working)

### Inferences
- The evolution is a refinement along the read/write axis: parallelism and fresh contexts are acceptable for reading, researching and reviewing; mutation of shared state stays serial. This is the cleanest reconciliation of the 2025 "single vs multi" dispute.
- The "reviewer works best with clean context" finding is a deliberate exception to "share full context": for verification, independence beats shared context.

### Gaps
- Full list of patterns in the April 2026 post was not retrieved (intro and review-loop section only). Whether it discusses a strong-model "advisor/oracle" pattern or manager-of-Devins is unverified.

## Manus: context engineering lessons and later material from Yichao "Peak" Ji

### Takeaway
Manus's July 2025 post is the canonical statement of cache-aware, append-only context design: optimise KV-cache hit rate, mask tools rather than removing them, use the file system as restorable external memory, recite goals via a todo file, and keep errors in context. By October 2025 Peak Ji added that the harness itself must be repeatedly simplified as models improve (five rewrites since March 2025) and that Manus runs on fewer than 20 atomic tools.

### Cited Findings

**Source: Yichao "Peak" Ji (Manus co-founder / chief scientist), "Context Engineering for AI Agents: Lessons from Building Manus" (company blog, 2025-07-18) — https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus**
- "If I had to choose just one metric, I'd argue that the KV-cache hit rate is the single most important metric for a production-stage AI agent." [SNIPPET] — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Cache-breaking mistakes: "A common mistake is including a timestamp—especially one precise to the second—at the beginning of the system prompt"; unstable JSON key ordering silently breaks the cache. Keep the prefix stable and the context append-only. [SNIPPET] — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Tool management: rather than dynamically adding/removing tools mid-task, Manus "masks the token logits during decoding to prevent (or enforce) the selection of certain actions based on the current context." [SNIPPET] — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- File system as memory: Manus treats "the file system as the ultimate context: unlimited in size, persistent by nature, and directly operable by the agent itself"; compression strategies "are always designed to be restorable" (e.g. keep the URL or path, drop the content). [SNIPPET] — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Recitation: a typical task takes about 50 tool calls; the agent creates a todo.md and updates it step by step, pushing the global plan into the model's recent attention span to counter goal drift. [SNIPPET] — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Keep errors in: failed actions and stack traces stay in context so the model updates away from repeating them. [SECONDARY] — [summary](https://agentic-ai.readthedocs.io/en/latest/ContextEngineering/manus/)
- "We've rebuilt our agent framework four times," each after finding a better way to shape context; context engineering is called an experimental science. [SNIPPET] — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [RECALL] Section headings are: "Design Around the KV-Cache", "Mask, Don't Remove", "Use the File System as Context", "Manipulate Attention Through Recitation", "Keep the Wrong Stuff In", "Don't Get Few-Shotted" (vary serialisation/phrasing so the agent does not fall into repetitive patterns), with the process nicknamed "Stochastic Graduate Descent". The search engine could not confirm these headings verbatim; only "Keep the Wrong Stuff In" appeared, in a third-party summary. — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)

**Source: Lance Martin, "Context Engineering in Manus" (personal blog notes on a LangChain + Manus webinar with Peak Ji, 2025-10-15) — https://rlancemartin.github.io/2025/10/15/manus/**
- Manus relies on three strategies: reduce, offload, and isolate context. [SNIPPET] — [Lance Martin](https://rlancemartin.github.io/2025/10/15/manus/)
- Action space: a small set (fewer than 20) of atomic functions including a Bash tool, filesystem tools and code execution; broader capability is reached through the shell/sandbox rather than more function-call tools. [SNIPPET] — [Lance Martin](https://rlancemartin.github.io/2025/10/15/manus/)
- The harness can limit performance as models improve; Manus "has been refactored five times since their launch in March." [SNIPPET] — [Lance Martin](https://rlancemartin.github.io/2025/10/15/manus/)
- Webinar event page (LangChain + Manus). [SNIPPET] — [Luma](https://luma.com/819i5ime)
- [RECALL] The webinar also distinguished compaction (reversible: strip content that can be re-fetched from files) from summarisation (irreversible; used only when compaction no longer helps, and done with a schema), and described a layered action space (function calls, sandbox utilities, packages/APIs). The phrase "layered action space" was not confirmed by search. — [Phil Schmid summary](https://www.philschmid.de/context-engineering-part-2)

### Inferences
- Manus's position evolved from "shape the context carefully" (July 2025) to "remove harness as models improve" (October 2025), the same "bitter lesson" direction Amp, Vercel and pi describe.
- "Mask, don't remove" and "few atomic tools plus a shell" both aim at the same thing: a stable, cacheable tool prefix with capability pushed out to the sandbox.
- Search results show manus.im page titles now reading "Manus is now part of Meta"; the acquisition context means later Manus engineering posts may be sparser. This is from page titles only, not a verified announcement.

### Gaps
- Could not read the Manus post; several quotes are snippet-level and headings are recall-level.
- No 2026 Manus/Peak Ji engineering post or talk was found.

## LangChain: context engineering, deep agents, harnesses, ambient agents

### Takeaway
LangChain supplied the shared vocabulary: Lance Martin's write / select / compress / isolate taxonomy (June 2025), then the "framework vs runtime vs harness" split (October 2025) and "deep agents" as a batteries-included harness with planning, a filesystem, subagents and context summarisation. By 2026 Harrison Chase frames "harness engineering" as the extension of context engineering.

### Cited Findings

**Source: Lance Martin, "Context Engineering for Agents" (personal blog, 2025-06-23; mirrored on LangChain blog) — https://rlancemartin.github.io/2025/06/23/context_engineering/ and https://www.langchain.com/blog/context-engineering-for-agents**
- Four buckets: write (save information outside the context window: scratchpads within a task, memories across sessions), select (pull relevant information in), compress (retain "only the tokens required to perform a task"), isolate (split context across agents or sandboxes). [SNIPPET] — [Lance Martin](https://rlancemartin.github.io/2025/06/23/context_engineering/); [langchain-ai/context_engineering repo](https://github.com/langchain-ai/context_engineering)
- Motivation: accumulated tool feedback exhausts the window, raises cost and latency, and degrades performance; linked to Chroma's "context rot" finding that performance deteriorates before the window is full. [SECONDARY] — [Latent Space wiki](https://latent.space/wiki/lance-martin-d6681b43/)
- Isolation example is Anthropic's multi-agent researcher, with the stated costs of more tokens and coordination; on compression, careful prompting worked for him while summarisation "can require substantial investment". [SECONDARY] — [Latent Space wiki](https://latent.space/wiki/lance-martin-d6681b43/)

**Source: Harrison Chase, "How and when to build multi-agent systems" (LangChain blog, June 2025) — https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems**
- Reconciles Cognition and Anthropic: the two posts have "seemingly opposite titles" but common ground. [SNIPPET] — [LangChain blog](https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems)
- [RECALL] His synthesis: context engineering is the crux in both; multi-agent is easier for "read" tasks than "write" tasks. Not verified this session. — [LangChain blog](https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems)

**Source: Harrison Chase, "Agent Frameworks, Runtimes, and Harnesses - oh my!" (LangChain blog, October 2025) — https://www.langchain.com/blog/agent-frameworks-runtimes-and-harnesses-oh-my**
- Layering: "LangChain is an agent framework, LangGraph is an agent runtime, DeepAgents is an agent harness." [SNIPPET] — [LangChain blog](https://www.langchain.com/blog/agent-frameworks-runtimes-and-harnesses-oh-my)

**Source: langchain-ai/deepagents README (GitHub repo, living; read 2026-11-23) — https://github.com/langchain-ai/deepagents**
- Describes itself as "an opinionated agent that runs out of the box"; "The agent can plan, read/write files, and manage its own context." [PAGE] — [deepagents](https://github.com/langchain-ai/deepagents)
- Filesystem: "read, write, edit, or search over pluggable local, sandboxed, or remote backends." Subagents: "delegate tasks to agents with isolated context windows." Context management: "summarize long threads and offload tool outputs to disk." You can "Extend, override, or replace any piece." [PAGE] — [deepagents](https://github.com/langchain-ai/deepagents)

**Source: Vivek Trivedy (LangChain), "The Anatomy of an Agent Harness" (LangChain blog, 2026-03-10) — https://www.langchain.com/blog/the-anatomy-of-an-agent-harness**
- Planning tools let agents "decompose tasks, track progress, and adapt as they learn." Note: this post is by Trivedy, not Chase. [SNIPPET] — [LangChain blog](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness)

**Other Chase material (2026)**
- VentureBeat reports Chase describing "harness engineering" as an extension of context engineering and arguing better models alone will not get agents to production. [SECONDARY] — [VentureBeat](https://venturebeat.com/orchestration/langchains-ceo-argues-that-better-models-alone-wont-get-your-ai-agent-to)
- Sequoia podcast "Owning Your Intelligence Starts With the Harness" covers when an off-the-shelf harness suffices vs building your own (subagents, middleware hooks). [SNIPPET] — [Sequoia](https://sequoiacap.com/podcast/owning-your-intelligence-starts-with-the-harness)
- An October 2026 Chase article reportedly lists five parts of a "company harness": Slack-based interface, identity handling, two memory tiers, sandboxed code execution, schedule-driven runs. Single weak secondary source. [SECONDARY, weak] — [FourWeekMBA](https://fourweekmba.com/ai-langchains-chase-names-5-parts-of-a-company-harness/)

### Inferences
- LangChain, historically the "framework" company, now sells a harness whose components (plan tool, filesystem, subagents, summarisation) mirror Claude Code and Manus. That is convergence on a common architecture rather than on a common framework.
- The schedule-driven, Slack-fronted "company harness" is the 2026 form of what LangChain called "ambient agents" in early 2025.

### Gaps
- No primary source retrieved for "ambient agents" (Chase's early-2025 posts and agent-inbox pattern); the term did not surface in search. Treat as older (early 2025) material that still needs a citation.
- Lance Martin's later posts (e.g. on the "bitter lesson" for agent harnesses) were not retrieved.

## HumanLayer: Dex Horthy's 12-factor agents and advanced context engineering

### Takeaway
Horthy's two contributions are (1) "12-factor agents" (2025): production agents are mostly ordinary software with LLM steps, so own your prompts, context window and control flow rather than adopting a framework wholesale; and (2) "frequent intentional compaction" (August 2025): keep context utilisation around 40-60%, run research → plan → implement with compacted artifacts between phases, and put human review on research and plans, not code.

### Cited Findings

**Source: Dex Horthy (HumanLayer), "12-Factor Agents" (GitHub repo / essay, 2025; living) — https://github.com/humanlayer/12-factor-agents**
- The factors, verbatim: 1 "Natural Language to Tool Calls"; 2 "Own your prompts"; 3 "Own your context window"; 4 "Tools are just structured outputs"; 5 "Unify execution state and business state"; 6 "Launch/Pause/Resume with simple APIs"; 7 "Contact humans with tool calls"; 8 "Own your control flow"; 9 "Compact Errors into Context Window"; 10 "Small, Focused Agents"; 11 "Trigger from anywhere, meet users where they are"; 12 "Make your agent a stateless reducer". Honourable-mention 13th: "Pre-fetch all the context you might need". [PAGE] — [12-factor-agents](https://github.com/humanlayer/12-factor-agents)
- On frameworks: "I've tried every agent framework out there"; "I don't see a lot of frameworks in production customer-facing agents." Teams reach roughly 70-80% quality with a framework and then must "reverse-engineer the framework" to go further. [PAGE] — [12-factor-agents](https://github.com/humanlayer/12-factor-agents)
- Most products called AI agents are "not all that agentic": mostly deterministic code with LLM steps at key points; the recommended path is adopting small modular agent concepts into existing products rather than a rewrite. [PAGE] — [12-factor-agents](https://github.com/humanlayer/12-factor-agents)
- Flag: written in spring 2025 for customer-facing workflow agents with weaker models; its scepticism of the plain "loop until done" pattern conflicts with later coding-agent experience (Amp, pi, Vercel) where a simple loop with strong models works well.

**Source: Dex Horthy, "Advanced Context Engineering for Coding Agents" (GitHub essay `ace-fca.md`, circa late August 2025; author/date inferred from page content, which references a talk "on August 20th" and screenshots dated 2025-08-29) — https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md**
- Frequent intentional compaction = "designing your ENTIRE WORKFLOW around context management"; early form was a prompt like "Write everything we did so far to progress.md." [PAGE] — [ace-fca.md](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md)
- Target: keep "utilization in the 40%-60% range," depending on problem complexity. [PAGE] — [ace-fca.md](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md)
- Research (understand the code, information flow, likely causes) → Plan (exact steps, files, "super precise" verification per phase) → Implement ("Step through the plan, phase by phase," compacting status back into the plan file after each verified phase). [PAGE] — [ace-fca.md](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md)
- Subagents are for context control, not role-play: a fresh context does the searching and summarising so the parent stays clean; getting them to return well-compacted output is "not trivial." [PAGE] — [ace-fca.md](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md)
- Review leverage: review research and plans because a bad plan line becomes hundreds of bad code lines: "you get more leverage than you do when you review the code"; "I can't read 2000 lines of golang daily. But I _can_ read 200 lines of a well-written implementation plan." [PAGE] — [ace-fca.md](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md)
- Self-reported results (anecdotal, no controls): a BAML bug-fix PR in about an hour, approved next morning; about 35k LOC in roughly 7 hours; a three-person team spending about $12k/month on Opus. Reported failures: research that wrongly concluded a bug was invalid; a 7-hour parquet-java Hadoop-removal attempt that failed because research "didn't go deep enough through the dependency tree." [PAGE] — [ace-fca.md](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md)

**Source: Dex Horthy, "No Vibes Allowed: Context Engineering for Complex Codebases" (AI Engineer Code Summit talk, late 2025) — https://ai.engineer/talks/context-engineering-for-complex-codebases**
- "Dumb zone": output quality degrades as the window fills, with onset around 40% utilisation, varying by task and model. [SECONDARY] — [dev.to summary](https://dev.to/ametel01/advanced-context-engineering-for-coding-agents-11p7)
- Horthy says research-plan-implement "may not remain the defining sequence"; the durable principles are compaction, context engineering and avoiding overloaded windows. Tenet: "Do not outsource the thinking." [SECONDARY] — [AI Engineer talk page](https://ai.engineer/talks/context-engineering-for-complex-codebases)
- Follow-up interview: [Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/context-engineering-with-dex-horthy). [SNIPPET]

### Inferences
- Horthy's own position shifted from "own your control flow, loops don't work that well" (workflow agents, early 2025) to a workflow built on top of an autonomous loop agent (coding, late 2025). The constant is "own your context window."
- The 40% figure is a practitioner heuristic, not a measured threshold; newer models and larger windows likely move it. Flag it as dated to 2025-era models.

### Gaps
- No verified 2026 update from Horthy on whether the 40-60% target or RPI workflow changed with newer models.

## Sourcegraph Amp, Cursor, Replit, Factory, Vercel, Cloudflare, Letta, and others

### Takeaway
Coding-agent builders converge on a minimal core (model, loop, a handful of tools) and differ mainly on how far to push parallelism: Amp and pi keep the harness thin and are sceptical of MCP bloat and custom subagents; Cursor reports that hierarchical planner/worker/judge systems scale to hundreds of agents; Vercel and Cloudflare both replace many discrete tools with a code/shell interface; Factory and Letta focus on repo readiness, compaction quality and agent-managed memory.

### Cited Findings

**Sourcegraph Amp**
- Source: Thorsten Ball, "How to Build an Agent, or: The Emperor Has No Clothes" (ampcode.com blog, April 2025) — https://ampcode.com/how-to-build-an-agent. A working coding agent is an LLM, a loop and three tools (read_file, list_files, edit_file) in under 400 lines of Go. [SECONDARY; primary not fetched] — [summary](https://howaiworks.ai/thorsten-ball-how-to-build-an-agent/)
- Latent Space writeup (2026): Amp began as a February 2025 experiment by Ball and Quinn Slack giving Claude 3.7 tools and broad freedom; philosophy is keeping the agent central and scaffolding disposable; the team declined custom subagents at the time because of extra tokens, convoluted handoffs and doubts about model support, and criticised verbose MCP servers. [SECONDARY] — [Latent Space](https://latent.space/reading/amp-emperor-podcast/)
- Amp docs recommend bundling MCP servers inside skills (via mcp.json) rather than global settings, and keeping tool counts small. [SNIPPET] — [Amp manual](https://ampcode.com/manual/mcp.md)
- The "oracle" is a second, more expensive high-reasoning model the main agent can consult. [SNIPPET; SECONDARY] — [Amp tools docs](https://ampcode.com/docs/tools); [Hamel Husain notes](https://hamel.dev/notes/coding-agents/amp.html)
- Ball reportedly uses one or two skills and no MCP servers. [SNIPPET] — [Amp "using Amp" page](https://ampcode.com/using-amp/thorsten)

**Cursor**
- Source: Wilson Lin (Cursor), "Scaling long-running autonomous coding" (blog, January 2026) — https://cursor.com/blog/scaling-agents. Flat self-coordinating agents churned without progress; "Our next approach was to separate roles": planners continuously explore the codebase and create tasks (and can spawn sub-planners), workers are "solely responsible for driving them to completion" and do not coordinate with each other, and a judge decides at the end of each cycle whether to continue. [SNIPPET] — [Cursor](https://cursor.com/blog/scaling-agents)
- "We still need periodic fresh starts to combat drift and tunnel vision." Hundreds of workers pushed to one branch with minimal conflicts; a one-week run built a browser from scratch (over 1 million lines across 1,000 files, company-reported); different models suit different roles (GPT-5.2 reported best for extended autonomous work). [SNIPPET] — [Cursor](https://cursor.com/blog/scaling-agents)
- Shared-file self-coordination caused lock contention and throughput collapse. [SECONDARY] — [ZenML summary](https://www.zenml.io/llmops-database/scaling-multi-agent-autonomous-coding-systems)
- Source: Cursor, "Towards self-driving codebases" (blog, February 2026) — https://cursor.com/blog/self-driving-codebases. Role iterations: planner → executor (sole lead responsible for completing the plan) → workers, with an independent judge; removing the planner overloaded the executor (sleeping, low delegation, premature completion per a secondary summary); the final design is recursive planning with a root planner delegating to subplanners and workers. [SNIPPET; pathologies SECONDARY] — [Cursor](https://cursor.com/blog/self-driving-codebases)

**Vercel**
- Source: Vercel, "We removed 80% of our agent's tools" (blog, 2025-12-22) — https://vercel.com/blog/we-removed-80-percent-of-our-agents-tools. The d0 text-to-SQL agent went from about 17 specialised tools to essentially one bash tool with file-system access to the semantic layer; result "fewer steps, fewer tokens, faster responses"; success went from 4/5 to 5/5 benchmark queries. Sample size is five queries. [SNIPPET] — [Vercel](https://vercel.com/blog/we-removed-80-percent-of-our-agents-tools)
- Reported detail: about 3.5x faster, 37% fewer tokens, 42% fewer steps; depends on well-documented, well-structured data. [SECONDARY] — [ZenML summary](https://www.zenml.io/llmops-database/simplifying-text-to-sql-agents-by-removing-80-of-tools)

**Cloudflare**
- Source: Cloudflare, "Code Mode: the better way to use MCP" (blog, 2025) — https://blog.cloudflare.com/code-mode/. Convert MCP tools into a TypeScript API and have the LLM write code that calls it, instead of exposing tools directly. [SNIPPET] — [Cloudflare](https://blog.cloudflare.com/code-mode/)
- Source: Matt Carey (Cloudflare), "Code Mode: give agents an entire API in 1,000 tokens" (blog, February 2026) — https://blog.cloudflare.com/code-mode-mcp/. The server exports two tools, `search()` and `execute()`, instead of thousands; Cloudflare reports a 99.9% input-token reduction for its API (vendor benchmark). [SNIPPET] — [Cloudflare](https://blog.cloudflare.com/code-mode-mcp/)

**Factory**
- "Agent Readiness" scores repositories across technical pillars (eight per the launch post, nine per one docs page; sources conflict) and five maturity levels from Functional to Autonomous, each unlocked at 80% of the previous. [SNIPPET] — [Factory](https://factory.ai/news/agent-readiness); [docs](https://docs.factory.ai/agent-readiness/overview)
- "Compressing Context": keep a lightweight persistent rolling summary; once key artifacts are summarised away the agent must re-fetch them; preserve file paths, function names and key identifiers. [SNIPPET] — [Factory](https://factory.ai/news/compressing-context)
- December 2025 evaluation: structured summarisation retained more continue-the-task information than OpenAI's and Anthropic's built-in compaction at similar compression rates (vendor-run); probe-based evaluation beats text-similarity metrics. [SNIPPET] — [Factory docs](https://docs.factory.ai/guides/power-user/evaluating-context-compression)

**Replit**
- Michele Catasta (Replit president), "The 3 Pillars of Autonomy" (AI Engineer Code Summit, late 2025) — https://ai.engineer/talks/the-3-pillars-of-autonomy-michele-catasta-replit. Autonomy is scoped: the agent takes all technical decisions while the user stays involved; internal evals found more than 30% of individual features broken on first build, motivating Agent 3's self-testing in a real browser with a reflection loop. [SECONDARY] — [AI Engineer](https://ai.engineer/talks/the-3-pillars-of-autonomy-michele-catasta-replit); [InfoQ](https://www.infoq.com/news/2025/09/replit-agent-3)
- Older (2024): "We don't strive for full autonomy. We want the user to stay involved and engaged." [SNIPPET; older] — [LangChain case study](https://langchain.com/breakoutagents/replit)

**Letta**
- Core memory is a set of in-context memory blocks managed by the agent itself or by other agents; sleep-time agents consolidate memory in the background. [SNIPPET] — [Letta docs](https://docs.letta.com/guides/agents/architectures/memgpt); [Letta blog](https://www.letta.com/blog/agent-memory)
- A March 2026 Letta post argues memory cannot be bolted on as a plugin: context and state management are core responsibilities of the harness. [SNIPPET; exact post URL not isolated] — [Letta blog index](https://www.letta.com/blog)

**pi (Mario Zechner, Armin Ronacher)**
- Mario Zechner, "What I learned building an opinionated and minimal coding agent" (personal blog, 2025-11-30): four default tools (read, write, edit, bash), system prompt under 1,000 tokens, no MCP, no built-in subagents or plan mode; a minimal malleable core the user and agent can modify. [SNIPPET] — [Zechner](https://www.mariozechner.at/posts/2025-11-30-pi-coding-agent/)
- Armin Ronacher (2026-01-31) says Pi is the coding agent he uses almost exclusively. [SNIPPET] — [Ronacher](https://lucumr.pocoo.org/2026/1/31/pi/)

### Inferences
- "Fewer, more general tools" is now the majority position among coding-agent builders (Amp, pi, Manus, Vercel, Cloudflare), with a shell or code sandbox as the universal tool.
- Cursor is the main credible counterweight on multi-agent: it reports that large hierarchical swarms work, but its own lessons (workers isolated from each other, strict role separation, a judge, periodic fresh starts) are constraints that keep each write path owned by one agent, so it does not really contradict Cognition's 2026 rule.
- Factory and OpenAI agree that repo quality (docs, linters, tests, fast feedback) is the main lever for agent performance; Factory turned it into a maturity score.

### Gaps
- Amp primary posts were not fetched; Quinn Slack's own statements (for example on model selection or pricing) are unverified. The "no model selector" claim was not confirmed by a primary source.
- No credible engineering write-up found this session for Replit beyond talks, and nothing substantial for Vercel's AI SDK agent guidance beyond the d0 post.
- Cursor "lessons" phrases such as "anti-fragile" could not be confirmed.

## Where these sources agree and where they conflict

### Takeaway
Agreement is broad on context discipline, simple cores, file-system offloading, verification loops and removing harness as models improve. The live conflicts are narrower than the headlines: multi-agent is contested only for parallel writers; MCP is contested as a context cost, not as a protocol; and frameworks are contested mainly at the orchestration layer, while "harnesses" are now accepted.

### Cited Findings

**Agreement**
- Context is the main engineering lever, and quality degrades before the window is full: Martin (context rot), Horthy (40-60% utilisation), Manus (restorable compression), Google (compiled working context), Factory (rolling summary). — [Lance Martin](https://rlancemartin.github.io/2025/06/23/context_engineering/); [ace-fca.md](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md); [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus); [Google Developers Blog](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/); [Factory](https://factory.ai/news/compressing-context)
- Use the file system / durable store as external memory and keep pointers in context: Manus, deepagents, Google artifacts, OpenAI docs-as-system-of-record, Horthy's plan and progress files. — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus); [deepagents](https://github.com/langchain-ai/deepagents); [OpenAI](https://openai.com/index/harness-engineering/)
- Short instruction file as a map with progressive disclosure: OpenAI (AGENTS.md about 100 lines, skills), Amp (skills bundling MCP). — [OpenAI](https://openai.com/index/harness-engineering/); [Codex best practices](https://developers.openai.com/codex/learn/best-practices); [Amp manual](https://ampcode.com/manual/mcp.md)
- Start with one agent; add agents only with a reason: OpenAI guide, Microsoft reference architecture, Cognition, 12-factor ("Small, Focused Agents"). — [OpenAI PDF](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf); [Microsoft](https://github.com/MicrosoftDocs/architecture-center/commit/04ff7691f5); [Cognition](https://cognition.ai/blog/dont-build-multi-agents)
- Subagents are primarily a context-isolation device: Horthy, deepagents, Martin's "isolate". — [ace-fca.md](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md); [deepagents](https://github.com/langchain-ai/deepagents)
- Independent verification with fresh context: Cognition's clean-context reviewer, Cursor's judge and fresh starts, Replit's self-testing. — [Cognition](https://cognition.ai/blog/multi-agents-working); [Cursor](https://cursor.com/blog/scaling-agents); [InfoQ on Replit Agent 3](https://www.infoq.com/news/2025/09/replit-agent-3)
- Harness should shrink as models improve: Manus (five refactors), Vercel (removed 80% of tools), pi, Amp ("scaffolding disposable"). — [Lance Martin on Manus](https://rlancemartin.github.io/2025/10/15/manus/); [Vercel](https://vercel.com/blog/we-removed-80-percent-of-our-agents-tools); [Zechner](https://www.mariozechner.at/posts/2025-11-30-pi-coding-agent/); [Latent Space on Amp](https://latent.space/reading/amp-emperor-podcast/)

**Conflict 1: single agent vs multi-agent**
- Cognition 2025: default to "a single-threaded linear agent"; Cognition 2026: multi-agent works when "writes stay single-threaded". — [Cognition 2025](https://cognition.ai/blog/dont-build-multi-agents); [Cognition 2026](https://cognition.ai/blog/multi-agents-working)
- Cursor 2026: hundreds of concurrent workers under planners and a judge, "a more optimistic answer than we expected" on whether more agents help. — [Cursor](https://cursor.com/blog/scaling-agents)
- OpenAI Agents SDK: prefer specialised agents for LLM-driven orchestration, with agents-as-tools vs handoffs. — [Agents SDK docs](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)
- Amp: declined custom subagents (tokens, handoffs) in 2025, later shipped subagents with full tool access. — [Latent Space](https://latent.space/reading/amp-emperor-podcast/)

**Conflict 2: MCP / many tools vs CLI, shell and code**
- Against tool sprawl: Vercel (one bash tool beat about 17 tools), Manus (fewer than 20 atomic functions), pi (no MCP), Ball (no MCP servers), Cloudflare (wrap MCP in code: two tools instead of thousands). — [Vercel](https://vercel.com/blog/we-removed-80-percent-of-our-agents-tools); [Lance Martin on Manus](https://rlancemartin.github.io/2025/10/15/manus/); [Zechner](https://www.mariozechner.at/posts/2025-11-30-pi-coding-agent/); [Cloudflare](https://blog.cloudflare.com/code-mode-mcp/)
- For MCP as integration layer: Google's whitepaper series treats tools/MCP as a core topic; Gemini CLI and Amp support MCP; Amp scopes MCP inside skills. — [Google Cloud blog](https://cloud.google.com/blog/products/ai-machine-learning/a-devs-guide-to-production-ready-ai-agents); [Amp manual](https://ampcode.com/manual/mcp.md)
- Manus-specific tension: dynamic tool loading breaks the KV-cache, so Manus masks logits instead, whereas skills-style progressive disclosure loads tool descriptions on demand. — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus); [Codex best practices](https://developers.openai.com/codex/learn/best-practices)

**Conflict 3: frameworks vs plain loops**
- Plain loop: Ball (agent in under 400 lines), pi (four tools), Horthy ("I don't see a lot of frameworks in production customer-facing agents"). — [summary of Ball](https://howaiworks.ai/thorsten-ball-how-to-build-an-agent/); [12-factor-agents](https://github.com/humanlayer/12-factor-agents)
- But Horthy also says the bare loop pattern does not work as well as hoped for production workflow agents and wants code-owned control flow, while Ball/pi/Vercel trust the model-driven loop. — [12-factor-agents](https://github.com/humanlayer/12-factor-agents)
- Framework/harness side: OpenAI Agents SDK, Google ADK, LangChain deepagents ("an opinionated agent that runs out of the box"), with Chase's framework / runtime / harness distinction. — [deepagents](https://github.com/langchain-ai/deepagents); [LangChain blog](https://www.langchain.com/blog/agent-frameworks-runtimes-and-harnesses-oh-my)
- OpenAI SDK offers both: LLM orchestration or code orchestration, the latter "more deterministic and predictable". — [Agents SDK docs](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)

**Conflict 4: errors and history in context**
- Manus keeps failed actions in context; 12-factor says "Compact Errors into Context Window"; Horthy and Cursor favour fresh starts and compaction to escape bad trajectories. — [Manus summary](https://agentic-ai.readthedocs.io/en/latest/ContextEngineering/manus/); [12-factor-agents](https://github.com/humanlayer/12-factor-agents); [Cursor](https://cursor.com/blog/scaling-agents)

### Inferences
- Single vs multi resolves on a read/write axis: parallel or isolated agents for research, exploration and review are widely endorsed; multiple concurrent writers on coupled state are endorsed only by Cursor, and only under strict hierarchy and worker isolation.
- MCP vs CLI resolves on context cost: the objection is to loading many tool schemas into the prompt, and the emerging compromise is progressive disclosure (skills, code mode, search-then-execute) or a shell.
- Frameworks vs loops resolves on what is abstracted: builders reject frameworks that hide prompts, context and control flow, and accept harnesses that expose them. Task type also matters: customer-facing workflow agents favour code-owned control flow; coding agents favour a model-driven loop.
- Errors-in-context resolves on timescale: keep recent errors visible within a trajectory, but reset or compact when a trajectory has gone bad.
- Outdated or dated items to flag in the report: OpenAI's practical guide (2025, pre-harness framing); 12-factor agents' loop scepticism (early 2025 models); Cognition's June 2025 absolute stance (superseded April 2026); Horthy's 40% dumb-zone figure (2025 models); Amp's 2025 rejection of subagents (later relaxed); Replit's 2024 "we don't strive for full autonomy" (superseded by Agent 3's longer autonomous runs).

### Gaps
- Most primary blogs could not be opened, so many quotes are snippet-level; a pass with openai.com, cognition.ai, manus.im, cursor.com, ampcode.com, langchain.com and rlancemartin.github.io allowed would upgrade them.
- No quantitative, independent comparison exists in these sources for any of the conflicts; evidence is builder-reported (Vercel n=5 queries, Cloudflare and Factory vendor benchmarks, Cursor and HumanLayer self-reported output).

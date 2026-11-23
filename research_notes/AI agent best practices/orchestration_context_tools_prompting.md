# Multi-agent orchestration, harness design, context engineering, tool design and prompting for agents

Research date: 2026-11-23. Scope: how experts structure the system around the model.

**Verification legend (applies to every bullet):**
- **[V]** = read from the page itself in this session (page fetched). Only anthropic.com, claude.com, platform.claude.com and code.claude.com could be fetched.
- **[S]** = taken from web-search result summaries only; the page could not be opened (DNS-blocked host). Exact wording and numbers should be re-checked before quoting.
- **[M]** = from the researcher's prior knowledge of a well-known source, not re-verified in this session. Treat as lowest confidence.

**Environment limits that shaped these notes:** cognition.ai, research.google and mariozechner.at failed to resolve (one attempt each, not retried). Other non-Anthropic hosts were not attempted, in line with the brief. The session's web-search budget ran out before four planned searches (Anthropic context-editing benchmark numbers; Zechner's MCP-vs-CLI post; Hamel Husain; Philipp Schmid on harnesses). Those are listed under Gaps.

---

## Source register (author, title, type, date, URL, practices)

Dates are publication dates. "Status" flags what is outdated as of November 2026.

### Anthropic

| # | Author(s) | Title | Type | Date | URL | Verif. | Status |
|---|---|---|---|---|---|---|---|
| A1 | Prithvi Rajasekaran, Ethan Dixon, Carly Ryan, Jeremy Hadfield | Effective context engineering for AI agents | Engineering blog | 2025-09-29 | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | V | Core concepts current; "context is the main constraint" framing softened by 2026 Claude Code docs (A11) |
| A2 | Jeremy Hadfield, Barry Zhang, Kenneth Lien, Florian Scholz, Jeremy Fox, Daniel Ford | How we built our multi-agent research system | Engineering blog | 2025-06-13 | https://www.anthropic.com/engineering/multi-agent-research-system | V | Numbers measured on Opus 4 / Sonnet 4; patterns still cited; the 200K truncation detail is outdated |
| A3 | Ken Aizawa (+ contributors) | Writing effective tools for agents — with agents | Engineering blog | 2025-09-11 | https://www.anthropic.com/engineering/writing-tools-for-agents | V | Current |
| A4 | Barry Zhang, Keith Lazuka, Mahesh Murag | Equipping agents for the real world with Agent Skills | Engineering blog | 2025-10-16 | https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills | V | Current |
| A5 | Adam Jones, Conor Kelly | Code execution with MCP | Engineering blog | 2025-11-04 | https://www.anthropic.com/engineering/code-execution-with-mcp | V | Current |
| A6 | Bin Wu (+ contributors) | Introducing advanced tool use (Tool Search Tool, Programmatic Tool Calling, Tool Use Examples) | Engineering blog | 2025-11-24 | https://www.anthropic.com/engineering/advanced-tool-use | V | Accuracy numbers are for Opus 4 / 4.5; features current |
| A7 | Justin Young | Effective harnesses for long-running agents | Engineering blog | 2025-11-26 | https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | V | Pattern still in the prompting docs (A10); partly superseded by A8 |
| A8 | Prithvi Rajasekaran | Harness design for long-running application development | Engineering blog | 2026-03-24 | https://www.anthropic.com/engineering/harness-design-long-running-apps | V | Current; explicitly shows scaffolding being removed between Sonnet 4.5, Opus 4.5 and Opus 4.6 |
| A9 | Cara Phillips (+ Paul Chen, Andy Schumeister, Brad Abrams, Theo Chu) | Building multi-agent systems: when and how to use them | Blog / best practices | 2026-01-23 | https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them | V | Current |
| A10 | Anthropic docs | Prompting best practices (all current Claude models) | Documentation | Live page, read 2026-11-23 | https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices | V | Current. Some paragraphs name Opus 4.5/4.6 and say to re-check on newer models |
| A11 | Anthropic docs | Prompting Claude Fable 5.2 | Documentation | Live page, read 2026-11-23 | https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2 | V | Newest model-specific guidance found |
| A12 | Anthropic docs | Best practices for Claude Code | Documentation | Live page, read 2026-11-23 | https://code.claude.com/docs/en/best-practices | V | Current; contains a "What you can stop doing" table that retires older advice |
| A13 | Anthropic docs | Orchestrate teams of Claude Code sessions (agent teams) | Documentation | Live page, read 2026-11-23 | https://code.claude.com/docs/en/agent-teams | V | Feature still experimental |
| A14 | Anthropic docs | Manage costs effectively (Claude Code) | Documentation | Live page, read 2026-11-23 | https://code.claude.com/docs/en/costs | V | Current |
| A15 | Anthropic docs | Context editing | Documentation | Live page, read 2026-11-23 | https://platform.claude.com/docs/en/build-with-claude/context-editing | V (first 100K chars) | Current |

### Other labs, vendors and practitioners

| # | Author(s) | Title | Type | Date | URL | Verif. | Status |
|---|---|---|---|---|---|---|---|
| B1 | Tobi Lütke | X post endorsing "context engineering" | Social post | 2025-06-19 | Quoted at https://simonwillison.net/2025/Jun/27/context-engineering/ | S | Definition still the standard citation |
| B2 | Andrej Karpathy | X post "+1 for context engineering" | Social post | 2025-06-25 | Quoted at https://simonwillison.net/2025/Jun/27/context-engineering/ | S | Same |
| B3 | Kelly Hong, Anton Troynikov, Jeff Huber (Chroma) | Context Rot: How Increasing Input Tokens Impacts LLM Performance | Research report + repo | 2025-07-14 | https://research.trychroma.com/context-rot | S | Models tested (GPT-4.1, Claude 4, Gemini 2.5, Qwen3) are now 1-2 generations old; no replication on 2026 models found |
| B4 | Lance Martin (LangChain) | Context Engineering for Agents | Blog | 2025-06-23 (LangChain version July 2025) | https://rlancemartin.github.io/2025/06/23/context_engineering and https://www.langchain.com/blog/context-engineering-for-agents | S | Taxonomy still widely used |
| B5 | Drew Breunig | How Contexts Fail / How to Fix Your Context | Blog | 2025-06-22 (+ follow-up, June 2025) | https://dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html | S | Failure-mode names still used; evidence partly from small 2024-era models |
| B6 | Yichao "Peak" Ji (Manus) | Context Engineering for AI Agents: Lessons from Building Manus | Engineering blog | 2025-07-18 | https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus | S | Cache guidance now mirrored by Anthropic's append-only advice (A10/A11) |
| B7 | Dex Horthy (HumanLayer) | Advanced context engineering for coding agents; "Context Engineering for Complex Codebases" (AI Engineer talk) | Essay + talk | Aug 2025 essay; 2025 talk | https://ai.engineer/talks/context-engineering-for-complex-codebases | S | The ~40% "dumb zone" rule of thumb predates 1M-token windows; treat as heuristic |
| B8 | Walden Yan (Cognition) | Don't Build Multi-Agents | Blog | June 2025 | https://cognition.ai/blog/dont-build-multi-agents | S | Refined, not retracted, by B9 |
| B9 | Walden Yan (Cognition) | Multi-Agents: What's Actually Working | Blog | 2026-04-22 | https://cognition.ai/blog/multi-agents-working | S | Current |
| B10 | Yubin Kim et al. (Google Research, Google DeepMind, MIT) | Towards a Science of Scaling Agent Systems | arXiv paper + Google Research blog | arXiv Dec 2025; blog 2026-01-28 | https://arxiv.org/abs/2512.08296 and https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/ | S | Current |
| B11 | Steve Yegge | Welcome to Gas Town; Gas Town: from Clown Show to v1.0 | Blog (Medium) + open-source project | Open-sourced 2026-01-01; v1.0 April 2026 | https://steve-yegge.medium.com/gas-town-from-clown-show-to-v1-0-c239d9a407ec | S | Outdated as a working system: Yegge says it "fell apart" on Opus 4.7 (B12) |
| B12 | Simon Willison (quoting Yegge) | Weblog entry on Steve Yegge | Blog | 2026-08-04 | https://simonwillison.net/2026/Aug/4/steve-yegge/ | S | Current |
| B13 | OpenAI docs | Codex subagents | Documentation | Live 2026 | https://developers.openai.com/codex/subagents | S | Current |
| B14 | Ryan Lopopolo (OpenAI) | Harness engineering: leveraging Codex in an agent-first world | Blog | 2026-02-11 | https://openai.com/index/harness-engineering/ | S | Current |
| B15 | LangChain | Improving Deep Agents with harness engineering | Blog | 2026 (exact date not seen) | https://blog.langchain.com/improving-deep-agents-with-harness-engineering | S | Result is for GPT-5.2-Codex on Terminal Bench 2.0 |
| B16 | (survey authors not captured) | From Question Answering to Task Completion: A Survey on Agent System and Harness Design | arXiv | 2026 (id 2606.20683) | https://arxiv.org/pdf/2606.20683 | S | Current |
| B17 | Mario Zechner | What I learned building an opinionated and minimal coding agent (Pi) | Blog | 2025-11-30 | https://www.mariozechner.at/posts/2025-11-30-pi-coding-agent/ | S | Current; also AI Engineer Europe 2026 talk |
| B18 | Armin Ronacher | Pi (blog post on the Pi harness) | Blog | 2026-01-31 | https://lucumr.pocoo.org/2026/1/31/pi/ | S (title/URL only) | Current |
| B19 | SWE-agent team (Princeton/Stanford) | mini-SWE-agent | Open-source repo | 2025, updated 2026 | https://github.com/SWE-agent/mini-swe-agent | S | Scores keep rising with newer models |
| B20 | OpenAI | GPT-5 prompting guide | Cookbook | Aug 2025 | https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide | S | Superseded by B21 for current models |
| B21 | OpenAI | GPT-5.5 prompting guide (and a later GPT-5.6 guide) | Documentation | 2026 | https://developers.openai.com/api/docs/guides/prompt-guidance.md | S | Current |
| B22 | Google | Gemini 3 prompting guide; Gemini 3 developer guide | Documentation | Late 2025, live | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/gemini-3-prompting-guide and https://ai.google.dev/gemini-api/docs/gemini-3 | S | Current for Gemini 3 |
| B23 | Philipp Schmid | Gemini 3 Prompting: Best Practices for General Usage | Blog | Late 2025 | https://www.philschmid.de/gemini-3-prompt-practices | S | Current for Gemini 3 |
| B24 | Arize; Smithery; Scalekit (vendor blogs) | MCP vs CLI evaluations | Vendor blogs | Feb-Mar 2026 | https://arize.com/blog/mcp-vs-cli-skills-for-agents-what-our-eval-found-and-which-you-should-use.md and https://smithery.ai/blog/mcp-vs-cli-is-the-wrong-fight | S | Small samples; vendor-run; indicative only |

---

## Key question 1 — Context engineering: definitions, context rot, techniques, how much context to use

### Takeaway
Context engineering is the practice of choosing what goes into the context window at each step. All major sources agree that more tokens are not free, because recall and reasoning degrade as input grows ("context rot"). The 2025 toolkit (compaction, notes on disk, sub-agent isolation, just-in-time retrieval, tool-result clearing, stable cacheable prefixes) is still current. What changed in 2026 is the emphasis: with 1M-token windows and automatic compaction, Anthropic's own Claude Code docs now say running out of context is rare, and the reason to keep context small is focus and cost, not survival.

### Cited Findings

**Definitions**
- Tobi Lütke (2025-06-19) called context engineering a better name than prompt engineering and described it as "the art of providing all the context for the task to be plausibly solvable by the LLM." [S] — [Simon Willison, 2025-06-27](https://simonwillison.net/2025/Jun/27/context-engineering/)
- Andrej Karpathy (2025-06-25): "+1 for 'context engineering' over 'prompt engineering'"; it is "the delicate art and science of filling the context window with just the right information for the next step." [S] — [Simon Willison](https://simonwillison.net/2025/Jun/27/context-engineering/)
- Anthropic defines it as the strategies for "curating and maintaining the optimal set of tokens (information) during LLM inference" and frames it as the natural progression of prompt engineering. Context is "a finite resource with diminishing marginal returns"; models have an "attention budget." [V] — [Anthropic A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- LangChain / Lance Martin group the techniques into four buckets: **write** (save context outside the window), **select** (pull relevant context in), **compress** (keep only needed tokens), **isolate** (split context across agents or sandboxes). [S] — [LangChain](https://www.langchain.com/blog/context-engineering-for-agents); [Lance Martin](https://rlancemartin.github.io/2025/06/23/context_engineering). Note: one cached version of Martin's post names three buckets ("compress, persist, isolate"); the four-bucket version is the one most sources cite. [S]

**Context rot and failure modes**
- Chroma's "Context Rot" report (Kelly Hong, Anton Troynikov, Jeff Huber, 2025-07-14) tested 18 models including GPT-4.1, Claude 4, Gemini 2.5 and Qwen3. Core finding: models do not use context uniformly, and performance becomes increasingly unreliable as input length grows, even on simple tasks. [S] — [Chroma](https://research.trychroma.com/context-rot); replication code at [github.com/chroma-core/context-rot](https://github.com/chroma-core/context-rot)
- The report argues that Needle-in-a-Haystack overstates long-context ability because it tests lexical matching; Chroma's tasks used needle-question pairs that need inference. Secondary coverage reports that shuffled haystacks outperformed logically structured ones across the 18 models. [S] — [Chroma](https://research.trychroma.com/context-rot); [PromptHub summary](https://www.prompthub.us/blog/why-long-context-windows-still-dont-work)
- Reported model differences conflict between secondary sources: The Decoder says Claude models abstain more while GPT models give confident wrong answers; a Substack review ranks models differently. Not verified against the primary report. [S] — [The Decoder](https://the-decoder.com/yet-another-study-finds-that-overloading-llms-with-information-leads-to-worse-results/)
- Anthropic uses the same term: as tokens grow, "the model's ability to accurately recall information from that context decreases." [V] — [Anthropic A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Drew Breunig names four ways contexts fail: **poisoning** (a hallucination or error enters context and is repeatedly referenced), **distraction** (context so long the model over-relies on it instead of its training), **confusion** (superfluous content, including extra tools, degrades the answer), **clash** (new information or tools conflict with what is already there). [S] — [Breunig](https://dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html)
- Breunig's evidence for tool confusion: a "Less is More" paper where Llama 3.1 8B failed a benchmark with 46 tools and succeeded with 19. [S] — [secondary summary of Breunig](https://agentic-ai.readthedocs.io/en/latest/ContextEngineering/challenges/). Flag: small 2024 model; do not generalize the numbers to frontier models.
- Breunig's fixes are usually listed as RAG, tool loadout (select only relevant tools), context quarantine (isolate in separate threads/agents), context pruning, context summarization, context offloading (store outside the window). [M] Only "quarantine = multi-agent isolation, at the cost of conversational continuity" was confirmed in search results. [S] — [Simon Willison on "How to fix your context"](https://simonwillison.net/2025/Jun/29/how-to-fix-your-context/)

**Techniques (Anthropic A1, verified)**
- **Right-altitude system prompt.** "The right altitude is the Goldilocks zone between two common failure modes": brittle hardcoded if-else logic versus vague high-level guidance. Start with a minimal prompt on the best model, then add instructions and examples based on observed failures. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Examples.** Curate "a set of diverse, canonical examples" instead of listing every edge case. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Just-in-time retrieval.** Keep lightweight identifiers (file paths, stored queries, links) and load data at runtime with tools. Claude Code is the cited example, with CLAUDE.md as the up-front hybrid exception. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Progressive disclosure.** Agents "incrementally discover relevant context through exploration"; file sizes, names and timestamps act as signals. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Compaction.** Summarize a conversation nearing the limit and restart from the summary. Tune the compaction prompt for recall first, then precision. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Tool-result clearing.** "One of the safest lightest touch forms of compaction": once a tool result is deep in history the raw output is rarely needed. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Structured note-taking.** Notes "persisted to memory outside of the context window" (to-do lists, NOTES.md), pulled back in later. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Sub-agent isolation.** A sub-agent may explore "using tens of thousands of tokens or more" and return a condensed summary, "often 1,000-2,000 tokens." [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

**API mechanics of clearing and compaction (Anthropic, verified)**
- Context editing runs server-side; the client keeps full history. Tool-result clearing defaults: trigger at 100,000 input tokens, keep the last 3 tool uses, oldest results replaced by placeholders. [V] — [A15](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- Clearing tool results invalidates the cached prefix, so `clear_at_least` should be set so each clear removes enough tokens to justify the cache rewrite. [V] — [A15](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- When context nears the clearing threshold Claude gets an automatic warning and can write important results to memory files first. Server-side compaction is described as the primary strategy for long conversations. [V] — [A15](https://platform.claude.com/docs/en/build-with-claude/context-editing)

**Caching as a context-engineering constraint**
- Manus (Peak Ji, 2025-07-18): KV-cache hit rate is "the single most important metric for a production-stage AI agent." Keep the prompt prefix stable (no timestamp at the start of the system prompt), make context append-only, make serialization deterministic. [S] — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Manus: "mask, don't remove" tools. Removing tool definitions mid-run breaks the cache and confuses the model when history references tools that no longer exist; constrain choice by masking logits instead. [S] — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Manus agents keep rewriting a todo.md so the plan is "recited" into recent context to hold attention. [S, via secondary summary] — [DEV summary](https://dev.to/contextspace_/context-engineering-for-ai-agents-key-lessons-from-manus-3f83)
- Other Manus lessons usually cited (use the file system as external memory, keep failed actions in context, avoid few-shot ruts) were not confirmed in this session. [M]
- Anthropic's 2026 guidance converges on the same rule for a different reason: on the newest models, editing earlier messages, rebuilding `system` or `tools`, or summarizing in place invalidates later thinking blocks, so "keep the history append-only"; this "keeps the cache hit rate high, and no thinking is dropped." Use server-side context editing or compaction to shorten history. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices); [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Deferred tools (tool search) do not break prompt caching. [V] — [A6](https://www.anthropic.com/engineering/advanced-tool-use)
- Claude Code: prompt cache lasts about an hour on a subscription and five minutes on API keys; sub-agents get five minutes unless changed. A first message after a break longer than the cache lifetime reprocesses the full context. [V] — [A12](https://code.claude.com/docs/en/best-practices); [A14](https://code.claude.com/docs/en/costs)

**How much context before quality drops**
- No source gives a universal threshold. Chroma shows degradation is continuous and task-dependent, not a cliff. [S] — [Chroma](https://research.trychroma.com/context-rot)
- Dex Horthy's practitioner rule: stay out of the "dumb zone"; diminishing returns may start around 40% utilization in Claude Code (another account gives 40-50%), and he says the threshold varies by model and task. His method is "frequent intentional compaction": research, plan, implement, with each phase distilled into a written artifact that the next fresh session reads. [S] — [AI Engineer talk page](https://ai.engineer/talks/context-engineering-for-complex-codebases); [Pragmatic Engineer podcast listing](https://newsletter.pragmaticengineer.com/p/context-engineering-with-dex-horthy)
- Anthropic's Claude Code docs (2026) list "The context window is your main constraint, manage it aggressively" under old advice: "Most current models have a 1M-token window and compaction happens on its own, so running out is rare. `/clear` between tasks is still worth doing, because a focused context gets you better answers and costs less." [V] — [A12](https://code.claude.com/docs/en/best-practices)
- Same page: after two failed corrections, run `/clear` and write a better prompt, because a third correction "mostly adds to a conversation that already contains two failed attempts." This is context poisoning handled by reset. [V] — [A12](https://code.claude.com/docs/en/best-practices)
- CLAUDE.md is loaded every session, so "every line in it takes a little of Claude's attention on every task." Keep only what Claude cannot work out from the code; the memory guide recommends under 200 lines. Move workflow-specific instructions to skills, which load on demand. [V] — [A12](https://code.claude.com/docs/en/best-practices); [A14](https://code.claude.com/docs/en/costs)
- Reset versus compaction: for multi-window work, "consider starting with a brand new context window rather than using compaction. Claude's latest models are extremely effective at discovering state from the local filesystem." [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Context anxiety: Sonnet 4.5 would "begin wrapping up work prematurely" as it approached what it believed was its limit; full context resets were needed. Opus 4.5 largely removed the behavior, and compaction alone then sufficed. [V] — [A8](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- A token budget shown to the model changes how much the newest model spends; on coding tasks at high/xhigh effort, removing a large displayed budget lowered cost with equivalent performance. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)

### Inferences
- The field moved from "fit within the window" (2025) to "keep the window focused and the prefix stable" (2026). The techniques are the same; the justification shifted from hard limits to attention quality, cost and cache economics.
- Two independent constraints now push toward append-only histories: cache hit rate (Manus, 2025) and preserved thinking blocks (Anthropic, 2026). Harness code that rewrites history mid-run is the main thing to audit.
- "How much context" has no fixed number. A defensible rule from the sources: compact or reset at task boundaries rather than at a percentage, and prefer a reset plus on-disk state when the model can rediscover state from files.
- Horthy's 40% heuristic and Anthropic's "running out is rare" are not in direct conflict: one is about quality as context fills, the other about hitting the limit. But Anthropic's wording does retire aggressive manual context management as a default habit.

### Gaps
- Could not open the Chroma report; all numbers and per-model behavior are from secondary summaries. No evidence found of Chroma (or others) re-running context-rot tests on 2026 frontier models with 1M windows.
- Anthropic's published performance numbers for context editing plus the memory tool (search was cut off by budget) were not retrieved; the context-editing docs page contains no benchmarks.
- Breunig's fix list and several Manus lessons are from memory, not verified here.
- No primary-source text from Dex Horthy's essay was read; the 40% figure comes from talk summaries.

---

## Key question 2 — Multi-agent orchestration: patterns, when it helps or hurts, costs, briefing, merging

### Takeaway
The 2025 dispute (Anthropic's multi-agent research system versus Cognition's "Don't build multi-agents") has largely resolved into one rule: split work only where context can be truly isolated, parallelize reads (research, review, verification), and keep writes single-threaded. Multi-agent systems win mainly by spending more tokens on breadth; they cost roughly 3-15x a single agent and amplify errors when agents work independently without a checking orchestrator.

### Cited Findings

**Anthropic's research system (2025-06-13, verified)**
- Architecture: orchestrator-worker. A lead agent plans, spawns parallel sub-agents, synthesizes; a separate CitationAgent attributes claims. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- Result: Opus 4 lead + Sonnet 4 sub-agents beat single-agent Opus 4 by 90.2% on an internal research eval. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- Why it works: on BrowseComp, three factors explained 95% of variance; token usage alone explained 80%. "Multi-agent systems work mainly because they help spend enough tokens to solve the problem." [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- Cost: agents use about 4x the tokens of chat; multi-agent systems about 15x. Only worth it for high-value tasks. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- Poor fit: domains needing shared context or many inter-agent dependencies; "most coding tasks" have fewer truly parallelizable parts. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- How to brief a sub-agent: each needs "an objective, an output format, guidance on the tools and sources to use, and clear task boundaries." Vague one-line briefs caused duplicated and misread work. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- Scale effort in the prompt: simple fact-finding = 1 agent with 3-10 tool calls; direct comparisons = 2-4 sub-agents with 10-15 calls each; complex research = more than 10 sub-agents. An early failure was spawning 50 sub-agents for simple queries. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- Search strategy: start with short broad queries, then narrow. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- Parallelism: lead spawns 3-5 sub-agents in parallel and sub-agents call 3+ tools in parallel; research time cut by up to 90% on complex queries. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- Merging results: sub-agents write outputs to the filesystem and pass lightweight references to the coordinator, avoiding a "game of telephone" and token overhead. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- Known bottleneck: execution is synchronous; the lead waits for each batch, cannot steer mid-task, and one slow sub-agent blocks the system. Async adds problems of result coordination, state consistency and error propagation. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- Evaluation: start with about 20 real queries; judge outcomes, not paths; a single LLM-judge call with a rubric (factual accuracy, citation accuracy, completeness, source quality, tool efficiency) and a 0.0-1.0 score worked best; human testing still caught failures such as preference for SEO content farms. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
- A tool-testing agent that rewrote flawed MCP tool descriptions cut task completion time by 40%. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)

**Anthropic's 2026 decision guide (2026-01-23, verified)**
- Start with one agent; a well-designed single agent "can accomplish far more than many developers expect." Teams have spent months on multi-agent designs that better single-agent prompting matched. [V] — [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
- Three cases where multi-agent consistently wins: **context protection**, **parallelization**, **specialization**. "Outside these situations, the coordination costs typically exceed the benefits." [V] — [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
- Cost: "multi-agent implementations typically use 3-10x more tokens than single-agent approaches" for equivalent tasks, from duplicated context, coordination messages and handoff summaries. Wall-clock time is often longer overall. [V] — [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
- The "primary benefit of parallelization is thoroughness, not speed." [V] — [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
- Context isolation pays when a subtask generates more than about 1,000 tokens that are mostly irrelevant to the main task. [V] — [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
- Specialization signals: 20+ tools, confusion across unrelated domains, conflicting behavioral modes. Try the Tool Search Tool first for the many-tools case. [V] — [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
- **Context-centric decomposition**: split by context boundaries, not by role. Role splits (one writes features, one writes tests, one reviews) create a "telephone game." Good boundaries: independent research paths, components with clean interfaces, black-box verification. Bad boundaries: sequential phases of the same work, tightly coupled components, shared state. "Work should only be split when context can be truly isolated." [V] — [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
- Verification sub-agent: works because verification "requires minimal context transfer by nature." Main failure is the "early victory problem" (runs a few tests, declares success); fix with concrete criteria, negative tests, and an explicit instruction to run the complete suite. More capable orchestrators increasingly evaluate sub-agent work directly. [V] — [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)

**Cognition's position and its 2026 refinement (search summaries only)**
- "Don't Build Multi-Agents" (Walden Yan, June 2025) argued that splitting work fragments context and decisions; its commonly cited principles are "share context, including full agent traces" and "actions carry implicit decisions, and conflicting decisions carry bad results." [M for the exact principles; S for the post's existence and thrust] — [Cognition B8](https://cognition.ai/blog/dont-build-multi-agents)
- "Multi-Agents: What's Actually Working" (2026-04-22) keeps the objection to parallel-writer swarms ("most of the sexy ideas in that space still don't see meaningful adoption") but says a narrower class works: agents "contribute intelligence while writes stay single-threaded." [S] — [Cognition B9](https://cognition.ai/blog/multi-agents-working)
- Same post: sub-agents "mostly resemble tool calls rather than true multi-agent collaboration"; a coder + reviewer loop works best "when the coding and review agents do not share any context beforehand." [S] — [Cognition B9](https://cognition.ai/blog/multi-agents-working)
- No separate 2026 retraction was found; B9 is a refinement. [S] — [Latent Space wiki on Walden Yan](https://latent.space/wiki/walden-yan-7c572c23/)

**Scaling research (Google Research / DeepMind / MIT, search summaries only)**
- "Towards a Science of Scaling Agent Systems" (Yubin Kim et al., arXiv Dec 2025; blog 2026-01-28) ran 180 configurations across five architectures and three model families. Multi-agent gains are task-dependent; "more agents" often plateaus or degrades. [S] — [arXiv 2512.08296](https://arxiv.org/abs/2512.08296); [Google Research blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
- Error amplification: independent agents amplified errors about 17.2x versus baseline; centralized (orchestrator) systems held it to about 4.4x, because the orchestrator acts as a "validation bottleneck." [S] — [MIT Media Lab project page](https://www.media.mit.edu/projects/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/overview/)
- The paper's predictive model picked the best coordination strategy for 87% of unseen task configurations. [S] — [Google Research blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
- The widely repeated details (large gains on parallelizable tasks, large losses on sequential planning tasks, a single-agent accuracy level above which adding agents stops helping, a tool-heavy-task coordination penalty) were not returned by search in this session and are omitted rather than quoted from memory.

**Claude Code: sub-agents, agent teams, messaging (verified)**
- Sub-agents: own context window, return a result to the caller, lower token cost. Agent teams: independent sessions that share a task list, claim work and message each other directly; higher cost. [V] — [A13](https://code.claude.com/docs/en/agent-teams)
- Teams are experimental and off by default (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`). Components: team lead, teammates, shared task list (pending / in progress / completed, with dependencies and file-locked claiming), and a mailbox (a JSON inbox file per agent). [V] — [A13](https://code.claude.com/docs/en/agent-teams)
- Best uses: research and review, new modules with separate ownership, debugging with competing hypotheses, cross-layer changes. Poor uses: sequential tasks, same-file edits, many dependencies. [V] — [A13](https://code.claude.com/docs/en/agent-teams)
- Competing-hypotheses pattern: have teammates try to disprove each other, "like a scientific debate"; the stated reason is that sequential investigation suffers from anchoring. [V] — [A13](https://code.claude.com/docs/en/agent-teams)
- Sizing: start with 3-5 teammates; 5-6 tasks per teammate; "three focused teammates often outperform five scattered ones." Token cost scales linearly with teammates. [V] — [A13](https://code.claude.com/docs/en/agent-teams)
- Briefing: teammates load CLAUDE.md, MCP servers and skills but do not inherit the lead's conversation, so put task-specific detail in the spawn prompt (target paths, focus areas, relevant facts, required report format). [V] — [A13](https://code.claude.com/docs/en/agent-teams)
- Conflict avoidance: each teammate owns a different set of files; two editing one file leads to overwrites. Start with read-only tasks (review, research) before parallel implementation. [V] — [A13](https://code.claude.com/docs/en/agent-teams)
- Trust boundary: a message from another agent is marked as coming from another Claude session, not the user; a teammate cannot approve a permission prompt or supply consent. [V] — [A13](https://code.claude.com/docs/en/agent-teams)
- Limits: no nested teams, one team per session, lead is fixed, task status can lag, no resumption of in-process teammates. Quality gates via hooks (`TeammateIdle`, `TaskCreated`, `TaskCompleted`). [V] — [A13](https://code.claude.com/docs/en/agent-teams)
- Cost: "Agent teams use approximately 7x more tokens than standard sessions when teammates run in plan mode." Use a cheaper model for teammates, keep teams small, shut teammates down when done. [V] — [A14](https://code.claude.com/docs/en/costs)
- What sub-agents are for in 2026: parallel work, keeping noisy output out of the main thread, a reviewer "that hasn't seen how the code was written," and tool restriction. "Handing the same task to a subagent doesn't make the answer any better." A fresh sub-agent must re-gather context and can be slower. [V] — [A12](https://code.claude.com/docs/en/best-practices)
- Reviewer briefing: tell the reviewer to "report gaps that affect correctness and skip style," because "a reviewer that's asked to find problems will always find something," and fixing every finding produces over-engineering. [V] — [A12](https://code.claude.com/docs/en/best-practices)
- Enforce a read-only reviewer with a tool allowlist (`tools: Read, Grep, Glob`), not with a prompt request. [V] — [A12](https://code.claude.com/docs/en/best-practices)
- Newer models over-delegate: Opus 4.6 "has a strong tendency to use subagents" where a direct grep is faster; Opus 5 delegates more readily still. Sample damping text: use sub-agents "when tasks can run in parallel, require isolated context, or involve independent workstreams that don't need to share state"; otherwise work directly. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- For the newest model, the sub-agent tool description should "state what a subagent adds to the cost of a session, say when delegation is worth it and when it isn't, and tell the model how to brief a subagent." A longer cost-focused description roughly halved the number of sub-agents at high effort. Early testers had good results using it as planner/orchestrator for other models; at xhigh/max it used sub-agents to review its own work. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Claude Code also has "dynamic workflows" in which each spawned agent sends its own requests on top of the main conversation. Only this cost note was seen; the workflows page itself was not read. [V for the cost note] — [A14](https://code.claude.com/docs/en/costs)

**OpenAI Codex sub-agents (search summaries only)**
- Config lives under an `[agents]` table: `agents.max_threads` defaults to 6 open agent threads; `agents.max_depth` defaults to 1 (root is depth 0; a child may spawn, deeper nesting is blocked). Custom agents are one TOML file each in `~/.codex/agents/` or `.codex/agents/`. [S] — [OpenAI Codex docs](https://developers.openai.com/codex/subagents)
- The docs warn that raising limits "can turn broad delegation instructions into repeated fan-out, which increases token usage, latency, and local resource consumption." [S] — [OpenAI Codex docs](https://developers.openai.com/codex/subagents)

**Gas Town (Steve Yegge; search summaries only)**
- Open-source orchestrator for many coding agents, built on Beads (a git-stored task ledger of JSON records under `.beads/`). Roles: a Mayor that turns intent into tasks, Polecats (workers with persistent identities and temporary sessions in isolated git worktrees), a Witness overseeing workers in a project, a Refinery handling merges; work is batched in "convoys." Beads released Oct 2025; Gas Town open-sourced 2026-01-01; both v1.0 in April 2026. [S] — [Yegge on Medium](https://steve-yegge.medium.com/gas-town-from-clown-show-to-v1-0-c239d9a407ec); [Steve Klabnik's explainer](https://steveklabnik.com/writing/how-to-think-about-gas-town/)
- Cost: about 20 concurrent Claude instances can burn hundreds of dollars in a weekend, per one community write-up. [S] — [community write-up](https://daviderutigliano.github.io/portfolio-website/posts/2026/01/gas-town-the-industrial-revolution-of-vibe-coding/)
- Outcome: Yegge, quoted by Simon Willison on 2026-08-04: "Gas Town fell apart at the seams with Opus 4.7. Up through 4.6 it was working brilliantly." He blames a "just two more things" tic that kept the model from converging on real work, and says the tool "only ever wound up" being used to build itself. A successor called Wheelhouse is mentioned in one secondary essay. [S] — [Simon Willison](https://simonwillison.net/2026/Aug/4/steve-yegge/)

### Inferences
- Anthropic and Cognition now state compatible rules in different words: Anthropic's "split only where context can be truly isolated" and Cognition's "writes stay single-threaded" both permit parallel readers, reviewers and verifiers and both reject parallel writers on coupled work. The Google/MIT error-amplification result (independent 17.2x versus centralized 4.4x) gives the quantitative reason an orchestrator that checks work matters.
- Published cost multipliers span 3-10x (Anthropic 2026, equivalent tasks), about 7x (Claude Code teams in plan mode) and about 15x versus chat (Anthropic 2025 research system). They use different baselines, so they should not be averaged; "several times a single agent" is the safe statement.
- A clean-context reviewer is the one multi-agent pattern endorsed by Anthropic, Cognition and Claude Code docs alike. Its value comes from independence, not from extra capability.
- Gas Town is a caution about heavy orchestration scaffolding tied to one model's behavior: it worked for months, then a model update broke convergence.
- Briefing checklist that the sources support: objective, output format, tools and sources to use, boundaries, enough task facts to stand alone (the sub-agent has no conversation history), effort budget, and for reviewers an explicit scope ("correctness, skip style").

### Gaps
- Cognition's two posts and the Google paper could not be opened; their content here is from search summaries. The Google paper's task-level gain/loss percentages and saturation threshold are omitted.
- No verified detail on Claude Code "workflows" beyond its cost note, and none on OpenAI's own guidance for briefing Codex sub-agents.
- No source found with controlled evidence for agent-to-agent messaging (peer mesh) outperforming hub-and-spoke on production tasks; the Claude Code docs describe the mechanism and use cases but give no measurements.
- Async orchestration (steering sub-agents mid-task) was flagged by Anthropic in 2025 as unsolved; no 2026 write-up with results was found.

---

## Key question 3 — Harness engineering: what it is, harness versus model, minimal harnesses, removing scaffolding, long-running designs

### Takeaway
The harness is everything around the model: system prompt, tools, context management, control loop, permissions, hooks and verification. Harness changes alone have produced double-digit benchmark gains with a fixed model, yet the strongest 2026 guidance is to treat every harness component as a bet on a model weakness and delete it when the model catches up. Minimal harnesses (bash plus a few file tools) are competitive on coding benchmarks, and long-running work is handled with on-disk state, verification tools and independent evaluators rather than elaborate control flow.

### Cited Findings

**What a harness is, and deterministic versus non-deterministic control**
- Claude Code docs split instructions into non-deterministic (prompt, CLAUDE.md, skills: the model decides) and deterministic (hooks, deny rules, managed settings: the harness enforces). "If you keep adding emphasis to a CLAUDE.md line and Claude still misses it, that's a sign the line belongs in a hook." The reverse also holds: delete hooks that only encode preferences. [V] — [A12](https://code.claude.com/docs/en/best-practices)
- "Give Claude a check to run": a pass/fail signal (tests, build exit code, linter, fixture diff, screenshot beside a mockup) lets the agent close its own loop. "If your repo already has one of these, the single most useful line in your CLAUDE.md is the command that runs it." [V] — [A12](https://code.claude.com/docs/en/best-practices)
- OpenAI's "Harness engineering" post (Ryan Lopopolo, 2026-02-11) describes building a product with no hand-written code from late August 2025; the engineers' job became designing environments and feedback loops. One large AGENTS.md "failed in predictable ways"; the fix was a short AGENTS.md (about 100 lines) used as a map / table of contents into a structured `docs/` directory treated as the system of record (design docs, execution plans, references, ARCHITECTURE.md). [S] — [OpenAI](https://openai.com/index/harness-engineering/). The "roughly one million lines" figure appeared only in a partial non-English snippet and is unverified.

**Harness versus model**
- LangChain improved its Deep Agents coding agent from 52.8% to 66.5% on Terminal Bench 2.0 with the model fixed (GPT-5.2-Codex), changing only system prompt, tools and middleware; it moved from outside the top 30 to the top 5. Reasoning budget mattered: xhigh-only scored 53.9% (timeouts) versus 63.6% at high. [S] — [LangChain](https://blog.langchain.com/improving-deep-agents-with-harness-engineering)
- Counterpoint: a 2026 arXiv survey on harness design argues the benchmark "does not support a harness-only interpretation"; terminal agents still need strong planning, coding and tool-use priors from the model. [S] — [arXiv 2606.20683](https://arxiv.org/pdf/2606.20683)
- Anthropic 2025: small refinements to tool descriptions helped Claude Sonnet 3.5 reach state of the art on SWE-bench Verified. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)

**Minimal harnesses**
- mini-SWE-agent: about 100 lines for the agent class, bash as the only tool, no tool-calling interface. The README claims >74% on SWE-bench Verified (PyPI text ties 74% to Gemini 3 Pro); earlier releases claimed 65% and >70%. Self-reported; the "100 lines" excludes environment and model code. [S] — [GitHub](https://github.com/SWE-agent/mini-swe-agent)
- Pi (Mario Zechner, 2025-11-30): four default tools (read, write, edit, bash) and a system prompt under 1,000 tokens. Thesis: frontier models already understand agentic coding, so the harness should stay out of the way; "context engineering is paramount"; existing harnesses inject content "behind your back that isn't even surfaced in the UI." Pi is designed so the agent can modify Pi itself. [S] — [Zechner](https://www.mariozechner.at/posts/2025-11-30-pi-coding-agent/); [ZenML summary](https://zenml.io/llmops-database/building-pi-a-minimal-extensible-coding-agent-framework)
- Zechner's Terminal-Bench claims for Pi are his own and were not independently checked. [S] — [second-brain summary](https://second-brain-nuxt.vercel.app/pi-coding-agent-minimal-agent-harness)
- Pi's deliberate omissions (no built-in MCP, sub-agents, plan mode or to-do tool; use files and bash instead) are widely reported but were not confirmed in this session. [M]

**The bitter lesson applied to harnesses (remove scaffolding as models improve)**
- "Every component in a harness encodes an assumption about what the model can't do on its own," and those assumptions go stale. [V] — [A8](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Worked example in the same post: Sonnet 4.5 needed full context resets; Opus 4.5 removed that need; with Opus 4.6 the sprint construct was deleted and the evaluator moved to one end-of-run pass. Planner and evaluator were kept. [V] — [A8](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Claude Code's "What you can stop doing" table retires: aggressive context management, ALL-CAPS emphasis, sub-agents purely to protect context, delegating "for a better answer," "think hard" phrases (effort is the control), "keep going / don't be lazy" lines (they now cause over-work), and describing file layout in CLAUDE.md. "Most were workarounds for something the model couldn't reliably do yet." [V] — [A12](https://code.claude.com/docs/en/best-practices)
- Anthropic's newest model guide has a section on reviewing "prompts and harnesses designed for earlier models": remove standing delegation instructions, displayed token budgets, history-editing harness code, "think less" lines, rigid style rules and accumulated cautionary notes. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Gas Town breaking on Opus 4.7 after working through 4.6 is a practitioner example of scaffolding that did not survive a model change. [S] — [Simon Willison](https://simonwillison.net/2026/Aug/4/steve-yegge/)
- No source located in this session explicitly uses Sutton's "bitter lesson" wording for harnesses; the principle is stated in Anthropic's own terms above.

**Long-running harness designs**
- Initializer + coding agent (Nov 2025): the first session writes a feature list (200+ features for a claude.ai clone, all marked failing, in JSON), an `init.sh`, a `claude-progress.txt` and an initial commit. Every later session does one feature, then commits and updates progress. The two "agents" differ only in the first user prompt. [V] — [A7](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Session start routine: `pwd`; read git log and progress file; read the feature list and pick the top unfinished item; start the server via `init.sh` and run a basic end-to-end test. [V] — [A7](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Failure modes addressed: one-shotting the app and running out of context mid-feature; declaring victory early; marking features done without testing. Guard instruction: "It is unacceptable to remove or edit tests because this could lead to missing or buggy functionality." JSON was chosen because the model is less likely to overwrite it than Markdown. [V] — [A7](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Browser automation for end-to-end testing "dramatically improved performance" over unit tests or curl. [V] — [A7](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- The same pattern is in the current prompting docs: a different prompt for the first context window, tests in a structured file, setup scripts, git for state, JSON for structured state and free text for progress notes. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Planner / generator / evaluator (Mar 2026): planner expands a 1-4 sentence prompt into a product spec and stays high-level to avoid cascading technical errors; generator builds; evaluator drives the running app with Playwright and grades against hard thresholds. Agents communicate through files. Before coding, generator and evaluator agree a "sprint contract" defining done and how to verify it. [V] — [A8](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Why a separate evaluator: "agents reliably skew positive when grading their own work"; a separate evaluator was easier to tune toward skepticism than a self-critical generator. Out of the box Claude was "a poor QA agent" that found issues and then talked itself into approving; fixing it took several rounds of reading evaluator logs and updating the QA prompt, with few-shot calibration. [V] — [A8](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Cost data: solo run 20 min / $9 with a broken core feature; full harness 6 hr / $200 with clearly better output (Opus 4.5). Simplified harness on Opus 4.6: about 3 hr 50 min and $124.70. The evaluator's value "depends on whether a task sits beyond what the model reliably does solo." [V] — [A8](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Newest-model behavior for long runs: the model may end a turn early (listing what remains, offering to continue), more often at low/medium effort. A "finishing work" system-prompt block roughly halved unfinished-but-continuable sessions at about 30% more output tokens. For unattended runs, Claude Code uses a goal condition checked by a separate model or a Stop hook. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2); [A12](https://code.claude.com/docs/en/best-practices)

### Inferences
- "Harness beats model" and "delete your harness" are both supported, at different layers. The durable harness parts are the ones that give the model information or feedback it cannot get otherwise (verification tools, on-disk state, an independent evaluator, deterministic enforcement). The perishable parts are those that script the model's process (sprints, forced resets, anti-laziness prompts, mandatory delegation).
- A practical test from A8: for each component, name the model limitation it compensates for, then re-run without it on each new model.
- The LangChain result and the survey's rebuttal can both be true: harness tuning moves a fixed model a long way on a benchmark, but the ceiling is set by the model.
- Benchmarked minimal harnesses are coding-only. Nothing found shows that a bash-only harness is sufficient for non-coding enterprise agents that need permissions, auditing or third-party systems.

### Gaps
- Zechner's post, Ronacher's post and the LangChain and OpenAI posts were not opened; details are from search summaries. Pi's omitted features are from memory.
- Philipp Schmid's writing on agent harnesses and Hamel Husain's on agent evals/error analysis were not retrieved (search budget exhausted).
- No independent replication found of "harness-only" benchmark gains across several models.

---

## Key question 4 — Tool design: few clear tools, token-efficient output, teaching errors, MCP vs CLI vs code execution, skills, tool search

### Takeaway
Design tools for how an agent works, not as wrappers over API endpoints: few, non-overlapping, clearly named and described, returning high-signal and bounded output, with errors that say how to fix the call. For large tool sets, load definitions on demand (tool search, skills, code execution) instead of up front. The MCP-versus-CLI argument is unresolved on data, but there is agreement that up-front schema cost is the real problem and that deferred loading narrows it.

### Cited Findings

**Choosing and shaping tools (Anthropic A3, verified)**
- "More tools don't guarantee better outcomes"; do not simply wrap existing API endpoints. Consolidate around tasks: `list_users` + `list_events` + `create_event` becomes `schedule_event`; `read_logs` becomes `search_logs` returning only relevant lines with context; `get_customer_by_id` + `list_transactions` + `list_notes` becomes `get_customer_context`. Prefer `search_contacts` over `list_contacts`. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Namespacing: group by service and resource (`asana_search`, `asana_projects_search`). Prefix versus suffix naming had measurable eval effects, so test it. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Return meaningful context: names and other high-signal fields rather than `uuid` or `mime_type`. Resolving UUIDs to natural-language names or 0-indexed IDs improved precision by reducing hallucinations. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Offer a `response_format` enum (`DETAILED` / `CONCISE`): the example detailed response was 206 tokens, concise 72. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Token efficiency: pagination, range selection, filtering and truncation with sensible defaults; Claude Code caps tool responses at 25,000 tokens by default. When truncating, tell the agent a better strategy (several small targeted searches). [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Errors: replace opaque codes and tracebacks with specific, actionable guidance including an example of correct input. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Descriptions: write them as you would brief a new hire (formats, terminology, relationships between resources); use unambiguous parameter names (`user_id`, not `user`). Small refinements have large effects. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Process: prototype, build an eval of realistic multi-step tasks (dozens of tool calls) with verifiable outcomes, read transcripts, then let the agent refactor the tools from the transcripts; check on a held-out set. Agent-optimized tools beat expert-written ones on held-out tests (charts only, no figures in text). Example of a description bug: Claude appending "2025" to search queries. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Tools should be "self-contained, robust to error, and extremely clear with respect to their intended use"; ambiguous choice between overlapping tools is a common failure. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

**Too many tools, and tool search (Anthropic A6, verified)**
- Example setup: 58 tools from five MCP servers cost about 55K tokens of definitions; Anthropic had seen 134K tokens of tool definitions internally. [V for 55K; S for 134K] — [A6](https://www.anthropic.com/engineering/advanced-tool-use)
- Tool Search Tool: about 77K tokens up front traditionally versus about 8.7K with search (about 500 for the search tool plus about 3K for the 3-5 tools found), an 85% reduction. The page's "191,300 vs 122,800 tokens preserved" figures use an unclear baseline. [V] — [A6](https://www.anthropic.com/engineering/advanced-tool-use)
- Accuracy on MCP evals with tool search: Opus 4 49% to 74%; Opus 4.5 79.5% to 88.1%. [V] — [A6](https://www.anthropic.com/engineering/advanced-tool-use)
- Use when definitions exceed about 10K tokens, with 10+ tools, multiple MCP servers, or tool-selection errors. Keep the 3-5 most-used tools always loaded. Less useful under 10 tools. [V] — [A6](https://www.anthropic.com/engineering/advanced-tool-use)
- Tool Use Examples (1-5 realistic examples per tool, focused on ambiguous cases) raised accuracy on complex parameters from 72% to 90%. [V] — [A6](https://www.anthropic.com/engineering/advanced-tool-use)
- In Claude Code, MCP tool definitions are now deferred by default: only tool names and server instructions enter context until a tool is used. [V] — [A14](https://code.claude.com/docs/en/costs)

**Code execution and programmatic tool calling (verified)**
- Present MCP servers as a filesystem of typed function files that the agent imports in code; the agent reads only the files it needs, and intermediate data stays in the sandbox. Headline example: 150,000 tokens down to 2,000 (98.7%). A 2-hour meeting transcript passed through the model twice adds about 50,000 tokens in the direct-call approach. [V] — [A5](https://www.anthropic.com/engineering/code-execution-with-mcp)
- Benefits: filter large results before the model sees them (5 rows instead of 10,000), loops and conditionals in code, privacy (tokenize PII before the model), persistent state and reusable functions that can become skills. Cost: a secure sandbox, resource limits and monitoring. [V] — [A5](https://www.anthropic.com/engineering/code-execution-with-mcp)
- Programmatic Tool Calling: average tokens fell from 43,588 to 27,297 (37%) on complex research tasks; accuracy 25.6% to 28.5% on internal knowledge retrieval and 46.5% to 51.2% on GAIA-style benchmarks (page text reads "GIA"); 20+ tool calls in one code block removes 19+ inference passes. Not worth it for single calls or when the model must reason over every intermediate result. [V] — [A6](https://www.anthropic.com/engineering/advanced-tool-use)

**Skills and progressive disclosure (verified)**
- A skill is a directory with a SKILL.md whose YAML frontmatter (`name`, `description`) is preloaded; the body loads only when relevant; linked files and scripts load only when needed. Three disclosure levels; no token counts are given in the post. Scripts give "deterministic reliability." [V] — [A4](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- Authoring: start from evals of where the agent struggles; split large SKILL.md files; put the most care into name and description because they decide triggering; have Claude capture what worked into the skill; audit third-party skills for code and network instructions. [V] — [A4](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- Skills load on demand, so move workflow instructions out of CLAUDE.md into skills. [V] — [A14](https://code.claude.com/docs/en/costs)
- Triggering language: models since Opus 4.5 are more responsive to the system prompt, so "CRITICAL: You MUST use this tool when..." now causes over-triggering; write "Use this tool when..." [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)

**MCP versus CLI**
- Anthropic's Claude Code docs: "Prefer CLI tools when available: Tools like `gh`, `aws`, `gcloud`, and `sentry-cli` are still more context-efficient than MCP servers because they don't add any per-tool listing." [V] — [A14](https://code.claude.com/docs/en/costs)
- The public debate peaked with Eric Holmes's "MCP is dead. Long live the CLI" (late Feb 2026). The most-cited number is one Scalekit comparison (1,365 tokens via CLI versus 44,026 via MCP for a trivial GitHub question); commentators note most articles cite that single data point. [S] — [search summary citing multiple blogs](https://smithery.ai/blog/mcp-vs-cli-is-the-wrong-fight)
- Contrary data: Smithery found successful CLI runs often used more total actions and billed tokens than MCP; a 48-trial benchmark scored CLI 90.08 versus MCP 86.95 on quality with MCP using fewer output tokens. Arize reported near-identical correctness across arms, with skills staying inside allowed tools more than 99% of the time. [S] — [Smithery](https://smithery.ai/blog/mcp-vs-cli-is-the-wrong-fight); [Arize](https://arize.com/blog/mcp-vs-cli-skills-for-agents-what-our-eval-found-and-which-you-should-use.md)
- All of these are vendor-run, small-sample studies; treat ratios as indicative. [S]
- Manus's cache-driven rule (do not add or remove tools mid-run; mask instead) is a constraint on any dynamic tool-loading design. [S] — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)

### Inferences
- The durable tool-design rules are about the agent's context: fewer choices, bounded outputs, human-readable identifiers, errors that steer. They hold regardless of transport (MCP, CLI or code).
- The three "too many tools" remedies sit on one spectrum of progressive disclosure: tool search (defer schemas), skills (defer instructions and scripts), code execution (defer both and keep intermediate data out of context). Choose by how much data flows between calls and whether a sandbox is affordable.
- MCP versus CLI is mostly a question of where the up-front token cost lands. With deferred loading now default in Claude Code, the strongest remaining CLI argument is that well-known CLIs need no schema at all because the model already knows them; the strongest MCP argument is typed, permissioned access to systems with no CLI.
- Emphatic tool-triggering language is now a liability on current models; it should be removed from tool descriptions and skills during migration.

### Gaps
- Zechner's "what if you don't need MCP" analysis and Ronacher's comments were not retrieved (search budget exhausted), so the practitioner side of the CLI argument is under-sourced here.
- No neutral, large-sample benchmark of MCP versus CLI versus code execution was found.
- No verified OpenAI or Google tool-design guide was retrieved for comparison with Anthropic's.

---

## Key question 5 — Prompting agents: system prompts and task briefs for current models

### Takeaway
Current guidance from Anthropic, OpenAI and Google converges: state the outcome, success criteria, constraints and the reason behind them in plain language, give the model a way to verify, and let it choose the path. Newer models follow instructions more literally, so old habits (emphasis words, anti-laziness lines, step-by-step process stacks, standing "be thorough" or "delegate" instructions) now cause over-triggering, over-work or excess caution and should be removed and re-tested per model.

### Cited Findings

**Anthropic general principles (verified)**
- Be clear and direct: treat the model as "a brilliant but new employee who lacks context on your norms and workflows." Test: show the prompt to a colleague with minimal context; "If they'd be confused, Claude will be too." If you want more than the literal task, ask for it explicitly. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Give context and motivation: "NEVER use ellipses" is weaker than explaining the output is read by a text-to-speech engine that cannot pronounce them; "Claude can generalize from the explanation." [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Examples: relevant, diverse, wrapped in `<example>` tags; 3-5 for best results; the model can critique or extend your example set. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Right altitude: avoid both hardcoded if-else prompts and vague guidance; start minimal and add from observed failures. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Action versus suggestion: "can you suggest some changes" yields suggestions; say "change this function" if you want edits. Sample `default_to_action` and `do_not_act_before_instructions` blocks are provided to set the default either way. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Avoid over-constraint and over-prompting: replace blanket defaults ("Default to using [tool]") with targeted guidance ("Use [tool] when it would enhance your understanding"); remove "If in doubt, use [tool]"; use the effort setting as the fallback control. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Over-engineering: Opus 4.5/4.6 tend to add files, abstractions and flexibility; a sample block limits scope, documentation, defensive coding and abstractions. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Parallel tool calls can be pushed to about 100% with an explicit instruction. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Research briefs: define success criteria, ask for cross-source verification, track competing hypotheses and confidence in a notes file. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Letting the model plan: with adaptive thinking and native sub-agent orchestration, "Claude handles most multistep reasoning internally"; explicit prompt chaining is for when you need to inspect intermediate outputs. The most common useful chain is draft, review against criteria, refine. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Decisiveness prompt for over-thinking: "choose an approach and commit to it. Avoid revisiting decisions unless you encounter new information." [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)

**Anthropic newest-model differences (verified, Claude Fable 5.2 guide)**
- Effort is the main dial: it sets thinking, number of tool calls, turn length and tokens. Start at `high`; most of the gain arrives by `high`; above it gains are uneven. Re-run the effort sweep on every model change. "Answer directly without deliberating" no longer stops thinking at medium and above; lower effort instead. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Ownership framing changes behavior: "You are the owner and long-term maintainer of this codebase" raised consolidation of duplicated code from under 10% of runs to about 75%, with 70-80% more output tokens and unrequested changes. Leave it out when minimal change is wanted. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Absolutes are taken literally: "The model treats an instruction that contains 'never' as absolute." Write "ask for approval before deleting files," not "never delete files." Name the routine steps that need no approval; "act more freely" is too vague. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Over-broad approval rules backfire: a rule listing "sending messages to people outside this conversation" made the model hold back about half of its routine requests to colleagues. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Caution accumulates in memory, notes and instruction files (some written by the model itself) and was seen "mostly when several agents worked together." Prune stale cautions; rewrite absolute rules with scope and reason. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Final reports: write "for an intelligent but low-context reader" who has not seen the tool calls; key points first; no invented abbreviations. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Progress updates: a harness-side reminder after several silent tool-call steps roughly halved the longest silent stretch at no noticeable token cost; a system-prompt version costs about 20% more output tokens. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Approvals: reply with the specific action approved ("Yes, run the migration on the staging database"), not "continue." [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Migration method: change one thing at a time and "measure its effect and its token cost on your own tasks." [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)

**Claude Code task-brief guidance (verified)**
- Tell the model what it cannot see from the repo (unwritten decisions, misleading code, how to verify), give it a check to run, and write a spec first, then build from it in a fresh session so the build starts "with a clean context and a written definition of done." [V] — [A12](https://code.claude.com/docs/en/best-practices)
- Specific beats vague: "improve this codebase" triggers broad scanning; "add input validation to the login function in auth.ts" does not. [V] — [A14](https://code.claude.com/docs/en/costs)
- Emphasis is retired: "piling on emphasis mostly makes Claude over-apply the rule"; if a rule cannot bend, make it a hook. [V] — [A12](https://code.claude.com/docs/en/best-practices)
- Meta-prompting / prompt audit: `/init` drafts CLAUDE.md from the codebase (then mostly delete lines); `/doctor prompt-audit` flags instructions written for older models, dead references and contradictions. [V] — [A12](https://code.claude.com/docs/en/best-practices)
- OpenAI's harness team had Codex write the initial AGENTS.md itself. [S] — [OpenAI](https://openai.com/index/harness-engineering/)

**OpenAI (search summaries only)**
- GPT-5 prompting guide (Aug 2025): calibrate "agentic eagerness" (proactivity versus waiting for direction); steer "tool preambles" (plans and progress notes before tool calls); use the model to improve its own prompts (metaprompting); the model follows instructions closely and is destabilized by vague or conflicting ones. [S] — [OpenAI cookbook](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide)
- GPT-5.5 guide (2026): shorter, outcome-first prompts usually beat process-heavy prompt stacks; set target outcome, success criteria, constraints and available context, then let the model choose the path. Reported secondary details: test low/medium effort first; treat old prompts as needing a fresh baseline. [S] — [OpenAI docs](https://developers.openai.com/api/docs/guides/prompt-guidance.md); [The Decoder](https://the-decoder.com/openai-says-old-prompts-are-holding-gpt-5-5-back-and-developers-need-a-fresh-baseline/)
- A later GPT-5.6 guide says GPT-5-class models "follow prompt contracts closely, so conflicting rules can create more instability than missing detail," and to review remaining instructions for contradictions. [S] — [OpenAI docs](https://developers.openai.com/api/docs/guides/prompt-guidance-gpt-5p6.md)
- Some blogs overstate this as OpenAI telling users to "drop" step-by-step prompting; the official wording seen is "usually work better." [S]

**Google (search summaries only)**
- Gemini 3: be concise and direct; the model "may over-analyze verbose or overly complex prompt engineering techniques used for older models." Keep temperature at the default 1.0; lowering it "may lead to unexpected behavior, looping, or degraded performance." Default output is terse, so request a chattier persona explicitly. Use thinking level LOW for latency. [S] — [Vertex AI guide](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/gemini-3-prompting-guide); [Gemini API guide](https://ai.google.dev/gemini-api/docs/gemini-3); [Philipp Schmid](https://www.philschmid.de/gemini-3-prompt-practices)

### Inferences
- All three labs now say the same thing in different words: outcome-first (OpenAI), concise and direct (Google), clear with context and motivation (Anthropic). Process scripts are a fallback, not a default.
- Literal instruction-following is the common cause behind most 2026 prompt failures in these docs: contradictions destabilize, "never" becomes absolute, emphasis over-triggers, ownership wording changes scope. Prompt maintenance (auditing for contradictions and stale rules) is now as important as prompt writing.
- A task brief that satisfies every source: goal and why it matters; what done looks like and how to check it; constraints with reasons; context the agent cannot discover; output format; what needs approval, stated as specific actions. Sub-agent briefs add tool/source guidance, boundaries and an effort budget.
- Reasoning effort has replaced prompt wording as the lever for depth on Anthropic and OpenAI models, and thinking level on Gemini.
- Anything in a prompt that was added to fix an older model's weakness should be re-tested, not carried over. This is the prompt-level form of the harness rule in key question 3.

### Gaps
- OpenAI and Google guides were not opened; exact wording, especially for GPT-5.5/5.6 and any Gemini agentic system-instruction template, is unverified.
- Meta-prompting (having the model rewrite a prompt) is supported here by the GPT-5 guide summary, Anthropic's example-generation tip and Claude Code's prompt-audit tooling; no source was found with measured gains from model-rewritten prompts.
- Practitioner guidance from Hamel Husain and others on prompt iteration through error analysis was not retrieved.

---

## Consolidated practices and open debates

### Takeaway
There is a stable core of about thirty practices that at least two independent credible sources support. The live disagreements are about how much structure to put around the model, not about the basic techniques.

### Cited Findings — consolidated practices
Each line cites the strongest source; verification tags as above.

**Context**
1. Treat context as a finite attention budget; add only what earns its place. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
2. Keep always-loaded instruction files short (under about 200 lines) and limited to what the model cannot infer; move procedures to on-demand skills. [V] — [A12](https://code.claude.com/docs/en/best-practices)
3. Retrieve just in time via identifiers and tools rather than preloading. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
4. Persist state outside the window: progress notes as free text, structured state as JSON, checkpoints in git. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
5. Clear old tool results before summarizing; tune compaction for recall first. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
6. Reset between unrelated tasks and after two failed corrections. [V] — [A12](https://code.claude.com/docs/en/best-practices)
7. Keep the prompt prefix stable and history append-only for cache hits (and, on the newest Claude models, valid thinking blocks). [S] [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus); [V] [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
8. Tell the model whether its context will be compacted so it does not wind down early. [V] — [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)

**Multi-agent**
9. Default to one agent; add agents only for context protection, parallel independent work, or specialization. [V] — [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
10. Split by context boundary, not by role. [V] — [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
11. Parallelize reads; keep writes single-threaded or give each writer its own files. [S] [Cognition B9](https://cognition.ai/blog/multi-agents-working); [V] [A13](https://code.claude.com/docs/en/agent-teams)
12. Brief every sub-agent with objective, output format, tools/sources, boundaries and effort scale; assume it knows nothing of the conversation. [V] — [A2](https://www.anthropic.com/engineering/multi-agent-research-system)
13. Have sub-agents return condensed summaries (about 1-2K tokens) or file references, not transcripts. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
14. Use an orchestrator that checks work; independent agents amplify errors far more. [S] — [Google/MIT](https://arxiv.org/abs/2512.08296)
15. Use a clean-context reviewer or verifier with explicit pass criteria and scope. [V] — [A12](https://code.claude.com/docs/en/best-practices)
16. Budget several times single-agent cost; use cheaper models for simple sub-agents; cap fan-out depth and width. [V] [A14](https://code.claude.com/docs/en/costs); [S] [OpenAI Codex docs](https://developers.openai.com/codex/subagents)
17. Small teams (3-5), right-sized tasks, shut agents down when done. [V] — [A13](https://code.claude.com/docs/en/agent-teams)

**Harness**
18. Give the agent a runnable pass/fail check; put its command in the instruction file. [V] — [A12](https://code.claude.com/docs/en/best-practices)
19. Enforce non-negotiable rules deterministically (hooks, permissions), not with stronger wording. [V] — [A12](https://code.claude.com/docs/en/best-practices)
20. For long runs: separate first-session setup from later sessions, one feature at a time, read-state-then-test at session start, commit and log at session end. [V] — [A7](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
21. Separate generation from evaluation; tune the evaluator for skepticism against real logs. [V] — [A8](https://www.anthropic.com/engineering/harness-design-long-running-apps)
22. Record which model limitation each harness component compensates for; re-test without it on every model upgrade. [V] — [A8](https://www.anthropic.com/engineering/harness-design-long-running-apps)
23. Keep the instruction file a short map into versioned docs rather than one large file. [S] — [OpenAI](https://openai.com/index/harness-engineering/)

**Tools**
24. Few, non-overlapping, task-shaped tools with namespaced names and new-hire-quality descriptions. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
25. Bounded, high-signal outputs: readable identifiers, concise/detailed modes, pagination, truncation with guidance. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
26. Error messages that state how to fix the call. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)
27. Above about 10 tools or 10K tokens of definitions, defer loading (tool search), keep 3-5 core tools loaded. [V] — [A6](https://www.anthropic.com/engineering/advanced-tool-use)
28. Use code execution when large intermediate data or many chained calls would otherwise pass through context. [V] — [A5](https://www.anthropic.com/engineering/code-execution-with-mcp)
29. Prefer a well-known CLI over an MCP server when one exists. [V] — [A14](https://code.claude.com/docs/en/costs)
30. Evaluate tools on realistic multi-step tasks and let the agent rewrite them from transcripts, checked on held-out tasks. [V] — [A3](https://www.anthropic.com/engineering/writing-tools-for-agents)

**Prompts**
31. State outcome, success criteria, constraints and reasons; let the model choose the path. [V] [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices); [S] [OpenAI GPT-5.5 guide](https://developers.openai.com/api/docs/guides/prompt-guidance.md)
32. Plain wording; no emphasis words; no contradictions; scoped rules instead of "never." [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
33. A few diverse canonical examples rather than exhaustive rules. [V] — [A1](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
34. Control depth with the effort / thinking setting, not with "think hard" or "be thorough." [V] — [A12](https://code.claude.com/docs/en/best-practices)
35. Audit prompts, skills and memory at each model migration; change one thing at a time and measure quality and token cost. [V] — [A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)

### Cited Findings — main open debates
1. **Is context still the binding constraint?** Chroma and Horthy say quality falls well before the limit and advise aggressive compaction ([Chroma](https://research.trychroma.com/context-rot) [S]; [Horthy talk](https://ai.engineer/talks/context-engineering-for-complex-codebases) [S]). Anthropic's 2026 Claude Code docs call aggressive management old advice given 1M windows and auto-compaction, while still recommending focused contexts ([A12](https://code.claude.com/docs/en/best-practices) [V]). No 2026 replication of context-rot on current models was found.
2. **Compaction versus reset.** Anthropic's own posts moved from "resets essential" (Sonnet 4.5) to "compaction suffices" (Opus 4.5+), while the prompting docs still suggest a fresh window plus file-based state in some cases ([A8](https://www.anthropic.com/engineering/harness-design-long-running-apps) [V]; [A10](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) [V]). Model-dependent; no general rule.
3. **Multi-agent: how far?** Agreement on readers/reviewers; disagreement on parallel writers and peer-to-peer swarms. Cognition says swarm ideas "still don't see meaningful adoption" ([B9](https://cognition.ai/blog/multi-agents-working) [S]); Claude Code ships agent teams with direct messaging but marks them experimental ([A13](https://code.claude.com/docs/en/agent-teams) [V]); Gas Town ran for months and then collapsed on a model change ([Willison](https://simonwillison.net/2026/Aug/4/steve-yegge/) [S]); Google/MIT find gains are task-dependent ([arXiv](https://arxiv.org/abs/2512.08296) [S]).
4. **Does multi-agent add intelligence or just tokens?** Anthropic 2025: token spend explains 80% of variance ([A2](https://www.anthropic.com/engineering/multi-agent-research-system) [V]). Claude Code 2026: delegating the same task "doesn't make the answer any better" ([A12](https://code.claude.com/docs/en/best-practices) [V]). Open question: whether a single agent given the same token budget would match.
5. **Harness versus model.** LangChain's 13.7-point fixed-model gain ([B15](https://blog.langchain.com/improving-deep-agents-with-harness-engineering) [S]) versus the survey's rejection of a harness-only reading ([B16](https://arxiv.org/pdf/2606.20683) [S]) versus minimal harnesses scoring 74%+ on SWE-bench Verified ([B19](https://github.com/SWE-agent/mini-swe-agent) [S]).
6. **Thick versus thin harness.** Anthropic's planner/generator/evaluator harness ($124-200 runs) and Claude Code's feature set versus Pi's four tools and sub-1,000-token prompt ([A8](https://www.anthropic.com/engineering/harness-design-long-running-apps) [V]; [B17](https://www.mariozechner.at/posts/2025-11-30-pi-coding-agent/) [S]). Both sides agree scaffolding should shrink as models improve; they disagree on where to start.
7. **MCP versus CLI versus code execution.** Conflicting small vendor benchmarks ([Smithery](https://smithery.ai/blog/mcp-vs-cli-is-the-wrong-fight) [S]; [Arize](https://arize.com/blog/mcp-vs-cli-skills-for-agents-what-our-eval-found-and-which-you-should-use.md) [S]); Anthropic itself recommends CLIs where available while building tool search and code execution for MCP ([A14](https://code.claude.com/docs/en/costs) [V]; [A5](https://www.anthropic.com/engineering/code-execution-with-mcp) [V]).
8. **Dynamic tool loading versus cache stability.** Manus says never change the tool list mid-run ([B6](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) [S]); Anthropic says deferred tools via tool search do not break caching ([A6](https://www.anthropic.com/engineering/advanced-tool-use) [V]). These may be reconcilable by implementation, but no source compares them directly.
9. **Autonomy versus control in prompts.** Outcome-first prompting gives the model the path; newest-model docs show that small wording (ownership, "never," approval lists) swings behavior and cost by large factors ([A11](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2) [V]). How much to specify remains an empirical, per-model question.
10. **Evaluator value.** Anthropic found the separate evaluator's worth depends on whether the task is beyond what the model does reliably alone, and notes stronger orchestrators can evaluate sub-agent work directly ([A8](https://www.anthropic.com/engineering/harness-design-long-running-apps) [V]; [A9](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them) [V]).

### Inferences
- The single most repeated meta-rule across 2026 sources: measure on your own tasks at each model change, and expect the right amount of scaffolding to go down over time.
- Sources flagged as outdated or partly outdated for a November 2026 reader: the 200K-token truncation detail in A2; Sonnet 4.5 context-anxiety workarounds; ALL-CAPS and anti-laziness prompting; `budget_tokens` thinking control (returns an error on Claude 4.7+ per A10) and prefilled assistant responses (unsupported from Claude 4.6 per A10); Horthy's 40% rule as a hard number; the GPT-5 (Aug 2025) guide where it conflicts with GPT-5.5/5.6 guidance; Gas Town as a recommended system; context-rot and tool-confusion numbers measured on 2024-2025 models.

### Gaps
- Not covered for lack of retrievable sources in this session: Hamel Husain (evals and error analysis for agents), Philipp Schmid on harnesses, Latent Space episode content, AI Engineer talk content beyond titles, Mario Zechner's and Armin Ronacher's own text, Cognition's and Google's primary pages, OpenAI and Google primary guide text.
- Hosts that must be allowed in the session's network settings to verify [S] items directly: cognition.ai, research.google, www.mariozechner.at (all confirmed blocked); likely also openai.com, developers.openai.com, research.trychroma.com, manus.im, dbreunig.com, blog.langchain.com, arxiv.org, simonwillison.net, steve-yegge.medium.com (not attempted).
- Four planned searches were not run because the web-search budget for the turn was exhausted.

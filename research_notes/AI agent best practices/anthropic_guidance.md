# Anthropic's published guidance and staff talks on building and using agents (as of Nov 2026)

Verification legend used on every bullet:
- **[V]** = read from the page itself in this session (anthropic.com, code.claude.com, platform.claude.com, claude.com, support.claude.com all fetched successfully). Quoted phrases under [V] come from the fetched page; where the fetch tool returned a model-written digest rather than raw text, quotes are marked "(digest)" and wording should be re-checked before being printed as verbatim.
- **[S]** = search-result summary only; page not opened. Treat quotes as unconfirmed.
- Hosts that could not be fetched (DNS blocked): youtube.com, simonwillison.net, howborisusesclaudecode.com, theneuron.ai, claude.dev. The web-search budget ran out part-way, so talks/interviews are thinner than the written guidance.

## 1. What do the Anthropic engineering blog posts recommend?

### Takeaway
Across 2024-2026 the posts converge on one line: start with the simplest thing that works, give the agent a way to verify its own work, treat context as a scarce curated resource, design tools and memory for the model rather than for humans, and delete harness scaffolding when a newer model no longer needs it. The 2026 posts (harness design, managed agents, auto mode, agent memory) explicitly retire earlier workarounds such as context resets and heavy sprint structures.

### Cited Findings

Index of posts and dates [V] — [anthropic.com/engineering](https://www.anthropic.com/engineering). Agent-relevant ones, newest first: The components of agent memory (Nov 6, 2026); How we contain Claude across products (May 25, 2026); Scaling Managed Agents (Apr 8, 2026); How we built Claude Code auto mode (Mar 25, 2026); Harness design for long-running application development (Mar 24, 2026); Building a C compiler with a team of parallel Claudes (Feb 5, 2026); Quantifying infrastructure noise in agentic coding evals (Feb 5, 2026); Demystifying evals for AI agents (Jan 9, 2026); Effective harnesses for long-running agents (Nov 26, 2025); Introducing advanced tool use (Nov 24, 2025); Code execution with MCP (Nov 4, 2025); Beyond permission prompts / sandboxing (Oct 20, 2025); Equipping agents with Agent Skills (Oct 16, 2025); Effective context engineering (Sep 29, 2025); Writing effective tools for agents (Sep 11, 2025); How we built our multi-agent research system (Jun 13, 2025). Not opened: "How we contain Claude across products", "Redesigning source control at Anthropic" (Oct 13, 2026), sandboxing post, infrastructure-noise post.

**Building effective agents** — Erik Schluntz and Barry Zhang, blog post, Dec 19, 2024. [V] — [Source](https://www.anthropic.com/engineering/building-effective-agents)
- Workflows = "systems where LLMs and tools are orchestrated through predefined code paths"; agents = LLMs that dynamically direct their own processes and tool usage. Both are "agentic systems".
- Find the simplest solution; add complexity only when needed. Often a single LLM call with retrieval and examples is enough. Agentic systems trade latency and cost for task performance.
- Start with LLM APIs directly; if using a framework, understand the code underneath (frameworks hide prompts/responses).
- Five workflow patterns with when-to-use: prompt chaining (fixed subtasks, gates between steps); routing (distinct categories, e.g. easy to Haiku, hard to Sonnet); parallelization (sectioning and voting); orchestrator-workers (subtasks not predictable in advance); evaluator-optimizer (clear criteria, iteration measurably helps).
- Agents: for open-ended problems where steps cannot be predicted; get "ground truth" from the environment each step; add human checkpoints and stopping conditions (max iterations); test in sandboxes.
- Three principles: keep design simple; show planning steps (transparency); carefully craft the agent-computer interface (ACI).
- Tool appendix: give tools as much prompt-engineering as the main prompt; keep formats close to natural internet text; "poka-yoke" tools (requiring absolute file paths removed a class of errors in the SWE-bench agent).
- Flag: the page itself now notes much of the tooling has changed since publication (digest). Still cited as the root principle by the Mar 2026 harness post.

**How we built our multi-agent research system** — Jeremy Hadfield, Barry Zhang, Kenneth Lien, Florian Scholz, Jeremy Fox, Daniel Ford, blog post, Jun 13, 2025. [V] — [Source](https://www.anthropic.com/engineering/multi-agent-research-system)
- Orchestrator-worker: lead agent plans, spawns 3-5 subagents in parallel, a CitationAgent runs last. Opus 4 lead + Sonnet 4 subagents beat single-agent Opus 4 by 90.2% on an internal research eval.
- Cost: agents use about 4x the tokens of chat; multi-agent about 15x. Token usage explained 80% of BrowseComp variance. Good fit: "valuable tasks that involve heavy parallelization", information exceeding one context window, many complex tools. Poor fit: shared context or many inter-agent dependencies ("most coding tasks").
- Eight prompting principles: think like your agents (simulate with the real prompts/tools); teach delegation (each subagent needs an objective, output format, tool/source guidance, clear boundaries); scale effort to complexity (1 agent with 3-10 calls for facts; 2-4 subagents with 10-15 calls for comparisons; 10+ subagents for complex research); design tools carefully; let agents improve themselves (a tool-testing agent rewrote tool descriptions, 40% faster task completion); start wide then narrow; guide thinking; parallelize (up to 90% faster).
- Evals: start with about 20 real queries; judge outcomes not paths; single LLM-judge call with a 0.0-1.0 score plus pass/fail; keep human testing (it caught a bias toward SEO content farms).
- Production: resume from checkpoints, tell agents when tools fail, full tracing, rainbow deployments; subagents write outputs to the filesystem and pass lightweight references back.
- Flag: the "coding is a poor fit" caveat is partly overtaken by 2026 material (C compiler with 16 parallel agents; Claude Code agent teams and dynamic workflows).

**Writing effective tools for agents - with agents** — Ken Aizawa, blog post, Sep 11, 2025. [V] — [Source](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Process: prototype (local MCP server), build an eval from realistic multi-call tasks with verifiable outcomes, read transcripts, then paste transcripts into Claude Code and let it refactor the tools; check on a held-out set.
- Do not just wrap API endpoints; build a few high-impact tools (`search_contacts` not `list_contacts`; `schedule_event` instead of three calls; `get_customer_context`).
- Namespace by service/resource (`asana_search`, `jira_search`); prefix vs suffix effects vary by model, so test.
- Return meaningful context: names over UUIDs; a `response_format` enum (concise vs detailed; example 72 vs 206 tokens).
- Token efficiency: pagination, filtering, truncation with sensible defaults (Claude Code caps tool responses at 25,000 tokens by default); error messages that say how to fix the call.
- Describe tools as to a new hire; unambiguous parameter names (`user_id` not `user`).

**Effective context engineering for AI agents** — Prithvi Rajasekaran, Ethan Dixon, Carly Ryan, Jeremy Hadfield (Applied AI), blog post, Sep 29, 2025. [V] — [Source](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Goal: the smallest set of high-signal tokens that makes the desired outcome likely; "context rot" means recall degrades as tokens grow.
- System prompt at the "right altitude" (neither brittle if-else logic nor vague); start minimal with the best model, add only to fix observed failures.
- Tools: minimal overlap; if a human engineer cannot say which tool applies, the agent cannot either. Examples: a few diverse canonical ones, not an edge-case list.
- Just-in-time retrieval via lightweight references (paths, queries, links); Claude Code is hybrid (CLAUDE.md up front, glob/grep on demand).
- Long-horizon techniques: compaction (maximise recall first, then precision; clearing old tool results is the lightest form); structured note-taking (NOTES.md, to-do lists, memory tool); sub-agents returning 1,000-2,000-token summaries. Compaction suits back-and-forth; notes suit milestone-driven work; multi-agent suits parallel research.
- "Do the simplest thing that works" (digest).
- Flag: Claude Code docs (2026) now list "the context window is your main constraint, manage it aggressively" under advice you can stop following, because of 1M-token windows and automatic compaction — [Source](https://code.claude.com/docs/en/best-practices) [V]. The curation principle stands; the scarcity framing is dated.

**Equipping agents for the real world with Agent Skills** — Barry Zhang, Keith Lazuka, Mahesh Murag, blog post, Oct 16, 2025. [V] — [Source](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- A skill is a folder with SKILL.md (YAML `name`, `description`); progressive disclosure in three levels (metadata at startup, SKILL.md when relevant, bundled files on demand); scripts run as deterministic tools without entering context.
- Develop: start with evaluation (find where agents struggle, add skills for those gaps); split unwieldy SKILL.md files; watch how Claude actually uses the skill; iterate with Claude and have it record what worked. Install only from trusted sources; audit bundled code.

**Code execution with MCP** — Adam Jones and Conor Kelly, blog post, Nov 4, 2025. [V] — [Source](https://www.anthropic.com/engineering/code-execution-with-mcp)
- Present MCP servers as code APIs the agent explores in a sandbox instead of loading every tool definition; example drops from about 150,000 to about 2,000 tokens (98.7%). Filter data in code, run loops/conditionals in code, keep intermediate and PII data out of the model, save working code as reusable skills. Cost: needs sandboxing, limits, monitoring.

**Introducing advanced tool use** — Bin Wu et al., blog post, Nov 24, 2025. [V] — [Source](https://www.anthropic.com/engineering/advanced-tool-use)
- Tool Search Tool (`defer_loading`): use above about 10 tools or 10K tokens of definitions; 85% context reduction in the example; keep the 3-5 most-used tools always loaded. Programmatic Tool Calling: for large data and 3+ dependent calls (37% fewer tokens on research tasks). Tool Use Examples: 1-5 realistic examples per tool (72% to 90% on complex parameters).

**The "think" tool** — Anthropic Engineering, Mar 20, 2025. [V] — [Source](https://www.anthropic.com/engineering/claude-think-tool)
- Useful for policy-heavy, sequential tool chains; pair with domain examples in the system prompt. Superseded: a Dec 15, 2025 note on the page says extended thinking is now recommended "instead of a dedicated think tool in most cases".

**Effective harnesses for long-running agents** — Justin Young, blog post, Nov 26, 2025. [V] — [Source](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Problem: each session starts with no memory. Fix: an initializer session writes `init.sh`, a progress file, an initial git commit and a JSON feature list (200+ items, all failing); every later session reads git log and progress, picks one feature, verifies end-to-end with browser automation "as a human user would", commits, and updates progress. Only the `passes` field may change: "It is unacceptable to remove or edit tests".
- Failure modes addressed: declaring victory early, leaving a broken undocumented state, marking features done without testing.

**Demystifying evals for AI agents** — Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares, Jiri De Jonghe, blog post, Jan 9, 2026. [V] — [Source](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Vocabulary: task, trial, grader, transcript, outcome (the end state, not what the agent claims), harness. Grader types: code (prefer), model (calibrate against humans, give an "Unknown" option), human (sparingly).
- Capability evals start low and give a hill to climb; regression evals stay near 100%; graduate one into the other. pass@k vs pass^k (75% per trial gives about 42% for 3 of 3).
- Roadmap: start with 20-50 tasks from real failures; write tasks two experts would grade identically; test both should-trigger and should-not cases; isolate trials; grade outputs not paths; give partial credit; read transcripts; watch saturation; give evals an owner.
- Pair with production monitoring, A/B tests, user feedback, manual transcript review.

**Building a C compiler with a team of parallel Claudes** — Nicholas Carlini, blog post, Feb 5, 2026. [V] — [Source](https://www.anthropic.com/engineering/building-c-compiler)
- 16 agents, nearly 2,000 sessions, about $20,000, 100,000 lines of Rust. A bash loop restarts Claude with the same prompt; no orchestrator; agents claim tasks with lock files in git.
- The verifier must be near-perfect or Claude "will solve the wrong problem". Design tests for the model: few lines of output, details to log files, greppable ERROR lines, a `--fast` sampled mode because the model cannot sense time. Use a known-good oracle (GCC) to split one giant failing task into parallelisable ones. Give agents specialised roles (dedup, performance, design critique, docs).

**Harness design for long-running application development** — Prithvi Rajasekaran, blog post, Mar 24, 2026. [V] — [Source](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Planner / generator / evaluator. Self-evaluation is unreliable: out of the box Claude was "a poor QA agent" that talked itself into approving work. A separate evaluator driving the app through Playwright, calibrated with few-shot scored examples and hard thresholds, fixed that. Generator and evaluator agree a "sprint contract" (what done means, how it is tested) through files.
- Cost: solo run 20 min / $9 with a broken core feature; full harness 6 hr / $200 and working.
- Context resets were needed for Sonnet 4.5's "context anxiety"; Opus 4.5 "largely removed that behavior", so resets were dropped in favour of compaction. With Opus 4.6 the sprint structure was removed and the evaluator moved to one final pass; the planner stayed.
- Rule: when a new model lands, remove pieces one at a time and re-test; the evaluator is worth its cost only "when the task sits beyond what the current model does reliably solo".

**How we built Claude Code auto mode** — John Hughes, blog post, Mar 25, 2026. [V] — [Source](https://www.anthropic.com/engineering/claude-code-auto-mode)
- Users accepted 93% of manual prompts (approval fatigue). Auto mode = prompt-injection probe on tool outputs plus a two-stage transcript classifier that sees user messages and tool calls but not the agent's prose ("reasoning-blind"). Blocked actions return a reason; 3 consecutive or 20 total denials escalate to the human.
- Numbers: 0.4% false positives on real traffic; 17% false negatives on 52 real overeager actions. "Not a drop-in replacement for careful human review" on high-stakes infrastructure.

**Scaling Managed Agents: Decoupling the brain from the hands** — Lance Martin, Gabe Cemaj, Michael Cohen, blog post, Apr 8, 2026. [V] — [Source](https://www.anthropic.com/engineering/managed-agents)
- Harnesses encode assumptions about what the model cannot do and those go stale ("The resets had become dead weight"). Split session (durable event log), harness (loop), sandbox (execution); keep credentials out of the sandbox via a vault/proxy; keep the full log outside the context window and let the harness choose what to load. TTFT p50 down about 60%, p95 down over 90%.

**The components of agent memory** — Lance Martin, blog post, Nov 6, 2026. [V] — [Source](https://www.anthropic.com/engineering/components-of-agent-memory)
- Four layers: transcripts, memory notes, representations (wikis), injected context. "Dreaming" = offline consolidation across all four.
- Tell Claude what memory is for, not how to use it. A fixed rule such as "record only failures" cut Opus 5's memory gain by 0.234. Weaker models need explicit steering (Sonnet 4.6 rose from 0.364 to 0.641 when told to read memory first and test its theories).
- Store only what exists because the agent worked it out, not what it can re-read. Sonnet 3.7 wrote 413 files and stalled; Fable 5 wrote nine files plus a `global.md` dashboard.
- Dreaming checks notes against transcripts (a wrong note caused a loop until corrected), builds wikis (Stainless corpus: wiki + search 61% vs search alone 53%; Harvey test 87% vs 73%), and promotes notes to always-loaded context. Keep the top layer small; a wrong injected note is carried into every session.
- Shared-memory pitfall: several agents writing one table lost 16 of 57 facts; one page per agent kept all 57.

**Building agents with the Claude Agent SDK** — Thariq Shihipar, claude.com blog (now redirected under Anthropic engineering), Sep 29, 2025. [V] — [Source](https://claude.com/blog/building-agents-with-the-claude-agent-sdk)
- Loop: "gather context -> take action -> verify work -> repeat". Give the agent a computer. Start with agentic search (grep/tail) and add semantic search only if needed. Verification in order of robustness: rules (lint, types), visual feedback (screenshots), LLM judge ("generally not a very robust method").

**Building always-on agents** — William Hu, claude.com article, Nov 13, 2026. [V] — [Source](https://claude.com/resources/articles/building-always-on-agents)
- Primitives: durable event inbox, timers/schedules, memory, permissions, outbox. One active worker per inbox; batch pending events; "memory should only hold what the agent can't look up again"; save to memory before compaction; credential proxy outside the agent; "start quiet and then earn more of the user's attention over time".

**New in Claude Managed Agents: dreaming, outcomes, multiagent orchestration** — claude.com product article, dated May 19, 2026 on the page. [V] — [Source](https://claude.com/resources/articles/new-in-claude-managed-agents)
- Dreaming (research preview): scheduled review of sessions and memory stores, with automatic or review-first updates. Outcomes (public beta): you write a rubric, a separate grader in its own context checks the output and sends the agent back; "up to 10 points" better task success. Harvey: completion rates "~6x". Search results date the announcement to Code with Claude, May 6, 2026 [S] — [SiliconANGLE](https://siliconangle.com/2026/05/06/anthropic-letting-claude-agents-dream-dont-sleep-job/).

### Inferences
- The stable core across two years is: simplicity first, an external check on the agent's work, curated context, model-oriented tools. Everything else (resets, sprints, think tool, emphasis words, context-protecting subagents) has been walked back at least once as models improved.
- Anthropic's own evidence says independent evaluation beats self-evaluation (harness post, Outcomes, `/code-review` in a fresh subagent), and that an imperfect verifier misdirects the agent (C compiler). Verification quality is the main lever, not prompt wording.
- Memory guidance shifted in 2026 from "write notes" to "write few notes, verify them later, keep the always-loaded layer tiny".

### Gaps
- "How the Claude Code team thinks about verification" (claude.dev blog, Nov 16, 2026; listed on claude.com/blog as covering end-to-end, property and differential tests) could not be opened: claude.dev is DNS-blocked here. Title/date/summary are [V] from the listing only — [listing](https://claude.com/blog). Likewise "Using a 'chaos loop' to harden the Claude Docs launch" (Nov 17, 2026).
- Original "Claude Code: best practices for agentic coding" post: the URL now 308-redirects to the docs page, so the original text, author and date were not re-verified (from memory: Boris Cherny, April 2025 — unverified here) — [redirect](https://www.anthropic.com/engineering/claude-code-best-practices).
- Not opened: "How we contain Claude across products", sandboxing post, infrastructure-noise post, "Redesigning source control at Anthropic".

## 2. What do Claude Code team members say in talks and interviews?

### Takeaway
Reported themes from Boris Cherny and Cat Wu in 2026 are: give Claude a way to verify its work, prefer auto mode to plan mode, run many sessions in parallel, move from prompting to writing loops and scheduled tasks, and avoid over-stuffing context. Almost all of this section rests on search-result summaries of secondary coverage, because YouTube and the write-up sites were unreachable; no quote here was checked against a recording.

### Cited Findings
- June 2026 one-year conversation, Boris Cherny (Head of Claude Code) and Cat Wu (Head of Product, Claude Code), Anthropic video, about 18 minutes, published around June 8, 2026; topics listed as verification, auto mode, routines, loops, and context minimalism. [S] — [X post by @hqmank](https://x.com/hqmank/status/2064337891321618663); [The Neuron explainer](https://www.theneuron.ai/explainer-articles/claude-code-creators-boris-cherny-and-cat-wu-explain-how-to-use-agent-loops/). Video id given in the brief (Hth_tLaC2j8) could not be opened: [YouTube](https://www.youtube.com/watch?v=Hth_tLaC2j8).
- Reported from that conversation: Cherny prefers auto mode over plan mode; Claude Code watches GitHub and X, proposes what to build, and runs its own security reviews; he keeps roughly 10-15 sessions running across terminal, web and phone. [S] — [pasqualepillitteri.it](https://pasqualepillitteri.it/en/news/4564/claude-code-boris-cherny-one-year-on-phone)
- Cat Wu, attributed: "When you give the model too much context, it's like you're micromanaging it." [S, wording unconfirmed] — [The Neuron](https://www.theneuron.ai/explainer-articles/claude-code-creators-boris-cherny-and-cat-wu-explain-how-to-use-agent-loops/)
- "Give Claude a way to verify its own work" is the most repeated Cherny tip in secondary compilations. [S] — [howborisusesclaudecode.com](https://howborisusesclaudecode.com/); [digitalstrategy-ai.com](https://digitalstrategy-ai.com/boris-cherny-claude-code-workflow). It matches the first-party docs heading "Give Claude a check to run" [V] — [Source](https://code.claude.com/docs/en/best-practices).
- "My job is to write loops": Cherny reportedly said he no longer prompts Claude directly but writes loops, with agents writing prompts for other agents, and called loops as big a step as the move from source code to agents. Named recurring loops include babysit-prs and post-merge-sweeper. [S] — [OfficeChai](https://officechai.com/ai/i-now-just-write-loops-to-prompt-claude-code-claude-code-creator-boris-cherny/); [The New Stack](https://thenewstack.io/loop-engineering/); [FourWeekMBA](https://fourweekmba.com/claude-code-loops-boris-cherny-next-paradigm/); [Business Today, Jun 22, 2026](https://www.businesstoday.in/technology/artificial-intelligence/story/what-comes-after-vibe-coding-claude-code-creator-boris-cherny-has-an-answer-538427-2026-06-22)
- Conflict: one report places the loops remark at WorkOS's Acquired Unplugged event on June 2, 2026; another at Meta's @Scale conference. Not resolved. "Loop engineering" as a name is attributed to Addy Osmani, after a similar June 7 post by Peter Steinberger. [S] — [The New Stack](https://thenewstack.io/loop-engineering/)
- Lenny's Podcast, Boris Cherny, February 2026: about 4% of GitHub commits made by Claude Code, forecast of about one-fifth by year end. [S] — [note.com transcript summary](https://note.com/ai_eng_tech/n/nf940a6dc47e6?hl=en)
- A later Cherny + Wu interview on the move from Claude Code to Claude Tag (persistent context and memory) is referenced. [S] — [X post by @rohanpaul_ai](https://x.com/rohanpaul_ai/status/2072906250661285895)
- Code w/ Claude 2026: San Francisco May 6, London May 19, Tokyo June 10, plus "Extended" days May 7 / May 20 / June 11. [V] — [Source](https://claude.com/code-with-claude)
- SF sessions confirmed (9 of 19 visible): Opening keynote (Ami Vora, Dianne Penn, Angela Jiang, Katelyn Lesse, Cat Wu, Boris Cherny); What's new in Claude Code (Dickson Tsai); Memory and dreaming for self-learning agents (Mahesh Murag, May 6, 10:30); Caching, harnesses, and advisors: Building on Claude at GitHub scale (Mario Rodriguez, Brad Abrams); The expanding toolkit (Lucas Gonzalez); How to get to production faster with Claude Managed Agents (Jess Yan, Lance Martin); Live coding session with Boris Cherny and Jarred Sumner (May 6, 13:50, "how the Claude Code team uses Claude Code themselves"). [V, titles/speakers only; pages carry no transcript] — [SF recordings](https://claude.com/code-with-claude/san-francisco); [Murag session](https://claude.com/code-with-claude/session/sf-memory-and-dreaming-for-self-learning-agents); [live coding session](https://claude.com/code-with-claude/session/sf-live-coding-session-with-boris-cherny-and-jarred-sumner)
- Murag session description: memory and dreaming turn Managed Agents into "self-learning systems"; dreaming verifies and adds to stored memory between sessions. [V] — [Source](https://claude.com/code-with-claude/session/sf-memory-and-dreaming-for-self-learning-agents)
- Thariq Shihipar, first-party written position (not a talk): the agent loop is gather context, act, verify, repeat; give agents a computer; prefer agentic search. [V] — [Source](https://claude.com/blog/building-agents-with-the-claude-agent-sdk)
- Erik Schluntz and Barry Zhang, first-party written position: prefer workflows or single calls until an agent is justified; three principles (simplicity, transparency, ACI). [V] — [Source](https://www.anthropic.com/engineering/building-effective-agents). A third-party talk listing shows this post was publicly criticised for its "anti-workflow" framing [S] — [ai.engineer talk page](https://ai.engineer/talks/8SUJEqQNClw-agents-vs-workflows-why-not-both).

### Inferences
- The staff talking points and the 2026 docs agree closely (verification first, auto mode as the default, loops and scheduled tasks for unattended work, short instruction files), so the docs can stand in as the verified version of what the talks are reported to say.
- "Context minimalism" (Wu) is consistent with the docs' test for CLAUDE.md content and with the memory post's finding that fixed rules reduce performance on stronger models.

### Gaps
- No verbatim quote from the June 2026 Cherny + Wu video was verified; YouTube is blocked and no transcript was retrievable. To print quotes, someone must check the recording.
- AI Engineer World's Fair 2026 fireside chat: no source found. The search budget was exhausted before this query ran. Unknown whether it happened as described, who took part, or what was said.
- Code w/ Claude 2025 sessions: nothing retrieved. Code w/ Claude 2026 session content beyond titles: not retrievable (session pages have no transcripts; 10 of 19 SF sessions and all London/Tokyo sessions not listed).
- Thariq Shihipar's talks, Barry Zhang / Mahesh Murag skills talk, Erik Schluntz podcast appearances, Alex Albert, Latent Space episodes, Simon Willison's event write-ups: not retrieved (search budget exhausted; simonwillison.net blocked). Do not cite specifics for these from this file.
- The venue and date of the "my job is to write loops" remark are in conflict between sources.

## 3. Recommended practices by feature (CLAUDE.md, skills, subagents, hooks, verification, modes, routines/loops, Dreaming, memory tool, compaction, prompt-writing)

### Takeaway
Current first-party guidance (Claude Code docs as of Nov 2026) says: write down only what Claude cannot see in the repo, give it a pass/fail check, enforce the few true invariants deterministically with hooks and deny rules, leave the defaults (auto mode, automatic subagents, automatic compaction) alone until a task costs too much, and bound unattended runs with a done-condition, a verifier, limited permissions, a spend cap and a human sign-off. The docs carry an explicit table of older advice to drop.

### Cited Findings
All from "Best practices for Claude Code" unless noted. [V] — [Source](https://code.claude.com/docs/en/best-practices)

**Framing**
- Two kinds of instruction: non-deterministic (prompt, CLAUDE.md, skills; the model decides) and deterministic (hooks, deny rules, managed settings; the harness enforces). "Most of the time, you want Claude to decide."
- "If you keep adding emphasis to a `CLAUDE.md` line and Claude still misses it, that's a sign the line belongs in a hook." The reverse also holds: delete hooks that are only preferences.

**CLAUDE.md / memory files**
- "Only write down what Claude can't find by reading the repo, or what it would otherwise spend time rediscovering every session." Test: could it find this by reading the code, and would it actually look? If both yes, leave it out.
- Contents that earn a place: exact test/typecheck/lint commands; places where the code is misleading; decisions not in the repo; "anything you've had to correct twice already".
- Target under 200 lines; longer files "reduce adherence". Be concrete enough to verify ("Use 2-space indentation" not "Format code properly"). Contradictions get resolved arbitrarily, so audit; `/doctor prompt-audit` finds stale, conflicting or old-model instructions. [V] — [memory docs](https://code.claude.com/docs/en/memory.md)
- Path-scoped rules in `.claude/rules/` with `paths:`; personal preferences in `CLAUDE.local.md`; `/init` to start and then mostly delete lines. AGENTS.md is read when no CLAUDE.md exists. [V] — [memory docs](https://code.claude.com/docs/en/memory.md)
- Auto memory: Claude writes its own notes to a per-repo directory; only the first 200 lines or 25KB of `MEMORY.md` load at start, with detail in topic files read on demand. "Claude treats CLAUDE.md files and auto memory as context. They aren't enforced configuration." [V] — [memory docs](https://code.claude.com/docs/en/memory.md)

**Skills**
- Put team procedures (release, incident runbook, doc style) in `.claude/skills/<name>/SKILL.md`; lighter than CLAUDE.md because only name and description load until needed. Same rule: only what Claude cannot find itself. For side-effecting workflows set `disable-model-invocation: true` so only a human can start it.
- Authoring: "Concise is key"; "Default assumption: Claude is already very smart"; match degrees of freedom to fragility (exact script for a migration, loose guidance for a review); descriptions in third person stating what it does and when to use it; SKILL.md body under 500 lines; references one level deep; build at least three evals before writing much; develop with one Claude writing and a fresh Claude testing; include validator-fix-repeat feedback loops; avoid offering many options. [V] — [Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)

**Subagents**
- Claude delegates on its own (Explore agent, forks, background). Subagents pay off for parallel independent work, keeping noisy output out of the main thread, a reviewer that has not seen the reasoning, and tool-restricted roles. They cost tokens and sometimes time, and "not a smarter answer".
- Cheap subagents: `model: haiku` for simple jobs; ask for short summaries; restrict a reviewer to `tools: Read, Grep, Glob`.
- Use one "when a side task would flood your main conversation with search results, logs, or file contents you won't reference again"; define a custom one when you keep spawning the same worker. Nesting is allowed up to three layers; default cap of 20 concurrent. [V, digest] — [subagent docs](https://code.claude.com/docs/en/sub-agents.md)
- API-side: newer models delegate readily; if overused, state when subagents are and are not warranted, and state their cost in the tool description. [V] — [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices); [Fable 5.2 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)

**Hooks and permissions**
- "If something has to happen every time, make it a hook, a deny rule, or a managed setting. `CLAUDE.md` is for guidelines." Examples: PostToolUse formatter, PreToolUse block on a protected folder, Stop hook that runs the typecheck and exits 2 to send Claude back.
- `prompt`/`agent` hooks ask another model and are not deterministic; use `command`, `http` or `mcp_tool` hooks for real enforcement. Deny rules hold in every mode including auto and bypass. Use the sandbox for OS-level limits.
- "Keep the enforced set small"; at forty hooks, most are preferences.

**Verification**
- "Give Claude something it can run that comes back pass or fail. Once it has that, it can keep going on its own and you don't have to watch." Without a check, "looks done" is all it has.
- Ladder of strictness: in the prompt ("keep going until they pass"); `/goal <condition>` checked by a separate model each turn; a Stop hook; review by a fresh subagent via `/code-review`. `/verify` builds and runs the app.
- Ask for evidence (the command and its output, a screenshot); "All tests pass" alone tells you little.
- Review against the spec and say "skip style": "A reviewer that's asked to find problems will always find something."
- Brainstorm first: have Claude interview you, write SPEC.md, then build in a fresh session.

**Plan mode vs auto mode**
- Auto mode is the built-in default for interactive terminal and VS Code sessions (v2.1.283+); `claude -p` and the SDK default to manual. [V, digest] — [permission modes](https://code.claude.com/docs/en/permission-modes.md)
- Mode table: default/manual for "sensitive work, unfamiliar code"; plan for "exploring a codebase before changing it"; auto for "long tasks, reducing prompt fatigue"; dontAsk for "locked-down CI and scripts"; bypassPermissions for "isolated containers and VMs only". [V, digest] — [permission modes](https://code.claude.com/docs/en/permission-modes.md)
- "Auto mode reduces permission prompts but does not guarantee safety. Use it for tasks where you trust the general direction, not as a replacement for review on sensitive operations." [V, digest] — [permission modes](https://code.claude.com/docs/en/permission-modes.md)
- Manual mode's cost: "after the tenth prompt you start approving without reading". "Never" rules belong in deny rules or hooks because the classifier is a judgment call. Conversation boundaries ("don't push") are honoured but can be lost at compaction.
- Tune auto mode in plain English (`autoMode.environment`, `allow`) and keep `"$defaults"` in each list.

**Scheduled tasks (formerly "routines") and loops**
- Naming: cloud "routines" were renamed "scheduled tasks"; `/schedule` (alias `/routines`), still at claude.ai/code/routines; research preview. Triggers: schedule, API, GitHub event. [V] — [routines docs](https://code.claude.com/docs/en/routines.md)
- "The prompt is the most important part: the task runs autonomously, so the prompt must be self-contained and explicit about what to do and what success looks like." [V] — [routines docs](https://code.claude.com/docs/en/routines.md)
- Runs start in auto mode, push to `claude/`-prefixed branches, act as you through connectors; remove connectors the task does not need; run status says how the session ended, not whether the work was done, so read the transcript. [V] — [routines docs](https://code.claude.com/docs/en/routines.md)
- Protect `main` with a branch rule the task cannot bypass and "watch the first few runs of anything new before trusting it unattended".
- Choosing: cloud tasks for work that must run without your machine (min interval 1 hour); Desktop tasks for local files; `/loop` for quick polling in an open session (min 1 minute, 7-day expiry, up to 50 tasks). `/loop` with no interval lets Claude pace itself; bare `/loop` runs a maintenance prompt (finish work, tend the PR, cleanup) or your `loop.md`. `/goal` for working toward a condition rather than on a timer. [V] — [scheduled-tasks docs](https://code.claude.com/docs/en/scheduled-tasks.md)
- Five questions before leaving a run unattended: does it know what done looks like; can it check its own work; what may it touch; what stops it (`--max-turns`, `--max-budget-usd`); where does the result go and who signs off.
- Headless batch pattern: `claude --bare -p ... --allowedTools ... --permission-mode dontAsk --max-turns 20`; try three items first.
- Parallel work: one worktree per session (`claude -w`), `/batch` for mechanical migrations (5-30 units, one PR each), a second session as independent reviewer.

**Dreaming**
- Managed Agents feature in research preview: scheduled review of past sessions and memory stores that curates memory, with automatic or review-first application. [V] — [claude.com](https://claude.com/resources/articles/new-in-claude-managed-agents)
- Design guidance: use dreaming to verify notes against transcripts, build linked wikis, and promote only a few notes into always-loaded context; tune cadence to how often information changes because dreaming costs compute. [V] — [components of agent memory](https://www.anthropic.com/engineering/components-of-agent-memory)
- Claude Code: the docs index has no "dreaming" page; the closest feature is auto memory. [V] — [docs index](https://code.claude.com/docs/llms.txt). Third-party claims that dreaming runs locally in the Claude Code CLI conflict with each other and are unconfirmed [S] — [MindStudio](https://www.mindstudio.ai/blog/what-is-claude-dreaming-anthropic-managed-agents).

**Memory tool (API)**
- Client-side file directory under `/memories`; the API auto-injects a protocol that begins "ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE" and "ASSUME INTERRUPTION". Add "keep its content up-to-date, coherent and organized" if files get cluttered; scope what gets written. Validate paths, cap sizes, expire stale files. [V] — [Source](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)
- Multi-session pattern: initializer session creates progress log, feature checklist and startup script; each session reads them first and updates the log at the end; one feature at a time, complete only after end-to-end verification. Pair with compaction: "memory preserves the information that must survive summarization". [V] — [Source](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)

**Context and compaction**
- Compaction is automatic; `/clear` between unrelated tasks; `/compact <focus>`; after two failed corrections, `/clear` and write a better prompt rather than a third correction. `/rewind` makes bold attempts cheap.
- Long sessions cost money through re-sent history and cache expiry even when context is not full.
- API-side: tell the model its context will be compacted so it does not wrap up early; consider starting fresh from files instead of compacting; JSON for structured state, free text for progress notes, git for checkpoints. [V] — [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Fable 5.2: keep history append-only apart from compaction/context editing, because editing earlier turns invalidates preserved thinking. [V] — [Fable 5.2 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)

**Prompt-writing for agents (coding)**
- Name the file, the symptom and what "fixed" looks like; hand over files, screenshots and logs rather than describing them; state a preference when two approaches are reasonable.
- Effort level is the control for depth; "think hard" and "be thorough" are ordinary text. "Be ambitious" still helps because it conveys scope. `ultrathink` works for one turn.
- Fable 5.2 specifics: start at `high` effort; add a "Finishing work" block if runs stop early (about half as many unfinished endings, about 30% more output tokens); approve with a specific sentence rather than "continue"; state code ownership if you want restructuring; prune cautions that have accumulated in memory and instruction files; write "ask for approval before deleting files" rather than "never delete files" because "never" is treated as absolute. [V] — [Fable 5.2 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)

**Superseded advice (the docs' own "What you can stop doing" table)** [V] — [Source](https://code.claude.com/docs/en/best-practices)
- Managing the context window aggressively; adding IMPORTANT or ALL CAPS; using subagents to protect context; delegating to a subagent "for a better answer"; "think hard / be thorough"; "keep going, don't be lazy"; describing architecture and file layout in CLAUDE.md.

### Inferences
- The working recipe for unattended agents is the five-question checklist plus a deterministic stop condition; it is the same shape as the harness posts (done-definition, independent verifier, bounded permissions, budget, human sign-off).
- Plan mode has been demoted from the default first step (as in 2025 guidance) to a tool for exploring unfamiliar code; the 2026 docs replace "explore, plan, code, commit" with "specific prompt or interviewed spec, then a check to run".
- Instruction files and memories are now treated as liabilities that accumulate: Anthropic recommends periodic audits and deletion in the Claude Code docs, the Fable 5.2 guide, and the memory post.

### Gaps
- Claude Code hooks reference, skills page, agent-teams and dynamic-workflows pages, `/goal` page and desktop scheduled tasks page were not opened; coverage of those is from the best-practices page and the index descriptions only.
- Whether Claude Code has a user-facing "dream" or session-review command could not be confirmed; no first-party page found.
- The subagent docs fetch was truncated before its best-practices section; permission-modes fetch was truncated at 100k characters.
- Usage limits for scheduled tasks (runs per hour/day) were not extracted.

## 4. Prompting guidance that applies to agents in general, non-coding use (Claude apps, Cowork, research, connectors)

### Takeaway
For non-coding agent work Anthropic's advice is: state the goal, the finished deliverable and when to stop and ask; explain why rather than issuing bare rules; connect the source systems instead of pasting; set approval per connector by risk; and ask for outputs that say what was done, what was found and what is needed from you. Emphasis words and "think hard" phrases are now counterproductive; effort level is the dial.

### Cited Findings
"Prompting best practices" (platform docs, covers Fable 5.2 back to Haiku 4.5) [V] — [Source](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- Treat Claude as "a brilliant but new employee who lacks context on your norms and workflows". Colleague test: if a colleague with minimal context would be confused by the prompt, Claude will be too.
- Give the reason behind an instruction; "Claude can generalize from the explanation" (text-to-speech example instead of "NEVER use ellipses").
- Be explicit about acting versus advising: "can you suggest some changes" yields suggestions; say "change this" to get action. System-prompt blocks exist for defaulting to action or defaulting to caution.
- Drop aggressive language: replace "CRITICAL: You MUST use this tool when..." with "Use this tool when..."; newer models overtrigger on emphasis. Tune down anti-laziness and "if in doubt, use [tool]" prompts.
- Research: define success criteria, ask for verification across sources, and for complex work have the model keep competing hypotheses, track confidence, self-critique and maintain a research-notes file.
- Autonomy and safety: have the agent weigh reversibility; proceed on local reversible actions; ask before destructive, hard-to-reverse or externally visible actions (messages, posts, shared infrastructure).
- Finance and records: recommended system-prompt block says do not copy full account or ID numbers unless needed, never release a payment itself, claim a check was done only if a tool result in the session showed it, and do not send a message in a person's name unless approved or labelled as from an assistant. The block "guides Claude and doesn't enforce anything", so require human approval on money-moving tools.
- Parallel tool calls can be pushed to about 100% with an explicit instruction. Prompt chaining remains useful mainly for inspectable pipelines; the common chain is draft, review against criteria, refine.

Fable 5.2 guide, applicable beyond code [V] — [Source](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2)
- Effort: `medium` for chat and everyday knowledge work; `high` for research, agentic work and long unattended runs.
- An approval rule listing "sending messages to people outside this conversation" made the model hold back about half of routine requests to colleagues in unattended tests; be specific about which actions need approval.
- Progress: ask for a one-line statement of intent before starting, brief updates, and a closing recap that stands alone (about 20% more output tokens).
- Reports: "Write for an intelligent but low-context reader" who has not seen the tool calls; key points first; no invented abbreviations.

"Using Claude Opus 5.5 as your daily driver for knowledge work" — Aleksandra Todorova, claude.com, Nov 12, 2026. [V, digest] — [Source](https://claude.com/resources/articles/using-claude-opus-5-5-as-your-daily-driver-for-knowledge-work)
- In the first message "describe your goal, the finished output, and when you want it to stop and ask". "Ask for a finished file" rather than an outline.
- Start at medium effort; "try a higher effort first before moving up a model". Remove "think hard" phrases.
- "Connect Claude to your work files and systems" instead of pasting. Attach charts and screenshots rather than retyping. Prefer outputs that state what was done, what was found, and what Claude needs from you. Ask Claude to check long drafts for inconsistencies.

"Getting started with Cowork" — Anthropic Help Center. [V, digest] — [Source](https://support.claude.com/en/articles/13345190-getting-started-with-cowork)
- "Describe an outcome, step away, and come back to finished work." Group related work in one task; start a new conversation for unrelated work; say when you only want a quick answer.
- Global instructions for tone, format and your role; folder instructions for project context, which Claude can update.
- `/schedule` creates cloud scheduled tasks; a task needing connector approval may stall unless set to "Allow for all scheduled runs".
- Set sensitive connector tools (for example, sending email) to "Always ask". Auto mode "can miss risky actions"; stay close or use Manual for money, messages sent as you, and important files. Plugins bundle skills, connectors and sub-agents.

Research agents specifically [V] — [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system); [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Brief each subagent with objective, output format, sources/tools and boundaries; start with broad short queries then narrow; scale agents and calls to query difficulty; prefer authoritative sources over SEO content.
- Grade research on groundedness, coverage and source quality, and recalibrate LLM rubrics against experts often.

Always-on assistants [V] — [Source](https://claude.com/resources/articles/building-always-on-agents)
- "Permissions are where always-on agents carry the most risk"; defer risky tool calls until the user answers; start quiet and tune message volume with evals.

### Inferences
- The non-coding advice mirrors the coding advice with the check replaced by a self-describing report and by per-connector approvals: Claude cannot run a test suite on a memo, so Anthropic asks for evidence-bearing output and independent review (Outcomes-style rubric grading) instead.
- For personal agent setups the highest-leverage steps implied by these sources are: a short standing-instructions file containing only what the agent cannot discover; risk-tiered connector approvals; scheduled tasks with self-contained prompts and a named deliverable; and periodic pruning of accumulated instructions and memories.

### Gaps
- No first-party page specific to prompting Claude apps' Research mode or to writing project/custom instructions was opened; the claude.ai system-prompt and "Claude Tag" guidance were not reviewed.
- The Cowork and Opus 5.5 pages came back as model-written digests; short quotes there should be re-checked before being printed as verbatim.
- Cowork-specific skills, memory and "dreaming" behaviour: not found in the sources opened.
- Network note for the coordinator: youtube.com, simonwillison.net, howborisusesclaudecode.com, theneuron.ai and claude.dev must be allowed in the session's network settings to close the talk and verification-post gaps; more web searches would also be needed.

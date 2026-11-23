# Talks, videos and podcast episodes on using and building LLM agents (2025-2026): watch-and-listen catalogue

Research date: 23 Nov 2026. Search budget used: 45 of 45 web searches.

**Provenance tags used on every item**
- `[P]` = verified from a page I actually fetched (only claude.com session pages and GitHub-hosted files were fetchable; theneuron.ai, simonwillison.net, workos.com, techcrunch.com, ai.engineer and youtube.com all failed DNS and were not retried).
- `[S]` = from a web-search result summary/snippet only (not opened). Treat quotes and numbers as unverified.
- `[B]` = my own background knowledge (training data up to mid-2026); no page seen in this session. No URL is given for `[B]`-only items.
- `TITLE-ONLY` = I know the item exists and its title/speaker but have no evidence of its content.

**URL rule followed:** every URL below appeared in a search result or on a fetched page. YouTube links come from fetched GitHub pages (the lsimons keynote index, the PiLastDigit transcript repo, the cocodedk source list) or from search results; none were built from memory.

**Two very useful secondary indexes found (both fetched, `[P]`)**
- AI Engineer World's Fair 2026 keynote index with YouTube links, view counts (as of 2026-08-23) and transcript-based summaries: https://github.com/lsimons/ai-world-fair-keynotes/blob/main/keynotes.md
- Full transcripts + chapter timestamps for all 19 Code w/ Claude San Francisco 2026 recordings: https://github.com/PiLastDigit/Code-With-Claude (summaries there are Gemini-generated; speaker labels are machine-assigned)

---

## Objective: ranked top 30, then further items

### Takeaway
The highest-value 2026 material clusters around four ideas: (1) verification is the bottleneck and the enabler of autonomy, (2) the unit of work is moving from prompt to loop/routine, (3) harness and context should get thinner as models improve (delete prompt, give context not constraints), and (4) a visible counter-current (Dex Horthy, Mario Zechner, Addy Osmani) says humans must still read code and own judgment. The ranking below is my editorial judgment, weighted by citation frequency in search results, view counts where available, and density of concrete lessons; no authoritative "best talks" ranking exists.

### Cited Findings

#### Top 30 (ranked)

**1. Boris Cherny + Cat Wu — "Reflecting on a year of Claude Code"** — Anthropic YouTube channel, ~8-9 June 2026, ~18 min. `[S]` for content; URL `[P]` via source list.
URL: https://www.youtube.com/watch?v=Hth_tLaC2j8 — [cocodedk source list](https://github.com/cocodedk/loop-engineering/blob/main/docs/10-sources.md); also indexed at [aibriefing.dev](https://aibriefing.dev/library/yt-Hth_tLaC2j8/)
- Verification is the first topic (chapter "How to make Claude good at verification" at 1:10): give Claude a way to check its own work. — [X/@hqmank](https://x.com/hqmank/status/2064337891321618663), search summary of YouTube description
- Cherny now uses auto mode instead of plan mode; newer models do not need a separate planning step. — [X/@bcherny](https://x.com/bcherny/status/2064034799711588805)
- Routines fix bugs before he sees them (babysitting PRs, stale bugs); The Neuron calls routines the biggest practical idea in the video. — [The Neuron](https://theneuron.ai/explainer-articles/claude-code-creators-boris-cherny-and-cat-wu-explain-how-to-use-agent-loops/)
- "Why loop is the next leap" (chapter at 10:24): interface moved from source code, to agent, to loop or routine. — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md)
- Context minimalism: do not over-specify; too much context is micromanaging the model (attributed to Cat Wu by secondary sources, attribution unconfirmed). — [The Neuron](https://theneuron.ai/explainer-articles/claude-code-creators-boris-cherny-and-cat-wu-explain-how-to-use-agent-loops/)
- See the dedicated key-question section below for the reconciliation.

**2. Cat Wu + Thariq Shihipar with Simon Willison — "Claude Fable, Claude Tag, and Anthropic's Culture" (fireside)** — AI Engineer World's Fair 2026 (SF, 29 Jun-2 Jul 2026); video 51:30; annotated ~8,000-word transcript published 21 Jul 2026. `[S]`
URL: https://simonwillison.net/2026/Jul/21/cat-and-thariq/ ; video listing https://ai.engineer/orgs/datasette
- Claude Code's system prompt was cut by ~80%; example-heavy few-shot prompts now make the newest models (named in summaries as Fable 5 and Opus 4.8) perform worse. — [ai-tldr.dev](https://ai-tldr.dev/releases/simonw-cat-thariq-claude-code-jul21/)
- Let the model use its own judgment instead of dictating procedure; example: for testing, the model itself picks a lower-power model and runs it in a subagent. — [Simon Willison tag page](https://simonwillison.net/tags/cat-wu/)
- More context beats a pile of prohibitions. — [clauding.de](https://clauding.de/en/posts/claude-code-team-fireside-chat-einblicke/)
- Credential injection: agents call authenticated APIs without ever seeing the key. — [ai-tldr.dev](https://ai-tldr.dev/releases/simonw-cat-thariq-claude-code-jul21/)
- "Claude Tag" (Claude Code via Slack) reportedly lands ~65% of the team's product-engineering PRs. — [ai-tldr.dev](https://ai-tldr.dev/releases/simonw-cat-thariq-claude-code-jul21/)
- Honest eval limits: Cat Wu, "We haven't been able to eval it — we don't have any hard data to show it" (which feature is unclear from the snippet). — [simonwillison.net](https://simonwillison.net/2026/Jul/21/cat-and-thariq/)

**3. Thariq Shihipar — "Field Guide to Fable" (keynote)** — AI Engineer World's Fair 2026, Day 3 (1 Jul 2026), 20-min slot; 153,326 views by 23 Aug 2026 (most-viewed keynote of the event in that index). `[P]` (GitHub transcript-based summary)
URL: https://youtube.com/watch?v=9fubhllmsBU — [lsimons index](https://github.com/lsimons/ai-world-fair-keynotes/blob/main/keynotes.md)
- "Capability overhang": models get smarter in spiky ways; tools/harness decide which spikes you can reach (chat model fails "Pokemon ending in -aw", Claude Code writes a script).
- Claude Code cut 80% of its system prompt; heavy instructions and example lists constrain a model more imaginative than its examples. Give context, not constraints.
- "The map is not the territory": the new bottleneck is your unknowns. Techniques: ask for a "blind spot pass", ask for several visibly different prototypes, have the model interview you about your spec (prioritizing architecture-changing questions), give a reference implementation instead of a spec, have it log implementation notes at each unknown, have it quiz you afterwards.
- Tool evolution example: AskUserQuestion barely worked on Opus 4, interviewed well by Opus 4.5, generates full HTML questionnaires by Opus 4.8/Fable.
- All from [lsimons summary](https://github.com/lsimons/ai-world-fair-keynotes/blob/main/keynotes.md)

**4. Dex Horthy (HumanLayer) — "Harness Engineering is not Enough: Why Software Factories Fail"** — AI Engineer World's Fair 2026, Day 2 (30 Jun 2026), 20-min slot; 60,139 views by 23 Aug. `[P]`
URL: https://youtube.com/watch?v=Ib5GBkD555M — [lsimons index](https://github.com/lsimons/ai-world-fair-keynotes/blob/main/keynotes.md); also https://ai.engineer/talks/harness-engineering-is-not-enough-why-software-factories-fail
- He ran a "lights-off" factory (nobody reads code) from July 2025; it collapsed on an issue no prompting could fix in a codebase he had not read for three months.
- Root cause is training, not skill: benchmarks reward "tests pass", never maintainability, so models add needless try/catch and casts; no harness or loop fixes that.
- Claude Code's edge was a model trained against the harness it ships in; harness builders without weights are structurally disadvantaged (his claim).
- Prescription: turn the lights back on. Sequence: product review, system architecture, program design (types, signatures, call graphs), vertical slices. ~30 minutes of upfront alignment saves hours of review.
- Cites a Faros AI report (review quality down, unreviewed merges and incidents up).

**5. Ryan Lopopolo (OpenAI) — "Harness Engineering: How to Build Software When Humans Steer, Agents Execute"** — AI Engineer (listed under "popular talks"; exact event not confirmed, probably AI Engineer Europe, April 2026), ~46 min. `[S]`
URL: https://www.ai.engineer/talks/am_oeAoUhew-harness-engineering ; companion essay https://openai.com/index/harness-engineering/ (11 Feb 2026); a second recording transcript from AI Native DevCon London: https://tessl.io/registry/ainativedev/aidevcon-2026-ldn/0.100.8/files/talk-lopopolo-harness-engineering/transcript.md
- Agents write all the code; humans do delegation, system design and guardrails.
- Implementation is no longer the scarce resource.
- Make quality legible and enforceable (mechanical checks rather than taste in reviewers' heads).
- `[B]` The essay describes a ~1M-line product built with zero hand-written code, AGENTS.md as a table of contents into a docs tree, custom linters with remediation hints in error messages, and recurring "garbage collection" agents.
- Pair with #4 and #28; one write-up frames the field as a "Zechner-Lopopolo continuum" on whether to still read code. — [prismix.dev](https://prismix.dev/news/f736927f7b33)

**6. Boris Cherny with Lauren Reeder — "Anthropic's Boris Cherny: Why Coding Is Solved, and What Comes Next"** — Sequoia AI Ascent 2026 (event ~20-30 Apr; video published 4 May 2026), ~24-25 min; also on the Training Data podcast feed. `[S]`
URL: https://www.youtube.com/watch?v=SlGRN8jh2RI — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md); write-up [mindsleap.ai](https://www.mindsleap.ai/en/news/boris-cherny-coding-solved-ai-ascent-2026)
- /loop explained as Claude using cron to schedule a repeating job (reported at ~7:56).
- His named loops: babysit PRs, keep CI healthy, cluster Twitter feedback every 30 minutes.
- No hand-written code in 2026; dozens of PRs a day, many from his phone.
- "Coding is solved" is scoped: "at least for the code he writes".

**7. Boris Cherny with Ben Gilbert + David Rosenthal — "Claude Code & the Future of Engineering" (Acquired Unplugged, presented by WorkOS)** — 2 June 2026. Length not found. `[S]`
URL: https://www.youtube.com/watch?v=RkQQ7WEor7w — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md); recap https://workos.com/blog/boris-cherny-claude-code-acquired-interview-takeaways
- Source (per most write-ups) of "I don't prompt Claude anymore... My job is to write loops".
- Three stages: IDE autocomplete; 5-10 parallel manually prompted sessions; loops prompting Claude, with a couple of hundred agents reading his GitHub, Slack and Twitter to decide what to build.
- Loop defined live as a small program that prompts the agent, reads output, decides if done, re-prompts.
- Team's Claude-written share went from "10-20%" to ">90%"; every engineer scopes, talks to users, pulls data, ships design.
- Disputed: one analysis says the transcript has no mention of loops; see loops section.

**8. Andrej Karpathy with Stephanie Zhan — "From Vibe Coding to Agentic Engineering"** — Sequoia AI Ascent 2026; published ~29 Apr 2026. `[S]`
URL: YouTube id 96jN2OCOfLs as shown by [jc-aiwiki](https://jc-aiwiki.pages.dev/karpathy/from-vibe-coding-to-agentic-engineering); podcast version https://pocketcasts.com/podcast/training-data/4cd0e900-0a48-013d-5da1-0acc26574db2/andrej-karpathy-from-vibe-coding-to-agentic-engineering/d66ffbc1-c24d-403c-b39f-f93be691c862 ; event page https://sequoiacap.com/article/ai-ascent-2026
- Draws the line between vibe coding (don't look at code) and agentic engineering (professional, accountable use of agents).
- "You can outsource your thinking, but you can't outsource your understanding."
- Says he feels "more behind" as a programmer than ever.
- A third-party write-up lists five predictions — [mindstudio.ai](https://www.mindstudio.ai/blog/karpathy-sequoia-talk-5-predictions-agentic-engineering) (not opened).

**9. Dex Horthy — "No Vibes Allowed: Solving Hard Problems in Complex Codebases"** — AI Engineer Code Summit (NYC, Nov 2025), 20:31. `[S]` + `[B]`
URL: https://ai.engineer/talks/context-engineering-for-complex-codebases
- Treat the context window as a scarce engineering resource; use subagents to compress exploration.
- Research → Plan → Implement, with "frequent intentional compaction" between phases `[B]`.
- Keep humans on the highest-leverage review points (research and plan), not only the diff `[B]`.
- Secondary claim that quality drops past ~40% context use (the "dumb zone"); unverified figure. — search summary of https://ai.engineer/talks/context-engineering-for-complex-codebases

**10. Barry Zhang + Mahesh Murag (Anthropic) — "Don't Build Agents, Build Skills Instead"** — AI Engineer Code Summit, Nov 2025, ~16 min. `[S]`
URL: https://ai.engineer/talks/CEvIs9y1uog-agent-skills
- Stop building one agent per domain; package domain expertise as skills for a general agent.
- A skill is a folder: instructions, scripts, assets.
- `[B]` Progressive disclosure: only name/description sit in context until the skill is needed; code is the universal interface.

**11. Thariq Shihipar — "Claude Agent SDK [Full Workshop]"** — AI Engineer Code Summit workshop; published ~5 Jan 2026; ~1h52. `[S]`
URL: https://ai.engineer/talks/TqC1qOfiVcQ-claude-agent-sdk-workshop
- The agent loop: gather context → take action → verify work.
- Bash + filesystem agents beat tool-only designs for autonomous work.
- Security as "Swiss cheese" defence layers.
- Prototype in Claude Code first, then ship with the SDK.

**12. Boris Cherny + Jarred Sumner — "Live coding session"** — Code w/ Claude SF, 6 May 2026, 32:00. `[P]` (session page + transcript)
URL: https://www.youtube.com/live/DlTCu_pNDHE ; session page https://claude.com/code-with-claude/session/sf-live-coding-session-with-boris-cherny-and-jarred-sumner
- "RoboBun" reproduces every GitHub issue before a human looks; hard rule: no PR without a passing test that proves the fix (2:47).
- Layered review: CodeRabbit for style, Claude Code review for context-dependent bugs (5:01); agents fix review comments themselves.
- CLAUDE.md is what keeps it maintainable (8:19); let the agent read CI errors and build logs to close the full loop (10:23).
- Hill climbing: give a metric and a way to measure, and in auto mode it "will just keep going until it's done" (20:26).
- Auto mode exists because permission prompts killed overnight runs; Cherny: "Claude runs for hours and hours... run it almost every night" (22:21, ~29:00).
- Next bottleneck: deeper verification and communicating "sufficient proof", then planning (25:30).
- Transcript: [PiLastDigit repo](https://github.com/PiLastDigit/Code-With-Claude)

**13. Mahesh Murag (Anthropic) — "Memory and dreaming for self-learning agents"** — Code w/ Claude SF, 6 May 2026, 24:28 (repeated in London and Tokyo). `[P]`
URL: https://youtu.be/RtywqDFBYnQ ; https://claude.com/code-with-claude/session/sf-memory-and-dreaming-for-self-learning-agents
- Memory modelled as a filesystem the agent manages with bash/grep.
- Production features: read-only vs read-write scopes, optimistic concurrency, audit log with version history; standalone API for PII scans and review.
- "Dreaming": asynchronous out-of-band pass over many sessions that finds patterns/mistakes, dedupes and updates memory, off the hot path.
- Reported results: Rakuten 90% fewer first-pass mistakes; Harvey 6x task completion on one legal-benchmark scenario (vendor-reported).
- From [PiLastDigit transcript summary](https://github.com/PiLastDigit/Code-With-Claude)

**14. Addy Osmani — Closing keynote ("The engineer of the future is the person who is able to choose what is worth doing")** — AI Engineer World's Fair 2026, Day 3; 82,070 views. `[P]`
URL: https://youtube.com/watch?v=n97BCfyFIvw — [lsimons index](https://github.com/lsimons/ai-world-fair-keynotes/blob/main/keynotes.md)
- Arc: harness engineering → loop engineering → software factory; inner loop (agents: investigate, implement, test, report) vs outer loop (humans: decide, verify, approve, own).
- "Distrust without bandwidth": per Sonar research 96% distrust AI code but about half always verify. Fix: make verification cheaper and harder to skip.
- Three hazards: cognitive debt, cognitive surrender, orchestration tax (parallel agents do not multiply your attention).
- Rule: "explain it or don't ship it".

**15. Alexander Embiricos + Romain Huet + Peter Steinberger (OpenAI) — "The Golden Age of AI Engineering"** — AI Engineer World's Fair 2026, Day 2; 74,185 views. `[P]`
URL: https://youtube.com/watch?v=pMggiOb18tc — [lsimons index](https://github.com/lsimons/ai-world-fair-keynotes/blob/main/keynotes.md)
- Bottleneck moved from code generation to human attention; "value maxing" not "token maxing".
- Capability arc: completion → edits the model could not test → models that test their own work → long unsupervised goals.
- Codex is built as open layers: Responses API, open-source harness, AGENTS.md, open App Server.
- Product shape: chat-first with a collaborative surface for when you need to inspect or steer.
- Steinberger segment: stronger harnesses, orchestration and loops over juggling terminals. — [search summary](https://ai.engineer/talks/the-golden-age-of-ai-engineering)

**16. Andrej Karpathy — "Software Is Changing (Again)" (Software 3.0)** — YC AI Startup School, SF, 16-17 June 2025, ~39 min. `[S]` + `[B]`
URL: https://www.youtube.com/watch?v=LCEmiRjPEtQ ; https://www.ycombinator.com/library/MW-andrej-karpathy-software-is-changing-again (both shown in search summary)
- Software 1.0 code, 2.0 weights, 3.0 prompts in English.
- Build partial-autonomy products with an "autonomy slider" and fast human verification through GUIs.
- Keep the AI "on a leash": small diffs, fast generate-verify loop `[B]`.
- Build for agents as a new consumer of software (docs for LLMs, llms.txt) .

**17. Dwarkesh Patel × Andrej Karpathy — "AGI is still a decade away"** — Dwarkesh Podcast, 17 Oct 2025. `[S]`
URL: https://www.dwarkesh.com/p/andrej-karpathy
- "Decade of agents", not year of agents; agents are capable but cognitively limited interns.
- No continual learning: you cannot tell them something and have it stick.
- RL is inefficient: one outcome reward smeared over a trajectory.
- Demo-to-product gap, from self-driving experience.
- Useful calibration against 2026 "coding is solved" claims.

**18. Lenny's Podcast — "Head of Claude Code: What happens after coding is solved | Boris Cherny"** — 19 Feb 2026. `[S]`
URL: https://www.lennysnewsletter.com/p/head-of-claude-code-what-happens ; Lenny's takeaways https://x.com/lennysan/status/2024896611818897438
- "I ship 10 to 20 to 30 PRs every day"; no hand-edited line since November 2025.
- At this date he said "I start about 80% of my tasks in plan mode" (contrast with June 2026). — [note.com near-verbatim transcript, via cocodedk](https://note.com/ai_eng_tech/n/nf940a6dc47e6?hl=en)
- "Claude reviews 100% of the PRs."
- The 4%-of-public-GitHub-commits figure is SemiAnalysis's, cited not measured by Cherny. — [cocodedk timeline](https://github.com/cocodedk/loop-engineering/blob/main/docs/07-receipts-and-timeline.md)
- Contains no "loops" framing.

**19. The Pragmatic Engineer — "Building Claude Code with Boris Cherny"** — ~4 Mar 2026. `[S]`
URL: https://newsletter.pragmaticengineer.com/p/building-claude-code-with-boris-cherny ; companion deep dive https://newsletter.pragmaticengineer.com/p/how-claude-code-is-built
- Parallel agents workflow; plan mode then auto-accept (as of early 2026).
- Agentic search beat RAG; PRDs abandoned in favour of prototypes. — [cocodedk source notes](https://github.com/cocodedk/loop-engineering/blob/main/docs/10-sources.md)
- Tag teammates' PRs to add learnings to the shared CLAUDE.md. — [InfoQ](https://infoq.com/news/2026/01/claude-code-creator-workflow/)
- Which engineer archetypes and skills still matter.

**20. Latent Space — "Extreme Harness Engineering for Token Billionaires" (Ryan Lopopolo, OpenAI)** — April 2026, ~1h12. `[S]`
URL: https://latent.space/p/harness-eng
- Long-form version of #5: harness engineering for Codex, agent orchestration, code generation at very high token volumes. Content beyond the title/topic list not seen.

**21. Dex Horthy — "12-Factor Agents: Patterns of reliable LLM applications"** — AI Engineer World's Fair, June 2025 (YouTube ~3 Jul 2025). `[S]` + `[B]`
URL: https://ai.engineer/talks/8kMaTybvDUw-12-factor-agents
- Production agents are mostly deterministic software with targeted LLM steps.
- `[B]` Own your prompts, own your context window, own your control flow; tools are just structured output; small focused agents; stateless reducer; contact humans via tool calls.

**22. Barry Zhang (Anthropic) — "How We Build Effective Agents"** — AI Engineer Summit, NYC, Feb 2025. `[S]` + `[B]`
URL: https://ai.engineer/talks/D7_ipDqhtwk-effective-ai-agents
- `[B]` Don't build agents for everything; keep it simple; think like your agent (look at exactly what is in its context window).
- Workflows (predefined code paths) vs agents (model directs its own process). — [search summary](https://ai.engineer/talks/effective-ai-agents)
- Companion: Anthropic video "Building more effective AI agents" with Erik Schluntz and Alex Albert (~Oct 2025): simplicity and taking the agent's perspective. — [neura.market listing](https://neura.market/directories/claude/videos/building-more-effective-ai-agents)

**23. Every "AI & I" (Dan Shipper) — "How to Use Claude Code Like the People Who Built It" (Cat Wu + Boris Cherny)** — 29 Oct 2025. `[S]`
URL: https://every.to/podcast/transcript-how-to-use-claude-code-like-the-people-who-built-it
- Cat Wu on memory synthesis and agentic search; Boris on plan mode and the CLAUDE.md correction loop (add an entry whenever Claude does something wrong). — [cocodedk source notes](https://github.com/cocodedk/loop-engineering/blob/main/docs/10-sources.md)
- How Anthropic dogfoods Claude Code; origin story.
- Predates the loops framing; good baseline for how advice changed.

**24. Lex Fridman Podcast #491 — "OpenClaw: The Viral AI Agent that Broke the Internet – Peter Steinberger"** — 11 Feb 2026. `[S]`
URL: https://lexfridman.com/peter-steinberger/ ; transcript https://lexfridman.com/peter-steinberger-transcript/
- Personal always-on agent wired to messaging apps that edits its own code; from one-hour prototype.
- Terminal workflow with multiple coding agents.
- Security burden of owning your data and giving an agent access.
- Prefers "agentic engineering"; calls "vibe coding" a slur.

**25. Mario Rodriguez (GitHub) + Brad Abrams (Anthropic) — "Caching, harnesses, and advisors: Building on Claude at GitHub scale"** — Code w/ Claude SF, 6 May 2026, 26:15. `[P]`
URL: https://www.youtube.com/live/y5TmF_6o6xk ; https://claude.com/code-with-claude/session/sf-caching-harnesses-and-advisors-building-on-claude-at-github-scale
- Target >94% prompt-cache hit rate: static system prompt, nothing dynamic in the prefix.
- Advisor strategy: cheap fast executor (Haiku) consults a stronger model (Opus) only when needed.
- "Rubber duck" critique model at fixed points: after the plan, after complex implementation, before tests.
- Measure outcomes (code survival rate), not activity (acceptance rate).
- New-model lifecycle: harness update, offline benchmarks, internal dogfooding.

**26. Replit — "Evaluating and improving Replit Agent at scale"** — Code w/ Claude SF, May 2026, 27:42. `[P]`
URL: https://youtu.be/snroDwX1-JU — [PiLastDigit repo](https://github.com/PiLastDigit/Code-With-Claude)
- Evals as a continuous engine, not a boolean gate.
- Two pillars: offline benchmark (open-source "ViBench", built from real traces, built-from-scratch apps) and online monitoring.
- "Vibe-on-vibe" tasks: models struggle to extend their own earlier output.
- "Telescope": cluster millions of daily traces to find failure patterns and generate patches.
- A/B tests give conflicting signals (cost, speed, sentiment); human product judgment decides.

**27. Latent Space — "Context Engineering for Agents" (Lance Martin, then LangChain)** — 11 Sep 2025, 57 min. `[S]` + `[B]`
URL: https://www.latent.space/p/context-engineering-for-agents-lance
- Managing context across multi-step tool-calling workflows.
- `[B]` Taxonomy: write/offload, select, compress, isolate; the Bitter Lesson applied to harnesses (remove structure as models improve).
- Follow-up: Chroma interview "Lance Martin on the Current State of Agents" (16 Jan 2026): context rot, tool bloat, sub-agents, offloading, continual learning. — https://trychroma.com/interviews/lance-martin-context-engineering-ep-3

**28. Mario Zechner — "Building pi in a World of Slop" (keynote)** — AI Engineer Europe, London, 10 Apr 2026. `[S]`, content thin.
URL: no talk page seen; coverage https://dabase.com/blog/2026/ai-engineer-europe/ and https://prismix.dev/news/f736927f7b33
- Tells engineers to slow down and treat agents as surgical tools; reportedly got a standing ovation.
- Warns of coding-agent-caused outages (as cited by Dex Horthy in #4). — [lsimons summary](https://github.com/lsimons/ai-world-fair-keynotes/blob/main/keynotes.md)
- Minimal-harness philosophy behind the pi agent `[B]`.

**29. Boris Cherny — "We Cut 80% of Claude Code's Prompt"** — YC Startup School 2026 conversation. Date and length not found. `[S]`
URL: https://www.ycombinator.com/library/UN-boris-cherny-building-claude-code ; transcript mirror https://sozai.app/transcript/boris-cherny-cut-80-percent-claude-code-prompt/
- Team deleted over 80% of the system prompt; framed as getting the product out of the model's way.
- Frequent ablations: delete and rebuild prompts and code for each new model. — [i10x.ai](https://i10x.ai/news/claude-code-cherny-agent-tutorials)

**30. Anthropic — "The capability curve"** — Code w/ Claude SF, May 2026, 15:05 (also Tokyo). Speaker not captured. `[P]`
URL: https://www.youtube.com/live/tP4MGcJ80Y0 — [PiLastDigit repo](https://github.com/PiLastDigit/Code-With-Claude)
- SWE-bench Verified 62% (Sonnet 3.7) → 87% (Opus 4.7) in twelve months.
- Gains land in planning before acting, error recovery (escaping doom loops) and attention over long runs.
- Re-audit scaffolding: removing legacy instructions and workflows often improves results and cuts tokens.
- Build evals that mirror your product and check they are not saturated.
- Give the model "room to work": adaptive thinking, controlled tool access (auto mode), ability to inspect its own output.

#### Further items (briefer)

AI Engineer World's Fair 2026 (all `[P]` from the [lsimons index](https://github.com/lsimons/ai-world-fair-keynotes/blob/main/keynotes.md)):
- swyx — "The Highest Loop" (opening, 5 min): arc token → chat → tools → goals → automations, now "loop stacking"; the skill is choosing which loop you work in. Livestream timestamp https://youtube.com/watch?v=htM02KMNZnk&t=50
- swyx — "6 Things to Know about AIE World's Fair 2026" (17:50). `[S]` https://www.ai.engineer/talks/0S8xe9ftGTM-6-things-know-about-aie-worlds-fair
- Theo Browne — closing keynote "Everything we knew about software has changed" (89,636 views): tool-call era (Sonnet 3.5) → long unsupervised tasks (Opus 4.5) → self-orchestration; "markdown file tier" products (a PR-triage service that is a markdown file on a 9am cron); build wider, drop sunk-cost habits. https://youtube.com/watch?v=xUnRQ9vLXxo
- Garry Tan — "Every company should have a Brain" (65,879 views). https://youtube.com/watch?v=eBUyTS7SzV4 TITLE + one-line description only.
- Emil Eifrem (Neo4j) — "Thinner Agents on a Smarter Substrate" (42,938 views). https://youtube.com/watch?v=VGN22pPpb-8
- Tariq Shaukat (Sonar) — "In the Land of AI Agents, the Verifiers Are King". Livestream https://youtube.com/watch?v=4sX_He5c4sI&t=2135
- Aparna Dhinakaran (Arize) — evals track intro, "The Future of Evals: From LLM as a Judge to Agent as a Judge" (6 min): evals must change as fast as the agent. https://youtube.com/watch?v=q2JrUKBMf0w
- Erik Meijer — "In Code They Act, In Proof We Trust". https://youtube.com/watch?v=-CnA2lGfymY
- Lee Robinson (Cursor) — "Recursive Model Improvement". https://youtube.com/watch?v=q4Tr-DknG2M
- Mike Krieger + swyx — "How Anthropic Builds: Lessons from Labs". Livestream https://youtube.com/watch?v=I2cbIws9j10&t=4338
- Barr Yaron — "The 2026 State of AI Engineering". https://youtube.com/watch?v=RGe6EjucbzI
- Geoffrey Huntley vs Ian Livingstone loops debate, moderated by Allie Howe. `[S]`, no URL seen.

Other AI Engineer talks (`[S]`):
- Geoffrey Huntley — Ralph loops talk. https://ai.engineer/talks/2TLXsxkz0zI-ralph-loops-build-dumb-ai-loops-that (TITLE-ONLY; `[B]`: a bash while-loop feeding the same prompt to a fresh context each iteration)
- swyx — "Why Agent Engineering" (AI Engineer Summit, Feb 2025). https://www.ai.engineer/talks/5N33E9tC400-why-agent-engineering
- Dan Shipper — "Dispatch from the Future: Building an AI-Native Company" (AIE Code 2025). https://www.ai.engineer/talks/MGzymaYBiss-dispatch-from-future-building-ai-native-company
- Sean Grove (OpenAI) — "The New Code" / specs as the new code (AIE WF 2025). `[B]`, no URL seen; Horthy credits it for his spec-driven approach per [Latent Space wiki](https://latent.space/wiki/dex-horthy/).
- Latent Space recap "5 Trends That Defined AI Engineering at World's Fair 2026": systems around agents, loop engineering as human control layer, software factories and FDEs, coding agents replacing IDEs, everything converging on skills. https://latent.space/p/aiewf26trends/

Code w/ Claude 2026 (SF 6-7 May, London 19-20 May, Tokyo 10-11 June), all `[P]` from claude.com pages or the transcript repo:
- Dickson Tsai — "What's new in Claude Code" (24:55): remote control, auto mode (classifier-approved tool calls), worktrees, auto memory, multi-agent /ultra-review, routines (cron, GitHub webhook, API triggers). https://www.youtube.com/live/IMZa42k6L6M
- Fiona Fung — "Running an AI-native engineering org" (28:37): bottleneck moved to verification, review and alignment; kill legacy process; settle debates with competing PRs. https://youtu.be/igO8iyca2_g
- Cursor — "Giving coding agents their own computers" (14:35): onboard agents like employees, give them eyes, invest in agent DevX. https://youtu.be/BbYSGxtsMic
- Panel — "Building AI-native: Cognition, Gamma, Harvey" (28:15, incl. Walden Yan): expect to re-architect every six months. https://www.youtube.com/live/OFDm3T7pVlc
- Jess Yan + Lance Martin — "How to get to production faster with Claude Managed Agents" (17:25). https://www.youtube.com/live/E9gaQHrw_rg
- Donny Greenberg (Tokyo listing) — "The thinking lever" (24:01): effort and thinking budgets. https://youtu.be/OXJO4LldSnc
- Lucas Gonzalez — "The expanding toolkit" (21:22). https://youtu.be/KLCuxMDZSDg
- Datadog — "How Datadog built a universal machine tool for Claude Code" (30:50). https://www.youtube.com/live/EdmuYPBt_EM
- Opening keynote (47:29). https://www.youtube.com/live/GMIWm5y90xA
- Workshops with session pages but no recording link seen (descriptions only):
  - Thariq Shihipar — "How we Claude Code": https://claude.com/code-with-claude/session/sf-ext-how-we-claude-code
  - Daisy Hollman — "Beyond the basics with Claude Code" (CLAUDE.md, MCP, skills, auto mode safely; page shows "Watch recording"): https://claude.com/code-with-claude/session/ldn-beyond-the-basics-with-claude-code
  - Sid Bidasaria — "Stop babysitting your agents": https://claude.com/code-with-claude/session/tyo-stop-babysitting-your-agents
  - Ash Prabaker + Christian Ryan — "Build a proactive agent workflow with Claude Code" (routines, /schedule): https://claude.com/code-with-claude/session/tyo-proactive-agent-workflow
  - Hannah Moran — "The prompting playbook": https://claude.com/code-with-claude/session/tyo-the-prompting-playbook
  - Will Steuk / Tanveer Mittal — "Tool, skill, or subagent? Decomposing an agent that outgrew its prompt": https://claude.com/code-with-claude/session/ldn-ext-tool-skill-or-subagent-decomposing-an-agent-that-outgrew-its-prompt
  - Felix Becker / Jiri De Jonghe — "Evals for taste: hill-climbing a slide-generation agent": https://claude.com/code-with-claude/session/sf-ext-eval-driven-agent-development
  - Descript — "Evals for subjective, stateful agents": https://claude.com/code-with-claude/session/sf-ext-evals-for-subjective-stateful-agents
  - Nick Khami (Mintlify) — "Designing multi-agent systems: when to split, when to sandbox, what to ship": https://claude.com/code-with-claude/session/sf-ext-designing-multi-agent-systems-when-to-split-when-to-sandbox-what-to-ship
  - Tina Vachovsky — "Agents that remember": https://claude.com/code-with-claude/session/sf-ext-agents-that-remember

Code w/ Claude 2025 (SF, 22 May 2025), `[S]`:
- Boris Cherny — "Mastering Claude Code in 30 minutes"; Jeremy Hadfield — "Prompting for Agents"; "Building headless automation with Claude Code" (Sid Bidasaria listed). Agenda https://anthropic.swoogo.com/codewithclauderegister/agenda ; summaries https://lawwu.github.io/posts/2025-08-14-claude-code-conference-summary . No official video URLs seen.

Podcasts and interviews (`[S]`):
- Latent Space — "Claude Code: Anthropic's Agent in Your Terminal" (Cat Wu + Boris Cherny, May 2025). https://www.latent.space/p/claude-code
- Latent Space — Thariq Shihipar episode. https://latent.space/p/thariq (TITLE-ONLY)
- Latent Space — "The Age of Async Agents" (Walden Yan + Cole Murray, 28 May 2026). https://latent.space/p/cognition ; related essay "Multi-Agents: What's Actually Working" (22 Apr 2026): keep writes single-threaded, let extra agents add intelligence. https://cognition.ai/blog/multi-agents-working
- Latent Space — "Codex from 0 to 10M Users: Building ChatGPT Work" (28 Jul 2026). No URL seen.
- Pragmatic Engineer — "The creator of Clawd: I ship code I don't read" (Peter Steinberger, 28 Jan 2026). https://newsletter.pragmaticengineer.com/p/the-creator-of-clawd-i-ship-code
- Pragmatic Engineer — "Context engineering with Dex Horthy". https://newsletter.pragmaticengineer.com/p/context-engineering-with-dex-horthy (TITLE-ONLY)
- Chroma context-engineering interviews: Dex Horthy https://trychroma.com/interviews/dex-horthy-context-engineering-ep-1 ; Drew Breunig https://www.trychroma.com/interviews/drew-breunig-context-engineering-ep-2
- Changelog — "Opus 4.5 changed everything" (Burke Holland, 27 Feb 2026, 104 min). https://www.podscan.fm/podcasts/changelog-interviews/episodes/opus-45-changed-everything
- YC Lightcone — "We're All Addicted To Claude Code" (Calvin French-Owen, 6 Feb 2026): Codex vs Claude Code vs Cursor. https://zeno.fm/podcast/y-combinator-startup-podcast/episodes/were-all-addicted-to-claude-code
- Every AI & I — "How Two Engineers Ship Like a Team of 15 With AI Agents" (Kieran Klaassen, Nityesh Agarwal; compound engineering). https://every.to/podcast/how-two-engineers-ship-like-a-team-of-15-with-ai-agents-7bc186bd-b5ea-40cd-9690-963845203f80
- Dev Interrupted — "Inventing the Ralph Wiggum Loop" (Geoffrey Huntley). https://linearb.io/dev-interrupted/podcast/inventing-the-ralph-wiggum-loop
- The Developing Dev — Boris Cherny career interview. https://www.developing.dev/p/boris-cherny-creator-of-claude-code
- AWS re:Invent 2025 — "What Anthropic learned building AI agents in 2025 (AIM277)". https://dev.to/kazuya_dev/aws-reinvent-2025-what-anthropic-learned-building-ai-agents-in-2025-aim277-16lc (TITLE-ONLY)

### Inferences
- View counts in the keynote index put Thariq's "Field Guide to Fable" (153k) far ahead, then Theo Browne (90k), Addy Osmani (82k), OpenAI's Golden Age (74k), Garry Tan (66k) and Dex Horthy (60k); that is the best available proxy for "most praised" at World's Fair 2026.
- The Anthropic material (items 1, 2, 3, 12, 29, 30) agrees on one direction: remove scaffolding and prompt text, add verification and autonomy. Horthy/Zechner/Osmani are the organised counter-position, and all three accept loops but insist on human reading or accountability.
- Advice has a short shelf life: plan-mode-first (Oct 2025 to Feb 2026) was reversed by June 2026. Anything older than about six months should be read as model-version-specific.

### Gaps
- No authoritative ranking of "best" talks exists; ai.engineer and youtube.com could not be fetched, so view counts exist only for the 16 World's Fair 2026 keynotes with standalone videos.
- Lopopolo's AI Engineer talk: event and date not confirmed.
- Zechner keynote, Huntley talk, Latent Space Thariq/Lopopolo episodes: content known only from titles or one-line summaries.
- AI Engineer Code Summit 2026 (ai.engineer/code lists 11-12 Nov) may have happened days ago; nothing was found on its talks.

---

## Key question 1: The Cherny + Cat Wu one-year conversation (Hth_tLaC2j8): verification, auto vs plan mode, routines, loops, context

### Takeaway
The video is Anthropic's "Reflecting on a year of Claude Code", about 18 minutes, published around 8-9 June 2026. Secondary write-ups agree on five themes (verification first, auto mode replacing plan mode, routines, loops as the next interface, context minimalism) but I could not read a transcript, and several quote attributions (especially to Cat Wu) are unconfirmed even by the write-ups themselves.

### Cited Findings
- Title and URL: "Reflecting on a year of Claude Code — Anthropic, YouTube", https://www.youtube.com/watch?v=Hth_tLaC2j8, dated "~June 8, 2026". `[P]` — [cocodedk sources doc](https://github.com/cocodedk/loop-engineering/blob/main/docs/10-sources.md)
- Length: "an 18-min conversation"; topics "verification, auto mode, routines, loops, and context"; started as a Slack demo that got two emoji reactions. `[S]` — [X/@hqmank](https://x.com/hqmank/status/2064337891321618663)
- Cherny's own summary: "why I use auto mode instead of plan mode, how routines fix bugs before I see them, why I do most of my coding from my phone now". `[S]` — [X/@bcherny](https://x.com/bcherny/status/2064034799711588805)
- Chapter markers reported from the YouTube description: "How to make Claude good at verification" at 1:10; "Why loop is the next leap" at 10:24. `[S]` — search summary citing [aibriefing.dev](https://aibriefing.dev/library/yt-Hth_tLaC2j8/)
- Plan mode quote (from a fan-run site, unverified): "What it used to be is plan mode. I don't use that anymore. I use auto mode instead of plan mode. The newer models don't actually need a planning step." Planning "was really important for Opus 4 through 4.5, but starting with 4.6, and definitely with 4.7, it just doesn't need it." `[S]` — [howborisusesclaudecode.com](https://howborisusesclaudecode.com/)
- Loops/routines: "the interface moved from source code, to agent, to loop or routine"; a Korean recap says developers are moving to running agents through loops or routines, a second shift after source code → agent. `[P]`/`[S]` — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md); [brunch.co.kr](https://brunch.co.kr/@beingcognitive/42)
- Routines: The Neuron calls them the biggest practical idea; they "babysit PRs and stale bugs" and listen for tickets/issues. `[S]` — [The Neuron](https://www.theneuron.ai/explainer-articles/claude-code-creators-boris-cherny-and-cat-wu-explain-how-to-use-agent-loops/)
- Agent fleets: "agents prompting agents in trees of thousands"; "armies of agents". `[S]` — [The Neuron](https://www.theneuron.ai/explainer-articles/claude-code-creators-boris-cherny-and-cat-wu-explain-how-to-use-agent-loops/)
- Context: Wu's stance is to avoid giving the model too much context, which amounts to micromanaging; The Neuron labels this "context minimalism". `[S]` — [The Neuron](https://www.theneuron.ai/explainer-articles/claude-code-creators-boris-cherny-and-cat-wu-explain-how-to-use-agent-loops/)
- A different reading of the same video: central thesis is that product and engineering roles are merging ("Product Taste Beats Prompting"). `[S]` — [thekb.eu](https://www.thekb.eu/en/fiches/cherny-wu-reflecting-year-claude-code-2026-07-17/)
- Caveats recorded by the write-ups: YouTube captions do not attribute speakers, so quotes are assigned by inference; "the prompt is becoming the smallest unit of work" is The Neuron's gloss, not a Cherny quote; "some Cat Wu attributions are unconfirmed against the primary transcript". `[P]` — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md)
- Verification background from Cherny's 2 Jan 2026 thread: verification is "probably the most important thing"; a feedback loop will "2-3x the quality"; for long runs use a background agent, a Stop hook or the ralph-wiggum plugin. `[P]` (via cocodedk docs) — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md)
- Auto mode facts: classifier-based permissioning; Anthropic's engineering post gives a 93% permission-acceptance figure versus a rhetorical "99%" in interviews. `[P]` (via cocodedk source note) — [anthropic.com/engineering/claude-code-auto-mode](https://anthropic.com/engineering/claude-code-auto-mode)
- Date conflict: sources give 8 June, 9 June and (one outlier) 30 Sept 2026; a separate ~47-second LinkedIn clip with a similar title exists. `[S]` — [thekb.eu](https://www.thekb.eu/en/fiches/cherny-wu-reflecting-year-claude-code-2026-07-17/)

### Inferences
- Reconciled reading: (1) verification is the precondition for everything else; (2) auto mode replaced plan mode because of model capability from Opus 4.6/4.7 onward, not because plan mode was wrong, which matches his Feb 2026 statement that he started ~80% of tasks in plan mode; (3) routines are the event/cron-triggered form and /loop the in-session form of the same idea; (4) context advice is "less, and descriptive rather than prescriptive", consistent with the 80% prompt cut reported in items 2, 3 and 29.
- The 30 Sept date is most likely a page-update or indexing date; Cherny's X post id sits next to his 9 June IDE post id, supporting 8-9 June.
- The "micromanaging" line should be reported as paraphrase attributed to Cat Wu, not as a quote.

### Gaps
- No transcript or first-hand viewing; exact wording, per-speaker attribution and the specifics of what routines Cherny runs in this video are unverified. theneuron.ai must be allowed in network settings to read the most detailed explainer.
- No X thread beyond the two posts above was retrievable.

---

## Key question 2: "My job is to write loops": which event, which date, what argument

### Takeaway
The sentence is most consistently attributed to Acquired Unplugged (presented by WorkOS, 2 June 2026, with Ben Gilbert and David Rosenthal). Meta @Scale (22 June 2026) is a different, later remark ("loops are just as important and as big a step"). A Sequoia AI Ascent interview (published 4 May 2026) contains earlier loops material and is named by one source as the firmest origin, so the true first utterance is unresolved without the videos.

### Cited Findings
- Quote as transcribed by several outlets: "I don't prompt Claude anymore. I have loops that are running. They're the ones that are prompting Claude and figuring out what to do. My job is to write loops." Attributed to Acquired Unplugged, 2 June 2026. `[P]`/`[S]` — [cocodedk timeline](https://github.com/cocodedk/loop-engineering/blob/main/docs/07-receipts-and-timeline.md); [oscargallegoruiz.com](https://www.oscargallegoruiz.com/en/blog/stop-prompting-write-loops/)
- Event recording: "Boris Cherny: Claude Code & the Future of Engineering — Acquired Unplugged (WorkOS)", https://www.youtube.com/watch?v=RkQQ7WEor7w; hosts Ben Gilbert and David Rosenthal confirmed by the WorkOS recap. `[P]` — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md)
- The loops passage is reported at about 11:14 of the Acquired video. `[S]` — search summary; [WorkOS recap](https://workos.com/blog/boris-cherny-claude-code-acquired-interview-takeaways)
- Contradiction: one analysis says the Acquired transcript has "zero mentions of loops" (it covers origin story and sub-agents) and places the loops passage in a separate ~24-minute Sequoia talk. `[S]` — [blakecrosley.com](https://blakecrosley.com/blog/loops-win-where-verification-is-cheap)
- Sequoia AI Ascent 2026 video "Anthropic's Boris Cherny: Why Coding Is Solved, and What Comes Next" (https://www.youtube.com/watch?v=SlGRN8jh2RI, published 4 May 2026) is called "the firmest first-party origin" of the statement in one document, while the same repo's timeline credits Acquired; neither transcript was machine-read by that author. `[P]` — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md)
- Meta @Scale, 22 June 2026: asked whether loops are hype, Cherny said "Yes, they're for real" and "As big as the step from source code to agents was, loops are just as important and as big a step." He described two of his loops: one hunting architectural improvements, one unifying duplicated abstractions, both opening PRs. `[S]` — [TechCrunch](https://techcrunch.com/2026/06/22/the-ai-world-is-getting-loopy/)
- The argument, assembled:
  - Three stages: IDE autocomplete; 5-10 parallel hand-prompted sessions; loops that prompt Claude, with a couple of hundred agents reading GitHub, Slack and Twitter. `[P]` — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md)
  - Loop = small program that prompts the agent, reads the output, decides if done, re-prompts. `[S]` — [The New Stack](https://thenewstack.io/loop-engineering/)
  - Abstraction ladder: punch cards → assembly → high-level languages → prompting → loops. `[S]` — [The New Stack](https://thenewstack.io/loop-engineering/)
  - Anatomy in later Fortune-based reporting: trigger, task, verification step, stopping condition; thousands of agents overnight. `[S]` — [startupfortune.com](https://startupfortune.com/boris-cherny-says-he-now-runs-thousands-of-claude-code-agents-at-once/)
  - Humans remain: "Someone has to prompt the Claudes, talk to customers, coordinate with other teams, decide what to build next." `[P]` — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md)
- "Loop engineering" is a third-party label (Addy Osmani, Peter Steinberger, journalists), not Cherny's. Steinberger's 7 June tweet ("you shouldn't be prompting coding agents anymore") drew ~2.2M views. `[P]`/`[S]` — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md); [X/@mvanhorn](https://x.com/mvanhorn/status/2063865685558903149)
- Critiques: loops have no spending ceiling; "loops win where verification is cheap"; Armin Ronacher's "The coming loop". `[S]` — [aichief.com](https://aichief.com/news/ais-looping-dilemma/); [blakecrosley.com](https://blakecrosley.com/blog/loops-win-where-verification-is-cheap); [lucumr.pocoo.org](https://lucumr.pocoo.org/2026/6/23/the-coming-loop.md)
- Lenny's Podcast (19 Feb 2026) does not contain the loops framing. `[P]` — [cocodedk](https://github.com/cocodedk/loop-engineering/blob/main/docs/01-who-is-boris-cherny.md)

### Inferences
- The WorkOS-versus-Meta conflict is two different quotes at two events twenty days apart, not one quote with disputed provenance. Cite "my job is to write loops" to Acquired Unplugged (2 June 2026) with a "per secondary transcriptions" caveat, and "as big a step" to Meta @Scale (22 June 2026).
- The idea was already public at Sequoia in late April/May (the /loop-as-cron explanation and named loops), so the June remark was a sharper restatement, not a new position.

### Gaps
- Nobody in the sources read an official transcript of the Acquired or Sequoia videos; the exact sentence and timestamp remain unverified. workos.com and youtube.com must be allowed to settle it.
- No recording URL for the Meta @Scale session was found.

---

## Key question 3: Most praised agent talks by event

### Takeaway
World's Fair 2026 and Code w/ Claude 2026 are well documented (keynote index with views; full SF transcripts). For AI Engineer 2025 events, DevDay and Google I/O I found talk pages and directories but no popularity data, and for YC Startup School 2026 only one talk.

### Cited Findings
- AI Engineer World's Fair 2026 (SF, 29 Jun-2 Jul): 25 keynotes, 16 with standalone videos; views as of 23 Aug 2026: Field Guide to Fable 153,326; Theo Browne 89,636; Addy Osmani 82,070; Golden Age of AI Engineering 74,185; Garry Tan 65,879; Dex Horthy 60,139; Emil Eifrem 42,938. `[P]` — [lsimons index](https://github.com/lsimons/ai-world-fair-keynotes/blob/main/keynotes.md)
- Latent Space's recap names five trends (systems around agents; loop engineering; FDEs and software factories; coding agents replacing IDEs; convergence on skills). `[S]` — [Latent Space](https://latent.space/p/aiewf26trends/)
- Simon Willison's fireside with Cat Wu and Thariq was at World's Fair 2026; video out and transcript published 21 Jul 2026. `[S]` — [X/@simonw](https://x.com/simonw/status/2079551159514411437)
- AI Engineer Code Summit (Nov 2025): "Don't Build Agents, Build Skills Instead" and "No Vibes Allowed" ran back to back; Thariq's Agent SDK workshop came from the same event. `[S]` — [ai.engineer agent-skills](https://ai.engineer/talks/agent-skills); [ai.engineer workshop](https://ai.engineer/talks/TqC1qOfiVcQ-claude-agent-sdk-workshop)
- AI Engineer World's Fair 2025: ~40 of ~1000 submissions were chosen for YouTube; 12-Factor Agents is on the "popular talks" list. `[S]` — [Latent Space](https://www.latent.space/p/aiewf-2025); [ai.engineer](https://ai.engineer/talks/12-factor-agents)
- AI Engineer Europe (London, 8-10 Apr 2026): Zechner keynote "Building pi in a World of Slop"; one attendee recap said no talk really stood out. `[S]` — [dabase.com](https://dabase.com/blog/2026/ai-engineer-europe/)
- Code w/ Claude 2026: SF 6-7 May, London 19-20 May, Tokyo 10-11 June; 19 SF recordings (~8 hours) on YouTube. `[P]` — [claude.com](https://claude.com/code-with-claude); [PiLastDigit repo](https://github.com/PiLastDigit/Code-With-Claude)
- Code w/ Claude 2025 (22 May 2025): Mastering Claude Code in 30 minutes (Boris Cherny), Prompting for Agents (Jeremy Hadfield), headless automation. `[S]` — [agenda](https://anthropic.swoogo.com/codewithclauderegister/agenda)
- OpenAI DevDay 2025: sessions include "Shipping with Codex", "OpenAI on OpenAI: Applying AI to Our Own Workflows", "Adopting Context Engineering with Cursor"; AgentKit announced. `[S]` — [openai.com/devday/directory](https://openai.com/devday/directory/)
- OpenAI DevDay 2026 (29 Sep 2026): 20+ announcements; "Dots" always-on agent; Agents API built on the Codex harness with multi-agent control; recap links session recordings. `[S]` — [OpenAI recap](https://openai.com/index/devday-2026-recap/)
- Google I/O 2026 (19-20 May): Antigravity 2.0 and CLI, Managed Agents in the Gemini API, Chrome DevTools for agents; 85+ sessions on demand; no agent-talk ranking found. `[S]` — [Google Developers Blog](https://developers.googleblog.com/en/all-the-news-from-the-google-io-2026-developer-keynote/)
- YC AI Startup School: 2025 Karpathy keynote; 2026 Cherny conversation "We Cut 80% of Claude Code's Prompt". `[S]` — [YC library](https://www.ycombinator.com/library/UN-boris-cherny-building-claude-code)
- Sequoia AI Ascent 2026: Karpathy fireside and Cherny interview; event page links a full playlist. `[S]` — [Sequoia](https://sequoiacap.com/article/ai-ascent-2026)
- swyx at World's Fair 2026: opening "The Highest Loop" (loop stacking; choose your loop and level of abstraction) plus a 17:50 "6 Things to Know" preview. `[P]`/`[S]` — [lsimons index](https://github.com/lsimons/ai-world-fair-keynotes/blob/main/keynotes.md); [ai.engineer](https://www.ai.engineer/talks/0S8xe9ftGTM-6-things-know-about-aie-worlds-fair)

### Inferences
- "Praised" is best evidenced by views plus repeat citation: Thariq's keynote, Horthy's two talks, Lopopolo's harness talk and the Zhang/Murag skills talk are the ones other speakers and recaps keep referencing.
- Vendor events (DevDay, I/O) produce announcements more than durable practice talks; the practice lessons come mostly from AI Engineer and Code w/ Claude sessions.

### Gaps
- No session-level detail for DevDay 2026 or Google I/O 2026 agent sessions; no video URLs for DevDay 2025 sessions or Code w/ Claude 2025 talks.
- Barry Zhang and Erik Schluntz: no 2026 talk found; Schluntz appears only via the Anthropic interview video and an older Latent Space episode.
- Code w/ Claude London and Tokyo recordings: session pages exist, but video URLs were not exposed in page HTML.

---

## Key question 4: Most cited podcast episodes and their lessons

### Takeaway
The Boris Cherny circuit (Latent Space May 2025, Every Oct 2025, Lenny Feb 2026, Pragmatic Engineer Mar 2026, Sequoia/Training Data May 2026, Acquired Jun 2026) is the most cited run and works as a time series of changing practice. Karpathy on Dwarkesh is the main sceptical anchor; Lopopolo on Latent Space and Steinberger on Lex Fridman and Pragmatic Engineer cover the OpenAI/Codex side.

### Cited Findings
- Lenny's Podcast, Cherny, 19 Feb 2026: 10-30 PRs/day, 80% of tasks start in plan mode, Claude reviews 100% of PRs, 200% per-engineer productivity (a different interview says ~70%). `[P]` via cocodedk — [cocodedk timeline](https://github.com/cocodedk/loop-engineering/blob/main/docs/07-receipts-and-timeline.md)
- Pragmatic Engineer, Cherny, ~4 Mar 2026: Meta lessons, origin, workflow, parallel agents, skills that still matter. `[S]` — [Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/building-claude-code-with-boris-cherny)
- Every AI & I, Wu + Cherny, 29 Oct 2025: official transcript; memory synthesis, agentic search, plan mode, CLAUDE.md correction loop. `[S]` — [Every](https://every.to/podcast/transcript-how-to-use-claude-code-like-the-people-who-built-it)
- Latent Space: Claude Code (May 2025), Context Engineering for Agents (11 Sep 2025, 57 min), Extreme Harness Engineering (Apr 2026, ~1h12), Age of Async Agents (28 May 2026), Codex 0 to 10M (28 Jul 2026). `[S]` — [latent.space/p/claude-code](https://www.latent.space/p/claude-code); [harness-eng](https://latent.space/p/harness-eng); [cognition](https://latent.space/p/cognition)
- Dwarkesh × Karpathy, 17 Oct 2025: decade of agents, no continual learning, RL critique, demo-to-product gap. `[S]` — [dwarkesh.com](https://www.dwarkesh.com/p/andrej-karpathy)
- Lex Fridman #491, Steinberger, 11 Feb 2026. `[S]` — [lexfridman.com](https://lexfridman.com/peter-steinberger/)
- Acquired: the relevant item is the live "Acquired Unplugged" interview (2 June 2026), not a regular episode. `[S]` — [WorkOS](https://workos.com/blog/boris-cherny-claude-code-acquired-interview-takeaways)
- Changelog, "Opus 4.5 changed everything" (27 Feb 2026). `[S]` — [podscan](https://www.podscan.fm/podcasts/changelog-interviews/episodes/opus-45-changed-everything)
- Sequoia Training Data carries both the Karpathy and Cherny AI Ascent conversations. `[S]` — [pocketcasts](https://pocketcasts.com/podcast/training-data/4cd0e900-0a48-013d-5da1-0acc26574db2/andrej-karpathy-from-vibe-coding-to-agentic-engineering/d66ffbc1-c24d-403c-b39f-f93be691c862)

### Inferences
- Listening in date order (Latent Space May 2025 → Every Oct 2025 → Lenny Feb 2026 → Sequoia May 2026 → Acquired/Anthropic June 2026) shows the shift from "plan first, review diffs" to "auto mode, loops, verify" most clearly.

### Gaps
- No Priors: no relevant 2026 episode surfaced in search; nothing to report.
- No download or popularity data for any podcast; "most cited" is judged from how often episodes recurred across search results.
- Lenny and Pragmatic Engineer transcripts are paywalled; lessons come from secondary recaps.

---

## Key question 5: Long-form courses and tutorials worth the time

### Takeaway
Three tiers stand out: Anthropic's free Academy courses for Claude Code and skills, DeepLearning.AI's short courses for memory, context and skills, and the Husain/Shankar Maven course for evals. Thariq's 1h52 Agent SDK workshop (item 11) is the best free long-form build-along.

### Cited Findings
- Anthropic Academy (Skilljar): "Claude Code in Action" (long hands-off sessions; steering, configuring, automating, verifying), "Introduction to agent skills", "Building with the Claude API" (also on Coursera), "Claude Platform 101" (agent loop, tool use, skills, MCP, context management, managed agents). `[S]` — [anthropic.skilljar.com](https://anthropic.skilljar.com/); [intro to agent skills](https://anthropic.skilljar.com/introduction-to-agent-skills); [Claude Platform 101](https://anthropic.skilljar.com/claude-platform-101)
- DeepLearning.AI "Agent Skills with Anthropic" (Elie Schoppik; 10 lessons, 2h19, beginner). `[S]` — [deeplearning.ai](https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/)
- DeepLearning.AI "Agent Memory: Building Memory-Aware Agents" (with Oracle; Richmond Alake, Nacho Martínez; 1h57, intermediate; ~March 2026). `[S]` — [deeplearning.ai](https://www.deeplearning.ai/short-courses/agent-memory-building-memory-aware-agents)
- DeepLearning.AI "LLMs as Operating Systems: Agent Memory" (Letta; Charles Packer, Sarah Wooders; 1h22). `[S]` — [deeplearning.ai](https://www.deeplearning.ai/courses/llms-as-operating-systems-agent-memory)
- DeepLearning.AI "Context Engineering" (Jeff Huber, Drew Breunig; includes hooks and traces in an agent harness). `[S]` — [deeplearning.ai](https://www.deeplearning.ai/short-courses/context-engineering/)
- DeepLearning.AI "Building Adaptive AI Agents" (Oracle; live 26 Aug 2026). `[S]` — [Oracle blog](https://blogs.oracle.com/developers/we-just-launched-a-new-course-on-how-to-build-adaptive-ai-agents)
- "AI Evals for Engineers & PMs" (Hamel Husain, Shreya Shankar; Maven; October 2026 cohort listed as the last of 2026). Lesson 1 fundamentals and lifecycle; Lesson 2 systematic error analysis. `[S]` — [Maven](https://maven.com/parlance-labs/evals?promoCode=hamel-dev)
- Their Lenny's Newsletter guides: three phases (error analysis to find what to measure; build the eval suite; operationalize into a flywheel); the Sept 2026 sequel renames error analysis "error discovery". `[S]` — [building eval systems](https://www.lennysnewsletter.com/p/building-eval-systems-that-improve); [advanced evals](https://lennysnewsletter.com/p/advanced-evals-how-to-find-and-fix)
- Free evals email course from Hamel. `[S]` — [ai.hamel.dev](https://ai.hamel.dev/eval-course)
- O'Reilly live event "Getting Started with Claude Agent SDK". `[S]` — [O'Reilly](https://www.oreilly.com/live-events/-/0642572273255/)

### Inferences
- Error analysis on real traces is the common core of the evals course, the Replit talk and the Code w/ Claude eval workshops; it is the most transferable eval lesson.

### Gaps
- Course quality could not be assessed first-hand; durations and instructors come from listing snippets.
- "Spec-Driven Development with Coding Agents" (DeepLearning.AI) and Stanford CS146S surfaced only as titles. A DeepLearning.AI Claude Code course exists in my background knowledge (2025) but did not surface in search, so no URL is given.

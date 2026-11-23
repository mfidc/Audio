# Foundational thinkers on using and building LLM agents (Karpathy, Willison, swyx, Weng, Huyen, Mollick, Howard, Chollet and comparable voices)

> **Verification status for the whole file (read first).** Direct page fetches failed for every primary host tried in this session (karpathy.bearblog.dev, simonwillison.net, latent.space, huyenchip.com, oneusefulthing.org, dwarkesh.com, lilianweng.github.io: all DNS failures; each tried once). **No claim below was verified against the page itself.** Everything is from web-search result summaries, which often paraphrase and which draw heavily on secondary write-ups. Labels used per item:
> - **[S-primary]** = search summary of the author's own page/post (URL is the primary source, wording may still be paraphrased).
> - **[S-secondary]** = search summary of a third-party write-up, transcript mirror or aggregator.
> - **"quoted"** text = wording the search results presented as a direct quote. Treat as *probable* wording; re-check before publishing a quote.
> - Items in "Inferences" marked **[recall]** are from the researcher's background knowledge, not found in this session's searches.
> Dates are as reported by search results; conflicts are flagged.

---

## 1. What are Karpathy's main ideas relevant to agents, and what does he actually recommend?

### Takeaway
Karpathy's line runs: LLMs are a new programmable computer (Software 3.0 / LLM OS) populated by fallible "people spirits", so build and use *partial-autonomy* products with a human verifying fast and the AI "on a leash"; in Oct 2025 he said robust agents are a decade-long project, but he reports a "phase shift" around Dec 2025 after which he programs "mostly in English", and in 2026 he reframed the professional practice as **agentic engineering** (spec, review, eval loops, security) and demonstrated two concrete loop patterns: **autoresearch** (agent iterates against an objective metric) and the **LLM-maintained wiki** (agent compiles raw sources into a persistent markdown knowledge base). His position visibly moved between Oct 2025 and early 2026; the earlier skepticism is partly superseded for coding but not for general autonomy.

### Cited Findings

**Source K1. Andrej Karpathy, "Software Is Changing (Again)" (a.k.a. "Software 3.0: Software in the Age of AI"), keynote talk, YC AI Startup School, San Francisco, June 2025** (sources conflict: June 17 vs June 18, 2025). Video: https://www.youtube.com/watch?v=LCEmiRjPEtQ ; slides/annotated write-up: https://www.latent.space/p/s3
- Three paradigms: Software 1.0 (hand-written code), 2.0 (neural-net weights), 3.0 (LLMs programmed with English prompts). [S-secondary] — [Latent Space](https://www.latent.space/p/s3); [HF blog](https://huggingface.co/blog/fdaudens/karpathy-software-3); [frenxt cable](https://www.frenxt.com/cables/claude-code/karpathy-05-software-3-0) (which titles it with the phrase "the hottest new programming language is English")
- LLMs are "stochastic simulations of people" / "people spirits"; they have "jagged intelligence", hallucinate, and have cognitive deficits (Rain Man comparison); "fallible systems that aren't yet perfect". [S-secondary] — [circleback notes](https://circleback.ai/view/mc2qsiqi8w2v7cng2e6); [conffab summary](https://conffab.com/elsewhere/andrej-karpathy-software-is-changing-again/)
- **Partial autonomy apps**: products where humans keep oversight and control; an **autonomy slider** lets the user choose how much the AI takes over. Cursor is the example: tab completion → Cmd+K (chunk) → Cmd+L (file) → Cmd+I (repo-level agent). Perplexity analog: search → research → deep research. [S-secondary] — [Latent Space, partial autonomy section](https://www.latent.space/i/166191505/part-partial-autonomy); [circleback notes](https://circleback.ai/view/mc2qsiqi8w2v7cng2e6)
- **Generation–verification loop**: AI generates, human verifies; the job of product builders and users is to make that loop extremely fast. To improve generation, "keep AI on tight leash" so the human is not swamped by huge diffs. [S-secondary] — [circleback notes](https://circleback.ai/view/mc2qsiqi8w2v7cng2e6); [Latent Space](https://www.latent.space/p/s3)
- **Iron Man suit, not Iron Man robot**: build augmentation with an autonomy dial rather than fully autonomous agents, because LLMs remain fallible. [S-secondary] — [Substack summary](https://saanyaojha.substack.com/p/llms-mainframes-and-the-iron-man); [Analytics India Mag](https://analyticsindiamag.com/global-tech/2025-is-not-the-year-of-ai-agents/)
- LLM-as-OS analogy (context window as RAM/working memory, model as CPU) is attributed to him by secondary sources, originally sketched ~Sept–Nov 2023 as the LLM being "the kernel process of a new Operating System". [S-secondary] — [MindStudio explainer](https://www.mindstudio.ai/blog/software-3-0-explained-karpathy-context-window-ram-model-weights-cpu); [arXiv 2512.05470](https://arxiv.org/pdf/2512.05470v1)

**Source K2. Karpathy, X post endorsing "context engineering", mid-2025 (post; primary URL not retrieved)**
- "Context engineering is the delicate art and science of filling the context window with just the right information for the next step." Two variants circulate ("next step" vs "each step"); he contrasted it with "prompt", which people associate with short instructions. [S-secondary] — [Addy Osmani](https://addyo.substack.com/p/context-engineering-bringing-engineering); [Context-Engineering repo](https://github.com/davidkimai/Context-Engineering)

**Source K3. Karpathy on the Dwarkesh Podcast, "AGI is still a decade away", podcast, Oct 17, 2025.** https://www.dwarkesh.com/p/andrej-karpathy
- "Decade of agents", not "year of agents": the line was a reaction to industry hype; agents like Claude and Codex are impressive and he uses them daily, but there is "so much work to be done". [S-primary transcript summary] — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy); [HappyScribe transcript](https://podcasts.happyscribe.com/dwarkesh-podcast/andrej-karpathy-agi-is-still-a-decade-away)
- What is missing: continual learning ("you can't just tell them something and they'll remember it"), plus other cognitive gaps; "They're cognitively lacking and it's just not working." Mental model for the target: an agent as "an employee or an intern that you would hire to work with you". [S-secondary] — [HappyScribe transcript](https://podcasts.happyscribe.com/dwarkesh-podcast/andrej-karpathy-agi-is-still-a-decade-away); [appliedaihub breakdown](https://appliedaihub.org/ai-digests/interview-briefs/andrej-karpathy-dwarkesh/)
- On coding while building nanochat: "autocomplete is my sweet spot"; agents "go off-rails sometimes"; in his nanochat release note he said Claude/Codex agents "just didn't work well enough at all and net unhelpful" for that novel, non-boilerplate repo. [S-primary/secondary] — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy); [HN nanochat thread](https://news.ycombinator.com/item?id=45569350)
- "Overall, the models are not there. I feel like the industry is making too big of a jump and is trying to pretend like this is amazing, and it's not. It's slop." [S-secondary, quote widely reported] — [Fortune](https://fortune.com/2025/10/21/andrej-karpathy-openai-ai-bubble-pop-dwarkesh-patel-interview); [Simon Willison link post](https://simonwillison.net/2025/Oct/18/agi-is-still-a-decade-away/)
- Reliability framed as a slow "march of nines". [S-secondary only; not confirmed in transcript excerpts] — [Zvi's review](https://thezvi.substack.com/p/on-dwarkesh-patels-podcast-with-andrej)
- Follow-up clarification thread on X, Oct 21, 2025: said "10 years" to AGI should read as optimistic, corrected media over-reading. [S-secondary] — [X post](https://x.com/karpathy/status/1979644538185752935); [dev.to account](https://dev.to/lhua0420/the-man-who-summoned-ghosts-chapter-5-summoning-ghosts-1gl7)

**Source K4. Karpathy, "2025 LLM Year in Review", blog post, Dec 19, 2025.** https://karpathy.bearblog.dev/year-in-review-2025/
- Claude Code is "the first convincing demonstration of what an LLM Agent looks like"; section title "Claude Code / AI that lives on your computer": the agent should run on localhost with your private environment, data and context; he argues early cloud-container agent efforts had the direction wrong. [S-secondary via translations] — [Simon Willison link post](https://feeds.simonwillison.net/2025/Dec/19/andrej-karpathy/); [36kr](https://eu.36kr.com/en/p/3608026572030976)
- "Ghosts vs animals": LLMs are summoned ghosts, not evolved animals; applying a biological lens to their intelligence misleads. [S-secondary] — [3QuarksDaily clipping](https://3quarksdaily.com/?p=294852)
- Vibe coding (term he coined Feb 2025) makes code "free" and "discardable"; it will "terraform software and alter job descriptions". [S-secondary] — [AOL/Business Insider](https://www.aol.com/news/guy-coined-vibe-coding-predicts-181404391.html)

**Source K5. Karpathy, X posts, Dec 26, 2025 – Jan 2026 (posts)**
- Dec 26, 2025: he has "never felt this much behind" as a programmer; the profession is being "refactored". [S-secondary] — [HN thread](https://news.ycombinator.com/item?id=46395714); [xyzbytes](https://www.xyzbytes.com/blog/karpathy-never-felt-this-much-behind)
- Jan 2026 notes: agent capabilities "crossed some kind of threshold of coherence around December 2025 and caused a phase shift in software engineering"; his workflow flipped from ~80% manual / 20% agent in November to ~80% agent in December; "I really am mostly programming in English now." Also predicted a 2026 "slopacolypse" of mostly-working, low-quality output across GitHub, arXiv, X etc. [S-secondary] — [Code with Andrea newsletter](https://codewithandrea.com/newsletter/january-2026/); [The Decoder](https://the-decoder.com/former-tesla-ai-chief-andrej-karpathy-now-codes-mostly-in-english-just-three-months-after-calling-ai-agents-useless/)

**Source K6. karpathy/autoresearch, open-source repo + posts, released ~Mar 7, 2026.** https://github.com/karpathy/autoresearch
- Design: human writes research direction in `program.md` (described as "a super lightweight 'skill'"; edited and iterated by the human); the agent edits only `train.py`; every experiment gets a fixed 5-minute wall-clock training budget so runs are comparable; a single metric (validation bits-per-byte, lower is better, vocab-size independent) decides; improvements are committed, regressions reverted with git; outcomes logged to `results.tsv`; loop repeats (~100 experiments overnight on one GPU). [S-primary README summary + secondary] — [GitHub README](https://github.com/karpathy/autoresearch); [DataCamp guide](https://www.datacamp.com/tutorial/guide-to-autoresearch); [NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/karpathy-open-sources-autoresearch-100-ai-experiments-overnight-on-one-gpu/)
- Generalization he draws: the pattern is "extremely suitable for anything with objective metrics that are easy to assess". [S-secondary, translated quote] — [Longbridge write-up of No Priors](https://longbridge.com/en/news/280013354)

**Source K7. Karpathy on No Priors with Sarah Guo, "Andrej Karpathy on Code Agents, AutoResearch, and the Loopy Era of AI", podcast, Mar 20, 2026.** https://podcasts.apple.com/us/podcast/andrej-karpathy-on-code-agents-autoresearch-and-the/id1668002688?i=1000756334966
- Reports he has written little code by hand since December 2025; coding agents "basically didn't work before December and basically work since" (paraphrase reported by Yahoo; exact wording not confirmed). [S-secondary] — [Yahoo Tech](https://tech.yahoo.com/ai/chatgpt/articles/ai-agents-wrote-80-karpathys-175030864.html); [pjfp breakdown](https://pjfp.com/andrej-karpathy-on-autoresearch-ai-agents-and-why-he-stopped-writing-code-full-breakdown-of-his-2026-no-priors-interview/)
- "Skill issue" framing: when agents fail, the cause is often the user's orchestration skill, not model capability (paraphrase from summary sites). Removing himself from the optimization loop let agents find nanoGPT improvements he had missed. [S-secondary] — [BigGo summary](https://finance.biggo.com/podcast/ff2f96f49e22b331); [podwise](https://podwise.ai/episodes/7559903)

**Source K8. Karpathy, "LLM Wiki" idea file, GitHub gist, ~Apr 3–4, 2026.** https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Pattern: instead of re-deriving answers from raw documents by RAG each time, an agent incrementally compiles a persistent, interlinked markdown wiki. Three layers: `raw/` (immutable sources, read-only to the LLM), `wiki/` (LLM-written summaries, concept/entity pages, cross-links), and a schema file (CLAUDE.md / AGENTS.md) with the conventions. Three operations: **ingest**, **query**, **lint** (audit for contradictions, orphan pages, stale claims). [S-secondary, consistent across many write-ups] — [knowledgework.ai guide](https://knowledgework.ai/guides/llm-wiki); [Starmorph guide](https://blog.starmorph.com/blog/karpathy-llm-wiki-knowledge-base-guide); [synthesis gist](https://gist.github.com/deanjstone/98141cb836bb97c555ae9d6ce2484b5f)
- "Obsidian is the IDE; the LLM is the programmer; the wiki is the codebase." The human browses while the LLM edits. It is published as an "idea file" to paste into your own agent, not as code. [S-secondary quote] — [Angelo Lima](https://angelo-lima.fr/en/karpathy-second-brain-obsidian-claude-en/); [unabyss](https://unabyss.com/blog/what-is-karpathys-llm-wiki)
- Critique: a 2026 paper calls open-source LLM-wiki projects personal-grade prototypes lacking "search write-back". [S-secondary] — [arXiv 2604.11243](https://arxiv.org/pdf/2604.11243)

**Source K9. Karpathy, Sequoia AI Ascent 2026 fireside chat with Stephanie Zhan (talk, late April 2026; one source says Apr 30, another Apr 29) and his own "Sequoia Ascent 2026 summary" blog post.** https://karpathy.bearblog.dev/sequoia-ascent-2026/
- "Vibe coding raises the floor. Agentic engineering raises the ceiling." Vibe coding lets anyone build; agentic engineering is the professional discipline of coordinating agents while preserving correctness, security and maintainability. [S-secondary] — [Karpathy's summary post (not fetched)](https://karpathy.bearblog.dev/sequoia-ascent-2026/); [techmeltd](https://techmeltd.com/en/blog/from-vibe-coding-to-agentic-engineering-karpathy-ai-ascent-2026/); [ai-wiki source note](https://businessdatasolutions.github.io/ai-wiki/sources/2026-04-29-andrej-karpathy-from-vibe-coding-to-agentic-engineering)
- Skills named in coverage: spec design, diff review, eval-loop construction, security oversight, quality judgment. Agents are "fallible and stochastic". Gains for those who master it can be "far more than 10x" (one summary). The agentic engineer "does not blindly accept generated code". [S-secondary] — [startuphub](https://www.startuphub.ai/ai-news/ai-figures/2026/figure-andrej-karpathy-agentic-engineering-sequoia-2026-07-31); [aibuilderclub](https://www.aibuilderclub.com/blog/karpathy-ai-engineering-playbook)
- Origin of the term in his usage is dated Jan or Feb 2026 depending on source. [S-secondary, conflicting] — [aibuilderclub](https://www.aibuilderclub.com/blog/karpathy-ai-engineering-playbook)

### Inferences
- What he recommends doing, assembled from K1–K9: (1) treat the model as a brilliant but jagged, memoryless collaborator; (2) choose the autonomy level per task and keep diffs small enough to verify; (3) invest in the verification side (tests, evals, objective metrics) because that is what makes long loops safe; (4) run agents locally with your real context; (5) write the durable instructions in markdown (`program.md`, schema files) and iterate on those rather than on individual prompts; (6) let agents maintain compounding artifacts (a wiki, a results log, git history) instead of relying on chat memory.
- Autoresearch and the LLM wiki are the same shape: fixed boundaries + a file the human owns + an artifact the agent owns + a check (metric or lint). This is his practical answer to "missing continual learning": externalize memory to files.
- His stance changed materially in ~10 weeks (Oct 2025 "net unhelpful"/"slop" → Dec 2025 "phase shift"). The Dwarkesh skepticism should now be read as applying to general-purpose autonomous agents and to novel/out-of-distribution code, not to agentic coding with a competent supervisor.
- [recall] The YC talk also contained a "build for agents" section (docs in markdown, `llms.txt`, replacing "click here" instructions with commands agents can run) and the line that 2025 is not the year but the decade of agents; neither was surfaced by English-language search results this session (only a Chinese-language wiki entry for the latter: [link](https://panqiwei.github.io/advanced-agentic-dev-patterns/en/wiki/entities/andrej-karpathy)).
- [recall] The original vibe-coding post (X, Feb 2025) described fully giving in to the vibes and forgetting the code exists; search confirmed only the date and coinage.

### Gaps
- No primary text was readable. Exact wording of the Sequoia summary post, the LLM-wiki gist, the Jan 2026 notes and the Oct 21 clarification thread are unverified; the karpathy gist search snippet appeared contaminated with a third-party implementation's text.
- Exact dates conflict for: the YC talk (June 17 vs 18, 2025), Sequoia Ascent (Apr 29 vs 30, 2026), the LLM-wiki gist (Apr 3 vs 4, 2026), first use of "agentic engineering" (Jan vs Feb 2026).
- "March of nines", "skill issue" and "basically didn't work before December" are attributed via secondary summaries only.
- I found nothing reliable on Karpathy material after ~May 2026; hosts to allow for a re-check: karpathy.bearblog.dev, gist.github.com, x.com, dwarkesh.com.

---

## 2. What does Simon Willison recommend?

### Takeaway
Willison supplies the field's working vocabulary: an agent "runs tools in a loop to achieve a goal"; the craft is designing the tools and the loop; the security rule is never to combine private data, untrusted content and external communication (the "lethal trifecta"), enforced by sandboxes and deterministic limits rather than prompts. His 2026 "Agentic Engineering Patterns" guide turns this into named practices (red/green TDD, first run the tests, code is cheap, fight cognitive debt).

### Cited Findings

**Source W1. Simon Willison, "The lethal trifecta for AI agents: private data, untrusted content, and external communication", blog post, June 16, 2025.** https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/
- The three ingredients: access to private data, exposure to untrusted content, ability to communicate externally. Combine all three and "you are ripe for exploitation". "The only way to reliably prevent that class of attacks is to cut off one of the three legs." MCP makes it worse because it encourages users to mix tools from different sources. [S-primary] — [Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/); [tag page](https://simonwillison.net/tags/lethal-trifecta)

**Source W2. Willison, agent definition (posts under his "definitions"/"ai-agents" tags, Sept 2025).** https://simonwillison.net/tags/definitions/?page=2
- Preferred definition: an LLM agent "runs tools in a loop to achieve a goal"; he credits the "tools in a loop" framing to Anthropic and likes pairing it with the classic academic definition. [S-primary] — [Willison tags](https://simonwillison.net/tags/ai-agents/?page=3)

**Source W3. Willison, "Designing agentic loops", blog post, Sept 30, 2025.** https://simonwillison.net/2025/Sep/30/designing-agentic-loops/
- "The art of using them well is to carefully design the tools and loop for them to use." Discusses YOLO (no-approval) mode risks; recommends restricting files, secrets and network; allow-listing trusted hosts "is a great way to prevent exfiltration attacks from stealing your private source code". [S-primary] — [Willison](https://simonwillison.net/2025/Sep/30/designing-agentic-loops/)

**Source W4. Willison, "Embracing the parallel coding agent lifestyle", blog post, Oct 5, 2025.** https://simonwillison.net/2025/Oct/5/parallel-coding-agents/
- Runs several Claude Code / Codex CLI instances at once; restricts YOLO mode to tasks where malicious instructions are unlikely to enter context; wants local agents in Docker to bound damage. [S-primary] — [Willison](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/)

**Source W5. Willison, "Living dangerously with Claude", talk write-up, Oct 22, 2025.** https://simonwillison.net/2025/Oct/22/living-dangerously-with-claude/
- A sandbox is the only "credible" solution for running agents unattended. [S-primary] — [Willison](https://simonwillison.net/2025/Oct/22/living-dangerously-with-claude/)

**Source W6. Willison, "2025: The year in LLMs", blog post, Dec 31, 2025.** https://simonwillison.net/2025/Dec/31/the-year-in-llms/
- Admits he runs YOLO mode "all the time" despite knowing the risks; floated "vibe engineering" as the professional counterpart to vibe coding and was unsure it would stick (later superseded by "agentic engineering"). [S-primary] — [Willison](https://simonwillison.net/2025/Dec/31/the-year-in-llms/)

**Source W7. Willison, "Agentic Engineering Patterns", multi-chapter guide, started Feb 23, 2026 (still being extended).** https://simonwillison.net/guides/agentic-engineering-patterns/ ; intro: https://simonwillison.net/2026/Feb/23/agentic-engineering-patterns/
- Framing: coding practices for getting good results from agents such as Claude Code and Codex, modelled loosely on the 1994 Design Patterns book; the defining feature of these tools is that they both generate *and execute* code, so they can test and iterate without turn-by-turn guidance. [S-primary] — [intro post](https://simonwillison.net/2026/Feb/23/agentic-engineering-patterns/)
- Chapter "Writing code is cheap now": the hardest adjustment is absorbing that initial working code now costs almost nothing, when many team habits assume a few hundred clean tested lines cost a day. [S-primary] — [guide](https://simonwillison.net/guides/agentic-engineering-patterns)
- Chapter "Red/green TDD" (Feb 24, 2026): have the agent write tests first, *confirm they fail*, then implement until they pass; "red/green TDD" works as a compact prompt good models understand; skipping the red step risks a test that never exercised the new code. [S-primary] — [chapter](https://simonwillison.net/guides/agentic-engineering-patterns/red-green-tdd/)
- Chapter "First run the tests" (Feb 24, 2026): old excuses for skipping automated tests no longer hold when an agent writes them in minutes. [S-primary] — [guide](https://simonwillison.net/guides/agentic-engineering-patterns)
- Chapter "Interactive explanations": **cognitive debt** = agent-written code becoming a black box the team no longer understands, slowing planning like technical debt; remedy: have agents build interactive/animated explanations of how the code works. [S-primary] — [chapter](https://simonwillison.net/guides/agentic-engineering-patterns/interactive-explanations/); [X announcement](https://x.com/simonw/status/2027885000432259567)

**Source W8. Willison on Lenny's Podcast, "An AI state of the union: We've passed the inflection point, dark factories are coming, and automation timelines", podcast, ~April 2026.** https://www.lennysnewsletter.com/p/an-ai-state-of-the-union
- November 2025 was the inflection where coding agents went from "mostly works" to "actually works". "Probably 95% of the code that I produce, I didn't type it myself." [S-secondary] — [Lenny's](https://www.lennysnewsletter.com/p/an-ai-state-of-the-union); [media summary gist](https://gist.github.com/shines/bcec0105fd6127e9cfa768fc12e1faaf)
- Three daily patterns: red/green TDD, templates, "hoarding" (keeping a stock of working examples to reuse). Verbose test suites are fine now because maintaining them is the agent's job. [S-secondary] — [show notes](https://www.lennysnewsletter.com/p/an-ai-state-of-the-union)
- Distinguishes hands-off vibe coding from professional agentic engineering; describes the "dark factory" pattern (nobody writes or reviews code; quality by automated gates; term from Dan Shapiro, Jan 2026; StrongDM example with agent swarms simulating users); warns the lethal trifecta remains unsolved; says mid-career engineers are most exposed. [S-secondary] — [media summary gist](https://gist.github.com/shines/bcec0105fd6127e9cfa768fc12e1faaf); [Arun Baby](https://www.arunbaby.com/ai-agents/0106-dark-factory-era-ai-engineering/)

**Source W9. Willison, fireside chat on agentic engineering, Pragmatic Summit, 2026 (talk write-up).** https://simonw.substack.com/p/fireside-chat-about-agentic-engineering
- Sandboxing is "the most important thing"; wants agents sandboxed by default; trusts deterministic restrictions over prompt-based protections. [S-secondary/primary newsletter] — [Substack](https://simonw.substack.com/p/fireside-chat-about-agentic-engineering)

### Inferences
- Willison's recommendations reduce to: give the agent a way to check itself (tests it can run), give it a tight box (sandbox, host allow-list, scoped credentials), and keep yourself able to understand the result (explanations, review). His security advice is structural: remove a leg of the trifecta, do not rely on the model resisting injection.
- He is candid that practice lags advice: he runs YOLO mode constantly. The honest version of his recommendation is "YOLO inside a sandbox, or only where untrusted content cannot enter".
- "Vibe engineering" (Oct–Dec 2025) is superseded by "agentic engineering" (Feb 2026 onward), converging with Karpathy's term.

### Gaps
- Could not read the guide's full table of contents; more chapters exist than the four listed.
- No results for "Rule of Two", "conformance suite" or his older workflow post "Here's how I use LLMs to help me write code" (Mar 2025); not covered.
- Exact formal definition sentence of "agentic engineering" in his guide not retrieved.

---

## 3. What are swyx's and the AI Engineer community's key frameworks for agents?

### Takeaway
swyx's contribution is taxonomy and industry framing: a compact agent formula (LLM + memory + planning + tools + loop), the six-element IMPACT checklist, the "Agent Labs" thesis (value accrues to companies that build the product/harness around frontier models), and in 2026 the elevation of **context engineering** and **harness engineering** as named disciplines, while openly staging the "Big Model vs Big Harness" debate.

### Cited Findings

**Source S1. swyx (Shawn Wang), "Why Agent Engineering", keynote, AI Engineer Summit (New York), Feb 2025; companion essay "Agent Engineering", Latent Space, ~Mar 24, 2025.** https://www.latent.space/p/agent ; talk: https://ai.engineer/talks/why-agent-engineering
- Explains why the summit pivoted to being an agent-engineering conference; covers the hype and competing definitions; credits Simon Willison for help defining agents. [S-primary] — [swyx on X](https://x.com/swyx/status/1904256213661192405); [AI Engineer](https://ai.engineer/talks/why-agent-engineering)
- Slide formula: "agent = llm + memory + planning + tools + while loop". [S-primary slide deck] — [Google Slides](https://docs.google.com/presentation/d/1SWoBIvTQu__uNEvSawmNcROiUx-n86O_fP0arZcTGb8/htmlpresent)
- "Six Elements of Agent Engineering" under the acronym **IMPACT** ("write agents with IMPACT!"). "Intent" was chosen over OpenAI's "Instructions" because intent combines instructions *and evals* to guide behaviour. A later summary lists the organizing words as intent, control flow, memory, planning and tool use. [S-primary, partial] — [Latent Space](https://latent.space/p/agent/); [AI Engineer talk page](https://ai.engineer/talks/IHkyFhU6JEY-designing-ai-intensive-applications)

**Source S2. swyx, "Agent Labs" essay, Latent Space, ~Nov 18, 2025; follow-ups in 2026.** https://x.com/swyx/status/1990886806250782876
- Thesis: "Agent Labs are clearly emerging in 2025 as a complement to Model Labs", which are all becoming AI cloud platforms. Playbook: start on frontier models, specialize for a domain, train own models only once data/workload justify cost and latency savings. [S-primary] — [swyx on X](https://x.com/swyx/status/1990886806250782876); [Latent Space crossover episode](https://www.latent.space/p/unsupervised-learning-2026)
- May 2026 update: "All Model Labs are now Agent Labs" (model labs ship agents as the product; the lines converge). [S-primary] — [AINews](https://swyx.substack.com/p/ainews-all-model-labs-are-now-agent)

**Source S3. Latent Space / AINews, "Is Harness Engineering real?" (newsletter, 2026) and "AIE Europe Debrief + Agent Labs Thesis" (podcast, 2026).** https://swyx.substack.com/p/ainews-is-harness-engineering-real ; https://www.latent.space/p/unsupervised-learning-2026
- Core loop of every production agent: execute the model's tool calls, capture results, append to context, call the model again. [S-primary] — [AINews](https://swyx.substack.com/p/ainews-is-harness-engineering-real)
- "Big Model vs Big Harness" debate; one cited benchmark found harness choice was within the margin of error for some models. [S-primary] — [AINews](https://swyx.substack.com/p/ainews-is-harness-engineering-real)
- AIE Europe 2026 had what swyx called the first Harness Engineering track; top interests were OpenClaw, harness engineering and context engineering, plus evals and infrastructure. Memory/personalization is argued to be the next bottleneck and wedge. [S-primary] — [Latent Space](https://www.latent.space/p/unsupervised-learning-2026); [AIE Europe wiki](https://latent.space/wiki/aie-europe-ed4b0876/)
- AI Engineer World's Fair 2026 trends piece: the field has moved from prompting models to engineering reliable systems around them. [S-primary] — [Latent Space](https://latent.space/p/aiewf26trends)

**Source S4. Dex Horthy (HumanLayer), "12-Factor Agents: Patterns of reliable LLM applications", talk, AI Engineer World's Fair 2025 (June 2025) + repo.** https://ai.engineer/talks/8kMaTybvDUw-12-factor-agents ; https://github.com/humanlayer/12-factor-agents
- Factors include "Own your context window" (Factor 3) and "Small, Focused Agents" (Factor 10; sources cite 3–10 or 3–20 steps). [S-secondary] — [AI Engineer](https://ai.engineer/talks/8kMaTybvDUw-12-factor-agents); [paddo.dev](https://paddo.dev/blog/12-factor-agents)
- "Dumb zone": model quality degrades as the context window fills; practitioners derive rules like keeping context under ~40% and clearing between tasks. Thresholds conflict across sources (40–60% of window vs ~100k tokens). [S-secondary, numbers unverified] — [dev.to](https://dev.to/diggidydale/escaping-the-dumbzone-part-4-advanced-patterns-151m); [garrit.xyz](https://garrit.xyz/posts/2026-05-06-dont-trust-large-context-windows)

### Inferences
- The AI Engineer community's centre of gravity moved in ~18 months from "what is an agent" (2025: definitions, IMPACT) to "what surrounds the model" (2026: context and harness engineering, memory). This parallels Karpathy's context-engineering remark and Mollick's models/apps/harnesses framing.
- swyx himself treats the importance of the harness as an open empirical question, not settled doctrine.

### Gaps
- Could not retrieve what each IMPACT letter stands for from the essay. [recall] It is Intent, Memory, Planning, Authority, Control flow, Tools; unverified this session.
- No session-level takeaways from AI Engineer World's Fair 2026 found; only the trends headline.

---

## 4. What do Lilian Weng, Chip Huyen, Ethan Mollick, Jeremy Howard, François Chollet and others add?

### Takeaway
Weng and Huyen supply the architecture and failure analysis (LLM core + planning + memory + tools; errors compound across steps; evaluate planning, tools and efficiency separately). Mollick translates agent use into management practice for non-coders (delegation math, a good brief, verification, the opacity and deskilling risks). Howard is the main dissenter favouring small human-driven steps; Chollet marks the capability boundary (agents still fail at novel, on-the-fly adaptation).

### Cited Findings

**Source L1. Lilian Weng, "LLM Powered Autonomous Agents", blog post, June 23, 2023 (OLDER; still the canonical architecture reference, not formally superseded but pre-dates tool-calling-in-a-loop coding agents).** https://lilianweng.github.io/posts/2023-06-23-agent/
- The LLM is the agent's core controller, complemented by three components: planning (task decomposition, reflection), memory (short-term in-context; long-term via external vector store and retrieval), and tool use. [S-primary] — [Lil'Log](https://lilianweng.github.io/posts/2023-06-23-agent); [2024 survey noting the industry follows this form](https://arxiv.org/html/2404.11584v1)

**Source L2. Lilian Weng, 2026 posts reported by aggregators: "Harness Engineering for Self-Improvement" and a post on recursive self-improvement (reported date July 4, 2026).** https://lilianweng.github.io/
- Reported definition: a harness is "the system surrounding a base model: it orchestrates execution, planning, tool calls, context, stored artifacts, and evaluation". [S-secondary, LOW confidence; aggregator sources only, post existence and date unverified] — [antoinebuteau lessons page](https://www.antoinebuteau.com/lessons-from-lilian-weng.md); [cryptobriefing](https://cryptobriefing.com/lilian-weng-rejoins-openai-recursive-self-improvement/)

**Source H1. Chip Huyen, "Agents", blog post, Jan 7, 2025 (adapted from her 2025 book *AI Engineering*).** https://huyenchip.com/2025/01/07/agents.html
- Sections: Agent Overview, Tools, Planning, Agent Failure Modes and Evaluation. Agents "with their new modes of operations, have new modes of failure"; the post ends with how to evaluate agents to catch them. [S-primary] — [Huyen](https://huyenchip.com/2025/01/07/agents); [Willison link post, Jan 11, 2025](https://simonwillison.net/2025/Jan/11/agents/)
- Compound mistakes: at 95% per-step accuracy, a 10-step task succeeds ~60% of the time and a 100-step task ~0.6%. [S-secondary quoting the book; one secondary source instead reports "roughly 18%" ten-step failure, which conflicts] — [Towards Data Science](https://towardsdatascience.com/the-math-thats-killing-your-ai-agent/); conflicting figure in [platform blog](https://platform-2o0.pages.dev/blog/planning-memory-tools-an-agents-three-jobs)

**Source M1. Ethan Mollick, "On Working with Wizards", essay, One Useful Thing, ~Sept 2025.** https://www.oneusefulthing.org/p/on-working-with-wizards
- New systems are "essentially agents, AI that can plan and act autonomously toward given goals"; you get sophisticated output from a vague request "but you have no part in the process"; you "can't know how the AI made its choices, nor confirm that everything is correct". [S-primary] — [Mollick](https://www.oneusefulthing.org/p/on-working-with-wizards)
- Deskilling: "Every time we hand work to a wizard, we lose a chance to develop our own expertise", and the judgment needed to evaluate the work. New literacy: learn "when to summon the wizard versus when to work with AI as a co-intelligence or to not use AI at all". [S-primary] — [Mollick](https://www.oneusefulthing.org/p/on-working-with-wizards)

**Source M2. Mollick, "The Shape of AI: Jaggedness, Bottlenecks and Salients", essay, Dec 20, 2025.** https://www.oneusefulthing.org/
- Exists and extends the jagged-frontier idea; content not retrieved. [S-primary index only] — [One Useful Thing](https://www.oneusefulthing.org/)

**Source M3. Mollick, "Management as AI superpower" (subtitle "Thriving in a world of agentic AI" / "...of agents"), essay, Jan 27, 2026.** https://www.oneusefulthing.org/p/management-as-ai-superpower
- Working with agents uses "management 101" skills; instructing AI "basically reinvents management". [S-primary] — [Mollick](https://www.oneusefulthing.org/p/management-as-ai-superpower)
- Delegation decision weighs how long the task takes you against **Probability of Success** (chance an attempt meets your bar) and **AI Process Time** (time to request, wait for and evaluate output). AI is fast whether right or wrong, so failures cost you prompting and review time. [S-secondary] — [John Talaguit summary](https://www.johntalaguit.com/management-skills-ai-advantage/); [Melody Koh](https://melodykoh.substack.com/p/the-leash-length-problem)
- A good brief answers: what are we trying to accomplish and why; what "done" looks like; limits of authority; interim outputs for course-correction; what must be verified before accepting. [S-secondary] — [John Talaguit summary](https://www.johntalaguit.com/management-skills-ai-advantage/)

**Source M4. Mollick, "A Guide to Which AI to Use in the Agentic Era", essay, Feb 18, 2026.** https://www.oneusefulthing.org/p/a-guide-to-which-ai-to-use-in-the
- "Using AI" now means choosing across three layers: **models, apps and harnesses**. "A harness is a system that lets the AI use tools, take actions, and complete multi-step tasks on its own" (horse-harness analogy). Calls this edition "a very large break with the past". [S-secondary/primary excerpts] — [Mollick on Threads](https://www.threads.com/@ethan_mollick/post/DU4gpdNjEHG/every-few-months-i-write-an-updated-idiosyncratic-guide-on-which-a-is-to-use); [Downes](https://www.downes.ca/post/78865)
- Later (July 2026) guide warns about permissions: with computer use on, the AI "can literally take over your mouse, browser, and computer". [S-secondary] — [Forbes](https://www.forbes.com/sites/johnwerner/2026/07/31/ethan-mollick-writes-ai-agent-guide/)

**Source J1. Jeremy Howard (Answer.AI / fast.ai), "A New Chapter for fast.ai: How To Solve It With Code", blog post, Nov 7, 2024 (OLDER; position restated in 2025 podcasts).** https://www.answer.ai/posts/2024-11-07-solveit.html
- Solveit is built for "Dialog Engineering": AI embedded in a step-by-step, notebook-like process in which the human crafts the reasoning before code is generated. [S-primary] — [Answer.AI](https://www.answer.ai/posts/2024-11-07-solveit.html); [Howard on X](https://x.com/jeremyphoward/status/1854659286179434681)
- Argues human-in-the-loop small steps beat fully autonomous agents on complex coding/writing; Latent Space episode titled "The antidote to AI fatigue"; earlier critique of Devin. [S-secondary] — [Latent Space episode](https://www.latent.space/p/answerai); [user account](https://christhomas.co.uk/blog/2025/09/24/the-human-is-the-agent-how-solveit-changed-my-programming-journey-after-25-years/)
- Related Answer.AI post (Jan 20, 2026), "The unauthorized tool call problem". Content not retrieved. [S-primary title only] — [Answer.AI](https://www.answer.ai/posts/2026-01-20-toolcalling.md)

**Source C1. François Chollet, ARC-AGI-3 launch posts, X, Mar 25, 2026, and technical report.** https://x.com/fchollet/status/2036861192619384989 ; https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf
- ARC-AGI-3 evaluates "agentic intelligence via interactive reasoning environments"; bar is human-level action efficiency on environments seen for the first time; frontier models scored under 1% at launch. [S-primary] — [Chollet on X](https://x.com/fchollet/status/2036861192619384989); [second post](https://x.com/fchollet/status/2036863769981403497)
- He reads ARC progress as a signal: the late-2025 jump on ARC-AGI-2 coincided with the rise of agentic coding; models remain weak at fluid intelligence (combining known patterns on the fly into a new model of a problem). [S-secondary] — [Fast Company](https://www.fastcompany.com/91516330/francois-chollet-ai-benchmark)

**Comparable voices found**
- **Geoffrey Huntley, "Ralph" loop (2025):** a bash loop that re-runs a coding agent with fresh context each iteration until done; memory lives in git, docs and task files, not the conversation. [S-secondary] — [LinearB](https://linearb.io/blog/ralph-loop-agentic-engineering-geoffrey-huntley); [understandingdata](https://understandingdata.com/posts/ralph-loop/)
- **Steve Yegge, Gas Town (Jan 2026):** orchestrator for 20–30 parallel coding agents; "100% vibe coded. I've never seen the code, and I never care to". Third-party meme coins attached to both projects are a credibility caution. [S-secondary] — [Wikipedia](https://en.wikipedia.org/wiki/Steve_Yegge); [Sean Goedecke](https://www.seangoedecke.com/gas-and-ralph/)
- **Hamel Husain (with Shreya Shankar), evals:** error analysis is "the most important activity in evals": read real traces, label failures, count failure modes, fix the biggest; do not start from generic metrics; most teams do not need an LLM judge to get value; 2026 post "Evals Skills for Coding Agents". [S-primary/secondary] — [hamel.dev](https://hamel.dev/blog/posts/evals-skills/); [evals FAQ](https://hamel.dev/blog/posts/evals-faq/evals-faq.pdf); [Arize interview](https://arize.com/blog/rise-of-the-ai-engineer-why-ai-evals-fail-before-the-evaluation-begins/)

### Inferences
- For non-coding knowledge work the most directly usable material is Mollick's: decide to delegate by success probability × review cost, write a brief with goal, definition of done, authority limits and checkpoints, and decide in advance what you will verify.
- Weng's 2023 triad maps one-to-one onto swyx's formula and onto the reported 2026 "harness" definition, so the vocabulary changed (components → harness) more than the architecture did.
- Huyen's compound-error arithmetic is the quantitative argument behind Karpathy's "leash", Horthy's "small, focused agents" and Howard's small steps.

### Gaps
- Huyen: could not verify her tool taxonomy, the "decouple planning from execution" advice, or her three-way failure taxonomy (planning / tool / efficiency) from the page; found no 2026 agent-specific writing by her.
- Weng: 2026 posts are known only through low-quality aggregators; treat as unconfirmed. Her reported career moves in 2026 are also unverified and omitted.
- Mollick: the "four moments to check back with a human" and a "40% faster / 60% cheaper" figure appear only in third-party summaries; exact equation wording unverified.
- Howard: no 2026 statement on agents found; his view may have shifted after the Dec 2025 capability jump. Attribution of vibe-coding criticism in an MLST transcript was unclear and is omitted.
- Chollet: no direct statements found on how to *use* coding agents; only capability framing. Podcast chapter titles ("Why Ndea isn't competing with coding agents") were not expanded.
- Not covered for lack of results: Andrew Ng's agentic design patterns, Harrison Chase ("deep agents"), Armin Ronacher, Boris Cherny.

---

## 5. Where do these voices agree, and where do they disagree?

### Takeaway
They agree on the definition (tools in a loop), on fallibility, on verification as the scarce human contribution, and on context/harness mattering as much as the model; they disagree on how much autonomy to grant today, on whether unreviewed "dark factory" code is acceptable, and on how much of the gain is harness versus model.

### Cited Findings

**Agreements**
- *Agent = model + tools + loop (+ memory, planning).* Willison "runs tools in a loop to achieve a goal" — [Willison](https://simonwillison.net/tags/definitions/?page=2); swyx "llm + memory + planning + tools + while loop" — [slides](https://docs.google.com/presentation/d/1SWoBIvTQu__uNEvSawmNcROiUx-n86O_fP0arZcTGb8/htmlpresent); Weng's planning/memory/tools — [Lil'Log](https://lilianweng.github.io/posts/2023-06-23-agent); Latent Space's core loop — [AINews](https://swyx.substack.com/p/ainews-is-harness-engineering-real).
- *Models are jagged and fallible, so the human's job is verification.* Karpathy's generation–verification loop — [Latent Space](https://www.latent.space/p/s3); Mollick's inability to "confirm that everything is correct" — [Mollick](https://www.oneusefulthing.org/p/on-working-with-wizards); Huyen's compounding errors — [Huyen](https://huyenchip.com/2025/01/07/agents).
- *Tests and objective checks are what make autonomy safe.* Willison's red/green TDD — [guide](https://simonwillison.net/guides/agentic-engineering-patterns/red-green-tdd/); Karpathy's autoresearch metric gate — [GitHub](https://github.com/karpathy/autoresearch); Husain's error analysis — [hamel.dev](https://hamel.dev/blog/posts/evals-faq/evals-faq.pdf).
- *Context is the lever.* Karpathy's context-engineering definition — [Addy Osmani](https://addyo.substack.com/p/context-engineering-bringing-engineering); Horthy's "own your context window" — [AI Engineer](https://ai.engineer/talks/8kMaTybvDUw-12-factor-agents); AIE Europe tracks — [Latent Space](https://www.latent.space/p/unsupervised-learning-2026).
- *Late 2025 was a real capability step for coding agents.* Karpathy "phase shift" ~Dec 2025 — [newsletter](https://codewithandrea.com/newsletter/january-2026/); Willison's November 2025 inflection — [Lenny's](https://www.lennysnewsletter.com/p/an-ai-state-of-the-union); Chollet ties the late-2025 ARC-AGI-2 jump to agentic coding — [Fast Company](https://www.fastcompany.com/91516330/francois-chollet-ai-benchmark).
- *Professional practice needs a name distinct from vibe coding.* Karpathy and Willison both settled on "agentic engineering" in early 2026 — [Karpathy summary](https://karpathy.bearblog.dev/sequoia-ascent-2026/); [Willison guide](https://simonwillison.net/2026/Feb/23/agentic-engineering-patterns/).
- *Externalize memory into files.* Karpathy's wiki and `program.md` — [gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f); Huntley's Ralph (memory in git/docs) — [understandingdata](https://understandingdata.com/posts/ralph-loop/).

**Disagreements / tensions**
- *How much autonomy now.* Howard: human-driven small steps beat autonomous agents — [Latent Space](https://www.latent.space/p/answerai). Karpathy (June 2025): partial autonomy, short leash — [Latent Space](https://www.latent.space/p/s3); Karpathy (2026): overnight unattended loops where a metric exists — [GitHub](https://github.com/karpathy/autoresearch). Yegge: 20–30 parallel agents, never reads the code — [Wikipedia](https://en.wikipedia.org/wiki/Steve_Yegge).
- *Is unreviewed code acceptable?* Karpathy's agentic engineer "does not blindly accept generated code" — [aibuilderclub](https://www.aibuilderclub.com/blog/karpathy-ai-engineering-playbook). Willison reports the "dark factory" pattern (no human review, automated gates) as emerging while warning of cognitive debt — [media summary](https://gist.github.com/shines/bcec0105fd6127e9cfa768fc12e1faaf); [chapter](https://simonwillison.net/guides/agentic-engineering-patterns/interactive-explanations/).
- *Security posture.* Willison: prompt injection is unsolved; only structural limits work — [Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/); yet he and most practitioners run YOLO mode — [Willison](https://simonwillison.net/2025/Dec/31/the-year-in-llms/). Mollick flags permissions for general users — [Forbes](https://www.forbes.com/sites/johnwerner/2026/07/31/ethan-mollick-writes-ai-agent-guide/).
- *Harness vs model.* Latent Space frames "Big Model vs Big Harness" as unresolved, with a benchmark showing harness choice within noise for some models — [AINews](https://swyx.substack.com/p/ainews-is-harness-engineering-real); Mollick treats harness as a first-class choice — [Threads](https://www.threads.com/@ethan_mollick/post/DU4gpdNjEHG/every-few-months-i-write-an-updated-idiosyncratic-guide-on-which-a-is-to-use).
- *How far current agents generalize.* Karpathy (Oct 2025): "cognitively lacking", a decade of work — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy); Chollet: under 1% on novel interactive tasks — [Chollet](https://x.com/fchollet/status/2036861192619384989); versus the "actually works" assessments for coding above.
- *Deskilling.* Mollick warns delegation erodes the expertise needed to judge output — [Mollick](https://www.oneusefulthing.org/p/on-working-with-wizards); Willison names the team-level version, cognitive debt — [chapter](https://simonwillison.net/guides/agentic-engineering-patterns/interactive-explanations/); Howard's Solveit is positioned as the remedy — [Answer.AI](https://www.answer.ai/posts/2024-11-07-solveit.html).

### Inferences
- The disagreements largely dissolve along one axis: **how cheaply and objectively the output can be verified**. Where a metric or test suite exists (autoresearch, TDD, dark factories with conformance tests) the same people endorse long unattended loops; where verification is by human judgment (novel code, analysis, writing) they recommend short leash, small steps and briefs with checkpoints.
- Much of the apparent conflict is also temporal: statements before ~Nov–Dec 2025 (Howard 2024–25, Karpathy Oct 2025, Huyen Jan 2025) describe weaker agents. Date every quote.
- A defensible composite for a practitioner: define done and the check first; give the agent a sandbox and the minimum of the trifecta; keep durable instructions and memory in files; size the task to what you can review; analyze failures from traces; deliberately keep your own understanding current.

### Gaps
- No source directly stages these people debating each other; the agreements and disagreements are assembled by comparison, not from an explicit exchange.
- Empirical evidence for productivity claims (e.g. the METR study, "far more than 10x") was not examined here.
- All of the above rests on search summaries; hosts that need allowing for verification: karpathy.bearblog.dev, simonwillison.net, latent.space, huyenchip.com, www.oneusefulthing.org, www.dwarkesh.com, lilianweng.github.io.

# Practitioner Workflows for Agentic Coding (Claude Code, Codex, Cursor, Amp, etc.)

**Research date:** 2026-11-23. **Verification status (read this first):** direct page fetches failed with DNS errors for all eight hosts tried (simonwillison.net, lucumr.pocoo.org, mitchellh.com, steipete.me, ghuntley.com, addyosmani.com, harper.blog, blog.fsck.com), so **no claim below was verified from the page itself**. Everything comes from web-search result summaries. Tags used per finding:

- **[S-primary]** = search summary that quoted or paraphrased the author's own page (the URL is the primary source, but I did not read it).
- **[S-secondary]** = search summary of a third-party write-up about the author. Treat wording and numbers as approximate.

The search budget also ran out before the last batch, so several topics are thin; they are listed under Gaps. Material from before about mid-2025 is flagged **[dated]** where the author's own practice has since moved on.

## 1. What do named practitioners actually do and recommend? (source register)

### Takeaway
The respected practitioners converge on a small set of habits (give the agent a way to run and verify its work, keep tasks small, encode every repeated mistake into an instruction file or tool, prefer CLIs over MCP) while differing sharply on how much to plan up front, how many agents to run, and whether humans must still read the code. Most of the strongest material is from October 2025 to April 2026; anything earlier describes weaker models and often a workflow the author has since changed.

### Cited Findings

**Simon Willison** (Django co-creator; Datasette)
- *Agentic Engineering Patterns* — living guide, announced 2026-02-23, still being extended. Defines "agentic engineering" as building software with coding agents whose defining feature is that they can both generate and execute code, which lets them iterate toward software that demonstrably works. Chapters include Principles ("writing code is cheap now"), How coding agents work, Using Git with coding agents, Subagents, Testing and QA (red/green TDD, "first run the tests", agentic manual testing), and Understanding code (linear walkthroughs). [S-primary] — [guide index](https://simonwillison.net/guides/agentic-engineering-patterns/); [announcement](https://simonwillison.net/2026/Feb/23/agentic-engineering-patterns/); [code is cheap](https://feeds.simonwillison.net/guides/agentic-engineering-patterns/code-is-cheap/)
- Session opener: he gives the agent the command to run the test suite first, then says "use red-green TDD" and gives the task. Tests are treated as effectively free and therefore non-optional. Fireside chat, Pragmatic Summit, 2026-03-14. [S-primary] — [Pragmatic Summit post](https://simonwillison.net/2026/Mar/14/pragmatic-summit/)
- Agentic manual testing: have the agent exercise the running product; bugs found that way get fixed with red/green TDD and can be promoted into permanent automated tests. [S-primary] — [Agentic manual testing chapter](https://simonwillison.net/guides/agentic-engineering-patterns/agentic-manual-testing/)
- Built two tools (2026-02-10): Showboat (agent builds a Markdown demo document via `note`/`exec`/`image` commands, recording what it actually ran) and Rodney (CLI browser automation). Reported caveat: agents sometimes edit the Markdown directly instead of using the commands, which would fake the evidence. [S-primary for tools; S-secondary for caveat] — [Showboat and Rodney](https://simonwillison.net/2026/Feb/10/showboat-and-rodney/); [gu-log write-up](https://gu-log.vercel.app/en/posts/en-mp-61-20260211-simonw-showboat-rodney)
- "Embracing the parallel coding agent lifestyle" (2025-10-05): several terminal windows with Claude Code and Codex CLI in different directories, in YOLO mode only where he is confident malicious instructions cannot enter the context; riskier tasks go to asynchronous cloud agents (usually Codex Cloud) where the worst case is source leakage. Review of the output is the bottleneck. He had not adopted worktrees and used fresh checkouts, often in /tmp. [S-primary; worktree detail S-secondary] — [parallel agents post](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/)
- "Designing agentic loops" (2025-09-30) and "Living dangerously with Claude" (2025-10-22): unattended YOLO risks include destructive shell commands and exfiltration; the "lethal trifecta" is private data + untrusted content + external communication; his position is that the only credible fix is a sandbox that limits files, secrets and network. Points to Anthropic's sandbox-runtime and macOS sandbox-exec. [S-primary] — [Designing agentic loops](https://simonwillison.net/2025/Sep/30/designing-agentic-loops); [Living dangerously](https://simonwillison.net/2025/Oct/22/living-dangerously-with-claude/)

**Armin Ronacher** (Flask, Jinja; ex-Sentry)
- "Agentic Coding Recommendations" (blog, 2025-06-12) **[partly dated: Claude Code Sonnet/Opus 4 era]**: simple languages and stable, low-churn ecosystems work best (Go, PHP, "basic Python"); Go helps because the toolchain decides which tests to run; tools must be fast, give clear errors, and be protected against misuse. [S-primary headings, S-secondary detail] — [post](https://lucumr.pocoo.org/2025/6/12/agentic-coding/); [Willison's note](https://simonwillison.net/2025/Jun/12/agentic-coding-recommendations/)
- Talk "Agentic Coding: The Future of Software Development with Agents" (mid-2025): runs Claude Code with `--dangerously-skip-permissions`; uses exactly one MCP server (Playwright) and CLIs for everything else; uses Gemini CLI as a sub-agent to save Claude's context. [S-secondary] — [talk notes](https://ikyle.me/blog/2025/agentic-coding-the-future-of-software-development-with-agents); [Willison on the MCP point](https://simonwillison.net/tags/armin-ronacher/)
- "Agentic Coding Things That Didn't Work" (2025-07-30): custom slash commands (/add-tests, /fix-bug, /commit) never stuck; hooks gave no efficiency gain for him. A Lobsters commenter reported the opposite for post-edit format/lint/typecheck hooks. [S-primary] — [post](https://lucumr.pocoo.org/2025/7/30/things-that-didnt-work/); [Lobsters thread](https://lobste.rs/s/yd42ea/agentic_coding_things_didn_t_work)
- "Agent Design Is Still Hard" (2025-11-21), "A Year of Vibes" (2025-12-22), "Pi: The Minimal Agent Within OpenClaw" (2026-01-31): he is "doubling down" on code generation, file systems, programmatic tool invocation through an interpreter, and skill-based learning; admits his vocal lack of success with MCP rested on little evidence; migrating former slash commands to skills; Pi's philosophy is that the agent extends itself rather than installing extensions. Also "Agent Psychosis: Are We Going Insane?" (2026-01-18), text not retrieved. [S-primary] — [Agent design](https://lucumr.pocoo.org/2025/11/21/agents-are-hard/); [A Year of Vibes](https://lucumr.pocoo.org/2025/12/22/a-year-of-vibes/); [Pi](https://lucumr.pocoo.org/2026/1/31/pi/); [Agent Psychosis](https://lucumr.pocoo.org/2026/1/18/agent-psychosis/)

**Mitchell Hashimoto** (HashiCorp co-founder; Ghostty)
- "My AI Adoption Journey" (blog, 2026-02-05). Six steps: Drop the Chatbot; Reproduce Your Own Work (redo your manual work with an agent to learn its limits); End-of-Day Agents; Outsource the Slam Dunks; Engineer the Harness; Always Have an Agent Running. [S-primary step names via aggregator listing] — [post](https://mitchellh.com/writing/my-ai-adoption-journey); [Willison's link](https://simonwillison.net/2026/feb/5/ai-adoption-journey)
- Harness engineering: every time an agent makes a mistake, engineer a fix so it cannot recur, either as AGENTS.md guidance or as a programmed tool (screenshot scripts, filtered test runs). Almost every line in Ghostty's AGENTS.md stems from an observed bad behaviour. A tool must be mentioned in the file or the agent will not use it. [S-primary / S-secondary mix] — [post](https://mitchellh.com/writing/my-ai-adoption-journey); [Towards AI summary](https://pub.towardsai.net/he-built-terraform-vagrant-and-ghostty-heres-how-he-stopped-fighting-ai-and-started-using-it-f621baa834f2)
- End-of-day agents: last part of the day launches agents on low-stakes work (deep research, parallel prototypes of vague ideas, read-only issue triage via `gh`); agents only produce reports, never reply publicly; he reviews in the morning for a "warm start". Background agents run about 10-20% of his working day; "I don't want to run agents for the sake of running agents." Turns off agent desktop notifications to control context switching. [S-secondary] — [serenitiesai summary](https://serenitiesai.com/articles/mitchell-hashimoto-ai-workflow); [mindbento summary](https://www.mindbento.com/mitchellh/my-ai-adoption-journey); [Zed interview](https://zed.dev/blog/agentic-engineering-with-mitchell-hashimoto)

**Steve Yegge** (ex-Amazon/Google/Sourcegraph; with Gene Kim, *Vibe Coding*, IT Revolution 2025)
- Beads (released October 2025): an issue tracker for agents that preserves tasks and context across sessions. Gas Town (open-sourced January 2026): Go orchestrator using tmux to manage roughly 20-30 parallel agents on top of Beads; backend later moved to Dolt. Both reached 1.0.0 per "Gas Town: from Clown Show to v1.0". [S-primary / S-secondary] — [v1.0 post](https://steve-yegge.medium.com/gas-town-from-clown-show-to-v1-0-c239d9a407ec); [SE Daily interview, 2026-02-12](https://softwareengineeringdaily.com/2026/02/12/gas-town-beads-and-the-rise-of-agentic-development-with-steve-yegge/); [repo](https://github.com/gastownhall/gastown)
- Roles: Mayor (coordinator you talk to, decomposes work into beads), Polecats (ephemeral workers on isolated branches), Refinery (merge queue, Bors-style bisecting, verification gates), Witness (patrols for stuck agents). [S-secondary/README] — [repo](https://github.com/gastownhall/gastown); [paddo analysis](https://paddo.dev/blog/gastown-two-kinds-of-multi-agent/)
- His 8-stage adoption ladder runs from autocomplete (1) to orchestration (8); Gas Town is a stage-8 tool, and he says most developers should not use it. [S-secondary] — [Maggie Appleton](https://maggieappleton.com/gastown); [HN discussion](https://news.ycombinator.com/item?id=46458936)
- Reported costs and breakage: about $100/hour in tokens at 12-30 agents (attributed to a DoltHub team), multiple $200/month accounts, auto-merged failing tests, force-pushes to main, duplicate fixes, lost designs. [S-secondary, not confirmed against Yegge's text] — [Gas Town: good, bad, ugly](https://tenzinwangdhen.com/posts/gastown-good-bad-ugly/); [Codex-oriented write-up](https://codex.danielvaughan.com/2026/04/08/gas-town-multi-agent-factory/)

**Geoffrey Huntley** (Ralph loop)
- "Ralph Wiggum as a 'software engineer'" (blog, 2025-07-14): a bash loop that feeds the same PROMPT.md to a fresh agent each iteration. One task per loop; the agent picks the most important item from fix_plan.md; specs/*.md hold requirements; AGENT.md holds learnings about how to build and run; progress lives on disk and in git, not in context. [S-primary / S-secondary] — [post](https://ghuntley.com/ralph/); [how-to repo](https://github.com/ghuntley/how-to-ralph-wiggum)
- Prompt details reported: up to 500 parallel Sonnet subagents for reading specs, but only one subagent for build/tests; update fix_plan.md via a subagent. "Backpressure" (tests, linters, type checks that reject wrong output automatically) is described as the real steering mechanism. [S-secondary] — [field study](https://github.com/ianhxu/agentic-engineering-field-study/blob/main/05-loop-engineering-multi-agent.md); [Ralph Playbook](https://claytonfarr.github.io/ralph-playbook)
- Cost: about $10/hour (about $10.50 for Sonnet) per The Register, January 2026; an anecdote of a $50,000 contract delivered for $297 in API costs is secondhand. [S-secondary] — [The Register](https://www.theregister.com/2026/01/27/ralph_wiggum_claude_loops/)

**Peter Steinberger** (PSPDFKit founder; OpenClaw; reported to have joined OpenAI in February 2026)
- "Just Talk To It - the no-bs Way of Agentic Engineering" (blog, 2025-10-14) **[tool specifics dated]**: 3-8 Codex CLI sessions in parallel, usually in the same folder (no worktrees), agents make their own atomic commits scoped to the files they edited; prompts often 1-2 sentences plus a screenshot; avoids formal plan mode and big spec documents, instead discusses options with the model; large personal AGENTS.md; CLIs over MCP because MCP servers are a constant context cost (he cites about 23k tokens for GitHub's MCP). [S-secondary plus Willison's summary] — [post](https://steipete.me/posts/just-talk-to-it); [Willison's link](https://simonwillison.net/2025/Oct/14/agentic-engineering/); [dev.to critique](https://dev.to/stevengonsalvez/peter-steinberger-says-just-talk-to-it-and-hes-mostly-right-37bh)
- "Shipping at Inference-Speed" (December 2025): output is capped by inference time; stack is TypeScript for web, Go for CLIs, Swift for Apple platforms; 3-8 projects in parallel using Codex's queue; still thinks hard about architecture, dependencies and UX. Pragmatic Engineer interview title: "I ship code I don't read". [S-secondary] — [summary](https://second-brain-nuxt.vercel.app/shipping-at-inference-speed); [Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/the-creator-of-clawd-i-ship-code/comments); [Latent Space profile](https://latent.space/wiki/peter-steinberger/)

**Kent Beck** (XP, TDD)
- "Augmented Coding: Beyond the Vibes" (newsletter, mid-2025; B+ tree library in Rust and Python): in augmented coding you still care about the code, tests and complexity, unlike vibe coding. Warning signs that the "genie" is off track: loops, and functionality he did not ask for (even reasonable next steps); also cheating such as disabling or deleting tests. He drives with strict TDD and precise step-by-step test instructions, and notes the AI lacks taste and keeps piling onto large functions. [S-primary/S-secondary] — [post](https://newsletter.kentbeck.com/p/augmented-coding-beyond-the-vibes); [author summary page](https://www.kentbeck.com/summaries/augmented-coding-beyond-the-vibes/); [Pragmatic Engineer interview](https://newsletter.pragmaticengineer.com/p/how-kent-beck-shapes-the-software)

**Addy Osmani** (Google Chrome DX per Zed; one source says Anthropic, unresolved)
- "My LLM coding workflow going into 2026" (blog, 2026-01-04): spec and plan before code; small iterative chunks paired with TDD; treat the model as a pair programmer needing oversight; sometimes run two or more models on the same problem to cross-check. [S-primary] — [post](https://addyosmani.com/blog/ai-coding-workflow/)
- "The 80% Problem in Agentic Coding" (2026-01-28): the 80% agent-written share is reachable on greenfield and solo work, much harder in large existing team codebases; beyond a threshold "review" becomes "rubber stamping". [S-primary] — [post](https://addyosmani.com/blog/the-80-problem-in-agentic-coding/)
- "Comprehension debt" (2026-03-14, also O'Reilly Radar): debt accrues when code is merged that nobody understands; cites an Anthropic study with 50% vs 67% comprehension-quiz scores for AI-assisted vs manual (figure from a secondary summary). The search summary's attribution of who coined the term is unverified. [S-primary/S-secondary] — [post](https://addyosmani.com/blog/comprehension-debt/); [O'Reilly](https://www.oreilly.com/radar/comprehension-debt-the-hidden-cost-of-ai-generated-code/)

**Harper Reed**
- "My LLM codegen workflow atm" (blog, 2025-02-16) **[dated: pre-agent, copy/paste + repomix + aider; the structure survives]**: brainstorm a spec by having the model ask one question at a time, save spec.md; have a reasoning model produce prompt_plan.md; generate todo.md that the tool checks off to keep state across sessions. A follow-up "Basic Claude Code" (2025-05-08) keeps the spec/plan front end with Claude Code executing. [S-primary; follow-up S-secondary] — [post](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/); [Willison's note](https://simonwillison.net/2025/Feb/21/my-llm-codegen-workflow-atm/); [explainer](https://howaiworks.ai/blog/harper-reed-llm-codegen-workflow)

**Jesse Vincent** (obra; Superpowers)
- "How I'm using coding agents" (September 2025) and "Superpowers: How I'm using coding agents in October 2025" (blog.fsck.com): brainstorm the design, create an isolated git worktree with a clean test baseline, write a plan of 2-5 minute micro-tasks with verification steps, dispatch each task to a fresh subagent, then two-phase review per task (spec compliance, then code quality); TDD and systematic-debugging skills; CodeRabbit for review. MIT-licensed; SKILL.md format predates Anthropic's native skills by about a week per Wikipedia. [S-secondary] — [index of his posts](https://tomrochette.com/agents/people-and-publications/jesse-vincent/); [guide](https://www.developersdigest.tech/blog/claude-code-superpowers-plugin-guide); [Wikipedia](https://en.wikipedia.org/wiki/Jesse_Vincent)

**Thorsten Ball** (Amp / Sourcegraph)
- "How to Build an Agent, or: The Emperor Has No Clothes" (2025-04-15): an agent is a model in a loop with a few tools, a few hundred lines of Go. "How I use Amp" (2025-05-15) and the talk "Think Harder: How I Prompt" exist but their content was not retrieved. Latent Space (September 2025): Amp favours deliberate context selection and fast feedback from tests and human review. [S-primary titles; S-secondary content] — [post](https://ampcode.com/how-to-build-an-agent); [How I use Amp](https://ampcode.com/notes/how-i-use-amp); [talk](https://thorstenball.com/talks/how-i-prompt/); [Latent Space write-up](https://latent.space/reading/amp-emperor-podcast/)

**Birgitta Böckeler / martinfowler.com / Thoughtworks**
- "Understanding Spec-Driven-Development: Kiro, spec-kit, and Tessl" (about October 2025): three levels, spec-first, spec-anchored, spec-as-source; compares spec-as-source to model-driven development; wants usage reports on real codebases before believing the claims. [S-primary via Fowler's post] — [article](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html); [Fowler](https://x.com/martinfowler/status/1978453531565695415)
- Harness engineering article (dated 2026-04-02 by search results; exact title unconfirmed): controls split into feedforward guides and feedback sensors; goal is to direct human attention, not remove it; functional behaviour is the weak point. Kief Morris, "Humans and Agents in Software Engineering Loops": humans work "on the loop" by building the harness. [S-secondary / S-primary for Morris] — [Morris](https://www.martinfowler.com/articles/exploring-gen-ai/humans-and-agents.html); [Thoughtworks podcast](https://www.thoughtworks.com/insights/podcasts/technology-podcasts/what-harness-engineering)

**Dan Shipper and Kieran Klaassen** (Every)
- "Compound Engineering: How Every Codes With Agents" (article + open-source Claude Code plugin, late 2025/2026): loop of Plan, Work, Review, Compound; about 80% of effort in plan and review, 20% in work and compound; the compound step codifies lessons into prompts, subagents and slash commands; later added ideation/brainstorming; multi-agent review. [S-secondary] — [syndicated article](https://www.msn.com/en-us/news/technology/compound-engineering-how-every-codes-with-agents/ar-AA1Vm2AW); [pattern entry](https://aipatternbook.com/compound-engineering); [plugin research](https://rywalker.com/research/compound-engineering-plugin)

**antirez** (Redis)
- "Don't fall into the anti-AI hype" (antirez.com/news/158, January 2026) and a follow-up proposing the term "automatic programming" (distinct from vibe coding, where you are not part of the process). Says current models complete medium-size projects with little supervision; still uses web-chat prompting when needed. [S-primary/S-secondary] — [post](https://antirez.com/news/158); [HN thread](https://news.ycombinator.com/item?id=46574276); [Lobsters](https://lobste.rs/~antirez/threads)

**DHH** (Rails, 37signals)
- Lex Fridman Podcast #501 (reported 2026-08-26): changed his mind around November 2025; Omarchy is now "100%" written by agents but steered by him; uses OpenCode; one summary mentions 16 parallel agents. Handing over Basecamp and HEY (large existing codebases with customers) has been much harder. [S-primary transcript snippet / S-secondary] — [transcript](https://lexfridman.com/dhh-2-transcript/); [Startup Fortune](https://startupfortune.com/dhh-admits-ai-agents-struggled-on-basecamp-even-as-they-built-all-of-omarchy/)

**Others of similar standing**
- Andrej Karpathy (X post, 2026-01-26): moved from 80% manual to 80% agent between November and December 2025; a few Claude Code sessions on the left, IDE on the right; "watch them like a hawk"; give success criteria rather than instructions; calls both "no IDE" and "agent swarm" hype excessive; failure list: wrong assumptions, no clarifying questions, sycophancy, bloated abstractions, dead code, unrelated edits, 1000 lines where 100 would do. [S-primary quote] — [post mirror](https://xcancel.com/karpathy/status/2015883857489522876)
- Boris Cherny (Claude Code creator; X threads, January 2026): about 5 numbered terminal sessions plus 5-10 web sessions; starts most sessions in plan mode, iterates the plan, then auto-accept; team-shared CLAUDE.md updated whenever Claude does something wrong; top tip is giving Claude a way to verify its work. Counts vary by source. [S-secondary] — [tips compilation](https://github.com/shanraisshan/claude-code-best-practice/blob/main/tips/claude-boris-13-tips-03-jan-26.md); [paddo](https://paddo.dev/blog/how-boris-uses-claude-code/)
- Dex Horthy (HumanLayer; "No Vibes Allowed" talk; Pragmatic Engineer interview 2026): research, plan, implement with "frequent intentional compaction". [S-secondary] — [Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/context-engineering-with-dex-horthy); [talk page](https://aietalks.com/talks/no-vibes-allowed-solving-hard-problems-in-complex-codebases)
- ThePrimeagen: January 2026 stunt in which a fake "I was wrong" post got followers to install an unread skill; read as a supply-chain and review-habit warning. [S-secondary] — [analysis](https://superframeworks.com/articles/theprimeagen-poisoned-ai-skill-supply-chain-warning); [commentary](https://tim-schipper.nl/en/blog/theprimeagen-was-right)

### Inferences
- The people with the most detailed, durable advice (Willison, Hashimoto, Ronacher, Beck) describe a human who stays in the loop and invests in the environment; the people at the autonomy extreme (Yegge, Huntley, Steinberger) are explicit that their setups are expensive, experimental, or unsuitable for most developers.
- Several authors reversed or softened positions within months (Ronacher on MCP and slash commands, DHH and Karpathy on agents overall, Cognition on multi-agents). Any single post should be read as a snapshot.

### Gaps
- Theo (t3.gg): no source retrieved; search budget ran out. No claims made.
- ThePrimeagen's substantive critiques beyond the skill stunt, Thorsten Ball's "How I use Amp" body, Yegge/Kim *Vibe Coding* book specifics, Yegge's own wording of Gas Town warnings, Ronacher's "Agent Psychosis" text, Willison's "cognitive debt" and anti-pattern chapters: not retrieved.
- Nothing here was read on the original page; exact quotes should be re-checked before publication.

## 2. Planning and specs

### Takeaway
Nearly everyone plans before letting an agent write code, but the weight of the plan ranges from a conversation (Steinberger) to formal spec files and task trackers (Reed, Vincent, Huntley, Yegge, Every). Heavy spec-driven tooling is viewed with scepticism by the Fowler-site authors.

### Cited Findings
- Spec, then plan, then execute with persisted files (spec.md, prompt_plan.md, todo.md), with the model interviewing you one question at a time. [S-primary] — [Harper Reed](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/)
- Define the problem and plan before generating; vague prompts are the common mistake. [S-primary] — [Osmani](https://addyosmani.com/blog/ai-coding-workflow/)
- Plan mode first, iterate the plan, then auto-accept so the implementation is usually one-shot. [S-secondary] — [Cherny tips](https://github.com/shanraisshan/claude-code-best-practice/blob/main/tips/claude-boris-13-tips-03-jan-26.md)
- Brainstorm, then a plan of 2-5 minute tasks each with a verification step. [S-secondary] — [Superpowers guide](https://www.developersdigest.tech/blog/claude-code-superpowers-plugin-guide)
- About 80% of effort in planning and review. [S-secondary] — [Compound engineering](https://aipatternbook.com/compound-engineering)
- Research, plan, implement as separate fresh-context phases, each handing off a verified Markdown artifact. [S-secondary] — [Horthy / Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/context-engineering-with-dex-horthy)
- Frozen specs plus a mutable fix_plan.md the loop consumes one item at a time. [S-secondary] — [Ralph field study](https://github.com/ianhxu/agentic-engineering-field-study/blob/main/05-loop-engineering-multi-agent.md)
- Issue tracker built for agents (Beads) so work survives sessions. [S-secondary] — [SE Daily](https://softwareengineeringdaily.com/2026/02/12/gas-town-beads-and-the-rise-of-agentic-development-with-steve-yegge/)
- Counter-position: avoid rigid plan mode and structured documents, talk to the model and ask for options; write a spec only for tricky work. [S-secondary] — [Steinberger summary](https://dev.to/stevengonsalvez/peter-steinberger-says-just-talk-to-it-and-hes-mostly-right-37bh)
- Karpathy: plan mode helps, but he wants a lighter inline plan mode; state success criteria and let the agent loop. [S-primary] — [Karpathy](https://xcancel.com/karpathy/status/2015883857489522876)
- Spec-as-source is likened to failed model-driven development; SDD tools remain unproven on real codebases. [S-primary via Fowler] — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)

### Inferences
- The stable core is "agree on intent and acceptance criteria before code, and keep the plan in a file outside the context window". The formality appears to scale with task length and with how many agents or sessions must share the plan.
- Steinberger's light planning coincides with strong 2025-26 models and a solo, mostly greenfield setting; it may not transfer to team codebases (compare Osmani's and DHH's greenfield caveats).

### Gaps
- No retrieved data comparing outcomes of heavy vs light planning.
- PRD-specific templates and GitHub-issue-as-task workflows were not covered by the sources retrieved.

## 3. Verification

### Takeaway
The single most repeated instruction is to give the agent a way to check its own work: tests first, then running the real product, then automated review. Human review is acknowledged as the bottleneck, and practitioners disagree about whether it can be dropped.

### Cited Findings
- Run existing tests first, then red/green TDD for every change. [S-primary] — [Willison](https://simonwillison.net/2026/Mar/14/pragmatic-summit/)
- Strict TDD with small steps; watch for disabled or deleted tests. [S-primary] — [Beck](https://newsletter.kentbeck.com/p/augmented-coding-beyond-the-vibes)
- Have the agent run the product and record evidence (Showboat), drive a browser (Rodney, Playwright). [S-primary] — [Willison](https://simonwillison.net/2026/Feb/10/showboat-and-rodney/); [Ronacher talk notes](https://ikyle.me/blog/2025/agentic-coding-the-future-of-software-development-with-agents)
- "Give Claude a way to verify its work" as the top tip. [S-secondary] — [Cherny](https://paddo.dev/blog/how-boris-uses-claude-code/)
- Build verification tools into the harness (screenshot scripts, filtered test runs). [S-secondary] — [Hashimoto summary](https://pub.towardsai.net/he-built-terraform-vagrant-and-ghostty-heres-how-he-stopped-fighting-ai-and-started-using-it-f621baa834f2)
- Backpressure: tests, linters and type checks that automatically reject wrong output steer the agent more than prose. [S-secondary] — [Ralph Playbook](https://claytonfarr.github.io/ralph-playbook)
- Second-agent review: per-task spec-compliance then code-quality review (Superpowers); multi-agent review (Every); CodeRabbit (Vincent); cross-checking with a second model (Osmani); a merge-queue agent with verification gates (Gas Town Refinery). [S-secondary] — [Superpowers](https://www.developersdigest.tech/blog/claude-code-superpowers-plugin-guide); [Every plugin](https://rywalker.com/research/compound-engineering-plugin); [Osmani](https://addyosmani.com/blog/ai-coding-workflow/); [Gas Town repo](https://github.com/gastownhall/gastown)
- Human review is the natural bottleneck of parallel agents. [S-primary] — [Willison](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/)
- Past a threshold, review turns into rubber stamping. [S-primary] — [Osmani](https://addyosmani.com/blog/the-80-problem-in-agentic-coding/)
- OpenAI's harness-engineering report (Ryan Lopopolo, 2026-02-11): about 1M lines and about 1,500 PRs over five months with no human-written code; human review reportedly optional for many merges; relied on agent-written linters and agent-readable docs. [S-secondary] — [ZenML summary](https://www.zenml.io/llmops-database/extreme-harness-engineering-building-production-software-with-zero-human-written-code); [agentastic](https://www.agentastic.dev/harness-engineering)

### Inferences
- Verification is layered in every serious workflow: deterministic checks first, agent self-demonstration second, agent review third, human review last and scarcest.
- Agent self-reports are not trusted on their own; Willison's Showboat exists precisely to tie claims to executed commands, and even that was reported as fakeable.

### Gaps
- CI-as-feedback-loop specifics (agents watching CI and auto-fixing) were mentioned only indirectly.
- No retrieved measurements of second-agent review precision.

## 4. Context management

### Takeaway
Keep sessions small and restart often, moving state into files; keep instruction files short and grown from observed failures rather than written up front.

### Cited Findings
- Frequent intentional compaction: distil the useful context into a Markdown artifact, verify it, start a fresh session. Thresholds differ by source: start thinking at about 40% of the window; some push to 60-70%; Horthy's talk reportedly gives about 100K tokens for smaller windows and 300-400K for a 1M window. [S-secondary, sources conflict] — [Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/context-engineering-with-dex-horthy); [Latent Space notes](https://latentspace.podhood.com/2b94b1fe-2689-4fac-8845-eaa30eec133a)
- Ralph: every iteration is a fresh context; memory is fix_plan.md, AGENT.md and git. [S-secondary] — [Huntley](https://ghuntley.com/ralph/)
- Fresh subagent per task so the coordinator's context stays clean. [S-secondary] — [Superpowers](https://www.developersdigest.tech/blog/claude-code-superpowers-plugin-guide)
- Use a different agent (Gemini CLI) as a sub-agent to save the main agent's context. [S-secondary] — [Ronacher talk notes](https://ikyle.me/blog/2025/agentic-coding-the-future-of-software-development-with-agents)
- Instruction files: add a line each time the agent repeats a mistake; mention tools or they are not used. [S-secondary] — [Hashimoto](https://serenitiesai.com/articles/mitchell-hashimoto-ai-workflow)
- Team-shared CLAUDE.md updated on every observed error. [S-secondary] — [Cherny](https://github.com/shanraisshan/claude-code-best-practice/blob/main/tips/claude-boris-13-tips-03-jan-26.md)
- Prefer self-describing CLIs to front-loaded docs in the agents file; MCP tool definitions are a standing context cost. [S-secondary] — [Steinberger](https://dev.to/stevengonsalvez/peter-steinberger-says-just-talk-to-it-and-hes-mostly-right-37bh)
- Instructions do not fully fix behaviour: Karpathy's listed failure modes persisted despite CLAUDE.md attempts. [S-primary] — [Karpathy](https://xcancel.com/karpathy/status/2015883857489522876)
- todo.md checked off by the tool to carry state across sessions. [S-primary] — [Reed](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/)

### Inferences
- "State on disk, not in the conversation" is the common mechanism behind compaction, Ralph, Beads, todo files and subagent dispatch.
- Specific percentage thresholds are folk heuristics that shift with model and window size; the 40% figure should not be quoted as a rule.

### Gaps
- No retrieved practitioner guidance on /clear vs /compact specifically, nor on recommended instruction-file length; Anthropic's official best-practices page was not retrieved.

## 5. Parallelism

### Takeaway
Experienced individuals typically run a handful of agents (about 3-8 interactive, sometimes 10-15 including cloud sessions); 20-30 is orchestrator territory with real cost and breakage. Isolation method (worktrees, separate checkouts, same folder) is unsettled, and human attention and review capacity are what break first.

### Cited Findings
- 3-8 Codex sessions in the same folder with atomic commits, no worktrees. [S-secondary] — [Steinberger](https://dev.to/stevengonsalvez/peter-steinberger-says-just-talk-to-it-and-hes-mostly-right-37bh)
- Multiple terminals in separate directories, fresh checkouts instead of worktrees; cloud async agents for riskier work. [S-primary] — [Willison](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/)
- 5 local plus 5-10 web sessions, some accounts say each in its own worktree; team guidance of 3-5. [S-secondary, counts vary] — [Cherny](https://paddo.dev/blog/how-boris-uses-claude-code/); [team tips gist](https://gist.github.com/joyrexus/e20ead11b3df4de46ab32b4a7269abe0)
- Worktree per plan with clean test baseline. [S-secondary] — [Superpowers](https://www.developersdigest.tech/blog/claude-code-superpowers-plugin-guide)
- "A small few" Claude Code sessions beside an IDE; swarm hype is too much for now. [S-primary] — [Karpathy](https://xcancel.com/karpathy/status/2015883857489522876)
- Background agents only 10-20% of the day; notifications off; no agents for their own sake. [S-secondary] — [Hashimoto](https://www.mindbento.com/mitchellh/my-ai-adoption-journey)
- 20-30 agents with Mayor/Polecat/Refinery roles; reported about $100/hour and merge and state corruption failures. [S-secondary] — [Gas Town analysis](https://tenzinwangdhen.com/posts/gastown-good-bad-ugly/)
- 16 parallel agents reported for DHH. [S-secondary] — [Runtime Wire](https://runtimewire.com/article/omarchy-quattro-dhh-ai-coding-convert)
- Cognition: "Don't Build Multi-Agents" (June 2025: share full context, do not split conflicting decisions) revised in "Multi-Agents: What's Actually Working" (April 2026): keep writes single-threaded, one main loop carries state, subagents are stateless narrow workers contributing intelligence not actions. [S-primary/S-secondary] — [Cognition 2026](https://cognition.com/blog/multi-agents-working); [Walden Yan](https://x.com/walden_yan/status/2047054554433462360)
- Carlini (Anthropic, February 2026): 16 parallel Claudes, about 2,000 sessions, just under $20,000, 100k-line Rust C compiler that builds Linux 6.9; regressions and merge conflicts still appear without supervision. [S-primary] — [Anthropic](https://www.anthropic.com/research/building-c-compiler)

### Inferences
- Parallel reading and research scale well; parallel writing to the same code needs serialization (merge queue, single writer, or atomic commits). Huntley's "500 readers, 1 builder", Cognition's single-threaded writes and Gas Town's Refinery are the same idea.
- Reported agent counts are self-reported and inconsistent across summaries; treat them as orders of magnitude.

### Gaps
- No retrieved comparison of worktrees vs separate clones vs same-folder.
- Cursor's long-running planner/worker experiments and orchestration tools other than Gas Town (Conductor, Herdr, etc.) were not retrieved.

## 6. Long-running autonomous loops

### Takeaway
Ralph-style loops work when progress is externalised to files, each iteration is small, and automated checks push back; they are cheap per hour but need caps and isolation, and the most careful practitioners restrict unattended runs to low-stakes or read-only work.

### Cited Findings
- Ralph mechanics, one task per loop, fresh context, disk state. [S-primary/S-secondary] — [Huntley](https://ghuntley.com/ralph/)
- Guardrails seen in implementations: one task per iteration, tests first, commit before exit, iteration cap (e.g. MAX=50), state-based completion checks, pre-flight token/cost limits. These are from the ecosystem, not confirmed as Huntley's own. [S-secondary] — [Steve Kinney](https://stevekinney.com/writing/the-ralph-loop); [DreamHost explainer](https://www.dreamhost.com/blog/ralph-wiggum/)
- Cost about $10/hour. [S-secondary] — [The Register](https://www.theregister.com/2026/01/27/ralph_wiggum_claude_loops/)
- End-of-day agents limited to research, prototypes and read-only triage, reports only. [S-secondary] — [Hashimoto](https://serenitiesai.com/articles/mitchell-hashimoto-ai-workflow)
- Unattended YOLO needs a sandbox limiting files, secrets and network. [S-primary] — [Willison](https://simonwillison.net/2025/Sep/30/designing-agentic-loops)
- Karpathy: models are exceptionally good at looping until they meet specific goals; 30-minute struggles that succeed. [S-primary] — [Karpathy](https://xcancel.com/karpathy/status/2015883857489522876)
- Pragmatic Engineer published "What is loop engineering" (2026), indicating the practice has a name; body not retrieved. [title only] — [Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/what-is-loop-engineering)
- Scale examples: OpenAI 1M lines / 5 months; Carlini $20k compiler. [S-secondary/S-primary] — [ZenML](https://www.zenml.io/llmops-database/extreme-harness-engineering-building-production-software-with-zero-human-written-code); [Anthropic](https://www.anthropic.com/research/building-c-compiler)

### Inferences
- The successful long runs share a strong external oracle (a compiler test suite, a spec to clone, an existing test corpus). Without one, loops tend to drift.
- Cost per hour is low; the real cost is review and cleanup of what the loop produced.

### Gaps
- Huntley's own statements on spend caps; built-in loop features in current tools (e.g. goal modes); documented overnight-run failures. An article about a $1.3M token bug surfaced in results but its details were not established and it is not relied on here.

## 7. Tooling choices

### Takeaway
The practitioner consensus through 2025-26 moved from MCP toward plain CLI tools and skills, with Playwright-style browser control the main MCP survivor; permission prompts are widely bypassed, and the responsible version of that is a sandbox; boring, fast, well-tooled stacks help agents.

### Cited Findings
- CLI over MCP; only Playwright MCP. [S-secondary] — [Willison on Ronacher](https://simonwillison.net/tags/armin-ronacher/)
- MCP as constant context cost (about 23k tokens for GitHub MCP). [S-secondary] — [Steinberger](https://dev.to/stevengonsalvez/peter-steinberger-says-just-talk-to-it-and-hes-mostly-right-37bh)
- Ronacher later conceded his MCP scepticism had little evidence, while still favouring code generation, file systems and skills; moved slash commands to skills. [S-primary] — [A Year of Vibes](https://lucumr.pocoo.org/2025/12/22/a-year-of-vibes/); [Pi](https://lucumr.pocoo.org/2026/1/31/pi/)
- Skills as packaged methodology (Superpowers; Every's plugin). [S-secondary] — [Superpowers](https://www.developersdigest.tech/blog/claude-code-superpowers-plugin-guide); [Every plugin](https://rywalker.com/research/compound-engineering-plugin)
- Skills as supply-chain risk: followers installed an unread skill. [S-secondary] — [Primeagen analysis](https://superframeworks.com/articles/theprimeagen-poisoned-ai-skill-supply-chain-warning)
- Hooks and slash commands did not pay off for Ronacher; others report useful post-edit lint/format hooks. [S-primary] — [Ronacher](https://lucumr.pocoo.org/2025/7/30/things-that-didnt-work/); [Lobsters](https://lobste.rs/s/yd42ea/agentic_coding_things_didn_t_work)
- Permissions: Ronacher runs skip-permissions; Willison uses YOLO only where injection is implausible and argues for sandboxes. [S-secondary/S-primary] — [talk notes](https://ikyle.me/blog/2025/agentic-coding-the-future-of-software-development-with-agents); [Willison](https://simonwillison.net/2025/Oct/22/living-dangerously-with-claude/)
- Stack: Go, PHP, basic Python (Ronacher); TypeScript, Go, Swift (Steinberger); Beck and Carlini used Rust successfully. [S-secondary] — [Ronacher](https://lucumr.pocoo.org/2025/6/12/agentic-coding/); [Steinberger](https://second-brain-nuxt.vercel.app/shipping-at-inference-speed); [Beck](https://newsletter.kentbeck.com/p/augmented-coding-beyond-the-vibes)
- Tools should be fast, emit clear errors, and resist misuse. [S-secondary] — [Ronacher](https://lucumr.pocoo.org/2025/6/12/agentic-coding/)
- Harness choice is fluid: Ronacher compares Claude Code, Amp and Pi; DHH uses OpenCode; Steinberger moved to Codex. [S-secondary] — [A Year of Vibes](https://lucumr.pocoo.org/2025/12/22/a-year-of-vibes/); [DHH](https://lexfridman.com/dhh-2-transcript/)

### Inferences
- The durable principle is "tools the model already knows, discoverable on demand, costing no context until used". Whether that is delivered as a CLI, a skill or a lazily loaded MCP server matters less than it did in 2025.
- Language advice from mid-2025 is partly model-dependent and should be treated as dated.

### Gaps
- No retrieved head-to-head of sandbox options or of permission/auto-approve modes in current tools.

## 8. Failure modes and criticism

### Takeaway
The recurring failures are over-complex code, silent wrong assumptions, test cheating, review overload and loss of understanding. The measured productivity evidence is mixed and weak: the famous 19% slowdown is now disowned as outdated by its authors, while organisation-level data shows more output and more review burden without clear delivery gains.

### Cited Findings
- Model failure list (wrong assumptions, no pushback, bloat, dead code, unrelated edits). [S-primary] — [Karpathy](https://xcancel.com/karpathy/status/2015883857489522876)
- Loops, unrequested features, disabled tests. [S-primary] — [Beck](https://newsletter.kentbeck.com/p/augmented-coding-beyond-the-vibes)
- Rubber-stamp review and comprehension debt. [S-primary] — [Osmani 80%](https://addyosmani.com/blog/the-80-problem-in-agentic-coding/); [Osmani comprehension debt](https://addyosmani.com/blog/comprehension-debt/)
- Agents much weaker on large existing commercial codebases than on greenfield. [S-secondary] — [DHH coverage](https://startupfortune.com/dhh-admits-ai-agents-struggled-on-basecamp-even-as-they-built-all-of-omarchy/)
- METR: 2025 RCT measured experienced open-source developers 19% slower while they believed they were faster. February 2026 update ("We are Changing our Developer Productivity Experiment Design"): that finding is "now outdated"; new estimates lean toward speedup (about -18% time, CI -38% to +9% for returning developers; -4%, CI -15% to +9% for new) but are not significant and are biased because 30-50% of developers declined to submit tasks they would not do without AI. [S-primary tweet / S-secondary numbers] — [METR on X](https://x.com/METR_Evals/status/2026355544668385373); [write-up](https://www.matthewswong.com/en/blog/metr-developer-productivity-2026-uplift-update/)
- GitClear: churn up, refactoring down, cloned code up (figures vary by write-up). CodeRabbit (December 2025, 470 PRs, vendor study): about 1.7x more issues per AI co-authored PR. Faros: 98% more PRs with flat DORA metrics and sharply higher review time. DORA 2024: about 7.2% reduction in delivery stability. [S-secondary, numbers unverified] — [DevClass on GitClear](https://www.devclass.com/ai-ml/2025/02/20/ai-is-eroding-code-quality-states-new-in-depth-report/1626250); [SoftwareSeni overview](https://www.softwareseni.com/the-ai-productivity-paradox-why-developers-feel-fast-but-deliver-slow/)
- Security: lethal trifecta and sandboxing. [S-primary] — [Willison](https://simonwillison.net/2025/Oct/22/living-dangerously-with-claude/)
- Hashimoto reportedly planted injection text in AGENTS.md to catch unreviewed AI contributions. [S-secondary, single source] — [kindapeak](https://kindapeak.com/tech-culture/mitchell-hashimoto-agents-md-prompt-injection-slop/)
- Orchestrator criticism: Gas Town called poorly designed; most developers told not to use it. [S-secondary] — [Appleton](https://maggieappleton.com/gastown)
- Mitigations practitioners name: TDD and small steps (Beck, Willison); IDE alongside to read diffs (Karpathy); harness fixes per mistake (Hashimoto); linear walkthroughs to understand agent code (Willison); keeping architecture decisions human (Steinberger); direct human attention to where it matters (Böckeler). [mixed] — [Willison guide](https://simonwillison.net/guides/agentic-engineering-patterns/); [Latent Space on Steinberger](https://latent.space/wiki/peter-steinberger/); [Thoughtworks](https://www.thoughtworks.com/insights/podcasts/technology-podcasts/what-harness-engineering)

### Inferences
- Individual expert testimony (large gains) and organisational telemetry (flat delivery, more review) are not contradictory: the bottleneck moves from writing to verifying and understanding.
- There is currently no reliable controlled estimate of productivity effect for 2026-era agents.

### Gaps
- Concrete 2025-26 security incidents with coding agents, skill-atrophy studies, and DORA 2025 were not retrieved (search budget exhausted). The comprehension-quiz figure is secondhand.

## 9. Consolidated: practices several independent practitioners agree on

### Takeaway
Nine practices have support from three or more independent practitioners in the sources above.

### Cited Findings
- **Give the agent a runnable feedback loop** (tests, build, linters, the app itself). Willison, Beck, Cherny, Huntley, Hashimoto, Karpathy. — [Willison](https://simonwillison.net/2026/Mar/14/pragmatic-summit/); [Karpathy](https://xcancel.com/karpathy/status/2015883857489522876); [Ralph Playbook](https://claytonfarr.github.io/ralph-playbook)
- **Tests first / TDD.** Willison, Beck, Osmani, Vincent, Karpathy. — [Beck](https://newsletter.kentbeck.com/p/augmented-coding-beyond-the-vibes); [Osmani](https://addyosmani.com/blog/ai-coding-workflow/)
- **Agree intent before code and persist it in a file.** Reed, Osmani, Cherny, Vincent, Klaassen, Horthy, Huntley. — [Reed](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/); [Horthy](https://newsletter.pragmaticengineer.com/p/context-engineering-with-dex-horthy)
- **Small tasks, fresh contexts, state on disk.** Huntley, Horthy, Vincent, Reed, Yegge. — [Huntley](https://ghuntley.com/ralph/); [SE Daily](https://softwareengineeringdaily.com/2026/02/12/gas-town-beads-and-the-rise-of-agentic-development-with-steve-yegge/)
- **Grow the instruction file from observed mistakes; turn repeated fixes into tools or skills.** Hashimoto, Cherny, Klaassen/Shipper, Huntley (AGENT.md). — [Hashimoto](https://mitchellh.com/writing/my-ai-adoption-journey); [Compound engineering](https://aipatternbook.com/compound-engineering)
- **Prefer CLIs (and now skills) to MCP servers.** Ronacher, Steinberger, Willison (reporting), Hashimoto (scripts). — [Willison on Ronacher](https://simonwillison.net/tags/armin-ronacher/); [Steinberger](https://simonwillison.net/2025/Oct/14/agentic-engineering/)
- **A handful of parallel agents, with writes serialized.** Willison, Steinberger, Cherny, Karpathy, Cognition, Huntley. — [Willison](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/); [Cognition](https://cognition.com/blog/multi-agents-working)
- **Autonomy needs isolation** (sandbox, cloud agent, read-only scope). Willison, Hashimoto. — [Willison](https://simonwillison.net/2025/Sep/30/designing-agentic-loops); [Hashimoto summary](https://serenitiesai.com/articles/mitchell-hashimoto-ai-workflow)
- **Humans keep architecture and taste; agents fare far better on greenfield.** Steinberger, Beck, Karpathy, Osmani, DHH. — [Latent Space](https://latent.space/wiki/peter-steinberger/); [DHH](https://lexfridman.com/dhh-2-transcript/)

### Inferences
- These nine form a coherent method: constrain the task, externalise state, let deterministic checks do the steering, and spend human time on intent and review.

### Gaps
- Agreement is assessed from summaries; some attributions may be weaker in the original texts.

## 10. Open disagreements

### Takeaway
Six questions remain contested among credible practitioners.

### Cited Findings
- **Must a human read the code?** Yes: Karpathy ("watch them like a hawk"), Beck, Osmani. No or not always: Steinberger ("I ship code I don't read"), OpenAI's harness team. — [Karpathy](https://xcancel.com/karpathy/status/2015883857489522876); [Pragmatic Engineer on Steinberger](https://newsletter.pragmaticengineer.com/p/the-creator-of-clawd-i-ship-code/comments); [OpenAI summary](https://www.zenml.io/llmops-database/extreme-harness-engineering-building-production-software-with-zero-human-written-code)
- **Heavy specs vs "just talk to it".** Reed, Vincent, Every vs Steinberger; Böckeler sceptical of spec-as-source. — [Reed](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/); [Steinberger](https://steipete.me/posts/just-talk-to-it); [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- **Swarms/orchestrators vs a few supervised agents.** Yegge and Huntley vs Karpathy and Hashimoto; Cognition in between. — [Yegge](https://steve-yegge.medium.com/gas-town-from-clown-show-to-v1-0-c239d9a407ec); [Karpathy](https://xcancel.com/karpathy/status/2015883857489522876); [Cognition](https://cognition.com/blog/multi-agents-working)
- **Worktrees vs checkouts vs same folder.** Vincent/Cherny vs Willison vs Steinberger. — [Superpowers](https://www.developersdigest.tech/blog/claude-code-superpowers-plugin-guide); [Willison](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/); [Steinberger](https://dev.to/stevengonsalvez/peter-steinberger-says-just-talk-to-it-and-hes-mostly-right-37bh)
- **Skip permissions vs sandbox-first.** Ronacher vs Willison. — [talk notes](https://ikyle.me/blog/2025/agentic-coding-the-future-of-software-development-with-agents); [Willison](https://simonwillison.net/2025/Oct/22/living-dangerously-with-claude/)
- **Do hooks, slash commands and MCP earn their keep?** Ronacher (no, then partly revised) vs teams using lint hooks and Playwright MCP. — [Ronacher](https://lucumr.pocoo.org/2025/7/30/things-that-didnt-work/); [Lobsters](https://lobste.rs/s/yd42ea/agentic_coding_things_didn_t_work)
- **Is there a net productivity gain?** Practitioner testimony vs METR's inconclusive data and Faros/GitClear telemetry. — [METR](https://x.com/METR_Evals/status/2026355544668385373); [SoftwareSeni](https://www.softwareseni.com/the-ai-productivity-paradox-why-developers-feel-fast-but-deliver-slow/)

### Inferences
- Most disagreements track context (solo greenfield vs team legacy code, risk tolerance, budget) more than incompatible beliefs.

### Gaps
- Theo's and ThePrimeagen's positions on these questions could not be sourced in this run.

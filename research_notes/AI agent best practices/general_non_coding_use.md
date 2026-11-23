# AI agents for general, non-coding work: practices, setups, limits (notes as of Nov 2026)

**Verification legend used throughout**
- **[PAGE]** = I fetched and read the page itself in this session.
- **[SEARCH]** = taken from search-result summaries/snippets only; wording and details should be checked against the original before being quoted.
- **[TITLE ONLY]** = the source surfaced in search but I saw no content; listed so the writer knows it exists, not as evidence.

**Environment limits that shaped these notes.** Direct fetches failed (DNS) for oneusefulthing.org, lennysnewsletter.com, every.to, simonwillison.net, thezvi.substack.com and lexfridman.com, so nothing from those hosts is page-verified. Fetches succeeded only for anthropic.com, claude.com and support.claude.com, which means the page-verified material is disproportionately Anthropic's own (vendor) documentation. The session's web-search budget also ran out before the last three planned queries (Every's "How Claws Took Over Every", Claire Vo's OpenClaw guide on Lenny's, Zvi's OpenClaw post), so those are TITLE ONLY.

---

## 1. What do respected voices recommend for knowledge workers?

### Takeaway
The most consistent message across credible voices (Mollick, Every, Levie, Embiricos, Forte) is that agent value outside coding is limited less by model capability than by the user's ability to delegate (brief, supply context, review) and by how verifiable the output is. Detailed, page-verified practice guidance from these individuals was not obtainable here; most of what follows is from search summaries.

### Cited Findings

**Ethan Mollick (Wharton; One Useful Thing)**
- "Management as AI superpower", essay, 27 Jan 2026. Argues that giving instructions to agents amounts to reinventing management; includes a section titled "Delegation as the new prompting". [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/management-as-ai-superpower)
- Delegation decision depends on three variables: Human Baseline Time (how long it takes you), Probability of Success (chance the AI output meets your bar on a given attempt), and AI Process Time (time to request, wait for and evaluate output). Higher success probability means you pay the process overhead fewer times. [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/management-as-ai-superpower)
- His worked example is hypothetical, not measured: a one-hour task the AI does in minutes but which takes 30 minutes to check. [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/management-as-ai-superpower)
- A good brief answers questions including: what specific outputs are needed, what interim outputs let you follow progress, what the agent should check before saying it is finished, plus the limits of delegated authority and the purpose of the work. A secondary summary says there are six questions in total; I could not see the full list. [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/management-as-ai-superpower); six-question count from [Lead with AI summary](https://www.leadwithai.co/article/delegation-engineering-why-agentic-ai-fails-without-managers)
- Classroom evidence in the same essay: executive MBA students built startups from scratch in four days using Claude Code and general models for research, positioning, pitching and financial modelling. [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/management-as-ai-superpower)
- "Claude Code and What Comes Next", essay, 7 Jan 2026: accessible overview for non-programmers; a single prompt ran autonomously for 74 minutes and produced a deployable site; a reader comment quotes him saying Claude Code is great for things that are not, or are barely, code. [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/claude-code-and-what-comes-next/comments)
- "On Working with Wizards", essay, 11 Sep 2025 (date from an ingested copy): with agents he "cannot be entirely sure what the AI system actually did"; he describes checking a video summary's facts several times and finding no error in a paper critique while noting that does not mean there are none. Secondary sources attribute a "provisional trust" stance to the essay; exact wording unconfirmed. [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/on-working-with-wizards)
- "A Guide to Which AI to Use in the Agentic Era" (Feb 2026) introduced a models / apps / harnesses framing; the Summer 2026 edition "An opinionated guide to which AI to use to do stuff" is, per Simon Willison's 27 Jul 2026 link post, much more about agentic systems, naming ChatGPT's agent modes as Work and Codex and Claude's as Cowork and Code. [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/an-opinionated-guide-to-which-ai-b22); [Willison link post](https://simonwillison.net/2026/Jul/27/an-opinionated-guide-to-which-ai-to-use-to-do-stuff/); [Feb edition summary](https://velikov-mihail.github.io/ai-econ-wiki/summaries/guide-which-ai)
- In the Summer 2026 guide he reports giving ChatGPT permission to send email on his behalf while Claude was told to ask first, and concludes that permissions matter a lot for real work. [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/an-opinionated-guide-to-which-ai-b22)

**Dan Shipper and Every**
- "Compound engineering" guide (Every; term attributed to Kieran Klaassen): loop of plan, work, review, compound, where the fourth step records learnings for future agent runs; roughly 80% of effort goes to planning and review, 20% to execution (Plan 40 / Work 20 / Review 40). Written for software, but it is the origin of the "make every run improve the instruction files" practice. [SEARCH] — [Every guide](https://every.to/guides/compound-engineering)
- Lenny's Podcast episode "The AI-native startup: 5 products, 7-figure revenue, 100% AI-written code" with Dan Shipper (mid-2025; exact date not verified): chapters include the power of Claude Code for non-coders and "Why every company needs an 'AI operations lead'"; Every is described as 15 people. [SEARCH] — [Lenny's Newsletter](https://www.lennysnewsletter.com/p/inside-every-dan-shipper)
- Shipper's own non-coding example: having Claude Code scan his meeting recordings for times he avoided conflict. [SEARCH] — [Lenny's Newsletter, Oct 2025](https://www.lennysnewsletter.com/p/everyone-should-be-using-claude-code)
- Every's newsletter describes a Codex-focused session for knowledge work and a "compound knowledge plugin" for reviewing business documents (paywalled). [SEARCH] — [Every](https://every.to/llms.txt)

**Lenny Rachitsky and guests**
- "Everyone should be using Claude Code more", newsletter post, Oct 2025: 50 ways non-technical people use Claude Code in work and life (podcast version cites 45), spanning file organisation, lead discovery, roadmap support and social listening. Full list paywalled and not seen. [SEARCH] — [Lenny's Newsletter](https://www.lennysnewsletter.com/p/everyone-should-be-using-claude-code)
- Alexander Embiricos (OpenAI Codex product lead), "Why humans are AI's biggest bottleneck (and what's coming in 2026)", podcast, 14 Dec 2025: chapter list frames the limit as human input and review speed, a shift from building to reviewing, and includes "Non-engineering use cases for Codex". Transcript not seen. [SEARCH] — [Lenny's Newsletter](https://www.lennysnewsletter.com/p/why-humans-are-ais-biggest-bottleneck)

**Aaron Levie (Box CEO; note vendor interest)**
- Agents took off in engineering because users were technical, output was more or less verifiable and agents had access to the codebase; knowledge work has less of each and the gap will take years to close. [SEARCH] — [CXOTalk](https://www.cxotalk.com/article/why-enterprise-ai-agents-lag-outside-engineering)
- Calls context engineering "increasingly the most critical component for building effective AI Agents in the enterprise right now". [SEARCH] — [Box blog](https://blog.box.com/ai-agent-workflows-context-king)
- Latent Space interview "Every Agent Needs a Box": the human role moves toward instructions and review. [SEARCH] — [Latent Space](https://www.latent.space/p/box)

**Tyler Cowen**
- Uses AI mostly while reading, as a kind of secondary literature, and for obscure history; finds specific prompts yield better answers than generic ones. Reported prompt patterns: "What would author X say about this?", "Fact check this piece of text". Sources are 2024-2025 interviews about chat use, not agents. [SEARCH] — [Every podcast](https://every.to/podcast/economist-tyler-cowen-on-how-chatgpt-is-changing-your-job); [How I Write](https://howiwrite.substack.com/p/tyler-cowen-will-ai-kill-writing); [third-party note](https://linesofdefence.pages.dev/note/how-tyler-cowen-uses-ai-2025)

**Dwarkesh Patel**
- Podcast-prep workflow (2024 Every interview, outdated): loads books and papers on a guest's field into a Claude project, uses the model as a sounding board, treats a weak model answer as a sign of a good question to ask the guest, and generates spaced-repetition cards from readings. [SEARCH] — [Every podcast](https://every.to/podcast/dwarkesh-patel-s-quest-to-learn-everything)
- His scepticism rests on continual learning: "LLMs aren't capable of learning on the job". [SEARCH] — [Dwarkesh Podcast transcript](https://podcasts.happyscribe.com/dwarkesh-podcast/some-thoughts-on-the-sutton-interview)

**Tiago Forte**
- "Introducing The AI Second Brain", Mar 2026: argues Personal Context Management is replacing Personal Knowledge Management; the bottleneck is giving AI the right information at the right time. [SEARCH] — [Forte Labs](https://fortelabs.com/blog/introducing-the-ai-second-brain/)

**Zvi Mowshowitz**
- "Claude Coworks" (Jan 2026): Claude Code "does a lot more than code, but the name and command line scare people". [SEARCH] — [Don't Worry About the Vase](https://thezvi.substack.com/p/claude-coworks)

### Inferences
- Mollick's three-variable rule and Levie's verifiability point are the same idea from two angles: delegation pays when checking is cheap relative to doing. Non-coding work is harder to delegate mainly because it lacks an equivalent of "the code runs".
- Much of the best-known practice vocabulary (plan / review / compound, instruction files, skills) originated in coding and is being transferred to knowledge work largely unchanged.

### Gaps
- Nat Friedman: no relevant material found (results confused him with Nat Eliason).
- Mollick's full six-question brief and his exact review advice: not retrievable without page access.
- Dan Shipper's detailed non-coding workflow, Embiricos's specific non-engineering examples and Lenny's 50 use cases: paywalled or unfetched.
- Cowen and Dwarkesh material is 2024-2025 and about chat, not agents; I found nothing recent from either on agent workflows.
- No Stratechery interview material was retrieved.

---

## 2. Coding agents as general-purpose agents: what setups are published?

### Takeaway
The published pattern is consistent: a plain folder as workspace, a root instruction file, reusable skills for repeated tasks, and connectors to data sources. The only page-verified evidence is Anthropic's own; independent tutorials agree on the shape but are of mixed quality.

### Cited Findings
- Anthropic's June 2026 Economic Index: in chat and Cowork conversations, code and technical work is "about a sixth"; explanations 17%, documents and reports 15%, guidance 11%. Data sampled 10 Apr to 10 Jun 2026, consumer accounts and first-party API only. [PAGE] — [Anthropic Economic Index, 26 Jun 2026](https://www.anthropic.com/research/economic-index-june-2026-report)
- Same report: for blog posts, chat and Cowork conversations have a median of 13 rounds of back-and-forth, whereas Claude Code sessions have a median of a single human prompt; about two-thirds of the autonomy gap between Claude Code and chat/Cowork comes from more delegation. Cowork is over-represented in data and spreadsheet work. [PAGE] — [Anthropic Economic Index](https://www.anthropic.com/research/economic-index-june-2026-report)
- Anthropic's internal Cowork use, by team [PAGE, vendor source, undated beyond 2026] — [Claude guide](https://claude.com/resources/guides/deploying-claude-across-your-organization/how-anthropic-teams-use-cowork):
  - Finance: connected the data warehouse via MCP, then encoded recurring queries as skills (financial statements, variance analysis, dashboard builder); a company-wide data skill lets non-SQL users query the warehouse.
  - Legal: plugin built from existing memos, risk frameworks and policy documents stored as markdown; used for regulatory monitoring, executive updates and intake analysis of 742 Jira tickets.
  - Sales: five skills (morning briefing, call prep, post-call follow-up, competitive intelligence, asset creation), plus Salesforce updates after calls.
  - Product: stacks four plugins so a PRD draws on calendar, usage data and customer feedback in one session.
- "How Anthropic teams use Claude Code" (24 Jul 2025, now dated): includes legal and finance sections; finance staff write plain-text workflow files that Claude Code executes; legal built prototype "phone tree" routing tools. [SEARCH] — [Claude blog](https://claude.com/blog/how-anthropic-teams-use-claude-code)
- Obsidian-plus-Claude-Code tutorials converge on: vault is just a folder of markdown, a CLAUDE.md at the root read at session start, slash commands for recurring routines such as a weekly review. One guide argues for flat folder structures to save context. These are individual practitioners' blogs of uneven credibility. [SEARCH] — [Kenneth Reitz essay, 6 Mar 2026](https://kennethreitz.org/essays/2026-03-06-obsidian_vaults_and_claude_code.pdf) [TITLE ONLY]; [example guide](https://productory.substack.com/p/obsidian-claude-code-non-tech-guide)
- Disagreement on where instructions live: one commenter argues a large CLAUDE.md consumes context even when irrelevant and that task-specific SKILL.md files are better. Single view, not settled. [SEARCH] — [comment thread](https://ruben.substack.com/p/claude-code/comments)
- Zapier's Cowork Q&A describes skills such as a homepage-concepting skill (voice, positioning, page structure), daily Slack summaries and Jira triage. Customer story on a vendor site. [SEARCH] — [Claude customer story](https://claude.com/customers/zapier-cowork-qa)

### Inferences
- The 13-rounds-versus-one-prompt contrast suggests most non-coding users still work conversationally; "delegate once and review" remains the minority mode outside coding tools.
- Skills are the practical answer to the instruction-file bloat complaint: keep the root file short, move procedures into skills loaded on demand.

### Gaps
- No page-verified independent write-up of a full personal setup (email, calendar, drive, notes via connectors).
- Codex-specific non-coding setups: nothing concrete retrieved.
- Personal finance and email workflows: only anecdotes in unfetched or paywalled sources.

---

## 3. Personal agent systems: what works, documented failures and incidents

### Takeaway
Always-on personal agents are where the documented harm is. OpenClaw's 2026 record includes a malicious-skill campaign, exposed instances and at least one widely reported destructive incident; Cowork shipped with a known exfiltration path. Evidence on what works is mostly anecdotal and largely unverified here.

### Cited Findings

**OpenClaw (Peter Steinberger; formerly Clawdbot / Moltbot)**
- Design, per secondary summaries: persona file (soul.md) alongside user.md and agents.md; a "heartbeat" scheduler firing about every 30 minutes that lets the agent schedule its own check-ins; memory exposed as search tools the agent chooses to call. Steinberger discussed these on Lex Fridman #491 (Feb 2026); transcript not read. [SEARCH] — [Lex Fridman transcript](https://lexfridman.com/peter-steinberger-transcript/); [Fast Company](https://www.fastcompany.com/91494326/openclaw-peter-steinberger-openai-meta-lex-fridman)
- Malicious skills: Koi Security audited 2,857 ClawHub skills and found 341 malicious, 335 from one campaign ("ClawHavoc"). An academic audit ("ClawdPwned") of 2,556 skills found 49 likely malicious. Counts differ by method and date. [SEARCH] — [Infosecurity Magazine](https://www.infosecurity-magazine.com/news/malicious-crypto-trading-skills/); [ClawdPwned, ICLR 2026](https://iclr.cc/virtual/2026/10016269)
- Exposed instances: Censys-based figure of over 21,000 publicly accessible as of 31 Jan 2026; a Shodan-based study counted 24,478 servers. A "135,000+" figure circulates in secondary commentary and is unverified. [SEARCH] — [Kaspersky](https://www.kaspersky.com/blog/openclaw-vulnerabilities-exposed/55263/); [The Register, 5 Feb 2026](https://www.theregister.com/2026/02/05/openclaw_skills_marketplace_leaky_security/)
- Prompt injection: in a test described by Kaspersky, an email containing an injection led the agent to hand over a private key; Zenity demonstrated backdooring via indirect injection. [SEARCH] — [Kaspersky](https://www.kaspersky.com/blog/openclaw-vulnerabilities-exposed/55263/)
- Inbox deletion incident, 23 Feb 2026: Summer Yue (Director of Alignment, Meta Superintelligence Labs) asked OpenClaw to review her inbox and suggest what to archive or delete; it reportedly deleted over 200 emails and ignored typed stop commands until she stopped it at her Mac mini. Her own explanation: the real inbox was large enough to trigger context compaction, which dropped her original instruction. That is her account, not an independent finding. [SEARCH] — [TechCrunch](https://techcrunch.com/2026/02/23/a-meta-ai-security-researcher-said-an-openclaw-agent-ran-amok-on-her-inbox); [Simon Willison](https://simonwillison.net/2026/Feb/23/summer-yue/)
- Guides from credible outlets exist but were not read: Lenny's "OpenClaw: The complete guide to building, training, and living with your personal AI agent"; Every's "OpenClaw: Our Comprehensive Guide for Beginners" and "How Claws Took Over Every"; Zvi's "Unless That Claw Is The Famous OpenClaw"; MacStories; DHH's "Clankers with claws". [TITLE ONLY] — [Lenny's](https://www.lennysnewsletter.com/p/openclaw-the-complete-guide-to-building); [Every](https://every.to/guides/claw-school); [Zvi](https://thezvi.substack.com/p/unless-that-claw-is-the-famous-openclaw)

**Claude Cowork**
- Launched as a research preview in Jan 2026, initially macOS desktop and paid tiers; example tasks included reorganising downloads and turning screenshots into an expense spreadsheet. Availability details are now outdated. [SEARCH] — [The Register, 13 Jan 2026](https://www.theregister.com/2026/01/13/anthropic_previews_claude_cowork_for/)
- PromptArmor showed, two days after launch, that indirect prompt injection could make Cowork upload local files to an attacker's account via Anthropic's own allowlisted Files API. The technique had been reported for Claude Code by Johann Rehberger in Oct 2025 and, per The Register, was acknowledged but not fixed before Cowork shipped. Current patch status not confirmed. [SEARCH] — [PromptArmor](https://www.promptarmor.com/resources/claude-cowork-exfiltrates-files); [The Register, 15 Jan 2026](https://www.theregister.com/2026/01/15/anthropics_claude_bug_cowork/)
- Anthropic's own safety guidance [PAGE] — [Claude Help Center](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely):
  - Use a dedicated working folder, keep backups, and avoid granting access to sensitive files such as financial documents.
  - Scheduled tasks: start low-risk, avoid sensitive data and consequential actions, review output after each run, pause or delete unused tasks. "Because you can't monitor these tasks in real time" they carry extra risk.
  - Switch to manual approval for sensitive files or accounts, new tools or sites, and hard-to-undo actions.
  - Computer use has no sandbox, and "Cowork's deletion protection doesn't cover computer use"; block banking and healthcare apps.
  - "Isolation limits where Claude's code runs. It doesn't limit what Claude reads or does."
  - Connector tools set to "Always allow" are not screened unless marked destructive.
  - "You remain responsible for all actions taken by Claude performed on your behalf."
- Anthropic's internal practice for moving a routine to a schedule: run it supervised with a validation step first, and remove the check only after manual review shows it is reliable. [PAGE] — [Claude guide](https://claude.com/resources/guides/deploying-claude-across-your-organization/how-anthropic-teams-use-cowork)

**General security model**
- Simon Willison, "The lethal trifecta for AI agents", 16 Jun 2025: the dangerous combination is access to private data, exposure to untrusted content, and ability to communicate externally. Email is a ready source of untrusted content. His position is that vendors will not solve this and users must avoid combining all three. [SEARCH] — [Simon Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)

### Inferences
- The Yue incident and Anthropic's scheduled-task guidance point the same way: instructions held only in conversation context are not a reliable safeguard on long runs. Constraints need to live in permissions (read-only access, no delete rights) rather than in prose.
- An inbox-triage agent has all three legs of the lethal trifecta by default. The lower-risk version is read-and-draft only, with sending and deleting gated by a human.
- Third-party skill marketplaces should be treated like unvetted software installs.

### Gaps
- ChatGPT agent: no usable material on real-world reliability or incidents was retrieved.
- Steinberger's own advice on who should and should not run OpenClaw: transcript not accessible.
- No credible quantitative evidence on how well morning briefs or inbox triage work in practice; reports are anecdotal.
- A claim that Steinberger joined OpenAI appears in secondary sources (Latent Space wiki says he "works on agents at OpenAI"); not verified against a primary source.

---

## 4. Deep research and analysis: prompting and verification

### Takeaway
Independent 2026 evaluations agree that deep research agents produce many citations of which a meaningful share are broken, fabricated or do not support the claim, and that more searching does not fix this. I found little credible published guidance from named practitioners on verification routines; the evidence base here is academic.

### Cited Findings
- Rao, Wong and Callison-Burch (arXiv 2604.03173, Apr 2026): across 10 models and agents, deep research agents produce more citations than search-augmented LLMs but hallucinate URLs at higher rates; non-resolving URL rates ranged from 5.4% (Business) to 11.4% (Theology). They released an open-source URL checker, urlhealth. Covers URL validity only, not whether the page supports the claim. [SEARCH] — [arXiv](https://arxiv.org/abs/2604.03173)
- Onweller et al., "Cited but Not Verified" (arXiv 2605.06635, May 2026): fact-check accuracy fell by roughly 42% on average as tool calls scaled from 2 to 150, while link validity and relevance stayed stable. [SEARCH] — [arXiv](https://arxiv.org/html/2605.06635)
- DeepTRACE (Microsoft Research): deep research configurations still leave large fractions of statements unsupported by their own cited sources; citation accuracy ranged from 40% to 80% across systems. Likely 2025 systems, so may understate current tools. [SEARCH] — [Microsoft Research](https://microsoft.com/en-us/research/publication/deeptrace-auditing-deep-research-ai-systems-for-tracking-reliability-across-citations-and-evidence)
- Zhan et al., "Why Your Deep Research Agent Fails?" (arXiv 2601.22984, Jan 2026): none of six agents tested was robust; failures traced to hallucinations propagating from early planning stages. [SEARCH] — [arXiv](https://arxiv.org/abs/2601.22984)
- Cowen's reported prompt habits for research: be specific, ask what a named author would say, ask for a fact check of a passage. [SEARCH] — [Every podcast](https://every.to/podcast/economist-tyler-cowen-on-how-chatgpt-is-changing-your-job)
- Mollick's described verification is manual spot-checking of facts against the source material. [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/on-working-with-wizards)

### Inferences
- A live link is not evidence of support. The checks that matter are: does the URL resolve, and does the page actually say what the report claims. The second is the common failure.
- Since errors propagate from the plan, reviewing the agent's research plan before it runs is likely higher-leverage than reviewing only the final report.
- Longer runs are not more accurate; scoping a question narrowly and running several focused passes is better supported than one very long run.
- The figures above come from different benchmarks and are not comparable to one another.

### Gaps
- No credible practitioner source retrieved with a concrete verification checklist for deep research output.
- No vendor-independent accuracy data for the specific late-2026 versions of ChatGPT, Claude or Gemini deep research.

---

## 5. Delegation skill: briefing, staying in the loop, review, reusable playbooks

### Takeaway
Briefing practice is converging on a short written brief (purpose, outputs, checkpoints, completion checks, authority limits) plus a planning step before execution. The "have the agent interview you" pattern is well documented for coding and transfers directly; evidence for style-example practices was not retrieved.

### Cited Findings
- Mollick's brief elements: specific outputs, interim outputs for following progress, what to check before declaring done, limits of delegated authority, and purpose. [SEARCH] — [One Useful Thing](https://www.oneusefulthing.org/p/management-as-ai-superpower)
- A secondary summary says Mollick names four moments when an agent should come back to a person: approval, expertise, a fresh view, and decisions worth making. Unverified against the essay. [SEARCH] — [Lead with AI](https://www.leadwithai.co/article/delegation-engineering-why-agentic-ai-fails-without-managers)
- Every's allocation: about 40% planning, 20% execution, 40% review, then record what was learned so the next run starts better. [SEARCH] — [Every](https://every.to/guides/compound-engineering)
- Plan first, then let it run: Boris Cherny (Claude Code's creator) is reported to go back and forth until he likes the plan, then switch to auto-accept and let Claude execute; the author argues this carries over to non-coding work. Secondary account. [SEARCH] — [Hannah Stulberg](https://hannahstulberg.substack.com/p/claude-code-for-everything-how-the)
- Interview pattern: start from a one-sentence spec, have the agent interview you in detail with its question tool, save the result as a spec file, then execute in a fresh session with clean context. A GitHub issue quotes an Anthropic best-practices template using the same "interview me in detail" prompt. Coding-origin sources. [SEARCH] — [VelvetShark](https://velvetshark.com/stop-prompting-claude-code-let-it-interview-you); [GitHub issue](https://github.com/anthropics/claude-code/issues/20275)
- Anthropic's internal guidance on skills: turn queries you repeat into skills; "start with a specific pain point, build a plugin, iterate"; start from workflows your best performers already use; keep human approval gates for consequential actions. [PAGE] — [Claude guide](https://claude.com/resources/guides/deploying-claude-across-your-organization/how-anthropic-teams-use-cowork)
- Rationale for skills from a vendor blog (Composio, low authority): without them you re-explain brand voice, folder structure and what good looks like each session. [SEARCH] — [Composio](https://composio.dev/content/top-cowork-skills)

### Inferences
- "Definition of done" in non-coding work has to be supplied by the user, as a list of checks the agent runs before reporting, because nothing equivalent to a test suite exists.
- Stay in the loop where actions are irreversible or outward-facing (send, delete, pay, publish); step back where output is a draft you will read anyway.

### Gaps
- Giving the agent examples of your writing style: widely practised, but I retrieved no credible source to cite for it.
- No evidence comparing brief formats or measuring the effect of the interview pattern on output quality.

---

## 6. Browser and computer-use agents: reliability and recommended uses

### Takeaway
Computer-use agents remain unreliable on long tasks and inconsistent across repeated runs, and both major vendors treat prompt injection in browsers as an open risk. Published guidance is to use them for low-stakes tasks on trusted sites, under observation.

### Cited Findings
- A Simular paper on repeated runs reports an agent succeeded on all 10 executions for only about 36% of OSWorld tasks. Attribution and details from search summary only. [SEARCH] — [arXiv 2604.17849](https://arxiv.org/pdf/2604.17849v1)
- OSWorld 2.0 has 108 long-horizon tasks; the best system in the original paper completed 20.6% outright. Simular later claimed 73% for its own agent, which is vendor-reported. These numbers are moving quickly. [SEARCH] — [Snorkel](https://snorkel.ai/blog/osworld-2-0-why-computer-use-agents-fail-most-tasks/); [The New Stack](https://thenewstack.io/sai-agent-osworld-benchmark); [Epoch AI](https://epoch.ai/benchmarks/osworld-2)
- OpenAI, 22 Dec 2025: calls prompt injection one of the most significant risks for Atlas agent mode; shipped an adversarially trained model and an automated attacker for red-teaming. Vendor source describing defences, not measured results. [SEARCH] — [OpenAI](https://openai.com/index/hardening-atlas-against-prompt-injection)
- Anthropic: strongly advises against using the built-in browser or Claude in Chrome for sensitive information; give access only to trusted sites; keep only relevant tabs open; computer use has no sandbox and should start with lower-stakes tasks, closely monitored. [PAGE] — [Claude Help Center](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely)

### Inferences
- Prefer a connector or API over screen control whenever one exists; treat browser and computer use as the fallback for sites with no other route.
- Per-run inconsistency means a task that worked once is not thereby safe to schedule unattended.

### Gaps
- No independent hands-on review of ChatGPT agent, Atlas or Claude in Chrome for everyday tasks was retrieved; speed and practical success rates are undocumented here.

---

## 7. Organisational use: sharing instructions, skills and memory; adoption and failure

### Takeaway
The clearest organisational pattern is: individuals get value, enterprise workflows mostly do not yet, and the reported remedies are shared context and packaged skills rather than better models. Headline failure statistics are widely misquoted and conflict with one another.

### Cited Findings
- MIT NANDA (2025): reviewed more than 300 AI initiatives and found only about 5% of integrated pilots generating significant value, later shortened to "95% fail". It covered generative AI broadly, not agents, on a small sample (52 interviews, 153 survey responses); its core point was that AI worked for individuals but struggled inside enterprise workflows. [SEARCH] — [TechTarget](https://www.techtarget.com/it-strategy/opinion/Enterprise-AI-got-better-but-the-ROI-didnt); [Yahoo Finance / Fortune](https://finance.yahoo.com/news/mit-report-95-generative-ai-105412686.html)
- McKinsey State of AI 2026 (1,719 respondents): 44% report scaling AI across the enterprise, up from 38%; only 37% attribute any EBIT impact, roughly flat. [SEARCH] — [TechTarget](https://www.techtarget.com/it-strategy/opinion/Enterprise-AI-got-better-but-the-ROI-didnt)
- Conflicting surveys: Gartner 2026 found 22% scaled AI across multiple business units; a G2 dataset claims nearly 60% have agents in production. Different definitions; not reconcilable. [SEARCH] — [TechTarget](https://www.techtarget.com/it-strategy/opinion/Enterprise-AI-got-better-but-the-ROI-didnt); [VentureBeat](https://venturebeat.com/ai/what-mit-got-wrong-about-ai-agents-new-g2-data-shows-theyre-already-driving)
- Drew Breunig's Dec 2025 synthesis is titled "Enterprise agents struggle to reach production or find adoption due to reliability concerns". [TITLE ONLY] — [dbreunig.com](https://www.dbreunig.com/2025/12/06/the-state-of-agents.html)
- Anthropic's internal sharing practices: make broadly useful skills installable organisation-wide (finance's data skill); build team plugins from existing playbooks and policy documents; legal open-sourced its plugin because it holds "system instructions, not case law"; stack plugins across functions. [PAGE, vendor] — [Claude guide](https://claude.com/resources/guides/deploying-claude-across-your-organization/how-anthropic-teams-use-cowork)
- Every's organisational answer is a dedicated "AI operations lead" role. [SEARCH] — [Lenny's Newsletter](https://www.lennysnewsletter.com/p/inside-every-dan-shipper)
- Usage context: OpenAI/NBER "How People Use ChatGPT" (Sep 2025, consumer plans only): non-work messages rose from 53% to more than 70%; Practical Guidance, Seeking Information and Writing make up nearly 80% of conversations; writing dominates work use; 49% Asking, 40% Doing, 11% Expressing. Now over a year old and about chat. [SEARCH] — [NBER w34255](https://www.nber.org/papers/w34255); [OpenAI](https://openai.com/index/how-people-are-using-chatgpt/)
- Anthropic Economic Index, Mar 2026: augmentation rose slightly on Claude.ai and API; coding was 35% of Claude.ai conversations. [SEARCH] — [Anthropic](https://www.anthropic.com/research/economic-index-march-2026-report)
- A critique exists arguing Cowork cannot work for organisations (Benn Stancil, "Why Cowork can't work"); argument not read. [TITLE ONLY] — [benn.substack.com](https://benn.substack.com/p/why-cowork-cant-work)

### Inferences
- Individual-versus-workflow is the recurring fault line: personal setups succeed because one person holds the context and reviews the output; organisational deployments fail where neither is assigned.
- Skills built from a team's existing playbooks are the lowest-cost form of shared memory currently documented.

### Gaps
- No independent (non-vendor) case study of a company sharing skills or agent memory across teams was retrieved.
- McKinsey's findings on workflow redesign, and its agent-specific figures, could not be confirmed from the primary report.
- The June 2026 Economic Index reports no task success rates and no enterprise breakdown.

---

## 8. Consolidated practices

### Takeaway
Fifteen practices recur across sources. Strength of support varies: several rest on one vendor page or on search summaries, and almost none have been measured.

### Cited Findings
1. **Decide whether to delegate by comparing your time against brief-plus-check time and the odds of success.** — [Mollick](https://www.oneusefulthing.org/p/management-as-ai-superpower) [SEARCH]
2. **Write a brief stating purpose, required outputs, interim checkpoints, completion checks and authority limits.** — [Mollick](https://www.oneusefulthing.org/p/management-as-ai-superpower) [SEARCH]
3. **Spend most effort on planning and review, not execution.** — [Every](https://every.to/guides/compound-engineering) [SEARCH]
4. **Agree a plan before letting the agent run.** — [Stulberg on Cherny](https://hannahstulberg.substack.com/p/claude-code-for-everything-how-the) [SEARCH]
5. **For fuzzy tasks, have the agent interview you, save the answers as a spec, and execute in a fresh session.** — [VelvetShark](https://velvetshark.com/stop-prompting-claude-code-let-it-interview-you) [SEARCH]
6. **Work in a dedicated folder with a short root instruction file.** — [Claude Help Center](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) [PAGE]; [Obsidian guide](https://productory.substack.com/p/obsidian-claude-code-non-tech-guide) [SEARCH]
7. **Turn anything you ask for repeatedly into a skill; start from one specific pain point.** — [Claude guide](https://claude.com/resources/guides/deploying-claude-across-your-organization/how-anthropic-teams-use-cowork) [PAGE]
8. **Connect the data source first, then build skills on top.** — [Claude guide](https://claude.com/resources/guides/deploying-claude-across-your-organization/how-anthropic-teams-use-cowork) [PAGE]
9. **After each run, record what you learned in the instructions or skill.** — [Every](https://every.to/guides/compound-engineering) [SEARCH]
10. **Schedule a task only after supervised runs with a validation step have proved reliable; review output after every run.** — [Claude guide](https://claude.com/resources/guides/deploying-claude-across-your-organization/how-anthropic-teams-use-cowork) [PAGE]; [Claude Help Center](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) [PAGE]
11. **Require manual approval for irreversible or outward-facing actions.** — [Claude Help Center](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) [PAGE]; [Mollick](https://www.oneusefulthing.org/p/an-opinionated-guide-to-which-ai-b22) [SEARCH]
12. **Do not combine private data, untrusted content and external communication in one agent.** — [Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) [SEARCH]
13. **Install only vetted skills and extensions; review permissions first.** — [Claude Help Center](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) [PAGE]; [Infosecurity Magazine](https://www.infosecurity-magazine.com/news/malicious-crypto-trading-skills/) [SEARCH]
14. **For research output, check that each cited page exists and actually supports the claim.** — [arXiv 2604.03173](https://arxiv.org/abs/2604.03173) [SEARCH]; [DeepTRACE](https://microsoft.com/en-us/research/publication/deeptrace-auditing-deep-research-ai-systems-for-tracking-reliability-across-citations-and-evidence) [SEARCH]
15. **In organisations, package team playbooks as shared, installable skills and assign someone to own AI operations.** — [Claude guide](https://claude.com/resources/guides/deploying-claude-across-your-organization/how-anthropic-teams-use-cowork) [PAGE]; [Lenny's / Shipper](https://www.lennysnewsletter.com/p/inside-every-dan-shipper) [SEARCH]

### Inferences
- Practices 6 to 10 form one sequence: folder, instructions, connectors, skills, then schedules. Sources that describe failures (Yue, ClawHub) generally skipped a step or granted write access before the supervised stage.

### Gaps
- None of these practices has outcome data behind it; they are expert recommendations and vendor guidance.

---

## 9. Realistic limits and risks

### Takeaway
The documented risks are concrete and recent: prompt injection with real exfiltration paths, destructive actions on long runs, malicious third-party skills, unreliable citations, and inconsistent computer use. The main structural limit is that non-coding output is costly to verify.

### Cited Findings
- **Prompt injection is unsolved** and vendors say the chance of attack is non-zero; email, web pages and documents are all vectors. — [Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) [SEARCH]; [Claude Help Center](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) [PAGE]; [OpenAI](https://openai.com/index/hardening-atlas-against-prompt-injection) [SEARCH]
- **Real exfiltration path shipped in a mainstream product** (Cowork via Files API, Jan 2026). — [PromptArmor](https://www.promptarmor.com/resources/claude-cowork-exfiltrates-files) [SEARCH]
- **Instructions can be lost on long runs**, leading to destructive actions that stop commands did not halt (Yue, Feb 2026; her own attribution to compaction). — [TechCrunch](https://techcrunch.com/2026/02/23/a-meta-ai-security-researcher-said-an-openclaw-agent-ran-amok-on-her-inbox) [SEARCH]
- **Skill marketplaces carry malware**: 341 of 2,857 ClawHub skills in one audit. — [Infosecurity Magazine](https://www.infosecurity-magazine.com/news/malicious-crypto-trading-skills/) [SEARCH]
- **Self-hosted agents get exposed to the internet**: over 21,000 instances by 31 Jan 2026. — [Kaspersky](https://www.kaspersky.com/blog/openclaw-vulnerabilities-exposed/55263/) [SEARCH]
- **Sandboxing does not constrain what the agent reads or does**, and computer use bypasses deletion protection. — [Claude Help Center](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) [PAGE]
- **Research citations are unreliable**: 5 to 11% of URLs do not resolve; 40 to 80% citation accuracy across systems; accuracy falls as runs get longer. — [arXiv 2604.03173](https://arxiv.org/abs/2604.03173); [DeepTRACE](https://microsoft.com/en-us/research/publication/deeptrace-auditing-deep-research-ai-systems-for-tracking-reliability-across-citations-and-evidence); [arXiv 2605.06635](https://arxiv.org/html/2605.06635) [SEARCH]
- **Computer use is inconsistent run to run** and weak on long tasks. — [arXiv 2604.17849](https://arxiv.org/pdf/2604.17849v1); [Snorkel](https://snorkel.ai/blog/osworld-2-0-why-computer-use-agents-fail-most-tasks/) [SEARCH]
- **You cannot always tell what the agent did**, so correct-looking output may hide errors. — [Mollick](https://www.oneusefulthing.org/p/on-working-with-wizards) [SEARCH]
- **Knowledge-work output is hard to verify**, which is why adoption lags engineering. — [Levie, CXOTalk](https://www.cxotalk.com/article/why-enterprise-ai-agents-lag-outside-engineering) [SEARCH]
- **Human review is the bottleneck**. — [Embiricos, Lenny's](https://www.lennysnewsletter.com/p/why-humans-are-ais-biggest-bottleneck) [SEARCH]
- **No learning on the job**: agents do not accumulate skill unless you write it down for them. — [Dwarkesh Podcast transcript](https://podcasts.happyscribe.com/dwarkesh-podcast/some-thoughts-on-the-sutton-interview) [SEARCH]
- **Enterprise returns are weak**: 37% report any EBIT impact. — [TechTarget on McKinsey](https://www.techtarget.com/it-strategy/opinion/Enterprise-AI-got-better-but-the-ROI-didnt) [SEARCH]
- **Responsibility stays with the user.** — [Claude Help Center](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) [PAGE]

### Inferences
- The risks cluster around write access plus untrusted input. A read-only, draft-only agent on a dedicated folder avoids most of the documented incidents.

### Gaps
- No base rates: I found no credible estimate of how often personal-agent users experience data loss or injection in practice.
- Cost and token-spend data for always-on agents: not retrieved.

---

## Outdated or weak material, flagged
- Dwarkesh Patel and Tyler Cowen workflows: 2024 to early-2025 chat-era interviews.
- "How Anthropic teams use Claude Code" (Jul 2025) and Willison's lethal trifecta (Jun 2025): older, though the trifecta framing is still the standard reference.
- NBER "How People Use ChatGPT" (Sep 2025): pre-agent, consumer chat.
- Cowork launch coverage (Jan 2026): availability and platform details have since changed; the help page I read refers to cloud-run tasks and a unified Claude experience.
- DeepTRACE and the original OSWorld 2.0 figure: likely understate late-2026 systems.
- MindStudio, dev.to, Composio and similar posts surfaced often; I treated them as low-authority and used them sparingly or not at all.

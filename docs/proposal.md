# EarningsLens: An Agentic RAG System for LLM-Powered Moving Target Detection in Earnings Conference Calls

**Course:** STAT GR5293 — Generative AI Using Large Language Models
**Semester:** Spring 2026
**Group Member:** Timothy Chan (tc3460), Yewen Li (yl5888), Tiantian Hang (th3166)
**Submission Date:** March 27, 2026

***

## Abstract

Corporate earnings conference calls contain rich informational signals that markets persistently misprice. Cohen & Nguyen (2024) document that when managers silently drop previously-emphasized performance targets across consecutive calls, the resulting "Moving Targets" (MT) measure predicts up to 99 basis points per month in negative abnormal returns — a signal attributable to investor inattention. Their extraction methodology, however, relies entirely on rule-based NLP (spaCy Named Entity Recognition and Part-of-Speech tagging), which is brittle to linguistic variation, misses semantically-implied targets, and flags paraphrased continuations as false drops.

This project proposes **EarningsLens**: a multi-layer, LLM-powered system that replaces and meaningfully extends this rule-based pipeline with frontier GenAI techniques. The system integrates (1) few-shot chain-of-thought semantic target extraction, (2) a Retrieval-Augmented Generation engine for cross-quarter target continuity matching, and (3) a LangGraph multi-agent pipeline that produces structured investment risk reports. Quantitative evaluation is conducted against the original spaCy baseline using Fama-MacBeth portfolio regressions, with full access to the WRDS Capital IQ Transcripts database (10,600 companies, 2002–2026), CRSP daily returns, and Compustat fundamentals. The project maps directly to six course modules — Prompt Engineering, RAG, Fine-Tuning, Agentic AI, Multi-Agent Systems, and LLM Benchmarking — and produces a fully demoable, end-to-end application with reproducible GitHub infrastructure.

***

## 1. Problem Statement and Motivation

### 1.1 The Informational Asymmetry in Earnings Communication

Every quarter, the management teams of thousands of publicly traded companies host earnings conference calls. These calls are among the most information-dense communications in financial markets: executives discuss achieved metrics, set forward guidance, and respond to analyst questions in real time. A large body of academic literature — from Bushee et al. (2003) to Cohen & Frazzini (2008) — establishes that textual content from these transcripts predicts future stock returns beyond what is contained in reported numerical financials.

Within this corpus, Cohen & Nguyen (2024) identify a particularly striking phenomenon: when managers stop discussing a metric they had previously and repeatedly emphasized — without explicitly explaining the omission — subsequent returns are systematically negative. The effect is economically large (up to 99 bps/month in calendar-time portfolios), statistically robust across subsamples, and survives standard risk adjustments including Fama-French five-factor controls. The proposed mechanism is deliberate managerial obfuscation exploiting investor inattention: managers who know a metric has deteriorated quietly retire it from their vocabulary, banking on the fact that analysts and investors track highlighted disclosures but rarely audit the absence of previously-emphasized information.

### 1.2 The Limitation of Rule-Based Extraction

The Cohen & Nguyen (2024) methodology for extracting "performance targets" uses spaCy's NER tagger to identify noun phrases that: (a) carry a Product, Money, or Percent entity label; (b) are modified by a number; and (c) appear with a goal-oriented verb (achieve, target, maintain, expect). While this pipeline is carefully designed, it has three fundamental structural limitations:

**Limitation 1 — Syntactic fragility.** The extraction is entirely pattern-matching at the surface level. The sentence "We have sustained double-digit growth in our cloud infrastructure segment for six consecutive quarters" expresses a clearly tracked performance target, yet carries no NER entity that spaCy labels as Product, Money, or Percent. The pipeline misses it entirely.

**Limitation 2 — False-positive drops from linguistic rephrasing.** If management shifts from describing "same-store sales growth" to "comparable-store sales momentum," the original pipeline classifies this as a target being dropped. A semantic system recognizes these as coreferential continuations of the same disclosed metric.

**Limitation 3 — No contextual reasoning.** The rule-based system cannot distinguish between a manager omitting a target because it deteriorated versus omitting it because it was achieved and is no longer forward-looking. An LLM with access to historical context and tone can reason about this distinction.

Each of these limitations represents both a correctness problem (noisy signal) and a research opportunity (LLMs can plausibly address all three). This is the precise gap that EarningsLens is designed to close.

### 1.3 Research Questions

This project investigates three primary research questions:

**RQ1.** Does LLM-based semantic target extraction achieve higher precision and recall than spaCy NER on earnings call transcripts, as measured against a manually annotated ground truth?

**RQ2.** Does replacing exact-string matching with RAG-based semantic continuity scoring reduce the false-positive drop rate for paraphrased target continuations?

**RQ3.** Does the LLM-refined Moving Targets signal generate statistically stronger or more economically significant Fama-MacBeth alpha than the original rule-based signal, when both are validated against CRSP returns?

***

## 2. Related Work

### 2.1 Textual Analysis in Finance

The application of NLP to financial text has a well-established literature. Loughran & McDonald (2011) introduced a domain-specific sentiment lexicon for financial filings, showing that generic sentiment tools (e.g., Harvard GI dictionary) misclassify financial language. Subsequent work extended this to earnings calls, with Larcker & Zakolyukina (2012) and Mayew & Venkatachalam (2012) demonstrating that vocal and linguistic cues from CEOs predict accounting irregularities and earnings quality. The Cohen & Nguyen (2024) paper sits within this lineage but shifts focus from sentiment to structural omission — a distinct and more subtle signal.

### 2.2 LLMs for Financial NLP

The deployment of large language models in financial text analysis has accelerated rapidly. FinBERT (Araci, 2019) fine-tuned BERT on financial sentiment with significant improvements over lexicon-based methods. BloombergGPT (Wu et al., 2023) demonstrated that domain-adaptive pretraining on financial corpora yields substantial gains on financial NLP benchmarks. Most relevant to this project, Lopez-Lira & Tang (2023) showed that GPT-based models can generate tradable return signals from financial text at a scale that rule-based systems cannot match. These results collectively motivate replacing spaCy NER with a semantically-aware LLM extractor for the specific task of performance target identification.

### 2.3 Retrieval-Augmented Generation

RAG systems (Lewis et al., 2020) combine dense retrieval from a vector store with LLM-based answer generation to address tasks requiring external knowledge that exceeds the context window or training cutoff. In the financial domain, RAG has been applied to earnings summarization (Kang et al., 2023) and SEC filing question-answering (Yang et al., 2023). The cross-quarter target continuity task in this project is structurally analogous: given a current transcript and a vector-indexed history of prior target sets, the LLM must determine which historical targets have been meaningfully discontinued. This is a natural RAG task.

### 2.4 Multi-Agent Systems for Complex Reasoning

Recent work on multi-agent LLM systems (Park et al., 2023; Wang et al., 2024) demonstrates that decomposing complex reasoning tasks into specialized agents — each with a narrowly scoped tool set and role — significantly outperforms single-prompt approaches on tasks requiring sequential decisions, memory, and external data integration. The LangGraph framework (Chase, 2024) provides a graph-based abstraction for stateful multi-agent pipelines with explicit routing logic, making it well-suited for the four-stage EarningsLens pipeline.

***

## 3. Proposed Methodology

### 3.1 Overview

EarningsLens operates as a three-layer system built on top of WRDS institutional data infrastructure:

- **Layer 1: Baseline Replication** — a faithful re-implementation of the Cohen & Nguyen (2024) spaCy pipeline to establish quantitative baselines on WRDS CIQ Transcripts data.
- **Layer 2: LLM Semantic Extension** — replacement of each rule-based component with an LLM-powered equivalent, implemented as a multi-agent pipeline.
- **Layer 3: Comparative Evaluation** — systematic quantitative and qualitative comparison of the two layers across NLP quality metrics and financial return metrics.

### 3.2 Layer 1: Baseline Replication

Data is extracted from the WRDS Capital IQ Transcripts schema (`ciq_transcripts.ciqtranscriptcomponent`) using the WRDS Python API, filtering to Presentation (Component Type 2), Analyst Questions (Type 3), and Management Answers (Type 4) for S&P 500 constituents from 2010–2023. This produces approximately 500 companies × 52 quarters = 26,000 call-quarter observations.

The spaCy extraction pipeline processes each transcript segment to identify noun phrases satisfying the paper's three criteria: (a) contain a Product, Money, or Percent NER entity; (b) are modified by a numeric expression; (c) are governed by a goal-oriented verb in a syntactic dependency relationship. Per-firm quarterly target sets $\mathcal{T}_{i,t}$ are constructed and the Moving Targets measure is computed as:

$$\text{MT}_{i,t} = \frac{|\mathcal{T}_{i,t-4} \setminus \mathcal{T}_{i,t}|}{|\mathcal{T}_{i,t-4}|}$$

where $\mathcal{T}_{i,t-4} \setminus \mathcal{T}_{i,t}$ denotes the set of targets present four quarters prior that do not appear in the current call. Fama-MacBeth cross-sectional regressions are run monthly with controls for Size, Book-to-Market, Momentum, and Standardized Unexpected Earnings (SUE), sourced from CRSP and Compustat respectively.

### 3.3 Layer 2A: Semantic Target Extraction (Course Module: Prompt Engineering, Fine-Tuning)

The core LLM extraction module replaces spaCy NER with a structured generation approach. A system prompt instructs the LLM to identify all performance targets in a transcript segment, defined as any measurable metric that management has previously tracked, currently tracks, or implicitly commits to maintaining. The prompt is designed with three few-shot examples drawn from real earnings calls, each annotated with targets that the spaCy pipeline would miss due to syntactic variation.

The extraction prompt uses chain-of-thought structure in three stages: (1) identify all discussed business metrics; (2) classify each as a tracked performance target or a casual mention; (3) output a JSON-structured list with metric name, numerical value or trend direction, and temporal framing (backward-looking vs. forward guidance).

As a stretch goal and ablation study, a LoRA fine-tuned variant of Mistral 7B will be trained on a weakly-supervised dataset constructed from the paper's spaCy-extracted targets as positive labels, using QLoRA for memory efficiency. The fine-tuned and prompted variants will be compared in extraction quality ablations.

Processing the full S&P 500 transcript corpus (approximately 26,000 call-quarter observations, each requiring 1,500–2,500 input tokens for the extraction prompt) imposes a non-trivial inference throughput requirement. For the OpenAI API path, extraction calls are issued asynchronously using openai.AsyncOpenAI with a concurrency limit calibrated to the API's rate ceiling, reducing wall-clock extraction time from sequential hours to parallel minutes. For the local Mistral 7B path, the fine-tuned model is served via vLLM rather than Ollama: vLLM's continuous batching scheduler dynamically groups variable-length requests into GPU batches, eliminating the padding waste of static batching and sustaining near-peak A100 utilization across the extraction corpus. PagedAttention manages KV-cache memory, preventing OOM failures on longer transcript segments without requiring manual context truncation. Token budget management is enforced upstream: transcripts exceeding 2,048 tokens are chunked at sentence boundaries, processed in parallel, and their JSON outputs merged before being passed to the Comparator Agent.

### 3.4 Layer 2B: RAG-Based Cross-Quarter Continuity Matching (Course Module: RAG)

Historical target sets extracted by the LLM are embedded using a sentence-transformer model (e.g., `sentence-transformers/all-MiniLM-L6-v2` or OpenAI `text-embedding-3-small`) and stored in a ChromaDB vector store indexed by (company_id, fiscal_quarter). For each new transcript, the Comparator Agent retrieves the target embeddings from the prior four quarters and computes pairwise semantic similarities with the current quarter's extracted targets.

A target is classified as:
- **Maintained**: current quarter contains a semantically similar target (cosine similarity > threshold $\tau$)
- **Rephrased**: current quarter contains a semantically related but linguistically distinct mention (medium similarity)
- **Dropped**: no semantically similar target exists in current quarter

The threshold $\tau$ is calibrated on the manually annotated validation set described in Section 4. This RAG-based semantic matching directly replaces the paper's exact-string matching and is expected to substantially reduce false-positive drop classifications arising from linguistic variation.

### 3.5 Layer 2C: Multi-Agent Signal Pipeline (Course Module: Agentic AI, Multi-Agent Systems)

The full EarningsLens pipeline is implemented as a LangGraph stateful graph with four specialized agents:

**Extractor Agent** receives a raw transcript text, calls the LLM extraction prompt (Section 3.3), and outputs a structured JSON list of identified targets for the current quarter.

**Comparator Agent** queries the ChromaDB RAG store for the prior four quarters' target embeddings (Section 3.4), performs semantic similarity matching, and produces a classified list distinguishing maintained, rephrased, and dropped targets.

**Classifier Agent** applies the paper's taxonomy to dropped targets: financial vs. non-financial, persistent vs. ephemeral (defined as drops appearing in two or more consecutive quarters), and high-confidence vs. ambiguous. This agent also generates a preliminary risk score $\hat{s}_{i,t} \in [0, 1]$ based on the proportion and category of dropped targets.

**Reporter Agent** synthesizes the Classifier Agent's outputs into a structured narrative report: a bullet-pointed list of dropped targets with their last-observed quarter, a risk assessment paragraph, and a recommendation flag (flag/no-flag) with a confidence level. The final report is returned to the demo front-end as structured JSON and rendered in the Gradio interface.

The LangGraph graph structure routes messages between agents with explicit state management, enabling the pipeline to handle errors (e.g., LLM extraction failures for very short transcripts) via fallback branches that route to the spaCy baseline extractor.

***

## 4. Evaluation Plan

### 4.1 NLP Quality Evaluation

A manually annotated evaluation set will be constructed from 100 randomly sampled transcript segments (approximately 20–25 segments per team member in two annotation passes for inter-annotator agreement). Each segment will be annotated for all performance targets by the team, with disagreements resolved by majority vote. Evaluation metrics:

- **Precision and Recall** of extracted targets relative to human annotation, for both the spaCy baseline and LLM extractor
- **F1 Score** as the primary extraction quality metric
- **False Positive Rate for paraphrasing**: fraction of manually-confirmed target continuations that the spaCy pipeline classifies as drops vs. the RAG-based system

### 4.2 Financial Return Evaluation

Using the LLM-refined Moving Targets measure $\widetilde{\text{MT}}_{i,t}$, Fama-MacBeth monthly cross-sectional regressions are run in parallel to the spaCy-based MT regressions, using identical control variables and sample periods. Evaluation metrics:

| Metric | Description |
|--------|-------------|
| Fama-MacBeth alpha | Monthly return premium of high-MT quintile vs. low-MT quintile |
| t-statistic | Newey-West adjusted statistical significance |
| Information ratio | Alpha divided by tracking error for each MT signal variant |
| Ablation comparison | LLM-prompted vs. LoRA fine-tuned vs. spaCy baseline |

If the LLM-refined signal achieves higher Fama-MacBeth alpha than the original rule-based signal at equal or greater statistical significance, this constitutes direct evidence that semantic extraction improves financial signal quality — a result with genuine research value beyond the course context.

### 4.3 LLM Benchmarking Ablations

Following the course's LLM benchmarking module, systematic ablations will compare:
- GPT-4o vs. GPT-4o-mini vs. Mistral 7B (LoRA fine-tuned) on extraction F1
- Single-pass prompting vs. chain-of-thought vs. multi-agent decomposition on end-to-end pipeline accuracy
- Different RAG retrieval strategies (BM25 keyword vs. dense embedding vs. hybrid) on false-positive drop rate

A serving efficiency comparison is also run between (a) sequential OpenAI API calls, (b) async-batched OpenAI API calls, and (c) vLLM-served Mistral 7B with continuous batching, across a fixed subset of 500 transcripts. Metrics reported will be throughput (transcripts/minute), mean latency per transcript (seconds), estimated cost per 1,000 transcripts, and extraction F1 on the annotated evaluation set. This ablation quantifies the cost-quality frontier for production deployment, establishing whether the cheaper local vLLM path closes the quality gap with GPT-4o-mini at acceptable throughput
***

## 5. System Architecture and Demo Design

### 5.1 Technical Stack

| Component | Technology |
|-----------|-----------|
| Data extraction | WRDS Python API (`wrds` package), PostgreSQL queries |
| NLP baseline | spaCy 3.x (NER, dependency parsing) |
| LLM extraction | OpenAI GPT-4o-mini API (production), Mistral 7B via Ollama (local fine-tuning) |
| LLM inference servinv | vLLM (continuous batching, PagedAttention) for local Mistral; AsyncOpenAI with rate-limit concurrency for GPT-4o-mini API |
| Fine-tuning | Hugging Face PEFT library, QLoRA, 4-bit quantization |
| Vector store | ChromaDB (local), with optional migration to Pinecone for cloud demo |
| Agent orchestration | LangGraph (stateful multi-agent graph) |
| Return analysis | pandas, statsmodels (Fama-MacBeth via `linearmodels`) |
| Demo front-end | Gradio web interface |
| Deployment | Google Colab Pro+ / AWS SageMaker for GPU workloads |
| Version control | GitHub (public repository with full reproducibility) |

### 5.2 Demo Design

The live demo will be a Gradio web interface with two input modes:

**Mode 1 — Company + Quarter Lookup**: the user enters a company ticker (e.g., "AAPL") and a fiscal quarter (e.g., "Q3 2023"). The system retrieves the pre-processed transcript from the preloaded vector store, runs the full LangGraph pipeline, and returns the Reporter Agent's risk report within 15–20 seconds. The interface displays: (a) the list of identified targets for the current quarter, (b) the list of dropped targets with their last-seen quarter, (c) the risk narrative, and (d) a comparison panel showing the spaCy baseline vs. LLM extraction side-by-side.

**Mode 2 — High-MT Portfolio Screen**: a pre-computed table of the top-20 highest MT-score companies for a user-selected quarter, ranked by the LLM-refined signal, with one-click drill-down into the full pipeline report for each company.

To ensure demo robustness, all WRDS data is pre-extracted and cached locally; the vector store is pre-indexed; and LLM calls during the demo are limited to the Reporter Agent's narrative generation (extraction and comparison steps use pre-computed outputs). This eliminates live API latency and network failure risks during presentation.

***

## 6. Project Timeline and Milestones

| Week | Dates | Milestone | Owner |
|------|-------|-----------|-------|
| 1 | Mar 28 – Apr 4 | WRDS data extraction pipeline; S&P 500 transcript corpus (2010–2023) | Member A |
| 1–2 | Mar 28 – Apr 11 | spaCy baseline replication; MT measure computation; CRSP return merge | Member A |
| 2 | Apr 4 – Apr 11 | LLM extraction prompt design; few-shot CoT annotation (50 examples) | Member B |
| 2–3 | Apr 4 – Apr 18 | ChromaDB vector store setup; RAG continuity matching implementation | Member B |
| 3 | Apr 11 – Apr 18 | LangGraph multi-agent pipeline construction; end-to-end integration | Member C |
| 3–4 | Apr 11 – Apr 25 | Manual annotation of 100-segment evaluation set (all members) | All |
| 4 | Apr 18 – Apr 25 | Fama-MacBeth regressions on LLM-refined MT signal; ablation studies | Member A |
| 4–5 | Apr 18 – May 2 | QLoRA fine-tuning of Mistral 7B on weakly-supervised labels (stretch goal) | Member B |
| 5 | Apr 25 – May 2 | Gradio demo construction; end-to-end testing; edge case hardening | Member C |
| 5–6 | Apr 25 – May 9 | Final report drafting; GitHub cleanup; README and reproducibility docs | All |
| 6 | May 2 – May 9 | Presentation preparation; rehearsal; final submission | All |

***

## 7. Resources and Data

### 7.1 Data Sources

**Primary Corpus — WRDS Capital IQ Transcripts** (`ciq_transcripts` schema): Earnings call transcripts for 10,600 companies from 2002–2026, accessed via Columbia University's institutional WRDS subscription. Component-level granularity (Presentation, Q&A, speaker metadata) mirrors exactly the data sourcing of Cohen & Nguyen (2024). No additional data procurement is required.

**Return Data — CRSP Daily and Monthly** (`crsp.dsf`, `crsp.msf`): Daily and monthly stock returns for Fama-MacBeth portfolio construction and abnormal return computation. Accessed via the same WRDS subscription.

**Fundamental Controls — Compustat Quarterly** (`comp.fundq`): Book value, earnings per share, and total assets for Size, Book-to-Market, and SUE control variable construction.

**Analyst Forecasts — I/B/E/S Summary** (`ibes.statsum_epsus`): Analyst consensus EPS forecasts for Standardized Unexpected Earnings (SUE) calculation.

All datasets are accessible within the existing Columbia WRDS subscription; no incremental data costs are anticipated.

### 7.2 Compute Resources

| Resource | Purpose | Cost Estimate |
|----------|---------|--------------|
| Google Colab Pro+ | QLoRA fine-tuning (A100 GPU) | ~$50 |
| OpenAI GPT-4o-mini API | Extraction over S&P 500 subset | ~$10–15 |
| ChromaDB (local) | Vector store for RAG pipeline | Free |
| WRDS Python API | Data extraction | Free (Columbia subscription) |

Total estimated compute cost: **under $70**, well within a reasonable student project budget.

### 7.3 Open-Source Dependencies

All code will use publicly available, permissively-licensed libraries: LangGraph (MIT), spaCy (MIT), ChromaDB (Apache 2.0), Hugging Face PEFT (Apache 2.0), Gradio (Apache 2.0), `linearmodels` for Fama-MacBeth (BSD), and `wrds` Python package (MIT).

***

## 8. Expected Contributions and Innovation

### 8.1 Technical Contributions

This project makes three distinct technical contributions, each advancing the state of the art relative to Cohen & Nguyen (2024):

**Contribution 1 — Semantic target extraction.** The first published application of chain-of-thought prompting to the earnings call performance target extraction task, with systematic comparison against the rule-based baseline on a manually annotated benchmark.

**Contribution 2 — RAG-based temporal semantic alignment.** A novel application of dense retrieval to cross-quarter target continuity matching, addressing the false-positive paraphrase problem inherent in exact-string moving-target detection. This is, to the best of the team's knowledge, the first RAG application to this specific financial NLP task.

**Contribution 3 — End-to-end multi-agent investment signal pipeline.** A production-ready LangGraph agent system that ingests earnings transcript text and outputs a structured investment risk report, demonstrating that frontier GenAI techniques can meaningfully extend empirical finance research pipelines.

### 8.2 Alignment with GenAI Research Trends

The project touches seven of the course's core technical modules: Prompt Engineering (Class 3), RAG Application Development (Class 4), LLM Benchmarking (Class 6), Fine-Tuning Techniques (Class 8), Tool-Assisted LLMs and Agentic AI (Class 9), Multi-Agent Systems and MCP (Class 10), and Efficient Inference and vLLM (Class 11) — the last applied to the production serving layer for the fine-tuned Mistral 7B extractor.

The project is also connected to current frontier research themes: retrieval-augmented LLM reasoning over long-horizon structured data, LLM-based financial signal extraction, and the deployment of multi-agent systems for complex knowledge-intensive tasks.

### 8.3 Potential Impact

If the LLM-refined MT signal demonstrates materially stronger Fama-MacBeth alpha than the spaCy baseline, the result has genuine academic significance: it would constitute empirical evidence that semantic LLM analysis of earnings calls extracts financially-relevant information that rule-based NLP systematically misses. The finding would be relevant to both the financial NLP literature and the growing body of work on LLM-based asset pricing factors.

***

## 9. Challenges and Risk Mitigation

| Challenge | Probability | Impact | Mitigation |
|-----------|-------------|--------|-----------|
| WRDS API rate limits slow transcript extraction | Medium | Medium | Pre-extract and cache all data in Week 1; use batch queries |
| LLM extraction precision insufficient for clean signal | Low–Medium | High | Fall back to hybrid approach: LLM flags candidates, spaCy validates syntax |
| QLoRA fine-tuning underfits on weak supervision labels | Medium | Low | Fine-tuning is a stretch goal; core system functions without it |
| LangGraph agent pipeline fails during live demo | Low | High | Pre-compute all extraction; demo only calls Reporter Agent live |
| Manual annotation creates disagreements on ambiguous targets | Medium | Medium | Use two-pass annotation with Cohen's kappa; discard low-agreement examples |
| API costs exceed budget at full S&P 500 scale | Low | Medium | Limit LLM extraction to S&P 500; use GPT-4o-mini not GPT-4o |

***

## 10. Conclusion

EarningsLens is a technically rigorous, fully feasible, and academically motivated project that places frontier GenAI techniques — chain-of-thought prompting, RAG, LoRA fine-tuning, and LangGraph multi-agent orchestration — in service of a concrete, validated research hypothesis from empirical finance. The project benefits from three structural advantages that distinguish it from generic GenAI course submissions: institutional-grade data infrastructure (WRDS CIQ Transcripts, CRSP, Compustat), a published, peer-reviewed baseline methodology against which LLM improvements can be quantitatively benchmarked, and an economically meaningful evaluation criterion (Fama-MacBeth portfolio alpha) that grounds the project's NLP contributions in real-world financial outcomes.

The system is designed to be demonstrable end-to-end, reproducible via a clean GitHub repository, and extensible beyond the course context. All required resources have been confirmed available, the technical scope has been calibrated to a three-person team over a five-week execution window, and potential failure modes have been identified with concrete mitigations. The team is prepared to begin data extraction immediately upon proposal approval.

***

## References

- Cohen, L., & Nguyen, N. (2024). *Moving Targets*. Working paper, Harvard Business School.
- Lewis, P., Perez, E., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
- Loughran, T., & McDonald, B. (2011). When is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.
- Wu, S., Irsoy, O., et al. (2023). BloombergGPT: A Large Language Model for Finance. *arXiv:2303.17564*.
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
- Lopez-Lira, A., & Tang, Y. (2023). Can ChatGPT Forecast Stock Price Movements? *arXiv:2304.07619*.
- Chase, H. (2024). LangGraph: Building Stateful Multi-Agent Applications. LangChain documentation.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*.
- Fama, E. F., & MacBeth, J. D. (1973). Risk, Return, and Equilibrium: Empirical Tests. *Journal of Political Economy*, 81(3), 607–636.
- Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*.

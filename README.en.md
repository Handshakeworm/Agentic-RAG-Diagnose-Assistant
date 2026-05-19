[简体中文](README.md) | [English](README.en.md)

# Agentic-RAG Medical Care Assistant

> Patient-side symptom self-check and initial diagnosis system, powered by a LangGraph Agent and multi-route RAG.
>
> **Personal portfolio project** (not production-deployed), covering data engineering → ML inference → Agent orchestration → backend → infrastructure → evaluation as a full end-to-end stack.
> Spec-driven design and implementation, single source of truth: [DEV_SPEC.md](DEV_SPEC.md) (4976 lines, Chinese).

---

## Evaluation Results (62 cases from Chinese licensed-physician exam, 2026-05-17)

> Two-layer evaluation: **① RAG retrieval** (does recall cover the right textbook section?) → **② Diagnosis** (can the LLM produce the correct diagnosis based on recalled chunks?)

### ① RAG retrieval quality (LLM Judge scores parents 0~3 as ground truth)

| Metric | Score | Meaning |
|---|---|---|
| **Hit@20 (≥3 highly relevant)** | **100%** | 62/62 cases — **no case misses a highly-relevant section** |
| **NDCG@20** | **0.774** | Ranking quality (medical-domain range typically 0.6~0.8) |
| **MRR (≥2)** | **0.954** | Top-1 is at least moderately relevant in almost every case |
| Spearman ρ vs LLM | +0.708 | RAG ranking strongly correlated with LLM judgment |

### ② Diagnosis accuracy (LLM diagnoses on RAG Top-20; Judge evaluates gold ↔ LLM equivalence)

> **Multi-gold case**: a case where the patient has multiple coexisting diseases/comorbidities (e.g. "splenic rupture + rib fracture", "acute MI + acute left heart failure" — about half of the 62 cases)

| Metric | Score | Meaning |
|---|---|---|
| **Top-1 clinical hit rate** | **93.5%** | LLM's first candidate is clinically equivalent to gold (58/62) |
| **Top-2 hit rate** | **100%** | Gold primary diagnosis always in top 2 — **62/62, the diagnostic direction is almost never missed** |
| **0 primary direction errors** | **0/62** | All cases produced the correct diagnostic direction (no `none`) |
| **Multi-gold average coverage** | **86.7%** | Multi-gold cases have 86.7% of gold diagnoses listed on average |
| Multi-gold full recall rate | 72.6% | Multi-gold cases where ALL gold diagnoses are covered (45/62) |

→ Detailed methodology, `match_type` distribution, key design decisions: see [Chinese README — Evaluation Results](README.md#评测结果) and [RETRIEVAL_EVAL.md](RETRIEVAL_EVAL.md) (Chinese, retrieval layer in 7 chapters)

---

## Project Positioning

**What it does**: User describes symptoms in natural language. The system clarifies history through multi-round follow-up, retrieves 13 medical textbooks as the knowledge base, and **gives an initial diagnosis + differential directions + recommended further exams**. 62-case Chinese licensed-physician exam evaluation: **Top-1 clinical-equivalent hit rate 93.5%, Top-2 100%, zero primary-direction errors** (see [Evaluation Results](#evaluation-results-62-cases-from-chinese-licensed-physician-exam-2026-05-17) above).

**Who it serves**: Patients needing initial diagnostic judgment and care navigation. All LLM outputs are filtered by a `safety_gate` node — **no prescriptions, no replacement for face-to-face physician care**. The system's role is "give the diagnostic directions a physician might consider + recommend how to investigate further"; final diagnosis and treatment still rest with a licensed physician.

**What it doesn't do**: Direct imaging interpretation, surgical planning, pediatric specialization, drug dosing calculation — these require specialized models or on-site physician judgment, out of scope.

---

## Highlights

> _Detailed in Chinese README §设计亮点 — 12 highlights covering full-stack ownership, multi-route RRF with multi-vector indexing, Small-to-Big parent/child chunking, single-GPU 16GB shared Embedding+Reranker, multimodal ingestion (text + tables + figures), LLM capability routing, idempotency + runtime degradation, 13-dimension HPI structured proactive questioning, three-step diagnosis chain with full-chain fallback, Safety Gate as hard rail, 15-field `rag_trace` audit, centralized runtime constants (`agent_limits`)._

---

## System Architecture

> _English content forthcoming. See [Chinese README §系统架构](README.md#系统架构) for the 13-container Docker Compose topology, GPU model sharing, and degradation paths._

---

## Agent Workflow

> _English content forthcoming. See [Chinese README §Agent 工作流](README.md#agent-工作流) for the LangGraph 16-node + 2-router state machine and interrupt-driven Human-in-the-Loop design._

---

## RAG Pipeline

> _English content forthcoming. See [Chinese README §RAG 流水线](README.md#rag-流水线) for ingestion (MinerU → chunking → LLM enrichment → multi-vector embedding) and retrieval (dense + sparse multi-route + weighted RRF) details._

---

## Tech Stack

> _English content forthcoming. See [Chinese README §技术栈与选型理由](README.md#技术栈与选型理由)._

---

## Quick Start

> _English content forthcoming. See [Chinese README §快速开始](README.md#快速开始) for `docker compose up -d` + database initialization commands._

---

## Roadmap

See [DEV_SPEC.md §8.4 progress table](DEV_SPEC.md#84-进度跟踪表). As of 2026-05-17:

| Phase | Content | Status |
|---|---|---|
| A | Engineering skeleton & infrastructure base | Done |
| B | Data layer & model clients | Done |
| C | Ingestion Pipeline | Main flow working (13 books ingested); production hardening pending |
| D | Terminology (data shelved) | ICD-10 alias 40k+ ingested; not used at runtime |
| E | Retrieval (Sparse / Dense / RRF / Reranker / Filter) | Done |
| F | Agent workflow (16 nodes + 2 routers) | Done |
| G | API layer & permissions (7 items) | Done |
| H | Infrastructure enhancements (Redis / Prometheus / Grafana / Loki / DCGM) | Done |
| I | Evaluation system | **Mostly done** (RAG retrieval + diagnosis closed-loop + dual-layer LLM Judge; Agent multi-round follow-up evaluation pending) |
| J | End-to-end acceptance & doc consolidation | J0 (Dockerization) done; J1-J6 pending |

**Test coverage**: 342 unit PASS / 71 integration PASS (real PG + Milvus + Redis) / e2e reserved for J1-J4; 17 skipped (GPU models + known Milvus race).

---

## Documentation

- [DEV_SPEC.md](DEV_SPEC.md) — full design specification (Chinese, 4976 lines, single source of truth)
- [CLAUDE.md](CLAUDE.md) — AI collaboration workflow, architectural rules, contract red lines
- [RETRIEVAL_EVAL.md](RETRIEVAL_EVAL.md) — RAG retrieval evaluation report (Chinese, 7 chapters)
- [EL_DESIGN_REVIEW.md](EL_DESIGN_REVIEW.md) — Entity Linking design review (Chinese)
- [scripts/METHODOLOGY.md](scripts/METHODOLOGY.md) — Chunking POC general methodology

---

## License

[MIT](LICENSE)

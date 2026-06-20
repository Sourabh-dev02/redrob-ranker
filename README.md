# Redrob — Intelligent Candidate Discovery & Ranking

Hackathon solution for the **Redrob India Runs Data & AI Challenge**.

## What this does

Ranks 100,000 candidates for a **Senior AI Engineer** role using a two-phase hybrid scoring system — structured signal scoring on all candidates followed by semantic embedding reranking on the top 500.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ranker
python rank.py --candidates candidates.jsonl --out submission.csv

# 3. Validate output
python validate_submission.py submission.csv
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB) from HuggingFace and caches it. Subsequent runs work offline.

Use `--no-embed` to skip the semantic step (faster, ~same quality):

```bash
python rank.py --candidates candidates.jsonl --out submission.csv --no-embed
```

## Approach

### Phase 1 — Structured scoring (all 100k candidates, ~2 min on CPU)

Each candidate gets a composite score from five dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Technical skills | up to 0.38 | Core required (embeddings, vector DBs, FAISS, NDCG, BM25, semantic search…) and nice-to-have (LoRA, RAG, learning-to-rank…), weighted by proficiency level and endorsements |
| Career history | up to 0.28 | Counts 40+ AI/IR/search signal keywords in job *description* text — catches plain-language candidates who built real systems but don't use jargon |
| Experience years | up to 0.15 | 5–9 yrs = full weight; penalty outside that range |
| Behavioral signals | up to 0.13 | Recruiter response rate, recency, notice period, GitHub activity, skill assessment scores, interview completion rate |
| Location | up to 0.09 | India required (no visa sponsorship); Pune/Noida/Delhi NCR preferred |

**Hard penalties applied before scoring:**
- Non-technical current title (Marketing Manager, HR, Customer Support, Project Manager…) → ×0.07
- Entire career at IT consulting firms (TCS, Infosys, Wipro, Accenture…) → ×0.20
- Honeypot profiles (multiple simultaneous current jobs, YoE vs graduation date impossible) → score ≈ 0

### Phase 2 — Semantic reranking (top 500 only, ~2 min on CPU)

- Encodes each candidate's headline + summary + career descriptions + skills into a 384-dim embedding using `all-MiniLM-L6-v2`
- Computes cosine similarity against a JD-derived query string
- Blends: **65% structured score + 35% semantic similarity**

This phase captures candidates who describe real systems in plain language (e.g., "built a system to rank search results by relevance") but don't necessarily list "NDCG" or "Pinecone" in their skills section.

### Why this works better than keyword matching

The JD explicitly warns that "a Marketing Manager with all the AI keywords is not a fit". The scoring handles this via:
- Title multiplier: non-technical roles get ×0.07 regardless of skill list
- Career history analysis: descriptions are searched for signals of *actually building* IR/ML systems
- Behavioral signals as multiplier: a perfect-on-paper candidate with 5% response rate and 6 months inactive is down-weighted to reflect real hiring probability

## Files

| File | Description |
|---|---|
| `rank.py` | Main ranker — run this |
| `requirements.txt` | Python dependencies |
| `submission.csv` | Final ranked output (top 100 candidates) |
| `submission_metadata.yaml` | Submission metadata for portal upload |

## Runtime

~4–5 minutes end-to-end on a standard CPU machine with 16 GB RAM.
Without `--no-embed`: ~2 min load + ~2 min structured scoring + ~2 min embedding = ~6 min.
With `--no-embed`: ~2 min load + ~2 min structured scoring = ~4 min.

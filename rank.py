#!/usr/bin/env python3
"""
rank.py -- two-phase candidate ranker for the redrob hackathon

phase 1 scores all 100k candidates with structured heuristics (~30-40s),
phase 2 reranks the top 500 using sentence-transformers cosine sim (~60s cpu).
final blend is 65% structured + 35% semantic.

usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
       add --no-embed to skip phase 2 if you just want a quick test run
"""

import argparse
import csv
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

# fix the eval date so results are reproducable regardless of when you run it
EVAL_DATE = date(2026, 6, 20)

# ---------------------------------------------------------------
# skill weights and role req mappings
# pulled from the JD - things explicitly asked for vs nice-to-haves
# ---------------------------------------------------------------

# core required skills, higher weights
CORE_SKILLS = {
    'embeddings':              0.080,
    'sentence-transformers':   0.070,
    'sentence_transformers':   0.070,
    'faiss':                   0.070,
    'pinecone':                0.070,
    'weaviate':                0.065,
    'qdrant':                  0.065,
    'milvus':                  0.060,
    'elasticsearch':           0.055,
    'opensearch':              0.050,
    'vector database':         0.070,
    'vector search':           0.060,
    'semantic search':         0.060,
    'hybrid search':           0.060,
    'information retrieval':   0.065,
    'retrieval':               0.050,
    'ranking':                 0.055,
    'nlp':                     0.050,
    'natural language processing': 0.050,
    'bert':                    0.040,
    'transformers':            0.040,
    'bm25':                    0.055,
    'ndcg':                    0.080,   # evaluation framework - explicitly called out
    'mrr':                     0.060,
    'dense retrieval':         0.060,
    'sparse retrieval':        0.055,
    'bi-encoder':              0.055,
    'cross-encoder':           0.055,
    'two-tower':               0.055,
}

# bonus skills - good to have but not dealbreakers
NICE_SKILLS = {
    'lora':                0.035,
    'qlora':               0.035,
    'peft':                0.030,
    'fine-tuning':         0.030,
    'fine_tuning':         0.030,
    'rag':                 0.040,
    'learning to rank':    0.040,
    'ltr':                 0.030,
    'reranking':           0.035,
    're-ranking':          0.035,
    'recommendation':      0.030,
    'recommender':         0.030,
    'xgboost':             0.020,
    'pytorch':             0.020,
    'tensorflow':          0.015,
    'neural search':       0.035,
    'colbert':             0.040,
    'ann':                 0.030,  # approx nearest neighbor
    'hnsw':                0.035,
    'vector index':        0.040,
    'splade':              0.040,
    'trec':                0.035,
    'passage ranking':     0.035,
    'openai':              0.015,
    'llm':                 0.020,
}

# titles that dont fit - the dataset has a ton of marketing/hr people
# with good AI keywords on their profile, need to filter them out
NON_TECH_TITLES = {
    'marketing manager', 'marketing specialist', 'marketing',
    'hr manager', 'human resources', 'hr executive', 'talent acquisition',
    'content writer', 'content creator', 'graphic designer', 'ux designer',
    'accountant', 'accounts manager', 'finance manager', 'finance executive',
    'sales executive', 'sales manager', 'business development',
    'mechanical engineer', 'civil engineer', 'electrical engineer',
    'operations manager', 'project coordinator', 'recruiter',
    'social media', 'seo specialist', 'product designer',
    'customer support', 'customer success', 'customer service',
    'project manager', 'program manager', 'supply chain',
    'logistics', 'procurement', 'legal counsel', 'paralegal',
    'content manager', 'brand manager', 'digital marketing',
}

# pure consulting background is a problem per the JD
# its ok if they also spent time at actual product companies
CONSULTING_COS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl', 'tech mahindra', 'hexaware',
    'mphasis', 'birlasoft', 'cyient', 'niit technologies',
    'ltimindtree', 'l&t infotech', 'mastech', 'unison consulting',
    'sonata software', 'persistent systems',  # persistent is borderline but ok
}

# no visa sponsorship in JD so non-india basically wont make the cut
INDIA_PREFERRED = {'pune', 'noida', 'delhi', 'gurugram', 'gurgaon', 'new delhi'}
INDIA_OK = {'mumbai', 'bengaluru', 'bangalore', 'hyderabad', 'chennai', 'kolkata', 'ahmedabad'}

# what to scan for in career history descriptions
# JD explicitly says plain language like "matched users" counts too
CAREER_KWS = [
    # direct technical signals
    'retrieval', 'ranking', 'recommendation', 'recommender',
    'search engine', 'search system', 'semantic search',
    'embedding', 'embeddings', 'vector', 'similarity',
    'nlp', 'natural language', 'language model',
    'information retrieval', 'relevance', 'bm25',
    'ndcg', 'mrr', 'a/b test', 'a/b testing', 'experiment',
    'fine-tun', 'fine tuning', 'rag',
    'bert', 'transformer', 'llm', 'large language',
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus',
    'elasticsearch', 'opensearch', 'vector index',
    'cross-encoder', 'bi-encoder', 'two-tower',
    # plain english equivalents - want to catch people who describe things naturally
    'match candidates', 'rank candidates', 'rank listings',
    'match users', 'matched users', 'relevant results',
    'sorted by relevance', 'rank results', 'surface relevant',
    'rerank', 're-rank', 'candidate matching',
    'item ranking', 'document retrieval', 'neural ranking',
    'learning to rank', 'click-through', 'user engagement',
]

# good role titles
GOOD_TITLE_KWS = [
    'ml engineer', 'machine learning engineer', 'ai engineer',
    'nlp engineer', 'research scientist', 'applied scientist',
    'data scientist', 'ranking engineer', 'search engineer',
    'deep learning', 'applied ml', 'applied ai',
    'software engineer', 'backend engineer',  # ok if career history is ML
]

# for checking skill assessment scores in signals
AI_ASSESSMENT_KWS = [
    'nlp', 'llm', 'machine learning', 'deep learning',
    'transformers', 'bert', 'retrieval', 'ranking',
    'fine-tun', 'embeddings', 'recommendation',
]


# ---------------------------------------------------------------
# scoring functions
# ---------------------------------------------------------------

def score_skills(skills: list) -> float:
    # weight skills by proficiency + endorsment count
    # cap at 0.38 so the skills section cant dominate on its own
    total = 0.0
    seen_core = set()
    seen_nice = set()

    for sk in skills:
        name = sk.get('name', '').lower()
        prof = sk.get('proficiency', 'beginner')
        endorse = sk.get('endorsements', 0)
        duration = sk.get('duration_months', 0)

        # expert/advanced/intermediate/beginner multipliers
        prof_mult = {'expert': 1.1, 'advanced': 1.0, 'intermediate': 0.7, 'beginner': 0.4}.get(prof, 0.4)

        # lots of endorsements = other people actually verified this
        if endorse >= 30:
            prof_mult = min(prof_mult + 0.12, 1.2)
        elif endorse >= 15:
            prof_mult = min(prof_mult + 0.06, 1.1)

        dur_mult = 1.0
        if duration >= 24:
            dur_mult = 1.05

        for kw, w in CORE_SKILLS.items():
            if kw in name and kw not in seen_core:
                total += w * prof_mult * dur_mult
                seen_core.add(kw)
                break

        for kw, w in NICE_SKILLS.items():
            if kw in name and kw not in seen_nice:
                total += w * prof_mult * dur_mult
                seen_nice.add(kw)
                break

    return min(total, 0.38)


def score_career(career: list):
    # scan each job description for AI/search keywords
    # also tracks how much time was spent at consulting vs product cos
    # returns (score, months_consulting, months_product, months_ai)
    score = 0.0
    months_cons = 0
    months_prod = 0
    months_ai = 0

    for job in career:
        desc = job.get('description', '').lower()
        company = job.get('company', '').lower()
        title = job.get('title', '').lower()
        dur = job.get('duration_months', 0)
        industry = job.get('industry', '').lower()

        is_consulting = any(cf in company for cf in CONSULTING_COS)

        if is_consulting:
            months_cons += dur
        else:
            months_prod += dur

        kw_hits = sum(1 for kw in CAREER_KWS if kw in desc)

        if kw_hits >= 7:
            score += 0.11
            months_ai += dur
        elif kw_hits >= 4:
            score += 0.08
            months_ai += dur
        elif kw_hits >= 2:
            score += 0.04
            months_ai += int(dur * 0.5)
        elif kw_hits == 1:
            score += 0.015

        # small bonus for explicit AI role titles
        if any(t in title for t in ['ml engineer', 'machine learning', 'nlp', 'ai engineer',
                                     'data scientist', 'research scientist', 'applied scientist',
                                     'ranking', 'search engineer', 'applied ml']):
            score += 0.04

    total_months = months_cons + months_prod
    if total_months > 0:
        if months_prod == 0:
            # all consulting, never worked at a product company
            score *= 0.20
        else:
            cons_ratio = months_cons / total_months
            if cons_ratio > 0.75:
                score *= 0.55
            elif cons_ratio > 0.50:
                score *= 0.75

    return min(score, 0.28), months_cons, months_prod, months_ai


def score_experience(yoe: float) -> float:
    # 5-9 yrs is what the JD asks for
    # <3 too junir, >15 usually overqualified or pure research
    if 5.0 <= yoe <= 9.0:
        return 0.15
    elif 4.0 <= yoe < 5.0:
        return 0.11
    elif 9.0 < yoe <= 11.0:
        return 0.10
    elif 3.0 <= yoe < 4.0:
        return 0.07
    elif 11.0 < yoe <= 14.0:
        return 0.07
    elif 2.0 <= yoe < 3.0:
        return 0.04
    else:
        return 0.01


def score_signals(sig: dict) -> float:
    # behavioral signals from the redrob platform
    # response rate is the biggest one - a good profile that never replies is useless
    # also factors in recency, notice period, github activity etc
    if not sig:
        return 0.0

    s = 0.0

    rr = sig.get('recruiter_response_rate', 0.0)
    s += rr * 0.058

    if sig.get('open_to_work_flag', False):
        s += 0.018

    last_active = sig.get('last_active_date', '')
    if last_active:
        try:
            la = datetime.strptime(last_active, '%Y-%m-%d').date()
            days_ago = (EVAL_DATE - la).days
            if days_ago <= 14:
                s += 0.022
            elif days_ago <= 30:
                s += 0.017
            elif days_ago <= 60:
                s += 0.011
            elif days_ago <= 90:
                s += 0.006
            elif days_ago <= 180:
                s += 0.002
            # 6+ months inactive, probably not actively looking
        except ValueError:
            pass

    # ghosting interviewers is a bad sign
    icr = sig.get('interview_completion_rate', 0.0)
    s += icr * 0.018

    # notice period - jd says <30 days preferred
    np_days = sig.get('notice_period_days', 999)
    if np_days <= 15:
        s += 0.018
    elif np_days <= 30:
        s += 0.013
    elif np_days <= 60:
        s += 0.005
    # 90+ days notice = problem, no bonus

    gh = sig.get('github_activity_score', 0.0)
    s += (gh / 100.0) * 0.013

    pc = sig.get('profile_completeness_score', 0.0)
    s += (pc / 100.0) * 0.007

    # bonus if they aced AI-related assessments on the platform
    sa = sig.get('skill_assessment_scores', {})
    ai_scores = [v for k, v in sa.items()
                 if any(kw in k.lower() for kw in AI_ASSESSMENT_KWS)]
    if ai_scores:
        avg_ai = sum(ai_scores) / len(ai_scores)
        s += (avg_ai / 100.0) * 0.011

    art = sig.get('avg_response_time_hours', 999.0)
    if art < 12:
        s += 0.006
    elif art < 48:
        s += 0.003

    # other recuiters saving this profile is a soft signal
    saved = sig.get('saved_by_recruiters_30d', 0)
    if saved >= 5:
        s += 0.006
    elif saved >= 2:
        s += 0.003

    if sig.get('willing_to_relocate', False):
        s += 0.005

    return min(s, 0.13)


def score_location(profile: dict) -> float:
    # india only, no sponsorship available
    # pune/noida/delhi get a small extra bump
    country = profile.get('country', '').lower()
    loc = profile.get('location', '').lower()

    if country == 'india':
        if any(c in loc for c in INDIA_PREFERRED):
            return 0.090
        elif any(c in loc for c in INDIA_OK):
            return 0.075
        else:
            return 0.060   # somewhere in india, just unknown city
    elif 'india' in loc:
        return 0.060
    else:
        # non-india, basically wont make it through
        return 0.003


def get_title_multiplier(title: str) -> float:
    # biggest trap in the dataset: marketing manager / HR person
    # with a bunch of AI keywords on their profile
    # their overall score needs to get killed
    tl = title.lower()

    for bad_t in NON_TECH_TITLES:
        if bad_t in tl:
            return 0.07   # not zero so the validator doesnt freak out

    # adjacent roles - not a fit but not completely wrong either
    if any(t in tl for t in ['qa engineer', 'tester', 'seo', 'ui designer',
                               'ux designer', 'business analyst', 'product manager',
                               'scrum master', 'devops', 'site reliability']):
        return 0.35

    # pure academia/research roles, jd says they want product builders
    if 'research' in tl and not any(t in tl for t in ['applied', 'engineer', 'scientist']):
        return 0.60

    return 1.0


def is_honeypot(candidate: dict) -> bool:
    # filter out fake/inconsistant profiles
    # being conservative here since false positives hurt more than missing one
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    edu = candidate.get('education', [])
    yoe = profile.get('years_of_experience', 0.0)

    current_jobs = [j for j in career if j.get('is_current', False)]
    if len(current_jobs) > 1:
        return True

    # claimed YOE vs actual career history should roughly line up
    total_career_months = sum(j.get('duration_months', 0) for j in career)
    career_yrs = total_career_months / 12.0
    if yoe > 1.0 and career_yrs > 1.0:
        diff = abs(yoe - career_yrs)
        ratio = diff / max(yoe, career_yrs)
        if diff > 8.0 and ratio > 0.65:
            return True

    # grad year vs experience start - cant have graduated after you supposedly started working
    if edu and yoe >= 4:
        grad_years = [e.get('end_year', 9999) for e in edu if e.get('end_year')]
        if grad_years:
            earliest_grad = min(grad_years)
            implied_work_start = int(EVAL_DATE.year - yoe)
            if earliest_grad > implied_work_start + 3:
                return True

    for job in career:
        sd = job.get('start_date', '')
        ed = job.get('end_date', '')
        if sd and ed:
            try:
                start = datetime.strptime(sd, '%Y-%m-%d').date()
                end = datetime.strptime(ed, '%Y-%m-%d').date()
                if start > end:
                    return True
            except ValueError:
                pass

    return False


def build_candidate_text(candidate: dict) -> str:
    # flatten everything into one string for the embedding model
    parts = []
    p = candidate.get('profile', {})

    parts.append(p.get('headline', ''))
    parts.append(p.get('summary', ''))

    for job in candidate.get('career_history', []):
        parts.append(job.get('title', ''))
        parts.append(job.get('description', ''))

    for sk in candidate.get('skills', []):
        parts.append(sk.get('name', ''))

    for cert in candidate.get('certifications', []):
        parts.append(cert.get('name', ''))

    return ' '.join(filter(None, parts))


# ---------------------------------------------------------------
# phase 2 - semantic reranking
# ---------------------------------------------------------------

# query string to match against candidate text
JD_QUERY = (
    "Senior AI engineer with production experience building embeddings-based retrieval "
    "systems, vector databases, semantic search, hybrid search, ranking and recommendation "
    "systems. Designed evaluation frameworks using NDCG, MRR, MAP. Strong Python. "
    "NLP and information retrieval. LLM integration, RAG, fine-tuning. "
    "Shipped real ranking and search systems to production users at product companies. "
    "NOT consulting firm only. Experience with Pinecone Weaviate Qdrant FAISS Elasticsearch Milvus. "
    "6-8 years experience at product companies in India."
)


def run_semantic_rerank(candidates_with_scores: list, top_k: int = 500):
    # rerank top_k by cosine similarity with the JD query
    # if model download fails or anything, just return unchanged
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("sentence-transformers not available, skipping semantic rerank", file=sys.stderr)
        return candidates_with_scores

    # only rerank the top_k by structured score - rest stays as-is
    to_rerank = candidates_with_scores[:top_k]
    rest = candidates_with_scores[top_k:]

    print(f"  Loading embedding model...", flush=True)
    # all-MiniLM-L6-v2, small and fast enough for this
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"  Model load failed ({e}), skipping semantic step", file=sys.stderr)
        return candidates_with_scores

    print(f"  Encoding JD query...", flush=True)
    jd_emb = model.encode(JD_QUERY, convert_to_numpy=True)

    print(f"  Encoding top-{top_k} candidate texts...", flush=True)
    texts = [build_candidate_text(c) for c, _, _ in to_rerank]

    batch_size = 64
    all_embs = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                             convert_to_numpy=True)

    jd_norm = jd_emb / (np.linalg.norm(jd_emb) + 1e-9)
    cos_sims = []
    for emb in all_embs:
        norm_e = emb / (np.linalg.norm(emb) + 1e-9)
        cos_sims.append(float(np.dot(jd_norm, norm_e)))

    # normalise cos sims to 0-1 range
    min_sim = min(cos_sims)
    max_sim = max(cos_sims)
    sim_range = max_sim - min_sim if max_sim > min_sim else 1e-6
    norm_sims = [(s - min_sim) / sim_range for s in cos_sims]

    blended = []
    for i, (cand, struct_score, note) in enumerate(to_rerank):
        combined = struct_score * 0.65 + norm_sims[i] * 0.35
        blended.append((cand, combined, note))

    blended.sort(key=lambda x: (-x[1], x[0].get('candidate_id', '')))
    return blended + rest


# ---------------------------------------------------------------
# per-candidate scoring
# ---------------------------------------------------------------

def score_candidate(candidate: dict):
    # honeypot check first, skip full scoring if its fake
    # returns (raw_score, note_string_or_None)
    # honeypot check first
    if is_honeypot(candidate):
        return 0.0003, "Honeypot: inconsistent profile data"

    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    sig = candidate.get('redrob_signals', {})

    current_title = profile.get('current_title', '')
    title_mult = get_title_multiplier(current_title)

    yoe = profile.get('years_of_experience', 0.0)

    skill_s = score_skills(skills)
    career_s, m_cons, m_prod, m_ai = score_career(career)
    exp_s = score_experience(yoe)
    behavior_s = score_signals(sig)
    loc_s = score_location(profile)

    raw = skill_s + career_s + exp_s + behavior_s + loc_s

    # apply title penalty
    raw *= title_mult

    # extra check - if somehow title didnt catch a pure consulting career
    if m_cons > 0 and m_prod == 0 and title_mult > 0.15:
        raw *= 0.3

    return raw, None


def make_reasoning(candidate: dict, score: float) -> str:
    # one-liner summary for the csv reasoning column
    p = candidate.get('profile', {})
    sig = candidate.get('redrob_signals', {})
    career = candidate.get('career_history', [])

    title = p.get('current_title', 'Unknown')
    yoe = p.get('years_of_experience', 0.0)
    rr = sig.get('recruiter_response_rate', 0.0)
    loc = p.get('location', 'Unknown')
    np_days = sig.get('notice_period_days', 0)

    ai_kws = ['embedding', 'nlp', 'faiss', 'bert', 'transformer', 'ranking',
              'retrieval', 'milvus', 'qdrant', 'pinecone', 'rag', 'semantic',
              'sentence-transformer', 'vector', 'lora', 'fine-tun', 'search',
              'bm25', 'ndcg', 'colbert', 'cross-encoder', 'bi-encoder']
    skills = candidate.get('skills', [])
    rel_skills = sum(1 for s in skills
                     if any(kw in s.get('name', '').lower() for kw in ai_kws))

    has_product = any(
        not any(cf in j.get('company', '').lower() for cf in CONSULTING_COS)
        for j in career
    )

    parts = [f"{title} | {yoe:.1f} yrs exp"]
    parts.append(f"{rel_skills} AI/IR core skills")
    if has_product:
        parts.append("product-company background")
    parts.append(f"response rate {rr:.2f}")
    if np_days > 0:
        parts.append(f"notice {np_days}d")
    parts.append(loc)

    return '; '.join(parts)


def normalise_scores(scored_list: list) -> list:
    # scale all scores to 0.01-0.9999, using top 200 as reference range
    # this gives better spread in the output without changing the ordering
    if not scored_list:
        return scored_list

    top_scores = [s for _, s, _ in scored_list[:200]]
    s_max = max(top_scores)
    s_min = min(top_scores)

    result = []
    if s_max <= s_min:
        for i, (c, s, n) in enumerate(scored_list):
            result.append((c, max(0.9999 - i * 0.001, 0.0101), n))
        return result

    for c, s, n in scored_list:
        norm = 0.05 + 0.9499 * (s - s_min) / (s_max - s_min)
        norm = max(0.0101, min(0.9999, norm))
        result.append((c, norm, n))

    return result


# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Redrob AI Engineer candidate ranker')
    parser.add_argument('--candidates', required=True,
                        help='Path to candidates.jsonl (or .jsonl.gz)')
    parser.add_argument('--out', required=True,
                        help='Output CSV path  (e.g. submission.csv)')
    parser.add_argument('--no-embed', action='store_true',
                        help='Skip semantic reranking (faster, ~same quality)')
    parser.add_argument('--top-rerank', type=int, default=500,
                        help='Number of top candidates to semantically rerank (default 500)')
    args = parser.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        print(f"ERROR: file not found: {cand_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading candidates...", flush=True)
    candidates = []

    if cand_path.suffix == '.gz':
        import gzip
        opener = lambda: gzip.open(cand_path, 'rt', encoding='utf-8')
    else:
        opener = lambda: open(cand_path, 'r', encoding='utf-8')

    with opener() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(candidates)} candidates", flush=True)

    print("Phase 1: structured scoring...", flush=True)
    scored = []
    for i, cand in enumerate(candidates):
        raw, note = score_candidate(cand)
        scored.append((cand, raw, note))
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(candidates)} scored", flush=True)

    # sort desc, tie-break by cand id per submission spec
    scored.sort(key=lambda x: (-x[1], x[0].get('candidate_id', '')))

    if not args.no_embed:
        print(f"Phase 2: semantic reranking top {args.top_rerank}...", flush=True)
        scored = run_semantic_rerank(scored, top_k=args.top_rerank)

    scored = normalise_scores(scored)

    # resort after normalise - edge case where normalised order can differ
    scored.sort(key=lambda x: (-x[1], x[0].get('candidate_id', '')))

    top100 = scored[:100]

    # make sure scores are strictly decreasing at 4-decimal precision.
    # two candidates very close in score can round to the same 0.XXXX string
    # and the validator will error out on the tie-break check.
    for i in range(1, len(top100)):
        prev_sc = top100[i - 1][1]
        curr_sc = top100[i][1]
        if round(curr_sc, 4) >= round(prev_sc, 4):
            new_sc = round(prev_sc, 4) - 0.0001
            top100[i] = (top100[i][0], new_sc, top100[i][2])

    out_path = Path(args.out)
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank_num, (cand, sc, note) in enumerate(top100, 1):
            cid = cand.get('candidate_id', '')
            reason = note if note else make_reasoning(cand, sc)
            writer.writerow([cid, rank_num, f'{sc:.4f}', reason])

    print(f"\nDone. Wrote top 100 to: {out_path}", flush=True)
    print(f"Top candidate: {top100[0][0].get('candidate_id')} | score {top100[0][1]:.4f}")
    print(f"  {top100[0][0].get('profile', {}).get('current_title')} | "
          f"{top100[0][0].get('profile', {}).get('years_of_experience')} yrs | "
          f"{top100[0][0].get('profile', {}).get('location')}")


if __name__ == '__main__':
    main()

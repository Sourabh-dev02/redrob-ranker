import streamlit as st
import json
import csv
import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# scoring functions copied inline so app.py works standalone

from datetime import date, datetime

EVAL_DATE = date(2026, 6, 20)

CORE_SKILLS = {
    'embeddings': 0.080, 'sentence-transformers': 0.070, 'sentence_transformers': 0.070,
    'faiss': 0.070, 'pinecone': 0.070, 'weaviate': 0.065, 'qdrant': 0.065,
    'milvus': 0.060, 'elasticsearch': 0.055, 'opensearch': 0.050,
    'vector database': 0.070, 'vector search': 0.060, 'semantic search': 0.060,
    'hybrid search': 0.060, 'information retrieval': 0.065, 'retrieval': 0.050,
    'ranking': 0.055, 'nlp': 0.050, 'natural language processing': 0.050,
    'bert': 0.040, 'transformers': 0.040, 'bm25': 0.055, 'ndcg': 0.080,
    'mrr': 0.060, 'dense retrieval': 0.060, 'sparse retrieval': 0.055,
    'bi-encoder': 0.055, 'cross-encoder': 0.055, 'two-tower': 0.055,
}
NICE_SKILLS = {
    'lora': 0.035, 'qlora': 0.035, 'peft': 0.030, 'fine-tuning': 0.030,
    'rag': 0.040, 'learning to rank': 0.040, 'ltr': 0.030, 'reranking': 0.035,
    're-ranking': 0.035, 'recommendation': 0.030, 'recommender': 0.030,
    'xgboost': 0.020, 'pytorch': 0.020, 'neural search': 0.035, 'colbert': 0.040,
    'hnsw': 0.035, 'vector index': 0.040, 'llm': 0.020,
}
NON_TECH_TITLES = {
    'marketing manager', 'marketing specialist', 'marketing', 'hr manager',
    'human resources', 'hr executive', 'talent acquisition', 'content writer',
    'content creator', 'graphic designer', 'ux designer', 'accountant',
    'accounts manager', 'finance manager', 'finance executive', 'sales executive',
    'sales manager', 'business development', 'mechanical engineer', 'civil engineer',
    'electrical engineer', 'operations manager', 'project coordinator', 'recruiter',
    'social media', 'seo specialist', 'product designer', 'customer support',
    'customer success', 'customer service', 'project manager', 'program manager',
    'supply chain', 'logistics', 'procurement', 'content manager', 'brand manager',
    'digital marketing', 'office manager', 'administrative',
}
CONSULTING_COS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'hcl', 'tech mahindra', 'hexaware', 'mphasis', 'birlasoft',
}
INDIA_PREFERRED = {'pune', 'noida', 'delhi', 'gurugram', 'gurgaon', 'new delhi'}
INDIA_OK = {'mumbai', 'bengaluru', 'bangalore', 'hyderabad', 'chennai', 'kolkata'}
CAREER_KWS = [
    'retrieval', 'ranking', 'recommendation', 'search', 'embedding', 'embeddings',
    'vector', 'similarity', 'nlp', 'natural language', 'language model',
    'information retrieval', 'relevance', 'bm25', 'ndcg', 'mrr', 'a/b test',
    'fine-tun', 'rag', 'bert', 'transformer', 'llm', 'faiss', 'pinecone',
    'weaviate', 'qdrant', 'milvus', 'elasticsearch', 'cross-encoder', 'bi-encoder',
    'two-tower', 'match candidates', 'rank results', 'relevant results',
    'learning to rank', 'rerank', 're-rank', 'neural ranking',
]


def score_skills(skills):
    total = 0.0
    seen_core, seen_nice = set(), set()
    for sk in skills:
        name = sk.get('name', '').lower()
        prof = sk.get('proficiency', 'beginner')
        endorse = sk.get('endorsements', 0)
        mult = {'expert': 1.1, 'advanced': 1.0, 'intermediate': 0.7, 'beginner': 0.4}.get(prof, 0.4)
        if endorse >= 30:
            mult = min(mult + 0.12, 1.2)
        elif endorse >= 15:
            mult = min(mult + 0.06, 1.1)
        for kw, w in CORE_SKILLS.items():
            if kw in name and kw not in seen_core:
                total += w * mult
                seen_core.add(kw)
                break
        for kw, w in NICE_SKILLS.items():
            if kw in name and kw not in seen_nice:
                total += w * mult
                seen_nice.add(kw)
                break
    return min(total, 0.38)


def score_career(career):
    score = 0.0
    months_cons, months_prod = 0, 0
    for job in career:
        desc = job.get('description', '').lower()
        company = job.get('company', '').lower()
        title = job.get('title', '').lower()
        dur = job.get('duration_months', 0)
        is_cons = any(cf in company for cf in CONSULTING_COS)
        if is_cons:
            months_cons += dur
        else:
            months_prod += dur
        kw_hits = sum(1 for kw in CAREER_KWS if kw in desc)
        if kw_hits >= 7:
            score += 0.11
        elif kw_hits >= 4:
            score += 0.08
        elif kw_hits >= 2:
            score += 0.04
        elif kw_hits == 1:
            score += 0.015
        if any(t in title for t in ['ml engineer', 'machine learning', 'nlp', 'ai engineer',
                                     'data scientist', 'research scientist', 'ranking', 'search engineer']):
            score += 0.04
    total = months_cons + months_prod
    if total > 0 and months_prod == 0:
        score *= 0.20
    elif total > 0:
        cons_ratio = months_cons / total
        if cons_ratio > 0.75:
            score *= 0.55
        elif cons_ratio > 0.50:
            score *= 0.75
    return min(score, 0.28)


def score_experience(yoe):
    if 5.0 <= yoe <= 9.0: return 0.15
    elif 4.0 <= yoe < 5.0: return 0.11
    elif 9.0 < yoe <= 11.0: return 0.10
    elif 3.0 <= yoe < 4.0: return 0.07
    elif 11.0 < yoe <= 14.0: return 0.07
    elif 2.0 <= yoe < 3.0: return 0.04
    else: return 0.01


def score_signals(sig):
    if not sig:
        return 0.0
    s = 0.0
    s += sig.get('recruiter_response_rate', 0.0) * 0.058
    if sig.get('open_to_work_flag', False):
        s += 0.018
    last_active = sig.get('last_active_date', '')
    if last_active:
        try:
            days_ago = (EVAL_DATE - datetime.strptime(last_active, '%Y-%m-%d').date()).days
            if days_ago <= 14: s += 0.022
            elif days_ago <= 30: s += 0.017
            elif days_ago <= 60: s += 0.011
            elif days_ago <= 90: s += 0.006
            elif days_ago <= 180: s += 0.002
        except ValueError:
            pass
    s += sig.get('interview_completion_rate', 0.0) * 0.018
    np_days = sig.get('notice_period_days', 999)
    if np_days <= 15: s += 0.018
    elif np_days <= 30: s += 0.013
    elif np_days <= 60: s += 0.005
    s += (sig.get('github_activity_score', 0.0) / 100.0) * 0.013
    s += (sig.get('profile_completeness_score', 0.0) / 100.0) * 0.007
    sa = sig.get('skill_assessment_scores', {})
    ai_scores = [v for k, v in sa.items() if any(kw in k.lower() for kw in ['nlp', 'ml', 'machine learning', 'deep learning', 'retrieval', 'ranking'])]
    if ai_scores:
        s += (sum(ai_scores) / len(ai_scores) / 100.0) * 0.011
    if sig.get('willing_to_relocate', False):
        s += 0.005
    return min(s, 0.13)


def score_location(profile):
    country = profile.get('country', '').lower()
    loc = profile.get('location', '').lower()
    if country == 'india':
        if any(c in loc for c in INDIA_PREFERRED): return 0.090
        elif any(c in loc for c in INDIA_OK): return 0.075
        else: return 0.060
    elif 'india' in loc:
        return 0.060
    return 0.003


def get_title_multiplier(title):
    tl = title.lower()
    for bad_t in NON_TECH_TITLES:
        if bad_t in tl:
            return 0.07
    if any(t in tl for t in ['qa engineer', 'tester', 'business analyst', 'product manager', 'scrum master', 'devops']):
        return 0.35
    if 'research' in tl and not any(t in tl for t in ['applied', 'engineer', 'scientist']):
        return 0.60
    return 1.0


def is_honeypot(candidate):
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    edu = candidate.get('education', [])
    yoe = profile.get('years_of_experience', 0.0)
    if sum(1 for j in career if j.get('is_current', False)) > 1:
        return True
    career_yrs = sum(j.get('duration_months', 0) for j in career) / 12.0
    if yoe > 1.0 and career_yrs > 1.0:
        diff = abs(yoe - career_yrs)
        if diff > 8.0 and diff / max(yoe, career_yrs) > 0.65:
            return True
    if edu and yoe >= 4:
        grad_years = [e.get('end_year', 9999) for e in edu if e.get('end_year')]
        if grad_years and min(grad_years) > int(EVAL_DATE.year - yoe) + 3:
            return True
    return False


def score_candidate(candidate):
    if is_honeypot(candidate):
        return 0.0003
    profile = candidate.get('profile', {})
    title_mult = get_title_multiplier(profile.get('current_title', ''))
    yoe = profile.get('years_of_experience', 0.0)
    skill_s = score_skills(candidate.get('skills', []))
    career_s = score_career(candidate.get('career_history', []))
    exp_s = score_experience(yoe)
    behavior_s = score_signals(candidate.get('redrob_signals', {}))
    loc_s = score_location(profile)
    raw = (skill_s + career_s + exp_s + behavior_s + loc_s) * title_mult
    return raw


# ui

st.set_page_config(page_title="Redrob AI Ranker", page_icon="🎯", layout="wide")
st.title("🎯 Redrob — AI Engineer Candidate Ranker")
st.caption("Hackathon demo · Ranks candidates for the Senior AI Engineer JD · Structured scoring + semantic signals")

st.info(
    "**How it works:** Upload a JSONL file (one candidate JSON per line) or paste raw JSON. "
    "The ranker scores each candidate on skills, career history, experience, behavioral signals, and location. "
    "Non-technical titles and consulting-only careers are penalised. Honeypot profiles are filtered out."
)

tab1, tab2 = st.tabs(["Upload JSONL", "Paste JSON"])

candidates = []

with tab1:
    uploaded = st.file_uploader("Upload candidates.jsonl (or any .jsonl sample)", type=["jsonl", "json", "txt"])
    if uploaded:
        content = uploaded.read().decode("utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if candidates:
            st.success(f"Loaded {len(candidates)} candidates from file.")

with tab2:
    sample_json = st.text_area(
        "Paste a JSON array or newline-delimited JSON objects:",
        height=200,
        placeholder='[{"candidate_id": "CAND_0000001", "profile": {...}, ...}]'
    )
    if sample_json.strip():
        try:
            parsed = json.loads(sample_json.strip())
            if isinstance(parsed, list):
                candidates = parsed
            else:
                candidates = [parsed]
            st.success(f"Loaded {len(candidates)} candidates from paste.")
        except json.JSONDecodeError:
            for line in sample_json.splitlines():
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if candidates:
                st.success(f"Loaded {len(candidates)} candidates (JSONL format).")

top_n = st.slider(
    "How many top candidates to show?",
    min_value=1,
    max_value=max(min(100, len(candidates)), 10),
    value=min(10, max(len(candidates), 1)),
    disabled=len(candidates) == 0,
)

if st.button("🚀 Run Ranker", type="primary", disabled=len(candidates) == 0):
    with st.spinner(f"Scoring {len(candidates)} candidates..."):
        scored = []
        for cand in candidates:
            sc = score_candidate(cand)
            scored.append((cand, sc))

        scored.sort(key=lambda x: (-x[1], x[0].get('candidate_id', '')))

        # normalise
        top_scores = [s for _, s in scored[:min(200, len(scored))]]
        s_max, s_min = max(top_scores), min(top_scores)
        s_range = s_max - s_min if s_max > s_min else 1e-6

        results = []
        for rank_i, (cand, raw) in enumerate(scored[:top_n], 1):
            norm = 0.05 + 0.9499 * (raw - s_min) / s_range
            norm = max(0.0101, min(0.9999, norm))
            p = cand.get('profile', {})
            sig = cand.get('redrob_signals', {})
            results.append({
                "Rank": rank_i,
                "Candidate ID": cand.get('candidate_id', ''),
                "Title": p.get('current_title', ''),
                "Exp (yrs)": p.get('years_of_experience', 0),
                "Location": p.get('location', ''),
                "Score": round(norm, 4),
                "Response Rate": round(sig.get('recruiter_response_rate', 0), 2),
                "Notice (days)": sig.get('notice_period_days', '?'),
                "Open to Work": '✅' if sig.get('open_to_work_flag') else '❌',
            })

    st.subheader(f"Top {top_n} Candidates")
    st.dataframe(results, use_container_width=True, hide_index=True)

    # download CSV
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
    for i, (cand, raw) in enumerate(scored[:top_n], 1):
        p = cand.get('profile', {})
        sig = cand.get('redrob_signals', {})
        norm = 0.05 + 0.9499 * (raw - s_min) / s_range
        norm = max(0.0101, min(0.9999, norm))
        reasoning = (f"{p.get('current_title','')} | {p.get('years_of_experience',0):.1f} yrs; "
                     f"response rate {sig.get('recruiter_response_rate',0):.2f}; "
                     f"{p.get('location','')}")
        writer.writerow([cand.get('candidate_id',''), i, f'{norm:.4f}', reasoning])

    st.download_button("⬇ Download CSV", buf.getvalue(), file_name="ranked_candidates.csv", mime="text/csv")

elif len(candidates) == 0:
    st.warning("Upload a JSONL file or paste candidate JSON above to get started.")

with st.expander("ℹ️ Scoring breakdown"):
    st.markdown("""
    | Dimension | Max weight | What it measures |
    |---|---|---|
    | Technical skills | 0.38 | Core required (FAISS, NDCG, embeddings, BM25, semantic search…) + nice-to-have (LoRA, RAG, LTR…) |
    | Career history | 0.28 | Keyword signals in job *descriptions* — catches plain-language candidates who built real AI/search systems |
    | Experience years | 0.15 | Sweet spot is 5–9 yrs per JD |
    | Behavioral signals | 0.13 | Response rate, recency, notice period, GitHub activity, skill assessments |
    | Location | 0.09 | India required; Pune/Noida/Delhi NCR preferred |
    
    **Hard penalties:** non-technical titles ×0.07 · consulting-only career ×0.20 · honeypot profiles ≈0
    """)

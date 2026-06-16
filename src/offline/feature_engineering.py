import polars as pl
import json
import math
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEARCH_RELEVANCE_TERMS = {
    "candidate matching": 1.0, "talent discovery": 1.0, "relevance optimization": 1.0,
    "marketplace ranking": 1.0, "query understanding": 1.0, "semantic search": 1.0,
    "inverted index": 0.8, "bm25": 0.8, "lucene": 0.7, "solr": 0.7, "tf-idf": 0.4
}

RETRIEVAL_TERMS = {
    "dense retrieval": 1.0, "bi-encoder": 1.0, "cross-encoder": 1.0, "hybrid search": 1.0,
    "approximate nearest neighbors": 1.0, "ann": 0.8, "hnsw": 0.9, "sentence-transformers": 0.9,
    "vector database": 0.8, "faiss": 0.8, "pinecone": 0.6, "embeddings": 0.6, 
    "rag": 0.2, "retrieval augmented generation": 0.2
}

RANKING_TERMS = {
    "learning-to-rank": 1.0, "lambdamart": 1.0, "re-ranking": 0.9, 
    "recommendation systems": 0.8, "personalization": 0.8, "collaborative filtering": 0.7,
    "xgboost": 0.7, "lightgbm": 0.7
}

EVALUATION_TERMS = {
    "ndcg": 1.0, "mean average precision": 1.0, "mrr": 1.0, "interleaving": 1.0,
    "offline evaluation": 0.9, "a/b testing": 0.8, "map": 0.5 
}

BUILDER_TERMS = {
    "productionized": 1.0, "architected": 1.0, "scaled": 0.8, "shipped": 0.8,
    "designed": 0.7, "deployed": 0.6, "built": 0.4
}

def extract_dictionary_score(text: str, terms_dict: dict) -> float:
    text_lower = text.lower()
    score = 0.0
    for term, weight in terms_dict.items():
        count = text_lower.count(term.lower())
        score += count * weight
    return score

def compute_product_exposure(row: dict) -> float:
    try:
        career = json.loads(row.get('career_history_json', '[]'))
    except:
        career = []
    
    if not career: return 0.5
    
    score = 0.5
    size = str(row.get('company_size', '')).lower()
    industry = str(row.get('industry', '')).lower()
    desc = " ".join([j.get('description', '').lower() for j in career])
    
    if any(s in size for s in ['1-10', '11-50', '51-200']): score += 0.2
    if any(i in industry for i in ['software', 'internet', 'saas', 'consumer']): score += 0.2
    if any(w in desc for w in ['saas', 'our product', 'b2b', 'b2c', 'startup', 'scale-up']): score += 0.3
    
    service_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "deloitte"]
    if any(firm in " ".join([j.get('company', '').lower() for j in career]) for firm in service_firms): score -= 0.5
    if any(w in desc for w in ['client', 'delivery', 'sow', 'consulting', 'offshore']): score -= 0.3
    
    return min(max(score, 0.0), 1.0)

def compute_trajectory_transition(row: dict) -> float:
    try:
        career = json.loads(row.get('career_history_json', '[]'))
    except:
        career = []
    
    if len(career) < 2: return 0.0
        
    service_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "deloitte"]
    oldest = career[-1].get('company', '').lower()
    newest = career[0].get('company', '').lower()
    
    started_service = any(firm in oldest for firm in service_firms)
    ended_product = not any(firm in newest for firm in service_firms)
    
    if started_service and ended_product:
        return 0.2
    return 0.0

def compute_contradiction_score(row: dict) -> float:
    score = 0.0
    yoe = row.get('years_of_experience', 0)
    dur_total = row.get('total_duration_months', 0)
    grad_year = row.get('grad_year', 0)
    
    if dur_total > (yoe * 12) * 1.2:
        score += 5.0
        
    if grad_year > 0:
        if (2026 - grad_year) < yoe - 2:
            score += 5.0
            
    sal_min = row.get('expected_salary_min', 0)
    if yoe <= 2 and sal_min > 40:
        score += 5.0
        
    views = row.get('profile_views_received_30d', 0)
    response_rate = row.get('recruiter_response_rate', 0.0)
    if response_rate == 1.0 and views == 0:
        score += 2.0
        
    try:
        skills = json.loads(row.get('skills_json', '[]'))
        for s in skills:
            if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0:
                score += 1.0
    except:
        pass

    return score

def compute_availability_score(row: dict) -> float:
    try:
        if row.get('last_active_date'):
            last_active = datetime.strptime(row.get('last_active_date')[:10], '%Y-%m-%d')
            days_inactive = (datetime(2026, 6, 16) - last_active).days
        else:
            days_inactive = 180
    except:
        days_inactive = 180
        
    recency_factor = math.exp(-max(0, days_inactive) / 180.0)
    notice = row.get('notice_period_days', 30)
    notice_factor = max(0.0, 1.0 - (max(0, notice - 30) / 60.0))
    engagement = (row.get('recruiter_response_rate', 0.0) + row.get('interview_completion_rate', 0.0)) / 2.0
    boost = 1.2 if row.get('open_to_work_flag', False) else 1.0
    
    return recency_factor * notice_factor * engagement * boost

def main():
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / 'artifacts'
    input_file = artifacts_dir / 'parsed_candidates.parquet'
    output_file = artifacts_dir / 'features.parquet'
    
    if not input_file.exists():
        logging.error(f"{input_file} not found.")
        return
        
    logging.info("Reading parsed candidates")
    df = pl.read_parquet(input_file)
    
    records = df.to_dicts()
    features = []
    for row in records:
        text_blob = row['full_profile_text']
        
        rel_ev = min(extract_dictionary_score(text_blob, SEARCH_RELEVANCE_TERMS) / 5.0, 1.0)
        ret_depth = min(extract_dictionary_score(text_blob, RETRIEVAL_TERMS) / 5.0, 1.0)
        rank_depth = min(extract_dictionary_score(text_blob, RANKING_TERMS) / 3.0, 1.0)
        eval_rigor = min(extract_dictionary_score(text_blob, EVALUATION_TERMS) / 3.0, 1.0)
        builder = min(extract_dictionary_score(text_blob, BUILDER_TERMS) / 5.0, 1.0)
        
        text_lower = text_blob.lower()
        feat_architect_no_coding = 1.0
        if "architect" in row.get('current_title', '').lower() or "manager" in row.get('current_title', '').lower():
            if extract_dictionary_score(text_blob, BUILDER_TERMS) < 0.5:
                feat_architect_no_coding = 0.7
                
        feat_wrapper_ai_only = 1.0
        if "langchain" in text_lower or "openai" in text_lower:
            if ret_depth < 0.2 and rank_depth < 0.2:
                feat_wrapper_ai_only = 0.4
                
        feat = {
            'candidate_id': row['candidate_id'],
            'feat_search_relevance_evidence': rel_ev,
            'feat_retrieval_depth': ret_depth,
            'feat_ranking_depth': rank_depth,
            'feat_evaluation_rigor': eval_rigor,
            'feat_builder_score': builder,
            'feat_product_exposure': compute_product_exposure(row),
            'feat_trajectory_transition': compute_trajectory_transition(row),
            'feat_availability_score': compute_availability_score(row),
            'feat_saved_boost': min(0.15, math.log1p(row.get('saved_by_recruiters_30d', 0)) * 0.05),
            'feat_search_appearance_boost': min(0.10, row.get('search_appearance_30d', 0) / 500.0),
            'feat_verified_search_skill': 1.03 if row.get('has_verified_search_skill') else 1.0,
            'feat_architect_no_coding': feat_architect_no_coding,
            'feat_wrapper_ai_only': feat_wrapper_ai_only,
            'contradiction_score': compute_contradiction_score(row)
        }
        features.append(feat)
        
    logging.info("Writing features parquet")
    feat_df = pl.DataFrame(features)
    
    final_df = df.join(feat_df, on='candidate_id', how='left')
    final_df.write_parquet(output_file)
    logging.info("Feature engineering complete.")

if __name__ == "__main__":
    main()

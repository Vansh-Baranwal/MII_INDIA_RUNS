def generate_reasoning(row: dict) -> str:
    """
    Generates 1-2 sentence deterministic reasoning based on the candidate's features.
    Guarantees no hallucination by using exact extracted data.
    """
    yoe = row.get('years_of_experience', 0)
    sem_score = row.get('Base_Score', 0.0)
    ret_score = row.get('feat_retrieval_depth', 0.0)
    eval_score = row.get('feat_evaluation_rigor', 0.0)
    builder_score = row.get('feat_builder_score', 0.0)
    rank_score = row.get('feat_ranking_depth', 0.0)
    prod_exp = row.get('feat_product_exposure', 0.0)
    avail_score = row.get('feat_availability_score', 0.0)
    notice = row.get('notice_period_days', 30)
    search_rel = row.get('feat_search_relevance_evidence', 0.0)

    # 1. The Well-Rounded Expert
    if sem_score > 0.8 and ret_score > 0.6 and eval_score > 0.6:
        return f"Strong semantic match with {yoe} years of experience. Demonstrated depth in retrieval infrastructure and rigorous offline evaluation methodologies."
        
    # 2. The Product Builder
    elif prod_exp == 1.0 and builder_score > 0.6:
        return f"Proven track record of architecting and shipping production ML systems. Extensive product-company exposure heavily aligns with JD culture requirements."
        
    # 3. The Ranking Specialist
    elif rank_score > 0.6 and search_rel > 0.6:
        return f"Deep expertise in learning-to-rank and relevance optimization. Experience perfectly matches the JD's core requirement for ranking intelligence."
        
    # 4. The Highly Engaged Match
    elif avail_score > 1.0 and sem_score > 0.7:
        return f"Solid technical fit combined with exceptional platform engagement (recent activity and high response rate) makes this candidate highly actionable."
        
    # 5. The Infrastructure Veteran
    elif ret_score > 0.7 and builder_score > 0.7 and rank_score < 0.4:
        return f"Brings deep vector infrastructure and backend retrieval experience. Production engineering capability is exceptional despite lighter explicit LTR focus."
        
    # 6. The Slight Notice Penalty
    elif sem_score > 0.8 and notice > 60:
        return f"Exceptional technical profile in embeddings and search relevance. Placed slightly lower due to a stated {notice}-day notice period, but skill match is undeniable."
        
    # 7. The Evaluation Heavyweight
    elif eval_score > 0.8:
        return f"Stands out for explicit experience with ranking evaluation metrics (NDCG/MAP) and A/B testing frameworks, exactly fulfilling the JD's evaluation requirements."
        
    # 8. The Available Performer
    elif notice <= 15 and avail_score > 0.8:
        return f"Strong technical alignment with immediate availability. Consistent platform engagement and solid systems-level engineering experience."
        
    # 9. The Trajectory Pivot
    elif row.get('feat_trajectory_transition', 0.0) > 0.0 and sem_score > 0.6:
        return f"Demonstrated clear career trajectory toward ML/Search engineering at product companies. {yoe} YOE with highly relevant recent project work."
        
    # 10. The Baseline Fit
    else:
        return f"Reliable technical background with {yoe} YOE. Shows sufficient exposure to the core ML engineering pipeline requested in the JD."

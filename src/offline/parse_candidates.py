import polars as pl
import json
import gzip
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_jsonl_gz(file_path: Path) -> List[Dict[str, Any]]:
    """Reads a gzipped JSONL file and yields parsed dictionaries."""
    logging.info(f"Reading candidates from {file_path}")
    if not file_path.exists():
        # Fallback to look at the parent directory just in case
        fallback = file_path.parent.parent.parent / file_path.name
        if fallback.exists():
            file_path = fallback
        else:
            raise FileNotFoundError(f"Could not find {file_path}")
    
    parsed_data = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                    parsed_data.append(extract_features(candidate))
                except json.JSONDecodeError as e:
                    logging.warning(f"Error decoding line {i}: {e}")
    except Exception as e:
         logging.error(f"Failed reading {file_path}: {e}")
         raise
    logging.info(f"Successfully parsed {len(parsed_data)} candidates.")
    return parsed_data

def extract_features(c: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens the candidate dictionary into a flat schema suitable for Polars/Parquet."""
    profile = c.get('profile', {})
    signals = c.get('redrob_signals', {})
    
    # Flatten Career History into a single string for text matching
    career = c.get('career_history', [])
    # Sort career by date, most recent first (assume start_date exists and can be sorted loosely)
    career.sort(key=lambda x: x.get('start_date', ''), reverse=True)
    
    career_texts = []
    total_duration = 0
    for job in career:
        title = job.get('title', '')
        desc = job.get('description', '')
        career_texts.append(f"{title}: {desc}")
        total_duration += job.get('duration_months', 0)
        
    recent_role_title = career[0].get('title', '') if career else ''
    recent_role_desc = career[0].get('description', '') if career else ''
    
    if len(career) >= 2:
        last_two_texts = f"{recent_role_title} - {recent_role_desc} | {career[1].get('title', '')} - {career[1].get('description', '')}"
    else:
        last_two_texts = f"{recent_role_title} - {recent_role_desc}"

    # Extract Skills
    skills = c.get('skills', [])
    skills_text = ", ".join([s.get('name', '') for s in skills])
    
    # Extract Education
    education = c.get('education', [])
    grad_year = max([ed.get('end_year', 0) for ed in education]) if education else 0

    return {
        'candidate_id': c.get('candidate_id'),
        'years_of_experience': profile.get('years_of_experience', 0),
        'current_title': profile.get('current_title', ''),
        'company_size': profile.get('current_company_size', ''),
        'industry': profile.get('current_industry', ''),
        'summary': profile.get('summary', ''),
        
        # Text block representations
        'recent_role_text': f"Title: {recent_role_title}. Role: {recent_role_desc}. Skills: {skills_text[:100]}",
        'last_two_roles_text': last_two_texts,
        'full_profile_text': f"Summary: {profile.get('summary', '')}. Experience: {' | '.join(career_texts)}. Skills: {skills_text}",
        
        # Honeypot checks
        'total_duration_months': total_duration,
        'grad_year': grad_year,
        'expected_salary_min': signals.get('expected_salary_range_inr_lpa', {}).get('min', 0),
        'expected_salary_max': signals.get('expected_salary_range_inr_lpa', {}).get('max', 0),
        
        # Behavioral signals
        'recruiter_response_rate': signals.get('recruiter_response_rate', 0.0),
        'last_active_date': signals.get('last_active_date', ''),
        'notice_period_days': signals.get('notice_period_days', 30), # default 30 if missing
        'github_activity_score': signals.get('github_activity_score', 0),
        'profile_views_received_30d': signals.get('profile_views_received_30d', 0),
        'open_to_work_flag': signals.get('open_to_work_flag', False),
        'interview_completion_rate': signals.get('interview_completion_rate', 0.0),
        
        # Dump career history JSON string for feature engineering step
        'career_history_json': json.dumps(career),
        'skills_json': json.dumps(skills)
    }

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    input_file = base_dir / 'candidates.jsonl.gz'
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_file = artifacts_dir / 'parsed_candidates.parquet'
    
    parsed_data = parse_jsonl_gz(input_file)
    
    logging.info("Converting to Polars DataFrame")
    df = pl.DataFrame(parsed_data)
    
    logging.info(f"Writing parquet to {output_file}")
    df.write_parquet(output_file)
    logging.info("Parsing complete.")

if __name__ == "__main__":
    main()

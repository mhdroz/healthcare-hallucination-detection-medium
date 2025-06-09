import numpy as np
import config as cfg




    

def _sanitize_numpy_types(data):
    """Recursively sanitize NumPy numeric types for JSON serialization."""
    if isinstance(data, dict):
        return {k: _sanitize_numpy_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_numpy_types(i) for i in data]
    if isinstance(data, (np.float32, np.float64)):
        return float(data)
    if isinstance(data, np.integer):
        return int(data)
    return data

def get_safety_interpretations(safety_result):
    """Get human-readable interpretations of safety scores."""
    interpretations = {}
    attr_score = safety_result.get('attribution_score', 0)
    if attr_score >= 0.7:
        interpretations['attribution'] = "Excellent - answer is well-grounded in sources"
    elif attr_score >= 0.6:
        interpretations['attribution'] = "Good - answer is mostly supported by sources"
    elif attr_score >= 0.4:
        interpretations['attribution'] = "Fair - some parts may lack source support"
    else:
        interpretations['attribution'] = "Poor - answer may contain unsupported claims"
    
    cons_score = safety_result.get('consistency_score', 0)
    if cons_score >= 0.8:
        interpretations['consistency'] = "High - very stable responses"
    elif cons_score >= 0.6:
        interpretations['consistency'] = "Good - mostly consistent responses"
    else:
        interpretations['consistency'] = "Low - responses vary significantly"
    
    entropy = safety_result.get('semantic_entropy', 0)
    if entropy < 1.0:
        interpretations['entropy'] = "Low uncertainty - confident answer"
    elif entropy < 2.0:
        interpretations['entropy'] = "Medium uncertainty - review recommended"
    else:
        interpretations['entropy'] = "High uncertainty - likely hallucination"
    
    return interpretations

def format_safety_response(safety_result):
    """Format safety check results for the frontend."""
    response = {
        'question': safety_result.get('question'),
        'answer': safety_result.get('answer'),
        'confidence': safety_result.get('confidence'),
        'safety_score': safety_result.get('safety_score'),
        'max_safety_score': safety_result.get('max_safety_score'),
        'source_chunks': safety_result.get('source_chunks', []),
        'attribution_score': safety_result.get('attribution_score', 0.0),
        'consistency_score': safety_result.get('consistency_score', 0.0),
        'semantic_entropy': safety_result.get('semantic_entropy', 0.0),
        'weak_sentences': safety_result.get('weak_sentences', []),
        'has_weak_sentences': len(safety_result.get('weak_sentences', [])) > 0,
        'multi_stage_used': safety_result.get('use_multi_stage', False),
        'fact_check_enabled': safety_result.get('external_fact_check_enabled', False)
    }
    
    fact_check_result = safety_result.get('fact_check_result')
    if fact_check_result and not fact_check_result.get('error'):
        sanitized_interpretation = _sanitize_numpy_types(fact_check_result.get('external_interpretation', {}))
        response['external_validation'] = {
            'score': float(fact_check_result.get('combined_score', 0.0)),
            'num_sources': fact_check_result.get('num_external_sources', 0),
            'query_used': fact_check_result.get('query_used', ''),
            'interpretation': sanitized_interpretation,
            'recommendation': fact_check_result.get('recommendation', '')
        }
    else:
        response['external_validation'] = None
    
    response['safety_interpretations'] = get_safety_interpretations(safety_result)
    
    return response 



def debug_print(*args, **kwargs):
    if cfg.DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)
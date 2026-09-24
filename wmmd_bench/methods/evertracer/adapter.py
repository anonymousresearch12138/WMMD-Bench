"""Benchmark member-oriented aggregation, recovered without upstream attack code."""
from __future__ import annotations

def empirical_fsr(member_scores: list[float], nonmember_scores: list[float], fpr_limit: float) -> dict[str, float]:
    """Maximum attainable TPR among thresholds whose empirical FPR is <= limit."""
    members = [float(x) for x in member_scores]
    nonmembers = [float(x) for x in nonmember_scores]
    if not members or not nonmembers:
        raise ValueError('member and nonmember score sets must be non-empty')
    thresholds = [float('inf'), *sorted(set(members + nonmembers), reverse=True)]
    eligible = []
    for threshold in thresholds:
        tpr = sum((score >= threshold for score in members)) / len(members)
        fpr = sum((score >= threshold for score in nonmembers)) / len(nonmembers)
        if fpr <= fpr_limit + 1e-12:
            eligible.append((tpr, threshold, fpr))
    best_tpr = max((row[0] for row in eligible))
    tpr, threshold, fpr = min((row for row in eligible if row[0] == best_tpr), key=lambda row: row[1])
    wins = sum((m > n for m in members for n in nonmembers))
    ties = sum((m == n for m in members for n in nonmembers))
    auc = (wins + 0.5 * ties) / (len(members) * len(nonmembers))
    return {'auc': float(auc), 'fsr': float(tpr), 'threshold': float(threshold), 'fpr': float(fpr), 'tpr': float(tpr), 'positive_count': len(members), 'negative_count': len(nonmembers)}

def corrected_detector_metrics(scores: list[dict[str, Any]], fpr_limit: float) -> dict[str, Any]:
    """Aggregate saved EverTracer scores using official and equivalent member directions."""
    members = [float(row['calibrated_score']) for row in scores if row['subset'] == 'dtr']
    nonmembers = [float(row['calibrated_score']) for row in scores if row['subset'] == 'dunseen']
    if not members or not nonmembers:
        raise ValueError('saved scores must contain both dtr members and dunseen non-members')
    member_metrics = empirical_fsr([-score for score in members], [-score for score in nonmembers], fpr_limit)
    official_auc = (sum((nonmember > member for nonmember in nonmembers for member in members)) + 0.5 * sum((nonmember == member for nonmember in nonmembers for member in members))) / (len(members) * len(nonmembers))
    if abs(official_auc - member_metrics['auc']) > 1e-12:
        raise AssertionError('official and member-oriented AUC must be equivalent')
    return {'official_definition': {'member_label': 0, 'nonmember_label': 1, 'score': 'C = suspect_variation - reference_variation', 'positive_threshold': 'C >= gamma predicts non-member'}, 'official_auc_nonmember_positive': float(official_auc), 'member_oriented_definition': {'member_label': 1, 'nonmember_label': 0, 'score': 'member_score = -C', 'positive_threshold': 'member_score >= gamma predicts member', 'note': 'Equivalent reparameterization of the official detector for benchmark readability.'}, 'member_oriented_auc': member_metrics['auc'], 'member_oriented_tpr_at_fpr_limit': member_metrics['tpr'], 'member_oriented_fpr': member_metrics['fpr'], 'member_oriented_threshold': member_metrics['threshold'], 'fpr_limit': float(fpr_limit), 'member_count': len(members), 'nonmember_count': len(nonmembers)}

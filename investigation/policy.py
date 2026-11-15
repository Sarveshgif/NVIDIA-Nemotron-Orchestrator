"""Independent evidence verdict plus an explicitly separate prediction policy."""
from . import VERSION


def routing(request, rf, settings):
    reasons = []
    if rf is None:
        reasons.append('RF_PREDICTION_UNAVAILABLE')
    else:
        if max(rf.p_silage, rf.p_grain) < settings.threshold:
            reasons.append('RF_SCORE_BELOW_PROVISIONAL_THRESHOLD')
        if rf.review_band and rf.review_band[0] <= rf.p_silage < rf.review_band[1]:
            reasons.append('WITHIN_ORIGINAL_MODEL_REVIEW_BAND')
        assigned_score = rf.p_silage if rf.predicted_label == 'SILAGE' else rf.p_grain
        if assigned_score < settings.threshold:
            reasons.append('ORIGINAL_MODEL_LABEL_HAS_LOW_CLASS_SCORE')
        if rf.provenance == 'saved_artifact' and not rf.barc_fitted and request.geography.country != 'NL':
            reasons.append('NETHERLANDS_MODEL_OUTSIDE_VALIDATED_GEOGRAPHY')
    if request.quality.observations_adequate is not True:
        reasons.append('OBSERVATION_QUALITY_INADEQUATE_OR_UNKNOWN')
    if request.quality.geographic_applicability != 'validated':
        reasons.append('GEOGRAPHIC_APPLICABILITY_UNVALIDATED_OR_UNKNOWN')
    reasons.extend(request.quality.flags)
    if request.physical_evidence.contradictory:
        reasons.append('EXISTING_PHYSICAL_EVIDENCE_CONFLICT')
    evidence_investigate = bool(reasons)
    investigate = evidence_investigate
    if settings.prediction_mode == 'always_predict':
        # Use the assigned-class score: author thresholds can differ from argmax.
        investigate = rf is None or assigned_score < settings.threshold
    return {'investigate': investigate, 'evidence_investigate': evidence_investigate,
            'reasons': sorted(set(reasons)), 'prediction_mode': settings.prediction_mode,
            'score_threshold': settings.threshold, 'threshold_provisional': True}


def physical_classification(request, weather, flags):
    """One shared field-evidence rule for decisions and evidence assessments."""
    p = request.physical_evidence
    development = weather.data.get('rules', {}).get('development_timing', {}).get('status') if weather else None
    risks = {'RAIN_MAY_CONFOUND_RADAR_OR_SURFACE', 'FROST_COULD_EXPLAIN_BROWNING',
             'RAINFALL_DEFICIT_COULD_EXPLAIN_BROWNING'} & set(flags)
    quality_ok = (request.quality.observations_adequate is True and not request.quality.flags
                  and request.quality.geographic_applicability == 'validated')
    if quality_ok and p.removal_confirmed is True and p.corroborated and p.source_ids and request.harvest and not p.contradictory and not risks:
        if p.canopy_condition == 'green' and development == 'BEFORE_PLAUSIBLE_GRAIN_MATURITY':
            return 'SILAGE'
        if p.canopy_condition == 'dry' and p.standing_drydown_confirmed is True and development == 'AT_OR_AFTER_PLAUSIBLE_GRAIN_MATURITY':
            return 'GRAIN'
    return None


def evaluate(request, rf, route, evidence):
    by_tool = {r.tool: r for r in evidence}
    flags, missing = set(request.quality.flags), []
    for result in evidence:
        flags.update(result.quality_flags)
        missing.extend(result.missing_inputs)
        missing.extend(result.errors)
    label, source, strength = 'REVIEW', 'review', 'insufficient'
    explanation = 'Available evidence does not resolve the RF uncertainty; contextual evidence alone cannot determine harvest use.'
    used = [r.evidence_id for r in evidence]
    if rf and rf.barc_fitted:
        flags.add('BARC_LABELS_USED_IN_MODEL_RECALIBRATION')
    if not route.get('evidence_investigate', route['investigate']) and rf:
        label, source, strength = rf.predicted_label, 'rf', 'model_only'
        explanation = 'Original RF decision retained: configured score gate and supplied quality/applicability checks passed. No investigation was required.'
    else:
        web = by_tool.get('search_public_harvest_evidence')
        records = web.data.get('verified_actual_harvest_records', []) if web else []
        classes = {r['harvest_use'] for r in records}
        if len(classes) > 1:
            flags.add('CONFLICTING_VERIFIED_HARVEST_RECORDS')
            explanation = 'Reviewed exact-field/season harvest records conflict; human review is required.'
            used.append('web')
        elif len(classes) == 1:
            label, source, strength = classes.pop(), 'verified_public_record', 'confirmed'
            explanation = 'A reviewed public record identifies actual harvest use for this field and season.'
            used.append('web')
        else:
            weather = by_tool.get('get_daymet_weather')
            p = request.physical_evidence
            inferred = physical_classification(request, weather, flags)
            if inferred:
                label = inferred
                source, strength = 'deterministic_existing_evidence', 'likely'
                used.extend(['weather', *p.source_ids])
                explanation = 'Existing corroborated physical-removal evidence and documented development timing support this likely classification; no additional imagery was retrieved.'
            if p.contradictory:
                flags.add('EXISTING_PHYSICAL_EVIDENCE_CONFLICT')
    return {'policy_version': VERSION, 'final_label': label, 'decision_source': source,
            'final_probability': None, 'evidence_strength': strength, 'evidence_ids': sorted(set(used)),
            'exception_flags': sorted(flags), 'missing_evidence': sorted(set(missing)), 'explanation': explanation}


def prediction_decision(decision, rf, route, hypothesis, agent_status, settings):
    """Always emit a binary prediction when RF exists; retain the evidence verdict.

    Tentative labels and failure fallbacks do not become verified evidence, and no
    fused probability or artificial lower confidence bound is invented.
    """
    result = dict(decision)
    result['prediction_mode'] = settings.prediction_mode
    result['evidence_decision'] = dict(decision)
    result['rf_assigned_class_score'] = (
        rf.p_silage if rf.predicted_label == 'SILAGE' else rf.p_grain) if rf else None
    result['prediction_is_evidence_supported'] = decision['decision_source'] in (
        'verified_public_record', 'deterministic_existing_evidence')
    result['nemotron_used_for_prediction'] = False
    if settings.prediction_mode != 'always_predict' or decision['final_label'] != 'REVIEW':
        result['prediction_status'] = 'evidence_supported' if result['prediction_is_evidence_supported'] else 'model_only' if decision['final_label'] != 'REVIEW' else 'unresolved'
        return result
    if rf is None:
        result['prediction_status'] = 'unavailable_no_rf'
        return result
    score = result['rf_assigned_class_score']
    low_score = score < settings.threshold
    use_nemotron = (low_score and agent_status == 'COMPLETED' and hypothesis
                    and hypothesis.get('citations_validated') is True
                    and hypothesis.get('label') in ('GRAIN', 'SILAGE'))
    if use_nemotron:
        label, source = hypothesis['label'], 'nemotron_best_effort'
        text = 'Tentative Nemotron prediction after low-score RF investigation; field evidence is insufficient for confirmation.'
    else:
        label = rf.predicted_label
        source = 'rf_fallback' if low_score else 'rf_score_gate'
        text = ('Original RF label retained as a fallback because Nemotron did not supply a valid completed binary prediction.'
                if low_score else 'Original RF label retained at or above the configured score gate; independent evidence limitations remain.')
    flags = set(result['exception_flags'])
    flags.add('PREDICTION_WITHOUT_SUFFICIENT_FIELD_EVIDENCE')
    if low_score and not use_nemotron:
        flags.add('NEMOTRON_BINARY_PREDICTION_UNAVAILABLE_RF_FALLBACK')
    if use_nemotron and label != rf.predicted_label:
        flags.add('TENTATIVE_NEMOTRON_OVERRIDE_UNVERIFIED')
    result.update(final_label=label, decision_source=source, final_probability=None,
                  evidence_strength=decision['evidence_strength'], exception_flags=sorted(flags),
                  explanation=text, prediction_status='tentative',
                  nemotron_used_for_prediction=bool(use_nemotron))
    return result

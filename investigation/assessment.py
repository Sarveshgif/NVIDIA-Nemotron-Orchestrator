"""Tool-backed findings and cited, non-authoritative Nemotron hypotheses.

Measurements and explanations are constructed from existing inputs/tool outputs.
The model can select a hypothesis and cite findings, but cannot submit new facts,
change evidence relationships or update RF probabilities.
"""
from hashlib import sha256

from .schemas import NemotronHypothesis
from .policy import physical_classification


RELATIONS = ('SUPPORTS', 'CONTRADICTS', 'DOES_NOT_RESOLVE')
REASONS = {
    'SATELLITE_TRAJECTORY': 'Dated field-masked optical/radar trajectories are available; they do not alone confirm harvest use.',
    'RF_HYPOTHESIS': 'The original RF label is a hypothesis, not independent verification.',
    'RF_INPUT_LIMITATIONS': 'Input quality or geographic applicability limits confidence in the RF result.',
    'CONTEXT_ONLY': 'Contextual evidence alone cannot establish field harvest use.',
    'MISSING_EVENT_INPUTS': 'Required event dates, planting information or documented maturity information are missing.',
    'FIELD_HARVEST_RECORD': 'A reviewed field-and-season actual-harvest record is available.',
    'CORROBORATED_PHYSICAL_EVIDENCE': 'Existing physical-removal evidence is corroborated.',
    'DEVELOPMENT_TIMING': 'Documented development timing is available for interpretation.',
    'CONFLICTING_EVIDENCE': 'Field evidence contains unresolved contradictions.',
    'NO_USABLE_EVIDENCE': 'No usable evidence is available to resolve harvest use.',
}


def build_assessment(request, rf, evidence):
    findings = []
    by_tool = {r.tool: r for r in evidence}
    flags = set(request.quality.flags)
    for result in evidence:
        flags.update(result.quality_flags)
    weather = by_tool.get('get_daymet_weather')
    physical_label = physical_classification(request, weather, flags)

    def add(finding_id, category, explanation, *, sources=None, dates=None, scope='field',
            status='OK', measurements=None, limitations=None, missing=None,
            inferred_label=None, decision_eligible=False, group=None, tool=None):
        relation = 'DOES_NOT_RESOLVE'
        if inferred_label and decision_eligible and rf:
            relation = 'SUPPORTS' if inferred_label == rf.predicted_label else 'CONTRADICTS'
        findings.append({
            'finding_id': finding_id, 'field_id': request.field_id, 'year': request.year,
            'reporting_cutoff': request.reporting_cutoff.isoformat(), 'category': category,
            'source_references': sources or [], 'relevant_dates': dates or {},
            'association': scope, 'status': status, 'relationship_to_rf': relation,
            'explanation': explanation, 'measurements': measurements or {},
            'limitations': limitations or [], 'missing_inputs': missing or [],
            'inferred_harvest_use': inferred_label, 'eligible_for_final_decision': decision_eligible,
            'independence_group': group or finding_id, 'tool': tool,
        })

    if rf:
        add('rf:prediction', 'rf', 'Original RF result, retained without probability adjustment.',
            sources=[{'source_id': rf.model_version, 'artifact_sha256': rf.artifact_sha256}],
            measurements=rf.model_dump(mode='json'),
            limitations=['A model score is not independent field evidence; calibration scope is preserved.'],
            group='rf_model')
    add('input:quality', 'quality', 'Supplied observation-quality and geographic-applicability checks.',
        status='PARTIAL' if request.quality.flags or request.quality.observations_adequate is not True
        or request.quality.geographic_applicability != 'validated' else 'OK',
        measurements=request.quality.model_dump(mode='json'),
        limitations=['Quality warnings cannot be cleared by a model hypothesis.'], group='existing_field_trajectory')
    p = request.physical_evidence
    missing_physical = []
    if p.removal_confirmed is not True:
        missing_physical.append('confirmed physical crop removal')
    if not p.corroborated or not p.source_ids:
        missing_physical.append('corroborating source references')
    if p.canopy_condition == 'uncertain':
        missing_physical.append('pre-removal canopy condition')
    add('existing:physical', 'physical', 'Existing field evidence; no additional imagery is fetched.',
        sources=[{'source_id': v} for v in p.source_ids],
        dates={'harvest_interval': request.harvest.model_dump(mode='json') if request.harvest else None},
        status='PARTIAL' if missing_physical else 'OK', measurements=p.model_dump(mode='json'),
        missing=missing_physical, inferred_label=physical_label,
        decision_eligible=bool(physical_label), group='existing_field_trajectory')
    for i, summary in enumerate(request.feature_summary):
        add(f'existing:feature:{i}', 'existing_feature', f'Supplied existing {summary.metric} measurement.',
            sources=[{'source_id': v} for v in summary.source_ids],
            dates={'observed_between': summary.observed_between.model_dump(mode='json') if summary.observed_between else None},
            measurements={'metric': summary.metric, 'value': summary.value, 'units': summary.units},
            limitations=['A single index or radar measurement is not proof of harvest use.'],
            group='existing_field_trajectory')

    for result in evidence:
        sources = result.sources
        if result.tool == 'get_daymet_weather':
            rules = result.data.get('rules', {})
            for name in ('development_timing', 'rain_context', 'frost_risk', 'drought_context', 'drydown_context'):
                rule = rules.get(name, {'status': 'UNKNOWN', 'reason': 'Weather rule unavailable'})
                explanation = {
                    'development_timing': 'Heat accumulation supports development timing only with supplied planting dates and documented local maturity information.',
                    'rain_context': 'Rain may explain radar changes; no rain does not exclude irrigation.',
                    'frost_risk': 'Freezing conditions may explain browning; they do not prove crop death or harvest.',
                    'drought_context': 'Low rainfall is an alternative explanation for browning; irrigation and soil moisture remain unknown.',
                    'drydown_context': 'Weather can explain faster or slower drying; it does not identify harvest product.',
                }[name]
                add('weather:'+name, 'weather', explanation, sources=sources,
                    dates={'planting': request.planting.model_dump(mode='json') if request.planting else None,
                           'harvest_interval': request.harvest.model_dump(mode='json') if request.harvest else None,
                           'greenness_decline_date': request.greenness_decline_date.isoformat() if request.greenness_decline_date else None,
                           'radar_observation_dates': [d.isoformat() for d in request.radar_dates],
                           'standing_drydown': request.standing_drydown.model_dump(mode='json') if request.standing_drydown else None},
                    scope='gridded_weather', status=rule.get('status', 'UNKNOWN'), measurements=rule,
                    missing=result.missing_inputs, limitations=result.errors + ['Weather alone does not label harvest use.'],
                    inferred_label=physical_label if name == 'development_timing' else None,
                    decision_eligible=bool(physical_label) and name == 'development_timing',
                    group='daymet_location_and_season', tool=result.tool)
            history = result.data.get('historical_comparison', {})
            for window in history.get('windows', []):
                add('weather:historical:'+str(window['window_days'])+'day', 'weather',
                    'Observed calendar-window weather compared with matching prior-year windows; window GDD is not crop age or maturity.',
                    sources=[s for s in sources if 'daymet' in s.get('url', '')], scope='gridded_weather',
                    dates={'window_end': history.get('anchor_date'), 'baseline_period': history.get('baseline_period')},
                    status=window['status'], measurements=window,
                    limitations=['Planting and hybrid maturity remain unknown unless explicitly supplied.'],
                    group='daymet_location_and_season', tool=result.tool)
            regional = result.data.get('nass_regional_context', {})
            if regional.get('status') not in (None, 'DISABLED'):
                add('nass:regional_history', 'agriculture', 'USDA NASS state crop progress and prior-year harvested acreage are regional context, not field truth.',
                    sources=regional.get('sources', []), scope='state', status=regional.get('status', 'UNKNOWN'),
                    dates={'context_cutoff': regional.get('context_cutoff')},
                    measurements={k:v for k,v in regional.items() if k not in ('sources', 'errors')},
                    limitations=regional.get('errors', [])+['Current API estimates are not a guaranteed archived publication vintage. No field label or actual planting date is inferred.'],
                    group='nass_state_and_season', tool=result.tool)
        elif result.tool == 'get_sentinel_spatial_evidence':
            for sensor in ('s2', 's1'):
                sensor_data=result.data.get(sensor, {})
                add('satellite:'+sensor, 'satellite',
                    'Field-masked Sentinel-2 optical indices and sub-field variation.' if sensor == 's2' else 'Field-masked Sentinel-1 RTC radar, using one relative orbit and pass.',
                    sources=[s for s in result.sources if s.get('collection', '').startswith('sentinel-2' if sensor == 's2' else 'sentinel-1')],
                    dates={'window':result.data.get('window')}, scope='field', status=sensor_data.get('status','UNKNOWN'),
                    measurements=sensor_data, limitations=result.errors+['Satellite changes are not confirmed crop removal or verified harvest use.'],
                    missing=result.missing_inputs, group='sentinel_'+sensor+'_field_trajectory', tool=result.tool)
            rule=result.data.get('rulebook_v4')
            if rule:
                add('satellite:rulebook_v4', 'satellite',
                    'Fixed first-match v4 interpretation of field-masked Sentinel observations; advisory only.',
                    sources=result.sources, dates={'window':result.data.get('window')}, scope='field',
                    status='OK' if rule.get('label') in ('GRAIN','SILAGE') else 'UNKNOWN',
                    measurements={**rule, 'summary':{'usable_observations':max(
                        result.data.get('s2',{}).get('summary',{}).get('usable_observations',0),
                        result.data.get('s1',{}).get('summary',{}).get('usable_observations',0))}},
                    limitations=['Rule output is not a verified harvest record or calibrated probability.'],
                    group='sentinel_field_trajectory', tool=result.tool)
        elif result.tool == 'get_elevation_context':
            add('elevation:context', 'elevation', 'Elevation helps assess whether gridded weather is representative.',
                sources=sources, scope='field', status=result.status, measurements=result.data,
                missing=result.missing_inputs, limitations=result.errors +
                ['Higher elevation does not automatically imply later harvest; no extra lapse-rate adjustment is applied.'],
                group='usgs_3dep', tool=result.tool)
        elif result.tool in ('search_public_harvest_evidence', 'get_agricultural_context'):
            documents = result.data.get('candidate_documents', []) + result.data.get('regional_report_candidates', [])
            for doc in documents:
                # The same report reached through two tools remains one finding/group.
                identity = doc.get('url', '')
                finding_id = 'public:'+sha256(identity.encode()).hexdigest()[:16]
                if any(f['finding_id'] == finding_id for f in findings):
                    continue
                add(finding_id, 'public_document', 'Public document is contextual until exact field/season harvest use is independently verified.',
                    sources=[{k: doc.get(k) for k in ('source_id', 'url', 'content_sha256', 'published_on')}],
                    dates={'published_on': doc.get('published_on'), 'declared_year': doc.get('year')},
                    scope=doc.get('scope', 'unverified'), status='UNKNOWN' if doc.get('fetch_error') else 'PARTIAL',
                    measurements={'title': doc.get('title'), 'text_excerpt': doc.get('text_excerpt'),
                                  'publication_date_basis': doc.get('publication_date_basis', 'unverified')},
                    limitations=[v for v in (doc.get('limitation'), doc.get('fetch_error'),
                                  'Declared field selectors and publication dates are not automatically verified.') if v],
                    group='public_document:'+(doc.get('content_sha256') or sha256(identity.encode()).hexdigest()), tool=result.tool)
            if result.tool == 'search_public_harvest_evidence':
                for i, record in enumerate(result.data.get('verified_actual_harvest_records', [])):
                    label = record['harvest_use']
                    add(f'verified:harvest:{i}', 'verified_record', 'Externally reviewed public record matches actual harvest use for this field and season.',
                        sources=[{'url': record.get('source_url'), 'source_sha256': record.get('source_sha256')}],
                        dates={'published_on': record.get('published_on'), 'harvest': record.get('harvest')},
                        inferred_label=label, decision_eligible=True,
                        measurements={'harvest_use': label}, group='verified_record:'+(record.get('source_sha256') or str(i)), tool=result.tool)
                if not documents and not result.data.get('verified_actual_harvest_records'):
                    add('web:unavailable', 'public_document', 'No matching usable public harvest record was retrieved.',
                        sources=sources, status='UNKNOWN', missing=result.missing_inputs, limitations=result.errors,
                        scope='unknown', tool=result.tool)
            else:
                add('agriculture:context', 'agriculture', 'Aggregate acreage is regional context, never an individual-field label.',
                    sources=sources, dates={'crop_year': request.year}, scope='county' if request.geography.county_fips else 'state',
                    status=result.status, measurements={'acreage_records': result.data.get('acreage_records', [])},
                    missing=result.missing_inputs, limitations=result.errors +
                    ['Do not force field predictions to match acreage totals; revisions may postdate the reporting cutoff.'],
                    group='nass_geography_and_year', tool=result.tool)
        else:
            add(result.evidence_id+':unknown', 'unavailable', 'Tool evidence unavailable.',
                sources=sources, status='UNKNOWN', missing=result.missing_inputs,
                limitations=result.errors, tool=result.tool)

    groups = {}
    for finding in findings:
        groups.setdefault(finding['independence_group'], []).append(finding['finding_id'])
    return {'rf_hypothesis': rf.predicted_label if rf else None, 'findings': findings,
            'relationship_counts': {relation: sum(f['relationship_to_rf'] == relation for f in findings) for relation in RELATIONS},
            'evidence_groups': groups,
            'note': 'Groups identify shared data provenance, not proven statistical independence. No vote-counting or probability fusion is applied.'}


def validate_hypothesis(proposal, assessment, request, rf, decision):
    proposal = NemotronHypothesis.model_validate(proposal)
    findings = {f['finding_id']: f for f in assessment['findings']}
    if any(key not in findings for key in proposal.evidence_ids):
        raise ValueError('Hypothesis must cite only received backend finding IDs')
    cited = [findings[key] for key in proposal.evidence_ids]
    categories = {f['category'] for f in cited}
    statuses = [f['measurements'].get('status') for f in cited if f['finding_id'] == 'weather:development_timing']
    labels = {f['inferred_harvest_use'] for f in findings.values() if f['eligible_for_final_decision']}
    allowed = {
        'SATELLITE_TRAJECTORY': any(f['category']=='satellite' and f['measurements'].get('summary',{}).get('usable_observations',0)>=2 for f in cited),
        'RF_HYPOTHESIS': 'rf' in categories and rf is not None,
        'RF_INPUT_LIMITATIONS': 'quality' in categories and (bool(request.quality.flags) or request.quality.observations_adequate is not True or request.quality.geographic_applicability != 'validated'),
        'CONTEXT_ONLY': bool(categories & {'weather', 'elevation', 'public_document', 'agriculture'}) and not labels,
        'MISSING_EVENT_INPUTS': bool(categories & {'physical', 'quality', 'weather'}) and (not request.planting or not request.harvest or not request.maturity),
        'FIELD_HARVEST_RECORD': 'verified_record' in categories,
        'CORROBORATED_PHYSICAL_EVIDENCE': 'physical' in categories and request.physical_evidence.removal_confirmed is True and request.physical_evidence.corroborated and bool(request.physical_evidence.source_ids),
        'DEVELOPMENT_TIMING': any(s in ('BEFORE_PLAUSIBLE_GRAIN_MATURITY', 'AT_OR_AFTER_PLAUSIBLE_GRAIN_MATURITY', 'MATURITY_RANGE_OVERLAP') for s in statuses),
        'CONFLICTING_EVIDENCE': (len(labels) > 1 and 'verified_record' in categories) or (request.physical_evidence.contradictory and 'physical' in categories),
        'NO_USABLE_EVIDENCE': rf is None and not labels and not any(f['status'] not in ('UNKNOWN', 'TIMING UNKNOWN') for f in findings.values() if f['category'] not in ('quality', 'physical')),
    }
    if any(not allowed[code] for code in proposal.reason_codes):
        raise ValueError('Hypothesis reason codes are not supported by the cited findings')
    eligible_cited = {f['inferred_harvest_use'] for f in cited if f['eligible_for_final_decision']}
    supported = proposal.label in eligible_cited and proposal.label == decision['final_label'] and decision['decision_source'] in ('verified_public_record', 'deterministic_existing_evidence')
    warnings = []
    if not supported:
        warnings.append('HYPOTHESIS_IS_NOT_A_VALIDATED_FINAL_CLASSIFICATION')
    if proposal.label not in ('UNRESOLVED', rf.predicted_label if rf else None) and not supported:
        warnings.append('CONTEXT_CANNOT_JUSTIFY_OVERTURNING_RF')
    if len(eligible_cited) > 1:
        warnings.append('CONFLICTING_FIELD_EVIDENCE')
    return {**proposal.model_dump(mode='json'), 'proposed_evidence_strength': proposal.evidence_strength,
            'evidence_strength': proposal.evidence_strength if supported else 'insufficient' if proposal.label == 'UNRESOLVED' else 'limited',
            'citations_validated': True, 'supported_for_final_decision': supported,
            'validation_flags': warnings,
            'explanation': ' '.join(REASONS[code] for code in proposal.reason_codes),
            'authority': 'Qualitative Nemotron hypothesis only. The backend owns the final decision.'}

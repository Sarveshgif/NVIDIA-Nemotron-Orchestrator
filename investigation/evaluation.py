"""Offline reference-label join, never imported by inference, tools, or server."""
from collections import defaultdict
from math import fsum


def metrics(rows, prediction_key):
    classes = ('GRAIN','SILAGE')
    columns = ('GRAIN','SILAGE','REVIEW')
    matrix = {truth:{pred:0 for pred in columns} for truth in classes}
    for row in rows:
        matrix[row['reference_label']][row[prediction_key]] += 1
    n = len(rows)
    resolved = sum(matrix[c][p] for c in classes for p in classes)
    correct = sum(matrix[c][c] for c in classes)
    per_class = {}
    for label in classes:
        tp = matrix[label][label]
        predicted = sum(matrix[c][label] for c in classes)
        actual = sum(matrix[label].values())
        actual_resolved = sum(matrix[label][c] for c in classes)
        per_class[label] = {'support':actual, 'precision':tp/predicted if predicted else None,
                            'recall_all_eligible':tp/actual if actual else None,
                            'recall_resolved_cases':tp/actual_resolved if actual_resolved else None}
    return {'n':n,'confusion_matrix':matrix,'coverage':resolved/n if n else None,
            'review_rate':(n-resolved)/n if n else None,
            'accuracy_all_eligible':correct/n if n else None,
            'accuracy_resolved':correct/resolved if resolved else None,'per_class':per_class}


def compare(reports, references):
    refs = {}
    for ref in references:
        field_id = ref.get('field_id')
        year = int(ref['year'])
        if not str(field_id).startswith('BARC-') and ('objectid' in ref or isinstance(field_id,int)):
            field_id = f'BARC-{year}-OBJ{ref.get("objectid",field_id)}'
        label = str(ref.get('reference_label',ref.get('label',''))).upper()
        if label not in ('GRAIN','SILAGE'):
            continue
        key = (str(field_id),year)
        if key in refs:
            raise ValueError('Duplicate reference field/year')
        refs[key] = label
    seen, rows, excluded, contaminated = set(), [], [], []
    for report in reports:
        key = (report['field_id'],int(report['year']))
        if key in seen:
            raise ValueError('Duplicate prediction field/year')
        seen.add(key)
        if key not in refs or not report.get('rf'):
            excluded.append({'field_id':key[0],'year':key[1],'reason':'Missing matching reference or RF prediction'})
            continue
        if report['rf'].get('barc_fitted') or 'BARC' in report['rf'].get('calibration_scope','').upper():
            contaminated.append(key[0])
        rows.append({'field_id':key[0],'year':key[1],'reference_label':refs[key],
                     'rf_label':report['rf']['predicted_label'],'investigated_label':report['final_label'],
                     'elapsed_ms':report['elapsed_ms'],'investigated':report['routing']['investigate']})
    by_year = {}
    for year in sorted({r['year'] for r in rows}):
        subset = [r for r in rows if r['year'] == year]
        by_year[str(year)] = {'rf_only':metrics(subset,'rf_label'),'rf_plus_investigation':metrics(subset,'investigated_label')}
    investigation_times = [r['elapsed_ms'] for r in rows if r['investigated']]
    return {'evaluation_status':'development_only_not_independent_validation',
            'note':'Previously inspected BARC data is development data. No threshold tuning or training was performed here. Repeated field-years are not independent fields.',
            'barc_recalibration_contamination':contaminated, 'same_eligible_field_years':len(rows),
            'rf_only':metrics(rows,'rf_label'),'rf_plus_investigation':metrics(rows,'investigated_label'),
            'by_year':by_year,'excluded':excluded,
            'unmatched_reference_count':sum(k not in seen for k in refs),
            'mean_investigation_elapsed_ms':fsum(investigation_times)/len(investigation_times) if investigation_times else None,
            'rows':rows}

"""Fixed v4 Sentinel interpretation for a single field-year.

This is an advisory observation rule, not a verified harvest record. It never
reads RF predictions or reference labels, and missing data never becomes grain.
"""
from datetime import date

VERSION = 'sentinel-v4-fixed-1'


def _day(value):
    return date.fromisoformat(value[:10])


def _ndvi(row):
    value = (row.get('indices') or {}).get('ndvi') or {}
    return value.get('mean')


def assess_v4(s2, s1, year):
    optical = sorted([( _day(x['date']), float(_ndvi(x)), x['date']) for x in s2
                      if x.get('usable') and _ndvi(x) is not None], key=lambda x: x[0])
    radar = sorted([(_day(x['date']), float(x['vh_db']), float(x['vv_db']), x['date']) for x in s1
                    if x.get('usable') and x.get('vh_db') is not None and x.get('vv_db') is not None], key=lambda x: x[0])
    limits = {'peak_window': ['06-01','09-30'], 'radar_vh_drop_strong_db': 2.0,
              'radar_vv_wet_rise_db': 1.0, 'green_ndvi': 0.55, 'dry_ndvi': 0.50,
              'radar_optical_gap_days': 20, 'optical_drop_strong': 0.20,
              'optical_drop_likely': 0.15, 'optical_gap_days': 15,
              'bare_ndvi': 0.35, 'rebound_window_days': 30, 'max_rebound': 0.10,
              'standing_radar_change_db': 0.5, 'stood_brown_images': 3,
              'stood_brown_span_days': 21, 'early_days': 45,
              'late_days': 70, 'radar_vh_drop_lean_db': 1.5}
    def result(label, tier, step, reason, evidence=None):
        return {'version': VERSION, 'label': label, 'tier': tier, 'step': step,
                'reason': reason, 'evidence': evidence or {}, 'thresholds': limits,
                'advisory_only': True, 'optical_dates': len(optical), 'radar_dates': len(radar)}
    in_peak = [x for x in optical if date(year,6,1) <= x[0] <= date(year,9,30)]
    if not in_peak:
        return result('UNRESOLVED', None, 0, 'No clear Sentinel-2 NDVI image from June 1 through September 30.')
    peak = max(in_peak, key=lambda x: x[1])
    peak_date = peak[0]
    def last_green_before(day):
        candidates = [x for x in optical if 0 < (day-x[0]).days <= 20]
        return candidates[-1] if candidates else None
    candidates = []
    for i in range(2, len(radar)-1):
        if radar[i][0] <= peak_date:
            continue
        before = radar[i-2:i]; after = radar[i:i+2]
        vh_drop = sum(x[1] for x in before)/2 - sum(x[1] for x in after)/2
        vv_rise = sum(x[2] for x in after)/2 - sum(x[2] for x in before)/2
        if vv_rise < 1.0:
            candidates.append((vh_drop, radar[i], vv_rise))
    strong = max((x for x in candidates if x[0] >= 2.0), default=None, key=lambda x:x[0])
    if strong:
        drop, event, vv_rise = strong
        prior = last_green_before(event[0])
        if prior and (prior[1] >= .55 or prior[1] <= .50):
            label = 'SILAGE' if prior[1] >= .55 else 'GRAIN'
            reason = (f'Sentinel-1 VH fell {drop:.2f} dB around {event[3]} without a wet-ground VV rise '
                      f'({vv_rise:+.2f} dB); clear Sentinel-2 NDVI was {prior[1]:.2f} on {prior[2]}.')
            return result(label, 'strong', 1, reason,
                          {'peak_date':peak[2], 'radar_date':event[3], 'vh_drop_db':drop,
                           'vv_rise_db':vv_rise, 'optical_date':prior[2], 'pre_ndvi':prior[1]})
    def corroborating_radar(start, end):
        before = [x for x in radar if 0 <= (start-x[0]).days <= 20]
        after = [x for x in radar if 0 <= (x[0]-end).days <= 20]
        if not before or not after:
            return None
        return abs(after[0][1]-before[-1][1])
    def optical_drops(min_drop):
        found=[]
        for before, after in zip(optical, optical[1:]):
            gap=(after[0]-before[0]).days
            if after[0] <= peak_date or gap <= 0 or gap > 15 or before[1]-after[1] < min_drop or after[1] > .35:
                continue
            later = next((x for x in optical if 0 < (x[0]-after[0]).days <= 30), None)
            if later is None or later[1]-after[1] > .10:
                continue
            radar_change = corroborating_radar(before[0], after[0])
            if radar_change is None or radar_change < .5:
                continue
            found.append((before, after, later, radar_change))
        return found
    strong_optical = optical_drops(.20)
    if strong_optical:
        before, after, later, radar_change = strong_optical[0]
        if before[1] >= .55 or before[1] <= .50:
            label = 'SILAGE' if before[1] >= .55 else 'GRAIN'
            return result(label, 'strong', 2,
                          f'Clear NDVI fell {before[1]-after[1]:.2f} from {before[2]} to {after[2]}, '
                          f'ended at {after[1]:.2f}, stayed low on {later[2]}, and radar VH changed {radar_change:.2f} dB.',
                          {'peak_date':peak[2], 'before_date':before[2], 'after_date':after[2],
                           'pre_ndvi':before[1], 'post_ndvi':after[1], 'radar_change_db':radar_change})
    brown = [x for x in optical if x[0] > peak_date and .30 <= x[1] <= .50]
    for i in range(len(brown)-2):
        span=(brown[i+2][0]-brown[i][0]).days
        if span >= 21:
            return result('GRAIN', 'likely', 3,
                          f'Three clear images from {brown[i][2]} through {brown[i+2][2]} span {span} days '
                          'with NDVI between 0.30 and 0.50 after the field peak.',
                          {'peak_date':peak[2], 'first_date':brown[i][2],
                           'third_date':brown[i+2][2], 'span_days':span})
    likely_optical = optical_drops(.15)
    if likely_optical:
        before, after, later, radar_change = likely_optical[0]
        age=(after[0]-peak_date).days
        if age <= 45 or age >= 70:
            label='SILAGE' if age <= 45 else 'GRAIN'
            return result(label, 'likely', 4,
                          f'An NDVI drop to {after[1]:.2f} ended {age} days after the {peak[2]} peak; '
                          f'no rebound was observed on {later[2]} and radar changed {radar_change:.2f} dB.',
                          {'peak_date':peak[2], 'before_date':before[2],
                           'after_date':after[2], 'days_after_peak':age})
    lean = max((x for x in candidates if x[0] >= 1.5 and last_green_before(x[1][0])
                and last_green_before(x[1][0])[1] >= .55), default=None, key=lambda x:x[0])
    if lean:
        drop,event,_=lean;prior=last_green_before(event[0])
        return result('SILAGE', 'lean', 5,
                      f'Sentinel-1 VH fell {drop:.2f} dB around {event[3]} while NDVI was '
                      f'{prior[1]:.2f} on {prior[2]}.',
                      {'peak_date':peak[2], 'radar_date':event[3], 'vh_drop_db':drop,
                       'optical_date':prior[2], 'pre_ndvi':prior[1]})
    return result('UNRESOLVED', None, 6,
                  'No v4 rule had complete supporting observations; missing clear dates, radar context, or an unambiguous harvest signal.',
                  {'peak_date':peak[2], 'peak_ndvi':peak[1]})

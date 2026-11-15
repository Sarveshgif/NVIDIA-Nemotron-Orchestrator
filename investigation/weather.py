from datetime import date, timedelta
import csv
import io
import math
import re

import numpy as np
from shapely.geometry import shape

from .schemas import ToolResult
from .http import DataUnavailable
from .historical_context import nass_regional_context, comparison_anchor

DAYMET = 'https://daymet.ornl.gov/single-pixel/api/data'
DAYMET_DOC = 'https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4.html'


def parse_daymet(text):
    lines = text.splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith('year,yday,')), None)
    if header_index is None:
        raise DataUnavailable('Daymet response has no daily CSV header')
    rows = []
    for row in csv.DictReader(io.StringIO('\n'.join(lines[header_index:]))):
        values = {}
        for key, value in row.items():
            name = key.split(' (')[0]
            try:
                number = float(value)
            except (ValueError, TypeError):
                number = float('nan')
            values[name] = number if math.isfinite(number) and number != -9999 else None
        if values.get('year') is None or values.get('yday') is None:
            continue
        year, doy = int(values['year']), int(values['yday'])
        if not 1 <= doy <= 365:
            raise DataUnavailable('Invalid Daymet day of year')
        # Feb 29 remains present; only Dec 31 is omitted in leap years.
        values['date'] = date(year, 1, 1) + timedelta(days=doy - 1)
        rows.append(values)
    if not rows:
        raise DataUnavailable('Daymet returned no weather observations')
    if len({r['date'] for r in rows}) != len(rows):
        raise DataUnavailable('Duplicate Daymet dates')
    elevation = re.search(r'^Elevation:\s*([-\d.]+) meters', text, re.M)
    xy = re.search(r'^X & Y on Lambert Conformal Conic:\s*([-\d.]+)\s+([-\d.]+)', text, re.M)
    tile = re.search(r'^Tile:\s*(\S+)', text, re.M)
    return rows, {'response_header': lines[:header_index],
                  'grid_elevation_m': float(elevation.group(1)) if elevation else None,
                  'requested_point_lcc_m': [float(xy.group(1)), float(xy.group(2))] if xy else None,
                  'tile': tile.group(1) if tile else None,
                  'grid_resolution_m': 1000,
                  'grid_cell_id': None,
                  'grid_note': 'Single-pixel service selects the cell; response supplies tile and requested-point coordinates, not an explicit cell index.'}


def corn_gdd_f(tmin_c, tmax_c):
    low, high = tmin_c * 9 / 5 + 32, tmax_c * 9 / 5 + 32
    # Standard modified corn GDD: upper Tmax=86F, lower Tmin=50F, negative daily sums zero.
    return max(0., (min(high, 86.) + max(low, 50.)) / 2 - 50.)


def approximate_vpd_kpa(row):
    if any(row.get(k) is None for k in ('tmin', 'tmax', 'vp')):
        return None
    saturation = lambda t: 0.6108 * math.exp(17.27 * t / (t + 237.3))
    return max(0., (saturation(row['tmin']) + saturation(row['tmax'])) / 2 - row['vp'] / 1000.)


def daily_window(rows, start, end, keys):
    by_date = {r['date']: r for r in rows}
    expected = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    complete = [by_date[d] for d in expected if d in by_date and all(by_date[d].get(k) is not None for k in keys)]
    return complete, [d.isoformat() for d in expected if d not in by_date or any(by_date[d].get(k) is None for k in keys)]


def weather_rules(request, rows, baseline, settings):
    unknown = lambda reason: {'status': 'UNKNOWN', 'reason': reason}
    rules = {name: unknown('Required dates, observations or maturity information unavailable')
             for name in ('development_timing', 'rain_context', 'frost_risk', 'drought_context', 'drydown_context')}
    flags = []
    if request.planting and request.harvest:
        totals, gaps = [], []
        for planting, harvest in ((request.planting.end, request.harvest.start), (request.planting.start, request.harvest.end)):
            part, missing = daily_window(rows, planting, harvest, ('tmin', 'tmax'))
            totals.append(sum(corn_gdd_f(r['tmin'], r['tmax']) for r in part))
            gaps.extend(missing)
        result = {'status': 'TIMING UNKNOWN', 'accumulated_gdd_f_range': totals if not gaps else None,
                  'units': 'Fahrenheit degree-days', 'missing_dates': sorted(set(gaps)),
                  'convention': 'Tmax capped at 86F; Tmin floored at 50F; negative daily GDD set to zero; planting and harvest endpoints included'}
        if not gaps and request.maturity:
            if totals[1] < request.maturity.minimum_grain_gdd_f:
                result['status'] = 'BEFORE_PLAUSIBLE_GRAIN_MATURITY'
            elif totals[0] >= request.maturity.maximum_grain_gdd_f:
                result['status'] = 'AT_OR_AFTER_PLAUSIBLE_GRAIN_MATURITY'
            else:
                result['status'] = 'MATURITY_RANGE_OVERLAP'
            result['maturity_assumption'] = request.maturity.model_dump(mode='json')
        else:
            result['reason'] = 'Missing temperature days or documented local maturity range'
        rules['development_timing'] = result
    anchors = sorted(set(request.radar_dates + ([request.harvest.start, request.harvest.end] if request.harvest else [])))
    if anchors:
        windows = []
        for anchor in anchors:
            part, missing = daily_window(rows, anchor-timedelta(days=2), anchor, ('prcp',))
            total = sum(r['prcp'] for r in part) if not missing else None
            wet = total >= settings.rain_mm if total is not None else None
            windows.append({'observation_date': anchor.isoformat(), 'precip_3day_mm': total, 'wet_flag': wet, 'missing_dates': missing})
        radar_windows = [w for w in windows if w['observation_date'] in {d.isoformat() for d in request.radar_dates}]
        wet_to_dry = any(a['wet_flag'] is True and b['wet_flag'] is False for a, b in zip(radar_windows, radar_windows[1:]))
        rain_flag = any(w['wet_flag'] is True for w in windows)
        rules['rain_context'] = {'status': 'PARTIAL' if any(w['precip_3day_mm'] is None for w in windows) else 'OK',
                                 'windows': windows, 'rain_threshold_mm': settings.rain_mm,
                                 'wet_to_dry_radar_transition': wet_to_dry,
                                 'note': 'Calendar-day windows include acquisition day; rainfall timing within that day and irrigation are unknown.'}
        if rain_flag or wet_to_dry:
            flags.append('RAIN_MAY_CONFOUND_RADAR_OR_SURFACE')
    if request.greenness_decline_date:
        anchor = request.greenness_decline_date
        part, missing = daily_window(rows, anchor-timedelta(days=settings.frost_window_days-1), anchor, ('tmin',))
        frosty = [r['date'].isoformat() for r in part if r['tmin'] <= 0]
        rules['frost_risk'] = {'status': 'FLAG' if frosty else 'UNKNOWN' if missing else 'NO_FLAG',
                               'possible_frost_dates': frosty, 'missing_dates': missing,
                               'note': 'Gridded daily Tmin indicates exposure risk, not confirmed crop death or removal.'}
        if frosty:
            flags.append('FROST_COULD_EXPLAIN_BROWNING')
    drought_anchor = request.greenness_decline_date or (request.harvest.start if request.harvest else None)
    if drought_anchor:
        part, missing = daily_window(rows, drought_anchor-timedelta(days=29), drought_anchor, ('prcp',))
        baseline_sums = []
        for year in range(settings.baseline_start, min(settings.baseline_end, request.year-1) + 1):
            try:
                end = date(year, drought_anchor.month, drought_anchor.day)
            except ValueError:
                continue  # Feb 29 has no exact calendar counterpart in non-leap years.
            historical, absent = daily_window(baseline, end-timedelta(days=29), end, ('prcp',))
            if not absent:
                baseline_sums.append(sum(r['prcp'] for r in historical))
        if not missing and len(baseline_sums) >= 20:
            precip = sum(r['prcp'] for r in part)
            threshold = float(np.percentile(baseline_sums, settings.drought_percentile))
            low = precip < threshold
            rules['drought_context'] = {'status': 'FLAG' if low else 'NO_FLAG', 'precip_30day_mm': precip,
                                         'baseline_years_used': len(baseline_sums),
                                         'baseline_period': [settings.baseline_start, min(settings.baseline_end, request.year-1)],
                                         'precip_threshold_mm': threshold, 'percentile_threshold': settings.drought_percentile,
                                         'note': 'Rainfall deficit only; irrigation and actual soil moisture are unknown.'}
            if low:
                flags.append('RAINFALL_DEFICIT_COULD_EXPLAIN_BROWNING')
        else:
            rules['drought_context'] = unknown(f'Need complete 30-day rainfall and >=20 historical matching seasonal windows; found {len(baseline_sums)}')
    if request.standing_drydown:
        part, missing = daily_window(rows, request.standing_drydown.start, request.standing_drydown.end, ('tmin', 'tmax', 'prcp', 'vp'))
        if part:
            rules['drydown_context'] = {'status': 'PARTIAL' if missing else 'OK',
                                        'interval': request.standing_drydown.model_dump(mode='json'),
                                        'days_observed': len(part), 'missing_dates': missing,
                                        'mean_temperature_c': float(np.mean([(r['tmin']+r['tmax'])/2 for r in part])),
                                        'precipitation_mm': sum(r['prcp'] for r in part) if not missing else None,
                                        'approximate_mean_vpd_kpa': float(np.mean([approximate_vpd_kpa(r) for r in part])),
                                        'note': 'Estimated daily VPD = mean saturation pressure at Tmin/Tmax minus vp/1000, clipped at zero. Not measured daily mean VPD or grain moisture. No universal drying-rate threshold.'}
    return rules, flags


def get_daymet_weather(request, settings, http, include_baseline=True):
    if request.geography.country == 'NL':
        raise DataUnavailable('Daymet does not cover the Netherlands; weather context is UNKNOWN')
    point = shape(request.geometry).representative_point()
    events = [d for interval in (request.planting, request.harvest, request.standing_drydown) if interval for d in (interval.start, interval.end)]
    events += request.radar_dates + ([request.greenness_decline_date] if request.greenness_decline_date else [])
    first = max(date(request.year, 1, 1), min(events or [date(request.year, 1, 1)]) - timedelta(days=30))
    last = min(request.reporting_cutoff, max(events or [request.reporting_cutoff]))
    if settings.historical_weather:
        first = max(date(request.year, 1, 1), min(first, comparison_anchor(request)-timedelta(days=89)))
    params = {'lat': round(point.y, 7), 'lon': round(point.x, 7), 'vars': 'tmin,tmax,prcp,vp,srad,dayl',
              'start': first.isoformat(), 'end': last.isoformat(), 'format': 'csv'}
    raw, cached = http.get(DAYMET, params)
    rows, metadata = parse_daymet(raw.decode('utf-8-sig'))
    rows = [r for r in rows if first <= r['date'] <= last and r['date'] <= request.reporting_cutoff]
    baseline, errors, sources = [], [], [{'url': DAYMET, 'documentation': DAYMET_DOC, 'parameters': params, 'cached': cached,
                                         'units': {'tmin': 'degC', 'tmax': 'degC', 'prcp': 'mm/day', 'vp': 'Pa'}, **metadata}]
    if include_baseline and (request.greenness_decline_date or request.harvest or settings.historical_weather):
        end_year = min(settings.baseline_end, request.year-1)
        if end_year-settings.baseline_start >= 19:
            baseline_params = {**params, 'vars': 'tmin,tmax,prcp,vp' if settings.historical_weather else 'prcp', 'start': f'{settings.baseline_start}-01-01', 'end': f'{end_year}-12-31'}
            try:
                historical, cached_baseline = http.get(DAYMET, baseline_params)
                baseline, _ = parse_daymet(historical.decode('utf-8-sig'))
                sources.append({'url': DAYMET, 'parameters': baseline_params, 'cached': cached_baseline, 'purpose': 'seasonally matched rainfall baseline'})
            except DataUnavailable as exc:
                errors.append(str(exc))
    rows, current_invalid = checked_weather_rows(rows)
    baseline, historical_invalid = checked_weather_rows(baseline)
    rules, flags = weather_rules(request, rows, baseline, settings)
    if current_invalid or historical_invalid:
        flags.append('INVALID_WEATHER_MEASUREMENTS_EXCLUDED')
    history = seasonal_comparison(request, rows, baseline, settings) if settings.historical_weather else {'status': 'DISABLED'}
    regional = nass_regional_context(request, settings, http) if settings.nass_context else {'status': 'DISABLED', 'sources': []}
    sources.extend(regional.get('sources', []))
    missing = []
    if not request.planting:
        missing.append('planting date or interval')
    if not request.harvest:
        missing.append('existing harvest interval')
    if not request.maturity:
        missing.append('documented local hybrid/maturity range')
    unknown = any(r['status'] in ('UNKNOWN', 'TIMING UNKNOWN', 'PARTIAL') for r in rules.values()) or (settings.historical_weather and history['status'] != 'OK')
    return ToolResult(evidence_id='weather', tool='get_daymet_weather', field_id=request.field_id, year=request.year,
                      status='PARTIAL' if unknown or errors else 'OK', sources=sources,
                      data={'rules': rules, 'daily_rows': [{**r, 'date': r['date'].isoformat()} for r in rows],
                            'grid_elevation_m': metadata['grid_elevation_m'], 'thresholds_provisional': True,
                            'historical_comparison': history, 'nass_regional_context': regional,
                            'weather_validation': {'current_invalid_measurements': current_invalid, 'historical_invalid_measurements': historical_invalid}},
                      quality_flags=flags, missing_inputs=missing, errors=errors)


def checked_weather_rows(rows):
    """Exclude impossible values without turning missing observations into zeros."""
    cleaned=[]; invalid=0
    for original in rows:
        row=dict(original)
        if row.get('tmin') is not None and row.get('tmax') is not None and row['tmin'] > row['tmax']:
            row['tmin']=row['tmax']=None;invalid+=2
        for key in ('prcp','vp'):
            if row.get(key) is not None and row[key]<0:
                row[key]=None;invalid+=1
        cleaned.append(row)
    return cleaned,invalid


def seasonal_comparison(request, rows, baseline, settings):
    """Compare observed windows, never infer a planting date or crop maturity."""
    anchor=comparison_anchor(request)
    end_year=min(settings.baseline_end,request.year-1)
    result={'status':'UNKNOWN','anchor_date':anchor.isoformat(),
            'anchor_basis':'Supplied candidate harvest endpoint or observation date; not confirmed removal',
            'baseline_period':[settings.baseline_start,end_year],
            'gdd_interpretation':'Calendar-window heat accumulation only; not planting-to-harvest GDD or hybrid maturity.',
            'windows':[]}
    # Index once: historical scans over 30 years must not be repeated per day.
    history_by_date={r['date']:r for r in baseline}
    current_by_date={r['date']:r for r in rows if r['date']<=request.reporting_cutoff}
    def values(index,end,days):
        start=end-timedelta(days=days-1)
        expected=[start+timedelta(days=i) for i in range(days)]
        absent=[d.isoformat() for d in expected if d not in index or any(index[d].get(k) is None for k in ('tmin','tmax','prcp'))]
        if absent:return None,absent
        part=[index[d] for d in expected]
        return {'mean_temperature_c':float(np.mean([(r['tmin']+r['tmax'])/2 for r in part])),
                'precipitation_mm':sum(r['prcp'] for r in part),
                'calendar_window_gdd_f':sum(corn_gdd_f(r['tmin'],r['tmax']) for r in part),
                'frost_risk_days':sum(r['tmin']<=0 for r in part)},[]
    for days in (30,60,90):
        current,missing=values(current_by_date,anchor,days)
        historical=[];years=[]
        for year in range(settings.baseline_start,end_year+1):
            try:end=date(year,anchor.month,anchor.day)
            except ValueError:continue
            summary,absent=values(history_by_date,end,days)
            if summary is not None:historical.append(summary);years.append(year)
        comparison={}
        if current is not None and len(historical)>=20:
            for key,value in current.items():
                distribution=[h[key] for h in historical]
                median=float(np.median(distribution))
                comparison[key]={'observed':value,'historical_median':median,
                    'historical_p10':float(np.percentile(distribution,10)),
                    'historical_p90':float(np.percentile(distribution,90)),
                    'anomaly_from_historical_median':value-median,
                    'historical_percentile_midrank':100*(sum(x<value for x in distribution)+.5*sum(x==value for x in distribution))/len(distribution)}
        result['windows'].append({'window_days':days,'window_start':(anchor-timedelta(days=days-1)).isoformat(),
            'status':'OK' if comparison else 'UNKNOWN','current':current,'missing_dates':missing,
            'complete_historical_years':years,'historical_year_count':len(years),'comparison':comparison,
            'minimum_historical_years':20})
    result['status']='OK' if all(w['status']=='OK' for w in result['windows']) else 'PARTIAL' if any(w['status']=='OK' for w in result['windows']) else 'UNKNOWN'
    return result

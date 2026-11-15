"""Bounded NASS regional context; never field labels, planting dates or weather."""
from datetime import date
import math
import statistics
from .http import DataUnavailable
from .tools import NASS, number_or_suppression


def comparison_anchor(request):
    candidate = (request.harvest.end if request.harvest else request.greenness_decline_date
                 or max(request.radar_dates, default=request.reporting_cutoff))
    return min(candidate, request.reporting_cutoff)


def _number(value):
    numeric, suppression = number_or_suppression(value)
    if numeric is not None and not math.isfinite(numeric):
        return None, str(value)
    return numeric, suppression


def _matching_rows(rows, request, kind, first_year, last_year, cutoff):
    """Do not trust server filters; reject future/revised-after-cutoff records."""
    selected = {}; ambiguous = set(); excluded = 0
    for r in rows:
        if not isinstance(r, dict):
            excluded += 1; continue
        try:
            year = int(r['year'])
            loaded = date.fromisoformat(str(r['load_time'])[:10])
            dimensions = {
                'source_desc':'SURVEY','commodity_desc':'CORN','state_alpha':request.geography.state_alpha,
                'agg_level_desc':'STATE','domain_desc':'TOTAL',
                'prodn_practice_desc':'ALL PRODUCTION PRACTICES',
                'statisticcat_desc':'PROGRESS' if kind == 'progress' else 'AREA HARVESTED',
                'freq_desc':'WEEKLY' if kind == 'progress' else 'ANNUAL',
            }
            if any(r.get(k) != v for k,v in dimensions.items()) or not first_year <= year <= last_year or loaded > cutoff:
                excluded += 1; continue
            if kind == 'progress':
                if r.get('unit_desc') not in ('PCT PLANTED','PCT MATURE','PCT HARVESTED'):
                    excluded += 1; continue
                observed = date.fromisoformat(r['week_ending'])
                if observed.year != year or observed > cutoff:
                    excluded += 1; continue
                expected_items = {
                    'CORN - PROGRESS, MEASURED IN PCT PLANTED',
                    'CORN - PROGRESS, MEASURED IN PCT MATURE',
                    'CORN, GRAIN - PROGRESS, MEASURED IN PCT HARVESTED',
                    'CORN, SILAGE - PROGRESS, MEASURED IN PCT HARVESTED',
                }
                if r.get('short_desc') not in expected_items:
                    excluded += 1; continue
            else:
                if r.get('reference_period_desc') != 'YEAR' or r.get('unit_desc') != 'ACRES' or r.get('short_desc') not in ('CORN, GRAIN - ACRES HARVESTED','CORN, SILAGE - ACRES HARVESTED'):
                    excluded += 1; continue
                observed = None
            value, suppression = _number(r.get('Value', ''))
            if value is not None and (value < 0 or (kind == 'progress' and value > 100)):
                excluded += 1; continue
            item = {'year':year,'data_item':r['short_desc'],'numeric_value':value,
                    'raw_value':str(r.get('Value','')),'suppression_flag':suppression,
                    'week_ending':observed.isoformat() if observed else None,
                    'load_time':str(r['load_time']),'unit':r['unit_desc']}
            key=(year,item['data_item'],item['week_ending'])
            prior=selected.get(key)
            if prior is None or item['load_time'] > prior['load_time']:
                selected[key]=item;ambiguous.discard(key)
            elif item['load_time'] == prior['load_time'] and item['raw_value'] != prior['raw_value']:
                ambiguous.add(key)
        except (KeyError,TypeError,ValueError,OverflowError):
            excluded += 1
    return [r for k,r in selected.items() if k not in ambiguous], excluded+len(ambiguous)


def nass_regional_context(request, settings, http):
    cutoff = comparison_anchor(request)
    result={'status':'UNKNOWN','state_alpha':request.geography.state_alpha,
            'context_cutoff':cutoff.isoformat(),'sources':[],'errors':[],
            'regional_only':True,'archived_publication_vintage_verified':False,
            'crop_progress':[],'historical_acreage':[],
            'note':'NASS supplies regional crop statistics, not daily weather or actual field use. Current database vintages may contain revisions.'}
    if request.geography.country != 'US' or not request.geography.state_alpha or not settings.nass_key:
        result['errors'].append('NASS regional context needs US state and configured NASS_API_KEY')
        return result
    common={'source_desc':'SURVEY','sector_desc':'CROPS','group_desc':'FIELD CROPS','commodity_desc':'CORN',
            'domain_desc':'TOTAL','prodn_practice_desc':'ALL PRODUCTION PRACTICES',
            'agg_level_desc':'STATE','state_alpha':request.geography.state_alpha,'format':'JSON'}
    filtered={};excluded={}
    for kind,start,end,extra in (
        ('progress',request.year-5,request.year,{'statisticcat_desc':'PROGRESS','freq_desc':'WEEKLY'}),
        ('acreage',request.year-10,request.year-1,{'statisticcat_desc':'AREA HARVESTED','unit_desc':'ACRES','freq_desc':'ANNUAL','reference_period_desc':'YEAR'}),
    ):
        params={**common,**extra,'year__GE':start,'year__LE':end}
        try:
            response,cached=http.json(NASS,{**params,'key':settings.nass_key},max_bytes=4_000_000)
            if not isinstance(response,dict) or 'error' in response or not isinstance(response.get('data'),list):
                raise DataUnavailable('NASS response has no usable data array')
            kept,rejected=_matching_rows(response['data'],request,kind,start,end,cutoff)
            filtered[kind]=kept;excluded[kind]=rejected
            result['sources'].append({'url':NASS,'parameters':params,'cached':cached,
                'returned_rows':len(response['data']),'retained_rows':len(kept),'excluded_rows':rejected})
        except DataUnavailable as exc:
            filtered[kind]=[];result['errors'].append(str(exc))
    progress=filtered.get('progress',[])
    items=sorted({r['data_item'] for r in progress})
    for item in items:
        matched=[]
        for year in range(request.year-5,request.year+1):
            try:seasonal_end=date(year,cutoff.month,cutoff.day)
            except ValueError:continue
            candidates=[r for r in progress if r['year']==year and r['data_item']==item
                        and 0 <= (seasonal_end-date.fromisoformat(r['week_ending'])).days <= 14]
            if candidates:matched.append(max(candidates,key=lambda r:r['week_ending']))
        current=next((r for r in matched if r['year']==request.year),None)
        past=[r for r in matched if r['year']<request.year and r['numeric_value'] is not None]
        median=statistics.median(r['numeric_value'] for r in past) if past else None
        result['crop_progress'].append({'data_item':item,'current':current,'matched_history':past,
            'historical_median_pct':median,
            'current_minus_historical_percentage_points':current['numeric_value']-median
                if current and current['numeric_value'] is not None and median is not None else None,
            'match_method':'Latest statewide observation within 14 days before the same month/day; no field planting date is imputed.'})
    acreage=filtered.get('acreage',[])
    for year in sorted({r['year'] for r in acreage}):
        records=[r for r in acreage if r['year']==year]
        values={r['data_item']:r['numeric_value'] for r in records}
        grain=values.get('CORN, GRAIN - ACRES HARVESTED');silage=values.get('CORN, SILAGE - ACRES HARVESTED')
        total=grain+silage if grain is not None and silage is not None else None
        result['historical_acreage'].append({'year':year,'grain_acres':grain,'silage_acres':silage,
            'grain_share_of_reported_harvest_acres':grain/total if total else None,'records':records})
    result['excluded_rows']=excluded
    if any(p['current'] or p['matched_history'] for p in result['crop_progress']) or result['historical_acreage']:
        result['status']='PARTIAL' if result['errors'] else 'OK'
    return result

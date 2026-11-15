"""Field-masked Sentinel observations for post-RF Nemotron assessment.

No training, label joins or remote code execution. Raw asset URLs and SAS tokens
never enter reports. All statistics are observations, not confirmed harvest use.
"""
from collections import Counter
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from hashlib import sha256
import json,math,time,zlib,struct
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from shapely.geometry import shape,mapping
from shapely.ops import transform as transform_geometry
from pyproj import Transformer

from .http import DataUnavailable
from .schemas import ToolResult

S2='https://earth-search.aws.element84.com/v1'
S1='https://planetarycomputer.microsoft.com/api/stac/v1'
COLLECTIONS={'s2':'sentinel-2-c1-l2a','s1':'sentinel-1-rtc'}


def interval(request):
    """Cover the entire visible growing season for peak and post-harvest rules."""
    end=min(request.reporting_cutoff,date(request.year,12,31))
    start=min(date(request.year,4,1),end)
    return start,end


def select_items(features,request,collection,limit):
    """Reject wrong collections, dates and footprints, then keep one geometry."""
    start,end=interval(request);field=shape(request.geometry);valid=[]
    for item in features:
        try:
            props=item['properties'];d=date.fromisoformat(props['datetime'][:10])
            footprint=shape(item['geometry'])
            if item.get('collection')!=collection or not start<=d<=end or not footprint.intersects(field):continue
            cover=footprint.intersection(field).area/field.area
            if cover<.95:continue
            if collection==COLLECTIONS['s2']:
                group=props.get('grid:code')
                if not group or float(props.get('eo:cloud_cover',100))>90:continue
            else:
                orbit=props.get('sat:relative_orbit');direction=props.get('sat:orbit_state')
                if orbit is None or direction not in ('ascending','descending'):continue
                group=(int(orbit),direction)
            valid.append((group,d,item))
        except (KeyError,ValueError,TypeError,AttributeError):continue
    if not valid:return [],{'candidate_count':0,'selection':'No matching dated field-covering scenes'}
    counts=Counter(g for g,d,i in valid);group=sorted(counts,key=lambda g:(-counts[g],str(g)))[0]
    dated={}
    for g,d,i in valid:
        if g!=group:continue
        prior=dated.get(d)
        if prior is None or float(i['properties'].get('eo:cloud_cover',0))<float(prior['properties'].get('eo:cloud_cover',0)):
            dated[d]=i
    ordered=sorted(dated.values(),key=lambda i:i['properties']['datetime'])
    count=len(ordered)
    if count>limit:
        indexes=set(np.linspace(0,count-1,limit,dtype=int).tolist())
        ordered=[ordered[i] for i in sorted(indexes)]
    return ordered,{'candidate_count':count,'selected_count':len(ordered),'consistent_geometry':group,
                    'selection':'Evenly spaced dated scenes from one tile/orbit and pass; bounded subset, not a complete season'}


def catalog(http,base,collection,request,deadline):
    start,end=interval(request)
    params={'collections':collection,'bbox':','.join(str(v) for v in shape(request.geometry).bounds),
            'datetime':f'{start}T00:00:00Z/{end}T23:59:59Z','limit':100}
    features=[];url=base+'/search';truncated=False
    for page in range(3):
        if time.monotonic()>=deadline:break
        data,_=http.json(url,params,max_bytes=6_000_000)
        if not isinstance(data,dict) or not isinstance(data.get('features'),list):
            raise DataUnavailable('Sentinel catalog returned no feature array')
        features.extend(data['features'])
        following=next((x for x in data.get('links',[]) if x.get('rel')=='next'),None)
        if not following:break
        if following.get('method','GET')!='GET' or urlsplit(following.get('href','')).netloc!=urlsplit(base).netloc:
            truncated=True;break
        url=following['href'];params={};truncated=page==2
    return features,truncated


def scaled_reflectance(array,asset):
    bands=asset.get('raster:bands',[])
    if not bands or 'scale' not in bands[0] or 'offset' not in bands[0]:
        raise DataUnavailable('Sentinel-2 reflectance scale/offset metadata missing')
    scale=float(bands[0]['scale']);offset=float(bands[0]['offset'])
    if not math.isfinite(scale) or not math.isfinite(offset) or scale<=0:
        raise DataUnavailable('Sentinel-2 reflectance scale/offset invalid')
    return np.ma.asarray(array,dtype=float).filled(np.nan)*scale+offset


def optical_statistics(bands,inside,scl,settings):
    total=int(inside.sum())
    clear=inside & np.isin(np.ma.filled(scl,0),(4,5))
    clear &= np.isfinite(bands['red']) & np.isfinite(bands['nir'])
    n=int(clear.sum());fraction=n/total if total else 0
    if total<settings.spatial_min_pixels or n<settings.spatial_min_pixels or fraction<settings.spatial_min_clear:
        return {'usable':False,'interior_pixels':total,'clear_pixels':n,'clear_fraction':fraction}
    def ratio(a,b):
        denominator=a+b
        with np.errstate(divide='ignore',invalid='ignore'):
            x=np.where(abs(denominator)>1e-6,(a-b)/denominator,np.nan)
        return np.where((x>=-1)&(x<=1),x,np.nan)
    indexes={'ndvi':ratio(bands['nir'],bands['red'])}
    for name,a,b in [('ndre','nir08','rededge1'),('lswi','nir08','swir16'),('ndti','swir16','swir22')]:
        if a in bands and b in bands:indexes[name]=ratio(bands[a],bands[b])
    if 'blue' in bands:
        denominator=bands['nir']+6*bands['red']-7.5*bands['blue']+1
        with np.errstate(divide='ignore',invalid='ignore'):
            indexes['evi']=np.where(abs(denominator)>1e-6,2.5*(bands['nir']-bands['red'])/denominator,np.nan)
    values={}
    for k,x in indexes.items():
        observed=x[clear & np.isfinite(x)]
        values[k]={'mean':float(np.mean(observed)),'p10':float(np.percentile(observed,10)),
                   'p90':float(np.percentile(observed,90)),'std':float(np.std(observed)),
                   'valid_pixels':len(observed)} if len(observed)>=settings.spatial_min_pixels else None
    rr,cc=np.where(inside);zones=[]
    for row in (0,1):
        for col in (0,1):
            zone=clear.copy();rgrid,cgrid=np.indices(inside.shape)
            zone &= ((rgrid>np.median(rr)) if row else (rgrid<=np.median(rr)))
            zone &= ((cgrid>np.median(cc)) if col else (cgrid<=np.median(cc)))
            data=indexes['ndvi'][zone & np.isfinite(indexes['ndvi'])]
            zones.append({'row':row,'column':col,'clear_pixels':len(data),
                          'ndvi_mean':float(np.mean(data)) if len(data)>=5 else None})
    return {'usable':values.get('ndvi') is not None,'interior_pixels':total,'clear_pixels':n,
            'clear_fraction':fraction,'indices':values,'ndvi_zones_2x2':zones}


def radar_statistics(vv,vh,inside,minimum):
    vv=np.ma.asarray(vv,dtype=float).filled(np.nan);vh=np.ma.asarray(vh,dtype=float).filled(np.nan)
    valid=inside & np.isfinite(vv)&np.isfinite(vh)&(vv>0)&(vh>0)
    n=int(valid.sum())
    if n<minimum:return {'usable':False,'valid_pixels':n}
    v=float(np.mean(vv[valid]));h=float(np.mean(vh[valid]))
    return {'usable':True,'valid_pixels':n,'vv_linear_mean':v,'vh_linear_mean':h,
            'vv_db':10*math.log10(v),'vh_db':10*math.log10(h),'vh_minus_vv_db':10*math.log10(h/v)}


class RasterReader:
    """Native field windows, explicit resampling and bounded HTTPS reads."""
    def __init__(self,request,settings,http,deadline):
        self.request=request;self.settings=settings;self.http=http;self.deadline=deadline
        self.reads=0;self.tokens={}

    def href(self,asset):
        url=asset['href'];p=urlsplit(url)
        if p.scheme!='https' or p.query or p.username or p.password:
            raise DataUnavailable('Unexpected Sentinel asset address')
        if p.hostname and p.hostname.endswith('.amazonaws.com'):
            self.http.public_url(url);return url
        if not p.hostname or not p.hostname.endswith('.blob.core.windows.net'):
            raise DataUnavailable('Sentinel asset origin not permitted')
        self.http.public_url(url)
        account=p.hostname.split('.')[0];container=p.path.strip('/').split('/')[0]
        if not account.isalnum() or not container or not all(c.isalnum() or c=='-' for c in container):
            raise DataUnavailable('Invalid Sentinel storage metadata')
        key=(account,container)
        if key not in self.tokens:
            response,_=self.http.json(f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}',cache=False)
            if not isinstance(response.get('token'),str):raise DataUnavailable('Sentinel asset access token unavailable')
            self.tokens[key]=response['token']
        return url+'?'+self.tokens[key]

    def check(self):
        if time.monotonic()>=self.deadline or self.http.remaining()<20:
            raise DataUnavailable('Sentinel extraction time budget reached')

    def grid(self,asset):
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.features import geometry_mask
        self.check()
        with rasterio.open(self.href(asset)) as src:
            if src.crs is None:raise DataUnavailable('Sentinel raster CRS unavailable')
            project=Transformer.from_crs(4326,src.crs,always_xy=True).transform
            polygon=transform_geometry(project,shape(self.request.geometry))
            interior=polygon.buffer(-10)
            if interior.is_empty:raise DataUnavailable('Field too small for a 10 metre interior buffer')
            win=from_bounds(*polygon.bounds,src.transform).round_offsets().round_lengths()
            h,w=int(win.height),int(win.width)
            if min(h,w)<1 or h*w>262144:raise DataUnavailable('Sentinel field window exceeds size bound')
            transform=src.window_transform(win)
            mask=geometry_mask([mapping(interior)],out_shape=(h,w),transform=transform,invert=True)
            return {'crs':src.crs,'transform':transform,'height':h,'width':w,'inside':mask}

    def read(self,asset,grid):
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling
        self.check()
        with rasterio.open(self.href(asset)) as src:
            with WarpedVRT(src,crs=grid['crs'],transform=grid['transform'],height=grid['height'],width=grid['width'],resampling=Resampling.nearest) as vrt:
                data=vrt.read(1,masked=True).astype(float)
        self.reads+=1
        return data

    @staticmethod
    def _png_rgba(array):
        height,width,_=array.shape
        raw=b''.join(b'\x00'+array[row].tobytes() for row in range(height))
        def chunk(kind,payload):
            return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
        return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',width,height,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b'')

    def preview(self,item):
        """Render real clear Sentinel-2 field pixels as compact true-colour and NDVI PNGs."""
        import rasterio,os
        assets=item.get('assets',{})
        if not all(name in assets for name in ('red','green','blue','nir','scl','rededge1')):
            raise DataUnavailable('Sentinel image bands are incomplete')
        env={'GDAL_DISABLE_READDIR_ON_OPEN':'EMPTY_DIR','GDAL_HTTP_TIMEOUT':'5',
             'GDAL_HTTP_CONNECTTIMEOUT':'4','GDAL_HTTP_MAX_RETRY':'0','CPL_VSIL_CURL_ALLOWED_EXTENSIONS':'.tif,.tiff',
             'CPL_LOG':os.devnull}
        with rasterio.Env(**env):
            grid=self.grid(assets['rededge1'])
            scl=self.read(assets['scl'],grid)
            clear=grid['inside'] & np.isin(np.ma.filled(scl,0),(4,5))
            bands={name:scaled_reflectance(self.read(assets[name],grid),assets[name]) for name in ('red','green','blue','nir')}
        valid=clear.copy()
        for values in bands.values(): valid &= np.isfinite(values)
        with np.errstate(divide='ignore',invalid='ignore'):
            ndvi=(bands['nir']-bands['red'])/np.maximum(bands['nir']+bands['red'],1e-6)
        valid &= np.isfinite(ndvi)
        h,w=valid.shape
        scale=min(220/max(1,w),220/max(1,h),12)
        out_w=max(1,min(220,int(round(w*scale))));out_h=max(1,min(220,int(round(h*scale))))
        yy=np.minimum((np.arange(out_h)*h/out_h).astype(int),h-1);xx=np.minimum((np.arange(out_w)*w/out_w).astype(int),w-1)
        mask=valid[np.ix_(yy,xx)]
        rgba=np.zeros((out_h,out_w,4),dtype=np.uint8)
        for channel,name in enumerate(('red','green','blue')):
            vals=np.nan_to_num(np.clip(bands[name][np.ix_(yy,xx)]*2.5,0,1),nan=0.0)
            rgba[:,:,channel]=np.clip(np.power(vals,1/2.2)*255,0,255).astype(np.uint8)
        rgba[:,:,3]=np.where(mask,255,0).astype(np.uint8)
        edge=mask.copy()
        if out_h>2 and out_w>2:
            eroded=mask.copy()
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                shifted=np.zeros_like(mask); y0=max(0,dy);y1=out_h+min(0,dy);x0=max(0,dx);x1=out_w+min(0,dx)
                shifted[y0:y1,x0:x1]=mask[y0-dy:y1-dy,x0-dx:x1-dx];eroded &= shifted
            edge=mask & ~eroded
        rgba[edge,:3]=255;rgba[edge,3]=255
        stops=np.array([[-.2,110,65,35],[.2,220,205,95],[.5,140,195,90],[.9,20,105,45]],dtype=float)
        scaled=np.clip(np.nan_to_num(ndvi[np.ix_(yy,xx)],nan=0.0),-.2,.9);nd=np.zeros((out_h,out_w,4),dtype=np.uint8)
        for c in range(3): nd[:,:,c]=np.interp(scaled,stops[:,0],stops[:,c+1]).astype(np.uint8)
        nd[:,:,3]=np.where(mask,255,0).astype(np.uint8);nd[edge,:3]=255;nd[edge,3]=255
        return {'true_colour_png':'data:image/png;base64,'+base64.b64encode(self._png_rgba(rgba)).decode('ascii'),
                'ndvi_png':'data:image/png;base64,'+base64.b64encode(self._png_rgba(nd)).decode('ascii')}

    def extract(self,item,sensor):
        import rasterio,os
        assets=item['assets']
        env={'GDAL_DISABLE_READDIR_ON_OPEN':'EMPTY_DIR','GDAL_HTTP_TIMEOUT':'5',
             'GDAL_HTTP_CONNECTTIMEOUT':'4','GDAL_HTTP_MAX_RETRY':'0','CPL_VSIL_CURL_ALLOWED_EXTENSIONS':'.tif,.tiff',
             'CPL_LOG':os.devnull}
        with rasterio.Env(**env):
            grid=self.grid(assets['rededge1' if sensor=='s2' else 'vv'])
            if sensor=='s1':
                values=radar_statistics(self.read(assets['vv'],grid),self.read(assets['vh'],grid),grid['inside'],self.settings.spatial_min_pixels)
            else:
                scl=self.read(assets['scl'],grid)
                if np.count_nonzero(grid['inside'] & np.isin(np.ma.filled(scl,0),(4,5)))<self.settings.spatial_min_pixels:
                    return {'usable':False,'reason':'Insufficient clear field pixels'}
                bands={}
                for name in ('red','nir','nir08','rededge1','swir16','swir22','blue'):
                    if name in assets:bands[name]=scaled_reflectance(self.read(assets[name],grid),assets[name])
                values=optical_statistics(bands,grid['inside'],scl,self.settings)
            p=item['properties']
            return {'date':p['datetime'][:10],'acquisition_datetime':p['datetime'],'item_id':item['id'],
                    'collection':item['collection'],'platform':p.get('platform'),
                    'relative_orbit':p.get('sat:relative_orbit'),'orbit_pass':p.get('sat:orbit_state'),
                    'native_grid_crs':str(grid['crs']),'grid_spacing_m':abs(grid['transform'].a),
                    **values}


def trajectory(observations,sensor,request):
    obs=sorted([r for r in observations if r.get('usable')],key=lambda r:r['date'])
    summary={'usable_observations':len(obs),'first_date':obs[0]['date'] if obs else None,
             'last_date':obs[-1]['date'] if obs else None,'changes':[]}
    metrics=('ndvi','ndre','lswi','ndti','evi') if sensor=='s2' else ('vv_db','vh_db','vh_minus_vv_db')
    for before,after in zip(obs,obs[1:]):
        delta={}
        for k in metrics:
            a=(before.get('indices',{}).get(k) or {}).get('mean') if sensor=='s2' else before.get(k)
            b=(after.get('indices',{}).get(k) or {}).get('mean') if sensor=='s2' else after.get(k)
            if a is not None and b is not None:delta[k]=b-a
        summary['changes'].append({'before':before['date'],'after':after['date'],
            'gap_days':(date.fromisoformat(after['date'])-date.fromisoformat(before['date'])).days,'delta':delta})
    summary['limitation']='Observed changes are not confirmed removal or harvest use; gaps, rain, soil and senescence can confound them.'
    return summary


def get_sentinel_spatial_evidence(request,settings,http,reader_factory=RasterReader):
    started=time.monotonic();deadline=started+min(settings.spatial_budget_seconds,max(0,http.remaining()-60))
    start,end=interval(request)
    signature=sha256(json.dumps({'geometry':request.geometry,'year':request.year,'field_id':request.field_id,
        'cutoff':str(request.reporting_cutoff),'window':[str(start),str(end)],'scenes':settings.spatial_max_scenes,
        'pixels':settings.spatial_min_pixels,'clear':settings.spatial_min_clear,'contract':'sentinel-evidence-v5-images-20m'},sort_keys=True).encode()).hexdigest()
    cache=settings.cache_dir/('sentinel-'+signature+'.json')
    if cache.exists() and time.time()-cache.stat().st_mtime<settings.cache_days*86400:
        result=ToolResult.model_validate_json(cache.read_text())
        if result.field_id==request.field_id and result.year==request.year:
            result.data['cache_hit']=True;result.data['new_satellite_data_fetched']=False;return result
    geometry=shape(request.geometry)
    epsg=32600+int((geometry.centroid.x+180)/6)+1 if geometry.centroid.y>=0 else 32700+int((geometry.centroid.x+180)/6)+1
    projected=transform_geometry(Transformer.from_crs(4326,epsg,always_xy=True).transform,geometry)
    data={'field_geometry_sha256':sha256(json.dumps(request.geometry,sort_keys=True).encode()).hexdigest(),
          'area_ha':projected.area/10000,'window':[start.isoformat(),end.isoformat()],
          'buffer_m':10,'cache_hit':False,'new_satellite_data_fetched':False,
          'note':'Contains modified Copernicus Sentinel observations. S2 clear-pixel NDVI is averaged on a 20 m field grid. Native 10 m bands are resampled to 20 m; no added detail. S1 RTC linear gamma0 is averaged before dB conversion.'}
    sources=[];errors=[];selected={};readers={}
    for sensor,base in [('s2',S2),('s1',S1)]:
        try:
            features,truncated=catalog(http,base,COLLECTIONS[sensor],request,deadline)
            selected[sensor],selection=select_items(features,request,COLLECTIONS[sensor],settings.spatial_max_scenes)
            data[sensor]={'selection':selection,'catalog_truncated':truncated,'observations':[]}
            sources.append({'url':base+'/collections/'+COLLECTIONS[sensor], 'collection':COLLECTIONS[sensor]})
        except (DataUnavailable,ValueError,KeyError,TypeError):
            selected[sensor]=[];data[sensor]={'observations':[]};errors.append(sensor.upper()+' catalog unavailable')
    def collect(sensor):
        reader=reader_factory(request,settings,http,deadline);readers[sensor]=reader
        records=[];failures=[]
        for item in selected[sensor]:
            if time.monotonic()>=deadline:failures.append('Extraction time bound reached');break
            try:records.append(reader.extract(item,sensor))
            except (DataUnavailable,OSError,ValueError,KeyError,RuntimeError) as exc:
                failures.append(type(exc).__name__+': scene unavailable')
        return records,failures
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={sensor:pool.submit(collect,sensor) for sensor in ('s2','s1')}
        for sensor,future in futures.items():
            records,failures=future.result();summary=trajectory(records,sensor,request)
            data[sensor].update(observations=records,summary=summary,status='OK' if summary['usable_observations']>=2 else 'PARTIAL' if summary['usable_observations'] else 'UNKNOWN',errors=failures)
            errors.extend(sensor.upper()+': '+x for x in failures)
    from .rulebook_v4 import assess_v4
    data['rulebook_v4']=assess_v4(data['s2']['observations'],data['s1']['observations'],request.year)
    usable=sorted((o for o in data['s2']['observations'] if o.get('usable')),key=lambda o:o['date'])
    previews=[]
    if usable:
        harvest=request.harvest
        if harvest:
            before=[o for o in usable if o['date']<=harvest.start.isoformat()]
            after=[o for o in usable if o['date']>=harvest.end.isoformat()]
            chosen=[('before',before[-1] if before else None),('after',after[0] if after else None)]
        else:
            chosen=[('before',usable[0]),('after',usable[-1] if len(usable)>1 else None)]
        item_by_id={item.get('id'):item for item in selected.get('s2',[])}
        image_reader=readers.get('s2')
        for position,observation in chosen:
            if not observation: continue
            item=item_by_id.get(observation.get('item_id'))
            try:
                if item is None or image_reader is None: raise DataUnavailable('Selected image scene is unavailable')
                previews.append({'position':position,'date':observation['date'],'ndvi_mean':observation.get('indices',{}).get('ndvi',{}).get('mean'),**image_reader.preview(item)})
            except (DataUnavailable,OSError,ValueError,KeyError,RuntimeError):
                previews.append({'position':position,'date':observation['date'],'unavailable':True})
        data['image_previews']={'source':'Sentinel-2 L2A clear field pixels; masked to the field boundary','items':previews,
            'missing_positions':[position for position,observation in chosen if observation is None]}
    else:
        data['image_previews']={'source':'Sentinel-2 L2A clear field pixels; masked to the field boundary','items':[],
            'missing_positions':['before','after'],'reason':'No usable clear Sentinel-2 observations'}
    data['raster_reads']=sum(r.reads for r in readers.values())
    data['new_satellite_data_fetched']=data['raster_reads']>0
    data['elapsed_seconds']=time.monotonic()-started
    status='OK' if all(data[s]['status']=='OK' for s in ('s2','s1')) else 'PARTIAL' if any(data[s]['summary']['usable_observations'] for s in ('s2','s1')) else 'UNKNOWN'
    result=ToolResult(evidence_id='sentinel',tool='get_sentinel_spatial_evidence',field_id=request.field_id,year=request.year,status=status,
        sources=sources,data=data,errors=errors,missing_inputs=[] if status=='OK' else ['At least two usable field-masked observations for each sensor'],
        quality_flags=['SATELLITE_OBSERVATIONS_DO_NOT_CONFIRM_HARVEST_USE'])
    if any(data[s]['summary']['usable_observations'] for s in ('s2','s1')):
        cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(result.model_dump_json(indent=2))
    return result

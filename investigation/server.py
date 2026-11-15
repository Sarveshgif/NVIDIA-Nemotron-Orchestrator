import json
import os
from pathlib import Path
import socket

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import VERSION
from .config import Settings
from .schemas import InvestigationInput
from .agent import run, runtime_configuration

settings = Settings.from_env()
app = FastAPI(title='Cornstellation investigation API', version=VERSION)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],
                   allow_methods=['GET','POST'],allow_headers=['Content-Type'])


@app.middleware('http')
async def local_requests(request: Request, call_next):
    # Local backend only; do not allow arbitrary websites to spend API credentials.
    origin = request.headers.get('origin')
    if origin and origin not in ('http://localhost:5173','http://127.0.0.1:5173','http://localhost:8788','http://127.0.0.1:8788'):
        from fastapi.responses import JSONResponse
        return JSONResponse({'error':'Origin not allowed'},status_code=403)
    if int(request.headers.get('content-length','0')) > 2_000_000:
        from fastapi.responses import JSONResponse
        return JSONResponse({'error':'Request too large'},status_code=413)
    return await call_next(request)


@app.get('/health')
def health():
    return {'status':'ok','workflow_version':VERSION,'nvidia_configured':bool(settings.nvidia_key),
            'runtime_configuration':runtime_configuration(settings),
            'nvidia_model':settings.nvidia_model,'search_configured':bool(settings.public_search_enabled and settings.google_key and settings.google_cx),
            'direct_reports_configured':bool(settings.public_report_sources and (settings.root/settings.public_report_sources).is_file()),
            'quickstats_configured':bool(settings.nass_key or settings.agricultural_csv),
            'rf_bundle_exists':(settings.root/settings.rf_bundle).is_file(),
            'satellite_retrieval_enabled':False}


@app.post('/api/investigate')
def investigate(payload: InvestigationInput):
    report = run(payload,settings)
    folder = settings.root / '.agent-runs' / 'investigation'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/(report['run_id']+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    import uvicorn
    port = int(os.getenv('INVESTIGATION_PORT','8788'))
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',port))
    except OSError:
        raise SystemExit(f'Port {port} is already in use. Stop the previous investigation server or set INVESTIGATION_PORT to another local port.') from None
    uvicorn.run(app,host='127.0.0.1',port=port)


if __name__ == '__main__':
    main()

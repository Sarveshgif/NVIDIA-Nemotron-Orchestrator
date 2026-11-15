import argparse
from hashlib import sha256
import json
from pathlib import Path

from .config import Settings
from .schemas import InvestigationInput


def main():
    parser = argparse.ArgumentParser(description='Scoped Cornstellation RF/Nemotron investigation')
    commands = parser.add_subparsers(dest='command',required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--input',required=True)
    run_parser.add_argument('--out',required=True)
    run_parser.add_argument('--mode',choices=['agent','context','offline'],default='agent')
    acreage_parser = commands.add_parser('aggregate-acreage', help='Retrieve regional NASS acreage separately; no field classification or NVIDIA calls')
    acreage_parser.add_argument('--input', required=True, help='One existing input JSON specifying year/state/county')
    acreage_parser.add_argument('--out', required=True)
    commands.add_parser('inspect-models')
    commands.add_parser('schema')
    evaluate_parser = commands.add_parser('evaluate')
    evaluate_parser.add_argument('--predictions',required=True)
    evaluate_parser.add_argument('--references',required=True)
    evaluate_parser.add_argument('--out',required=True)
    args = parser.parse_args()
    if args.command == 'schema':
        print(json.dumps(InvestigationInput.model_json_schema(),indent=2))
        return
    settings = Settings.from_env()
    if args.command == 'inspect-models':
        from .rf import inspect_bundle
        output = []
        for path in sorted((settings.root/'rf_silage_grain_models').glob('*.joblib')):
            try:
                bundle, checksum = inspect_bundle(str(path),str(settings.root))
                output.append({'artifact':path.name,'sha256':checksum,'feature_count':len(bundle['cols']),
                               'classes':{'0':'GRAIN','1':'SILAGE'},'threshold':bundle['threshold'],
                               'review_band':bundle.get('review_band'),'barc_fitted':'us_platt' in bundle,
                               'name':bundle.get('name'),'trained_on':bundle.get('trained_on')})
            except (RuntimeError,ValueError,OSError) as exc:
                output.append({'artifact':path.name,'error':str(exc)})
        print(json.dumps(output,indent=2,default=str))
        return
    if args.command == 'aggregate-acreage':
        from .http import PublicHTTP
        from .tools import get_agricultural_context
        request = InvestigationInput.model_validate(json.loads(Path(args.input).read_text()))
        evidence = get_agricultural_context(request, settings, PublicHTTP(settings))
        output = {'scope': 'aggregate_acreage_only', 'field_classification_performed': False,
                  'note': 'Regional acreage for separate comparisons; never a field label or forced acreage target.',
                  'evidence': evidence.model_dump(mode='json')}
    elif args.command == 'run':
        from .agent import run
        payload = json.loads(Path(args.input).read_text())
        inputs = payload if isinstance(payload,list) else [payload]
        # Validate entire batch before any paid model call or public retrieval.
        requests = [InvestigationInput.model_validate(item) for item in inputs]
        if len({(r.field_id,r.year) for r in requests}) != len(requests):
            raise ValueError('Duplicate field/year in batch')
        reports = [run(request,settings,args.mode) for request in requests]
        output = reports[0] if not isinstance(payload,list) else {'results':reports}
        frozen = {'threshold':settings.threshold,'rain_mm':settings.rain_mm,
                  'baseline':[settings.baseline_start,settings.baseline_end],
                  'drought_percentile':settings.drought_percentile,'model':settings.nvidia_model}
        output['frozen_configuration'] = frozen
        output['configuration_sha256'] = sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    else:
        from .evaluation import compare
        prediction = json.loads(Path(args.predictions).read_text())
        reference = json.loads(Path(args.references).read_text())
        reports = prediction.get('results',[prediction]) if isinstance(prediction,dict) else prediction
        references = reference.get('records',[]) if isinstance(reference,dict) else reference
        output = compare(reports,references)
    target = Path(args.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(f'Saved {target}')


if __name__ == '__main__':
    main()

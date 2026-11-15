import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    threshold: float = 0.80
    # Programmatic callers retain strict evidence policy unless explicitly selected.
    prediction_mode: str = 'evidence_only'
    rain_mm: float = 10.0
    frost_window_days: int = 7
    drought_percentile: float = 20.0
    historical_weather: bool = True
    nass_context: bool = True
    baseline_start: int = 1981
    baseline_end: int = 2010
    max_calls: int = 8
    max_rounds: int = 6
    max_corrections: int = 2
    nvidia_max_tokens: int = 2048
    nvidia_max_attempts: int = 3
    nvidia_read_timeout: float = 60
    nvidia_max_concurrency: int = 2
    nvidia_requests_per_minute: int = 40
    nvidia_circuit_failures: int = 3
    nvidia_circuit_cooldown: float = 20
    prefetch_evidence: bool = True
    max_run_seconds: float = 180
    http_timeout: float = 25
    cache_days: int = 30
    max_http_requests: int = 30
    nvidia_key: str = ''
    nvidia_reasoning_effort: str = ''
    nvidia_model: str = 'nvidia/nemotron-3.5-super-vl-120b-a12b'
    google_key: str = ''
    google_cx: str = ''
    public_search_enabled: bool = False
    public_report_sources: str = 'investigation/public_report_sources.json'
    nass_key: str = ''
    agricultural_csv: str = ''
    verified_records: str = ''
    rf_bundle: str = 'rf_silage_grain_models/rf_nl_console.joblib'

    @property
    def cache_dir(self):
        return self.root / '.cache' / 'investigation'

    @classmethod
    def from_env(cls, root=ROOT):
        from dotenv import load_dotenv
        load_dotenv(root / '.env', override=False)
        load_dotenv(root / '.env.investigation', override=False)
        cfg = cls(root=Path(root), threshold=float(os.getenv('INVESTIGATION_THRESHOLD', '.80')),
                  prediction_mode=os.getenv('INVESTIGATION_PREDICTION_MODE', 'always_predict').strip().lower(),
                  max_corrections=int(os.getenv('INVESTIGATION_MAX_CORRECTIONS', '2')),
                  nvidia_max_tokens=int(os.getenv('NVIDIA_MAX_TOKENS', '2048')),
                  nvidia_max_attempts=int(os.getenv('NVIDIA_MAX_ATTEMPTS', '3')),
                  nvidia_read_timeout=float(os.getenv('NVIDIA_READ_TIMEOUT_SECONDS', '60')),
                  nvidia_max_concurrency=int(os.getenv('NVIDIA_MAX_CONCURRENCY', '2')),
                  nvidia_requests_per_minute=int(os.getenv('NVIDIA_REQUESTS_PER_MINUTE', '40')),
                  nvidia_circuit_failures=int(os.getenv('NVIDIA_CIRCUIT_FAILURE_THRESHOLD', '3')),
                  nvidia_circuit_cooldown=float(os.getenv('NVIDIA_CIRCUIT_COOLDOWN_SECONDS', '20')),
                  prefetch_evidence=os.getenv('INVESTIGATION_PREFETCH_EVIDENCE', 'true').strip().lower() == 'true',
                  nvidia_key=os.getenv('NVIDIA_API_KEY', ''),
                  nvidia_model=os.getenv('NVIDIA_MODEL', cls.nvidia_model),
                  nvidia_reasoning_effort=os.getenv('NVIDIA_REASONING_EFFORT', '').strip().lower(),
                  google_key=os.getenv('GOOGLE_SEARCH_API_KEY', ''), google_cx=os.getenv('GOOGLE_SEARCH_CX', ''),
                  public_search_enabled=os.getenv('PUBLIC_SEARCH_ENABLED', 'false').strip().lower() == 'true',
                  public_report_sources=os.getenv('PUBLIC_REPORT_SOURCES', cls.public_report_sources),
                  nass_key=os.getenv('NASS_API_KEY', ''), agricultural_csv=os.getenv('AGRICULTURAL_CSV', ''),
                  verified_records=os.getenv('VERIFIED_HARVEST_RECORDS', ''),
                  rf_bundle=os.getenv('RF_BUNDLE', cls.rf_bundle),
                  historical_weather=os.getenv('INVESTIGATION_HISTORICAL_WEATHER', 'true').strip().lower() == 'true',
                  nass_context=os.getenv('INVESTIGATION_NASS_CONTEXT', 'true').strip().lower() == 'true',
                  baseline_start=int(os.getenv('DAYMET_BASELINE_START', '1981')),
                  baseline_end=int(os.getenv('DAYMET_BASELINE_END', '2010')),
                  rain_mm=float(os.getenv('WEATHER_RAIN_3DAY_MM', '10')),
                  drought_percentile=float(os.getenv('WEATHER_DROUGHT_PERCENTILE', '20')))
        for name in ('INVESTIGATION_HISTORICAL_WEATHER', 'INVESTIGATION_NASS_CONTEXT'):
            if os.getenv(name, 'true').strip().lower() not in ('true', 'false'):
                raise ValueError(name + ' must be true or false')
        if cfg.prediction_mode not in ('evidence_only', 'always_predict'):
            raise ValueError('INVESTIGATION_PREDICTION_MODE must be evidence_only or always_predict')
        if cfg.nvidia_reasoning_effort not in ('', 'low', 'medium', 'high'):
            raise ValueError('NVIDIA_REASONING_EFFORT must be blank, low, medium or high')
        if not 0 <= cfg.max_corrections <= 3:
            raise ValueError('INVESTIGATION_MAX_CORRECTIONS must be between zero and three')
        if not 256 <= cfg.nvidia_max_tokens <= 8192:
            raise ValueError('NVIDIA_MAX_TOKENS must be between 256 and 8192')
        if not 1 <= cfg.nvidia_max_attempts <= 3:
            raise ValueError('NVIDIA_MAX_ATTEMPTS must be between one and three')
        if not 5 <= cfg.nvidia_read_timeout <= 90:
            raise ValueError('NVIDIA_READ_TIMEOUT_SECONDS must be between 5 and 90')
        if not 1 <= cfg.nvidia_max_concurrency <= 8:
            raise ValueError('NVIDIA_MAX_CONCURRENCY must be between one and eight')
        if not 1 <= cfg.nvidia_requests_per_minute <= 40:
            raise ValueError('NVIDIA_REQUESTS_PER_MINUTE must be between one and forty')
        if not 2 <= cfg.nvidia_circuit_failures <= 5:
            raise ValueError('NVIDIA_CIRCUIT_FAILURE_THRESHOLD must be between two and five')
        if not 5 <= cfg.nvidia_circuit_cooldown <= 60:
            raise ValueError('NVIDIA_CIRCUIT_COOLDOWN_SECONDS must be between five and sixty')
        if os.getenv('INVESTIGATION_PREFETCH_EVIDENCE', 'true').strip().lower() not in ('true', 'false'):
            raise ValueError('INVESTIGATION_PREFETCH_EVIDENCE must be true or false')
        if not (0.5 < cfg.threshold <= 1 and cfg.rain_mm > 0 and 0 < cfg.drought_percentile < 100):
            raise ValueError('Invalid provisional thresholds')
        if not (1980 <= cfg.baseline_start <= cfg.baseline_end <= 2099 and cfg.baseline_end-cfg.baseline_start < 40):
            raise ValueError('Invalid historical baseline (maximum 40 years)')
        return cfg

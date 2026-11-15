from datetime import date
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from shapely.geometry import shape


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Interval(Strict):
    start: date
    end: date

    @model_validator(mode='after')
    def ordered(self):
        if self.start > self.end:
            raise ValueError('Interval start must precede end')
        return self


class RFResult(Strict):
    p_silage: float = Field(ge=0, le=1)
    p_grain: float = Field(ge=0, le=1)
    model_version: str = Field(min_length=1, max_length=200)
    calibration_scope: str = Field(min_length=1, max_length=400)
    decision_threshold: float = Field(default=0.5, gt=0, lt=1)
    predicted_label: Literal['SILAGE', 'GRAIN'] | None = None
    review_band: tuple[float, float] | None = None
    provenance: Literal['supplied', 'saved_artifact'] = 'supplied'
    artifact_sha256: str | None = None
    raw_p_silage: float | None = Field(default=None, ge=0, le=1)
    barc_fitted: bool = False

    @model_validator(mode='after')
    def consistent(self):
        if abs(self.p_silage + self.p_grain - 1) > 1e-6:
            raise ValueError('RF probabilities must sum to one')
        expected = 'SILAGE' if self.p_silage >= self.decision_threshold else 'GRAIN'
        if self.predicted_label and self.predicted_label != expected:
            raise ValueError('RF label inconsistent with its original model threshold')
        self.predicted_label = expected
        if self.review_band and not (0 <= self.review_band[0] < self.review_band[1] <= 1):
            raise ValueError('Invalid model review band')
        return self


class Quality(Strict):
    observations_adequate: bool | None = None
    geographic_applicability: Literal['validated', 'outside_validated_scope', 'unknown'] = 'unknown'
    flags: list[str] = Field(default_factory=list, max_length=30)


class PhysicalEvidence(Strict):
    """Only an existing processing result, never newly fetched imagery."""
    removal_confirmed: bool | None = None
    canopy_condition: Literal['green', 'dry', 'uncertain'] = 'uncertain'
    standing_drydown_confirmed: bool | None = None
    corroborated: bool = False
    contradictory: bool = False
    source_ids: list[str] = Field(default_factory=list, max_length=20)


class Maturity(Strict):
    minimum_grain_gdd_f: float = Field(gt=0, le=6000)
    maximum_grain_gdd_f: float = Field(gt=0, le=6000)
    source_url: str
    description: str = Field(min_length=1, max_length=300)

    @model_validator(mode='after')
    def ordered(self):
        if self.minimum_grain_gdd_f > self.maximum_grain_gdd_f:
            raise ValueError('Invalid maturity range')
        if not self.source_url.startswith('https://'):
            raise ValueError('Maturity assumptions require a public source URL')
        return self


class Geography(Strict):
    country: Literal['US', 'NL', 'other'] = 'US'
    state_alpha: str | None = Field(default=None, pattern=r'^[A-Z]{2}$')
    county_fips: str | None = Field(default=None, pattern=r'^\d{3}$')
    county_name: str | None = Field(default=None, max_length=100)
    farm_name: str | None = Field(default=None, max_length=150)


class FeatureSummary(Strict):
    metric: Literal['NDVI', 'NDRE', 'LSWI', 'NDTI', 'VH_DB', 'VV_DB', 'NDVI_DROP',
                    'PREHARVEST_NDVI', 'NDRE_PEAK_FRACTION', 'DAYS_PEAK_TO_HARVEST']
    value: float
    units: Literal['unitless', 'dB', 'days']
    observed_between: Interval | None = None
    source_ids: list[str] = Field(min_length=1, max_length=10)

    @model_validator(mode='after')
    def metric_units(self):
        expected = 'dB' if self.metric in ('VH_DB', 'VV_DB') else 'days' if self.metric == 'DAYS_PEAK_TO_HARVEST' else 'unitless'
        if self.units != expected:
            raise ValueError('Feature summary units disagree with the metric')
        return self


class InvestigationInput(Strict):
    field_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,128}$')
    year: int = Field(ge=1980, le=2100)
    reporting_cutoff: date
    geometry: dict
    geography: Geography = Field(default_factory=Geography)
    rf: RFResult | None = None
    rf_model: Literal['configured', 'barc_development'] = 'configured'
    features: dict[str, float | None] | None = None
    feature_contract: Literal['rf_s2s1_135_v1'] | None = None
    quality: Quality = Field(default_factory=Quality)
    planting: Interval | None = None
    harvest: Interval | None = None
    greenness_decline_date: date | None = None
    radar_dates: list[date] = Field(default_factory=list, max_length=20)
    standing_drydown: Interval | None = None
    maturity: Maturity | None = None
    physical_evidence: PhysicalEvidence = Field(default_factory=PhysicalEvidence)
    feature_summary: list[FeatureSummary] = Field(default_factory=list, max_length=40)

    @field_validator('geometry')
    @classmethod
    def polygon(cls, value):
        if value.get('type') not in ('Polygon', 'MultiPolygon'):
            raise ValueError('A WGS84 Polygon or MultiPolygon is required')
        g = shape(value)
        if not g.is_valid or g.is_empty:
            raise ValueError('Invalid field polygon')
        xmin, ymin, xmax, ymax = g.bounds
        if not all(isfinite(v) for v in g.bounds) or not (-180 <= xmin <= xmax <= 180 and -90 <= ymin <= ymax <= 90):
            raise ValueError('Invalid longitude/latitude')
        if xmax - xmin > 1 or ymax - ymin > 1:
            raise ValueError('Field extent exceeds one degree; submit individual fields')
        if len(str(value)) > 1_000_000:
            raise ValueError('Field polygon too large')
        return value

    @model_validator(mode='after')
    def dates_and_features(self):
        if self.reporting_cutoff < date(self.year, 1, 1) or self.reporting_cutoff > date(self.year + 1, 12, 31):
            raise ValueError('Cutoff must fall in the crop year or following year')
        event_dates = list(self.radar_dates)
        for interval in (self.planting, self.harvest, self.standing_drydown):
            if interval:
                event_dates.extend([interval.start, interval.end])
        for summary in self.feature_summary:
            if summary.observed_between:
                event_dates.extend([summary.observed_between.start, summary.observed_between.end])
        if self.greenness_decline_date:
            event_dates.append(self.greenness_decline_date)
        if any(d < date(self.year, 1, 1) or d > self.reporting_cutoff for d in event_dates):
            raise ValueError('Event outside crop year/reporting cutoff')
        if self.planting and self.harvest and self.planting.end > self.harvest.start:
            raise ValueError('Planting interval overlaps harvest interval')
        if self.features is not None and self.feature_contract is None:
            raise ValueError('Declare feature_contract=rf_s2s1_135_v1; use the existing feature builder')
        return self


class ToolResult(Strict):
    evidence_id: str
    tool: str
    field_id: str
    year: int
    status: Literal['OK', 'PARTIAL', 'UNKNOWN']
    sources: list[dict] = Field(default_factory=list)
    data: dict = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class NemotronHypothesis(Strict):
    """A proposal referencing backend findings, never new measurements or a final decision."""
    label: Literal['SILAGE', 'GRAIN', 'UNRESOLVED']
    evidence_strength: Literal['insufficient', 'limited', 'moderate', 'strong']
    evidence_ids: list[str] = Field(max_length=20)
    reason_codes: list[Literal['RF_HYPOTHESIS', 'RF_INPUT_LIMITATIONS', 'CONTEXT_ONLY',
                              'MISSING_EVENT_INPUTS', 'FIELD_HARVEST_RECORD',
                              'CORROBORATED_PHYSICAL_EVIDENCE', 'DEVELOPMENT_TIMING',
                              'CONFLICTING_EVIDENCE', 'NO_USABLE_EVIDENCE', 'SATELLITE_TRAJECTORY']] = Field(min_length=1, max_length=9)

    @model_validator(mode='after')
    def citations(self):
        if len(self.evidence_ids) != len(set(self.evidence_ids)) or len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError('Duplicate citations or reason codes are not independent evidence')
        if self.label != 'UNRESOLVED' and not self.evidence_ids:
            raise ValueError('A harvest hypothesis requires citations')
        return self


class VerifiedRecord(Strict):
    """A source reviewed outside the LLM; never accepted from a tool-call argument."""
    field_id: str
    year: int
    harvest_use: Literal['SILAGE', 'GRAIN']
    harvest: Interval
    published_on: date
    source_url: str
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    verified_by: str = Field(min_length=1)
    verification_method: str = Field(min_length=20)
    source_type: Literal['public_actual_harvest_record']

    @field_validator('source_url')
    @classmethod
    def public_url(cls, value):
        if not value.startswith('https://'):
            raise ValueError('Verified harvest records require a public HTTPS URL')
        return value

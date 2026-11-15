from datetime import date
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from investigation.config import Settings
from investigation.schemas import InvestigationInput
from investigation.http import DataUnavailable
from investigation.tools import direct_documents, public_documents, search_public_harvest_evidence, get_agricultural_context


def request():
    return InvestigationInput.model_validate({
        'field_id': 'TEST-FIELD', 'year': 2023, 'reporting_cutoff': '2023-12-31',
        'geometry': {'type': 'Polygon', 'coordinates': [[[-76.9, 39], [-76.89, 39], [-76.89, 39.01], [-76.9, 39]]]},
        'geography': {'country': 'US', 'state_alpha': 'MD', 'county_fips': '033', 'county_name': "Prince George's"}})


class DirectSourcesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'sources.json'
        self.settings = Settings(public_report_sources=str(self.path))
        self.row = {'source_id': 'fixture', 'url': 'https://example.org/report', 'year': 2023,
                    'state_alpha': 'MD', 'published_on': '2023-09-01', 'scope': 'regional'}
        self.http = MagicMock()
        self.http.get.return_value = (b'<html><p>Regional corn harvest report.</p></html>', False)

    def write(self, rows):
        self.path.write_text(json.dumps({'sources': rows}))

    def test_direct_reads_work_without_any_search_key(self):
        self.write([self.row])
        with patch('investigation.tools.search_documents', side_effect=AssertionError('Search must not run')):
            docs = public_documents('ignored query', request(), self.settings, self.http)
            web = search_public_harvest_evidence(request(), self.settings, self.http)
            agriculture = get_agricultural_context(request(), self.settings, self.http)
        self.assertEqual(len(docs), 1)
        self.assertIn('Regional corn harvest', docs[0]['text_excerpt'])
        self.assertEqual(docs[0]['field_association'], 'unverified')
        self.assertFalse(docs[0]['usable_as_direct_harvest_evidence'])
        self.assertEqual(web.data['verified_actual_harvest_records'], [])
        self.assertEqual(agriculture.data['acreage_records'], [])
        self.http.json.assert_not_called()

    def test_different_year_state_field_county_and_future_dates_are_excluded(self):
        variations = [{'year': 2022}, {'state_alpha': 'TX'}, {'field_id': 'OTHER'},
                      {'county_fips': '001'}, {'published_on': '2024-01-01'}]
        self.write([dict(self.row, **v) for v in variations])
        self.assertEqual(direct_documents(request(), self.settings, self.http), [])
        self.http.get.assert_not_called()

    def test_manifest_cannot_import_reference_labels(self):
        self.write([dict(self.row, reference_label='SILAGE')])
        with self.assertRaises(DataUnavailable):
            direct_documents(request(), self.settings, self.http)
        self.http.get.assert_not_called()

    def test_duplicate_urls_are_read_once_and_failures_are_explicit(self):
        self.write([self.row, self.row])
        self.http.get.side_effect = DataUnavailable('Public source HTTP 503')
        docs = direct_documents(request(), self.settings, self.http)
        self.assertEqual(len(docs), 1)
        self.assertIn('503', docs[0]['fetch_error'])
        self.assertFalse(docs[0]['usable_as_direct_harvest_evidence'])
        self.assertEqual(self.http.get.call_count, 1)

    def test_empty_manifest_returns_unknown_not_grain(self):
        self.write([])
        result = search_public_harvest_evidence(request(), self.settings, self.http)
        self.assertEqual(result.status, 'UNKNOWN')
        self.assertTrue(result.missing_inputs)


if __name__ == '__main__':
    unittest.main()

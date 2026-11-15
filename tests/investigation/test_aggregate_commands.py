"""Aggregate acreage stays usable without entering the field or model workflow."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from investigation.__main__ import main
from investigation.config import Settings
from investigation.schemas import ToolResult
from test_agent_protocol import request


class AggregateCommandTests(unittest.TestCase):
    def test_separate_acreage_command_never_runs_model_or_field_decision(self):
        req = request()
        result = ToolResult(tool='get_agricultural_context', evidence_id='agriculture',
                            field_id=req.field_id, year=req.year, status='PARTIAL',
                            data={'acreage_records': [{'numeric_value': 123, 'unit': 'ACRES'}]})
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'input.json', Path(folder)/'acreage.json'
            source.write_text(req.model_dump_json())
            with patch('sys.argv', ['investigation', 'aggregate-acreage', '--input', str(source), '--out', str(target)]), \
                 patch('investigation.__main__.Settings.from_env', return_value=Settings()), \
                 patch('investigation.tools.get_agricultural_context', return_value=result) as retrieve, \
                 patch('investigation.agent.run', side_effect=AssertionError('No field investigation')), \
                 patch('investigation.agent.requests.Session.post', side_effect=AssertionError('No NVIDIA request')):
                main()
            output = json.loads(target.read_text())
        retrieve.assert_called_once()
        self.assertEqual(output['scope'], 'aggregate_acreage_only')
        self.assertFalse(output['field_classification_performed'])
        self.assertEqual(output['evidence']['data']['acreage_records'][0]['numeric_value'], 123)
        self.assertNotIn('final_label', output)


if __name__ == '__main__':
    unittest.main()

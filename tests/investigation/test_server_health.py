"""A running server must show its loaded transport and field tool settings."""
import json
import unittest
from unittest.mock import patch

from investigation import server
from investigation.agent import runtime_configuration
from investigation.config import Settings


class HealthTests(unittest.TestCase):
    def test_health_reports_same_safe_effective_settings_as_field_reports(self):
        settings = Settings(nvidia_key='SECRET-NVIDIA', nass_key='SECRET-NASS',
                            nvidia_read_timeout=70, nvidia_max_concurrency=1)
        with patch.object(server, 'settings', settings):
            response = server.health()
        self.assertEqual(response['runtime_configuration'], runtime_configuration(settings))
        self.assertEqual(response['runtime_configuration']['nvidia_read_timeout_seconds'], 70)
        self.assertEqual(response['runtime_configuration']['transport_implementation'], 'rca-recovery-v2')
        self.assertTrue(response['quickstats_configured'])
        self.assertNotIn('SECRET', json.dumps(response))


if __name__ == '__main__':
    unittest.main()

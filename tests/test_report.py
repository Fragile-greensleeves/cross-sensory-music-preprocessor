import json
import re
import tempfile
import unittest
from pathlib import Path

from musicprep.report import write_report
from test_validation import package_fixture


class ReportTest(unittest.TestCase):
    def test_script_escape_and_json_round_trip(self):
        package = package_fixture()
        attack = '</script><script>alert("x")</script><img onerror="bad()"> & \u2028'
        package['track_id'] = attack
        package['warnings'].append(attack)
        package['descriptions'][0]['caption'] = attack
        with tempfile.TemporaryDirectory() as tmp:
            output = write_report(package, Path(tmp) / 'report.html')
            content = output.read_text()
        self.assertNotIn(attack, content)
        self.assertNotIn('<script>alert', content)
        embedded = re.search(r'<script id="package-data" type="application/json">(.*?)</script>', content, re.S).group(1)
        self.assertEqual(json.loads(embedded), package)
        self.assertNotIn('innerHTML', content)
        self.assertEqual(content.count('</script>'), 2)

    def test_offline_audio_relative_and_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audio = root / 'audio & demo.wav'; audio.write_bytes(b'not-used-for-analysis')
            output = write_report(package_fixture(), root / 'review' / 'report.html', audio)
            content = output.read_text()
            self.assertIn('src="../audio%20%26%20demo.wav"', content)
            self.assertNotIn(tmp, content)
            self.assertIn('type="file" accept="audio/*"', content)
            self.assertIn('URL.createObjectURL(file)', content)
            self.assertNotRegex(content, r'<script[^>]+src=')
            self.assertNotIn('fetch(', content)
            output = write_report(package_fixture(), root / 'no-audio.html')
            self.assertIn('src=""', output.read_text())

    def test_template_tokens_in_user_text_are_not_reprocessed(self):
        package = package_fixture(); package['track_id'] = '__PACKAGE_JSON__'
        with tempfile.TemporaryDirectory() as tmp:
            content = write_report(package, Path(tmp) / 'report.html').read_text()
        self.assertIn('<title>__PACKAGE_JSON__ · 音乐分析诊断</title>', content)
        self.assertEqual(content.count('<script id="package-data"'), 1)

    def test_status_and_source_data_remain_available(self):
        package = package_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            content = write_report(package, Path(tmp) / 'report.html').read_text()
        self.assertIn('test-only', content)
        self.assertIn('部分结果 / 有回退', content)
        self.assertIn('complete 仅表示必需任务执行完成', content)
        self.assertIn('audio.addEventListener', content)
        self.assertIn('requestAnimationFrame', content)


if __name__ == '__main__':
    unittest.main()

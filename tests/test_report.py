import copy
import json
import re
import shutil
import subprocess
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

    def test_public_view_is_allowlisted_audio_free_and_preserves_status_counts(self):
        package = package_fixture()
        secret = 'PRIVATE FULL LYRIC AND PROMPT'
        private_path = '/Users/private-owner/songs/private.flac'
        package['lyrics_context'] = {key: [{'text': secret, 'words': [{'text': secret, 'token': secret}]}]
                                    for key in ('lyrics', 'themes', 'narrative', 'imagery', 'background', 'sources')}
        package['extension'] = {'raw_output': secret}
        package['audio']['source_path'] = private_path
        package['melody']['contour'][0]['quality'] = {'raw_text': secret}
        package['warnings'].append(secret + private_path)
        package['descriptions'][0].update(caption=secret, raw_output=secret,
            genre=[{'label': 'pop', 'source': 'essentia', 'score': 0.7, 'raw_output': secret},
                   {'label': 'rock', 'source': 'qwen2_audio'}],
            model_results={'qwen2_audio': {'status': 'failed', 'error': secret, 'data': secret}, 'silent_example': {'status': 'succeeded', 'inference_status': 'silent_audio', 'data': secret}})
        counts = {'requested': 10, 'succeeded': 8, 'failed': 1, 'missing': 1, 'rejected': 2, 'generated': 7, 'silent_audio': 1}
        package['run']['tasks']['descriptions'] = {'status': 'partial', 'error': secret,
            'provenance': {'engine': 'essentia+qwen2_audio', 'model_id': private_path,
                'models': {'qwen2_audio': {'status': 'partial', 'counts': counts,
                    'error': secret, 'log_path': private_path,
                    'runtime': {'model_id': 'mlx-community/Qwen2-Audio', 'prompt': secret,
                                'raw_output': secret, 'audit_directory': private_path}}}}}
        before = copy.deepcopy(package)
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / 'audio.wav'; audio.write_bytes(b'audio')
            public = write_report(package, Path(tmp) / 'public.html', audio, public=True).read_text()
            local = write_report(package, Path(tmp) / 'local.html').read_text()
        embedded = json.loads(re.search(r'<script id="package-data" type="application/json">(.*?)</script>', public, re.S).group(1))
        self.assertNotIn(secret, public)
        self.assertNotIn(private_path, public)
        self.assertNotIn(tmp, public)
        self.assertIn('src=""', public)
        self.assertIn(secret, local)
        self.assertEqual(package, before)
        self.assertEqual(embedded['run']['status'], 'partial')
        self.assertEqual(embedded['run']['tasks']['descriptions']['provenance']['models']['qwen2_audio']['counts'], counts)
        self.assertEqual(embedded['lyrics_context']['lyrics'], [])
        self.assertEqual(embedded['publication']['lyrics_line_count'], 1)
        self.assertEqual(embedded['publication']['lyrics_word_count'], 1)
        self.assertEqual(embedded['descriptions'][0]['genre'][1]['source'], 'qwen2_audio')
        self.assertEqual(embedded['descriptions'][0]['model_results']['qwen2_audio'], {'status': 'failed'})
        self.assertNotIn('extension', embedded)
        self.assertEqual(embedded['descriptions'][0]['model_results']['silent_example'], {'status': 'succeeded', 'inference_status': 'silent_audio'})

    def test_sources_and_coverage_are_explicit_without_accuracy_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = write_report(package_fixture(), Path(tmp) / 'report.html').read_text()
        self.assertIn('已接受 ${c.succeeded} / 请求 ${c.requested} 片段', content)
        self.assertIn('拒绝结果 ${c.rejected', content)
        self.assertIn('模型分数不是识别准确率', content)
        self.assertIn('modelSummary(d.model_results', content)
        self.assertIn('sourceName(source)+score', content)
        self.assertIn('生成窗口 ${c.generated??0} · 静音跳过 ${c.silent_audio??0}', content)
        self.assertIn('静音规则返回空标签，未执行生成', content)

    @unittest.skipUnless(shutil.which('node'), 'JavaScript runtime not available')
    def test_current_lyrics_time_switch_gaps_line_only_and_public_guard(self):
        package = package_fixture()
        line = {'text': '测试 Do', 'start': 1.0, 'end': 3.0,
                'source': 'sofa_reference_lyrics', 'timing_status': 'estimated',
                'words': [{'text': '测', 'token': 'ce', 'start': 1.0, 'end': 1.4},
                          {'text': '试', 'token': 'shi', 'start': 1.6, 'end': 2.0},
                          {'text': 'Do', 'token': 'latin_do', 'start': 2.0, 'end': 3.0}]}
        package['lyrics_context']['lyrics'] = [line, {'text': 'only line', 'start': 5.0, 'end': 6.0}]
        with tempfile.TemporaryDirectory() as tmp:
            content = write_report(package, Path(tmp) / 'report.html').read_text()
        functions = content[content.index('function lyricState(time)'):content.index('function update(){')]
        script = "const P=" + json.dumps(package) + ";const lyrics=P.lyrics_context;" + r"""
const nodes={};const $=id=>nodes[id]||(nodes[id]={});
const text=(node,value)=>node.textContent=String(value);const labels={estimated:'估计'};
""" + functions + r"""
const assert=require('node:assert/strict');
showLyric(1.2);assert.equal(nodes.currentWord.textContent,'测');assert.match(nodes.lyricTiming.textContent,/SOFA 估计.*1.00–1.40 秒/);
showLyric(1.5);assert.equal(nodes.currentWord.textContent,'—');assert.match(nodes.lyricTiming.textContent,/词间空白/);
showLyric(2.0);assert.equal(nodes.currentWord.textContent,'Do');assert.ok(!nodes.currentWord.textContent.includes('latin_'));
showLyric(3.0);assert.equal(nodes.currentWord.textContent,'—');assert.match(nodes.currentLyric.textContent,/句间空白/);
showLyric(5.5);assert.equal(nodes.currentLyric.textContent,'only line');assert.match(nodes.lyricTiming.textContent,/仅行级时间/);
P.publication={redacted:true,lyrics_line_count:2,lyrics_word_count:3};
showLyric(1.2);assert.equal(nodes.currentWord.hidden,true);assert.match(nodes.currentLyric.textContent,/隐藏原文.*2 行.*3 个/);assert.equal(nodes.lyricTiming.textContent,'');
assert.deepEqual(lyricState(1.2),{hidden:true});
"""
        completed = subprocess.run([shutil.which('node'), '-e', script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()

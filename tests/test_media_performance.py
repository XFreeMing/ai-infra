"""Performance formulas must use generation time, not model-loading time."""
import math
import unittest

from lab_tools.media_performance import ratio, summarize


class MediaPerformanceTests(unittest.TestCase):
    def manifest(self, app):
        return {'app': app, 'run_id': 'test', 'status': 'passed', 'duration_seconds': 20,
                'configuration': {'text': '你好。', 'sampler': 'V4_TURBO_12'},
                'commands': [{'label': 'generate', 'duration_seconds': 18}],
                'result': {'passed': True, 'width': 256, 'height': 256,
                           'duration_seconds': 4, 'sample_rate': 22050, 'channels': 1}}

    def runtime(self):
        return {'phases': {name: {'status': 'succeeded', 'duration_seconds': value, 'duration_complete': True,
                                'gpu_synchronized_at_start': True, 'gpu_synchronized_at_end': True}
                          for name, value in [('runtime_check', 1), ('model_load', 13), ('generation', 2)]}}

    def test_audio_rtf_distinguishes_generation_from_end_to_end(self):
        report = summarize(self.manifest('indextts2'), self.runtime(), {'sample_count': 10})
        self.assertEqual(report['generation']['real_time_factor'], 0.5)
        self.assertEqual(report['generation']['audio_seconds_per_compute_second'], 2)
        self.assertEqual(report['generation']['end_to_end_real_time_factor'], 5)
        self.assertEqual(report['generation']['characters_per_second'], 1.5)
        self.assertEqual(report['timings']['model_load_seconds'], 13)
        self.assertEqual(report['measurement_status'], 'complete')

    def test_image_latency_and_throughput_are_pipeline_metrics(self):
        report = summarize(self.manifest('ideogram4'), self.runtime(), {'sample_count': 10})
        self.assertEqual(report['generation']['generation_seconds_per_image'], 2)
        self.assertEqual(report['generation']['images_per_second'], 0.5)
        self.assertEqual(report['generation']['megapixels_per_second'], 256 * 256 / 1e6 / 2)
        self.assertEqual(report['generation']['end_to_end_seconds_per_image'], 20)

    def test_failed_or_missing_phase_has_no_throughput(self):
        for status in ('failed', 'interrupted', 'running'):
            runtime = self.runtime()
            runtime['phases']['generation']['status'] = status
            report = summarize(self.manifest('indextts2'), runtime, {'sample_count': 10})
            self.assertIsNone(report['generation']['real_time_factor'])
            self.assertEqual(report['measurement_status'], 'partial')
        self.assertIsNone(summarize(self.manifest('ideogram4'), {}, {})['generation']['images_per_second'])

    def test_invalid_output_has_no_image_or_audio_throughput(self):
        for app, metric in [('ideogram4', 'images_per_second'), ('indextts2', 'real_time_factor')]:
            manifest = self.manifest(app)
            manifest['status'] = 'failed'
            report = summarize(manifest, self.runtime(), {'sample_count': 10})
            self.assertIsNone(report['generation'][metric])

    def test_blocked_without_worker_has_no_inference_measurements(self):
        manifest = self.manifest('ideogram4')
        manifest.update(status='blocked', commands=[], duration_seconds=0.68)
        report = summarize(manifest, {}, {})
        self.assertEqual(report['measurement_status'], 'not_started')
        self.assertEqual(report['timings']['hello_wall_seconds'], 0.68)
        self.assertIsNone(report['timings']['generation_seconds'])
        self.assertIsNone(report['generation']['images_per_second'])
        self.assertEqual(report['generation']['image_count'], 0)
        self.assertTrue(all(value is None for value in report['artifacts'].values()))

    def test_existing_worker_evidence_is_never_reported_as_not_started(self):
        for commands, runtime, resources in [
            ([{'label': 'generate', 'status': 'failed'}], {}, {}),
            ([], {'phases': {'generation': {'status': 'running'}}}, {}),
            ([], {}, {'sample_count': 1}),
        ]:
            manifest = self.manifest('ideogram4')
            manifest.update(status='blocked', commands=commands)
            report = summarize(manifest, runtime, resources)
            self.assertEqual(report['measurement_status'], 'partial')
        manifest.update(status='failed', commands=[])
        self.assertEqual(summarize(manifest, {}, {})['measurement_status'], 'partial')

    def test_unsynchronized_or_incomplete_timings_cannot_measure_throughput(self):
        for key in ('gpu_synchronized_at_start', 'gpu_synchronized_at_end', 'duration_complete'):
            for missing in (False, True):
                runtime = self.runtime()
                if missing:
                    del runtime['phases']['generation'][key]
                else:
                    runtime['phases']['generation'][key] = False
                report = summarize(self.manifest('indextts2'), runtime, {'sample_count': 10})
                self.assertIsNone(report['generation']['real_time_factor'])
                self.assertEqual(report['measurement_status'], 'partial')

    def test_invalid_divisors_and_nonfinite_values_are_unavailable(self):
        for value in (0, -1, None, math.nan, math.inf, True, '1'):
            self.assertIsNone(ratio(1, value))
        self.assertEqual(ratio(0, 1), 0)


if __name__ == '__main__':
    unittest.main()

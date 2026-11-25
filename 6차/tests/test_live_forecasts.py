"""운영 저장본 예측을 학습 채택과 분리하는 경계를 검사합니다."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import modeling
from src.core import save_json, read_json
import run_pipeline


class LiveForecastTests(unittest.TestCase):
    def test_loop_predicts_before_rejected_or_waiting_training(self):
        for status in ('rejected', 'waiting_new_data'):
            events = []
            with tempfile.TemporaryDirectory() as folder, \
                 patch.object(run_pipeline, 'STOP', Path(folder)/'stop'), \
                 patch.object(run_pipeline, 'load_market', return_value={'series': {'A': []}}), \
                 patch.object(run_pipeline, 'reconcile_forward'), \
                 patch.object(run_pipeline, 'reconcile_paper'), \
                 patch.object(run_pipeline, 'update_state'), \
                 patch.object(modeling, 'refresh_live_forecasts', create=True, side_effect=lambda: events.append('predict')), \
                 patch.object(run_pipeline, 'train_round', side_effect=lambda: events.append(status)):
                run_pipeline.run_loop('train', {}, False, float('inf'), 0)
            self.assertEqual(events, ['predict', status])

    def test_new_input_publishes_once_and_unchanged_input_skips_loader(self):
        refresh = getattr(modeling, 'refresh_live_forecasts', None)
        self.assertIsNotNone(refresh, '기존 운영 모델의 독립 예측이 필요합니다')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_json(root/'latest.json', {'checkpoint': 'approved', 'sha256': {'m': 'a'}})
            market = {'series': {'A': [{'date': '2026-11-23', 'close': 10}]}}
            with patch.object(modeling, 'PRIVATE', root), patch.object(modeling, 'LATEST', root/'latest.json'), \
                 patch.object(modeling, 'STOP', root/'stop'), patch.object(modeling, 'load_market', return_value=market), \
                 patch.object(modeling, 'load_config', return_value={'context_length': 32}), \
                 patch.object(modeling, 'update_state'), \
                 patch.object(modeling, '_publish_saved_forecasts', create=True) as publish:
                self.assertEqual(refresh()['status'], 'ready')
                self.assertEqual(refresh()['status'], 'unchanged')
                self.assertEqual(publish.call_count, 1)
                market['series']['A'].append({'date': '2026-11-24', 'close': 11})
                self.assertEqual(refresh()['status'], 'ready')
                self.assertEqual(publish.call_count, 2)

    def test_stop_or_prediction_failure_does_not_mark_input_complete(self):
        refresh = getattr(modeling, 'refresh_live_forecasts', None)
        self.assertIsNotNone(refresh)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_json(root/'latest.json', {'checkpoint': 'approved', 'sha256': {'m': 'a'}})
            with patch.object(modeling, 'PRIVATE', root), patch.object(modeling, 'LATEST', root/'latest.json'), \
                 patch.object(modeling, 'STOP', root/'stop'), \
                 patch.object(modeling, 'load_market', return_value={'series': {'A': [{'date': '2026-11-23', 'close': 10}]}}), \
                 patch.object(modeling, 'load_config', return_value={'context_length': 32}), \
                 patch.object(modeling, 'update_state'), \
                 patch.object(modeling, '_publish_saved_forecasts', create=True, side_effect=RuntimeError('예측 실패')):
                with self.assertRaises(RuntimeError):
                    refresh()
                self.assertIsNone(read_json(root/'운영예측_6차.json'))
                (root/'stop').touch()
                with self.assertRaises(InterruptedError):
                    refresh()

    def test_tampered_checkpoint_is_blocked_before_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                modeling._publish_saved_forecasts({'checkpoint': str(root),
                    'sha256': {'model.safetensors': 'wrong'}}, {}, 32, {})

    def test_first_continuous_cycle_refreshes_even_soon_after_pc_boot(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(run_pipeline, 'STOP', Path(folder)/'stop'), \
             patch.object(run_pipeline, 'load_market', return_value={'series': {'A': []}}), \
             patch.object(run_pipeline, 'refresh_market', return_value={'series': {'A': []}}) as refresh, \
             patch.object(run_pipeline.time, 'monotonic', return_value=100), \
             patch.object(run_pipeline, 'reconcile_forward'), patch.object(run_pipeline, 'reconcile_paper'), \
             patch.object(run_pipeline, 'train_round'), patch.object(modeling, 'refresh_live_forecasts'), \
             patch.object(run_pipeline, 'wait_cooperatively', return_value=False), \
             patch.object(run_pipeline, 'update_state'):
            run_pipeline.run_loop('start', {'refresh_seconds': 21600}, True, 0, 0)
            refresh.assert_called_once()


if __name__ == '__main__':
    unittest.main()

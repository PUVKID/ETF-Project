"""치명 CUDA 재시작과 사용자 중지의 별도 프로세스 경계를 검사합니다."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

import run_pipeline as pipeline
from src import runtime


class SupervisionTests(unittest.TestCase):
    def test_only_known_fatal_cuda_errors_request_a_new_process(self):
        self.assertTrue(pipeline.fatal_cuda(RuntimeError('CUDA error: unspecified launch failure')))
        self.assertTrue(pipeline.fatal_cuda(RuntimeError('CUDA error: device-side assert triggered')))
        self.assertFalse(pipeline.fatal_cuda(RuntimeError('CUDA out of memory')))
        self.assertFalse(pipeline.fatal_cuda(RuntimeError('일반 네트워크 실패')))

    def test_supervisor_restarts_only_recovery_exit_and_preserves_worker_argv(self):
        with tempfile.TemporaryDirectory() as folder:
            stop=Path(folder)/'중지요청'
            outcomes=[subprocess.CompletedProcess([],75),subprocess.CompletedProcess([],0)]
            with patch.object(pipeline,'STOP',stop), patch.object(pipeline,'PRIVATE',Path(folder)), \
                 patch.object(pipeline,'worker_lock'), patch.object(pipeline,'update_state'), \
                 patch.object(pipeline,'log'), patch.object(pipeline,'wait_cooperatively',return_value=True) as waiting, \
                 patch.object(pipeline.subprocess,'run',side_effect=outcomes) as launch:
                self.assertEqual(pipeline.supervise(),0)
            self.assertEqual(launch.call_count,2)
            self.assertIn('--worker',launch.call_args.args[0])
            self.assertEqual(launch.call_args.args[0][-2:],['start','--worker'])
            self.assertFalse(launch.call_args.kwargs.get('shell',False))
            waiting.assert_called_once()

    def test_stop_during_failed_worker_prevents_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            stop=Path(folder)/'중지요청'
            def finished(*args,**kwargs):
                stop.write_text('사용자 중지',encoding='utf-8')
                return subprocess.CompletedProcess([],75)
            with patch.object(pipeline,'STOP',stop),patch.object(pipeline,'PRIVATE',Path(folder)), \
                 patch.object(pipeline,'worker_lock'),patch.object(pipeline.subprocess,'run',side_effect=finished) as launch, \
                 patch.object(pipeline,'update_state'),patch.object(pipeline,'log'):
                self.assertEqual(pipeline.supervise(),0)
            self.assertEqual(launch.call_count,1)

    def test_fatal_cuda_exits_worker_instead_of_retrying_same_context(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(pipeline,'STOP',Path(folder)/'중지요청'), \
                 patch.object(pipeline,'load_market',return_value={'series':{'A':[]}}), \
                 patch.object(pipeline,'reconcile_forward'),patch.object(pipeline,'reconcile_paper'), \
                 patch.object(pipeline,'train_round',side_effect=RuntimeError('CUDA error: unspecified launch failure')), \
                 patch.object(pipeline,'record_failure'),patch.object(pipeline,'log'),patch.object(pipeline,'update_state'), \
                 patch.object(pipeline,'wait_cooperatively',return_value=False) as waiting:
                self.assertEqual(pipeline.run_loop('start',{},True,float('inf'),0),75)
            waiting.assert_not_called()

    def test_supervisor_lock_is_visible_to_duplicate_start_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            private=Path(folder)
            with patch.object(runtime,'PRIVATE',private),patch.object(runtime,'LOCK',private/'작업잠금'):
                self.assertFalse(runtime.is_busy())
                with runtime.worker_lock(private/'운용감시잠금'):
                    self.assertTrue(runtime.is_busy())
                self.assertFalse(runtime.is_busy())

    def test_state_probe_never_touches_worker_lock_while_supervisor_is_starting(self):
        with tempfile.TemporaryDirectory() as folder:
            private=Path(folder)
            with patch.object(runtime,'PRIVATE',private):
                with runtime.worker_lock(private/'운용감시잠금'):
                    with patch.object(runtime,'LOCK') as forbidden_worker:
                        forbidden_worker.open.side_effect=AssertionError('학습자 시작 경계에서 작업잠금 접근 금지')
                        self.assertTrue(runtime.is_busy())
                        forbidden_worker.open.assert_not_called()


if __name__=='__main__':
    unittest.main()

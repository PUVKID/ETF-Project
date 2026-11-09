import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from src import core


class AtomicStorageTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows의 실제 파일 공유 규칙 검사")
    def test_a_brief_real_reader_handle_does_not_abort_atomic_save(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "상태.json"
            core.save_json(path, {"value": "원본"})
            ready = threading.Event()
            release = threading.Event()
            errors = []
            actual_replace = os.replace

            def reader():
                with path.open("rb"):
                    ready.set()
                    release.wait(3)

            def attempt_replace(source, target):
                try:
                    return actual_replace(source, target)
                except OSError as error:
                    errors.append(getattr(error, "winerror", None))
                    release.set()
                    raise

            thread = threading.Thread(target=reader)
            thread.start()
            try:
                self.assertTrue(ready.wait(2))
                with patch("src.core.os.replace", side_effect=attempt_replace):
                    core.save_json(path, {"value": "새값"})
            finally:
                release.set()
                thread.join(3)
            self.assertIn(errors[0], (5, 32))
            self.assertEqual(core.read_json(path), {"value": "새값"})
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_permanent_windows_denial_is_bounded_and_preserves_the_original(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "상태.json"
            core.save_json(path, {"value": "원본"})
            denied = PermissionError("검사용 영구 접근 거부")
            denied.winerror = 5
            with patch("src.core.os.replace", side_effect=denied) as replacement, patch("time.sleep"):
                with self.assertRaises(PermissionError) as caught:
                    core.save_json(path, {"value": "새값"})
            self.assertIs(caught.exception, denied)
            self.assertGreaterEqual(replacement.call_count, 2)
            self.assertLessEqual(replacement.call_count, 10)
            self.assertEqual(core.read_json(path), {"value": "원본"})
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_unrelated_windows_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "상태.json"
            core.save_json(path, {"value": "원본"})
            denied = PermissionError("검사용 별도 권한 오류")
            denied.winerror = 1314
            with patch("src.core.os.replace", side_effect=denied) as replacement:
                with self.assertRaises(PermissionError) as caught:
                    core.save_json(path, {"value": "새값"})
            self.assertIs(caught.exception, denied)
            self.assertEqual(replacement.call_count, 1)
            self.assertEqual(core.read_json(path), {"value": "원본"})
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_windows_sharing_violation_can_finish_after_the_reader_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "상태.json"
            core.save_json(path, {"value": "원본"})
            actual_replace = os.replace
            failures = []

            def temporarily_busy(source, target):
                if len(failures) < 2:
                    self.assertEqual(core.read_json(path), {"value": "원본"})
                    denied = PermissionError("검사용 일시 공유 충돌")
                    denied.winerror = 32
                    failures.append(denied)
                    raise denied
                return actual_replace(source, target)

            with patch("src.core.os.replace", side_effect=temporarily_busy), patch("time.sleep"):
                core.save_json(path, {"value": "새값"})
            self.assertEqual(core.read_json(path), {"value": "새값"})
            self.assertEqual(list(Path(folder).iterdir()), [path])


class PipelineRecoveryTests(unittest.TestCase):
    def test_backoff_is_cleared_before_the_next_training_attempt(self):
        import run_pipeline as pipeline
        from src import runtime
        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder)
            stop = temporary / "중지요청"
            attempts = []

            def train():
                attempts.append(runtime.load_state()["job"])
                if len(attempts) == 1:
                    raise RuntimeError("제어된 첫 회차 실패")
                stop.write_text("검사 종료", encoding="utf-8")

            data = {"series": {"A": []}}
            with patch.object(runtime, "PRIVATE", temporary), patch.object(runtime, "STATE", temporary / "상태.json"), \
                 patch.object(runtime, "LOCK", temporary / "작업잠금"), patch.object(pipeline, "STOP", stop), \
                 patch.object(pipeline, "load_config", return_value={"refresh_seconds": 21600}), \
                 patch.object(pipeline, "load_market", return_value=data), patch.object(pipeline, "refresh_market", return_value=data), \
                 patch.object(pipeline, "reconcile_forward"), patch.object(pipeline, "reconcile_paper"), \
                 patch.object(pipeline, "wait_cooperatively", return_value=True), patch.object(pipeline, "train_round", side_effect=train):
                pipeline.run("start")
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[1]["status"], "running")
            self.assertIsNone(attempts[1]["next_run_at"])


if __name__ == "__main__":
    unittest.main()

"""날짜별 학습 예산과 운영 저장본 보호 경계를 검사합니다."""

import importlib
import math
from contextlib import ExitStack
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from src.core import read_json, save_json
from src import modeling, runtime
import run_pipeline


def corpus(days, value=10):
    return {"segments": [{"symbol": "QQQM", "period": "historical", "eligible": len(days) >= 2,
                          "rows": [{"date": day, "close": value} for day in days]}]}


class OverfitGuardTests(unittest.TestCase):
    def guard(self):
        self.assertIsNotNone(importlib.util.find_spec("src.overfit_guard"), "영속 과적합 방어가 필요합니다")
        return importlib.import_module("src.overfit_guard")

    def test_reserved_dates_survive_restart_and_value_or_fetch_changes_do_not_retrain(self):
        guard = self.guard()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "원장.json"
            first = guard.reserve_training(path, corpus(["2024-01-02", "2024-01-03"]), 1000, 2)
            self.assertEqual(first["status"], "reserved")
            self.assertEqual(first["budget_steps"], 1)
            second = guard.reserve_training(path, corpus(["2024-01-02", "2024-01-03"], 99), 1000, 2)
            self.assertEqual(second["status"], "waiting_new_data")
            self.assertEqual(read_json(path)["attempts"][0]["status"], "reserved")
            third = guard.reserve_training(path, corpus(["2024-01-02", "2024-01-03", "2024-01-04"]), 1000, 2)
            self.assertEqual(third["new_observations"], 1)
            self.assertEqual(third["budget_steps"], 1)
            self.assertEqual(len(read_json(path)["attempts"]), 2)

    def test_first_bootstrap_and_large_new_batch_are_capped_at_25(self):
        guard = self.guard()
        from datetime import date, timedelta
        days = [(date(2020, 1, 1) + timedelta(days=n)).isoformat() for n in range(200)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "원장.json"
            first = guard.reserve_training(path, corpus(days[:100]), 1000, 2)
            self.assertTrue(first["bootstrap"])
            self.assertEqual(first["budget_steps"], 25)
            second = guard.reserve_training(path, corpus(days), 1000, 2)
            self.assertFalse(second["bootstrap"])
            self.assertEqual(second["budget_steps"], 25)

    def test_corrupt_or_incomplete_ledger_fails_closed(self):
        guard = self.guard()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "원장.json"
            for content in ("{", "null", "{}", '{"schema_version":1,"consumed_dates":{},"attempts":[]}'):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "원장"):
                        guard.reserve_training(path, corpus(["2024-01-02", "2024-01-03"]), 1000, 2)
                    self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_pending_singleton_does_not_spend_the_bootstrap(self):
        guard = self.guard()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "원장.json"
            result = guard.reserve_training(path, corpus(["2024-01-02"]), 1000, 2)
            self.assertEqual(result["status"], "waiting_new_data")
            self.assertFalse(path.exists())
            result = guard.reserve_training(path, corpus(["2024-01-02", "2024-01-03"]), 1000, 2)
            self.assertTrue(result["bootstrap"])

    def test_failed_and_rejected_attempts_never_refund_dates(self):
        guard = self.guard()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "원장.json"
            rows = corpus(["2024-01-02", "2024-01-03"])
            attempt = guard.reserve_training(path, rows, 1000, 2)
            guard.finish_attempt(path, attempt["attempt_id"], "rejected", {"reason": "검증 악화"})
            self.assertEqual(read_json(path)["attempts"][0]["status"], "rejected")
            self.assertEqual(guard.reserve_training(path, rows, 1000, 2)["status"], "waiting_new_data")

    def test_validation_gate_rejects_each_worsening_and_invalid_metric(self):
        guard = self.guard()
        before = {"n": 144, "mae": 1.0, "mape": 0.01}
        self.assertTrue(guard.validation_decision(before, dict(before))["accepted"])
        for after in ({**before, "mae": 1.00001}, {**before, "mape": 0.01001},
                      {**before, "mae": math.nan}, {**before, "mape": math.inf},
                      {**before, "n": 0}, {**before, "n": 143}):
            with self.subTest(after=after):
                self.assertFalse(guard.validation_decision(before, after)["accepted"])
        for invalid in ({**before, "mae": math.inf}, {**before, "n": 0}):
            self.assertFalse(guard.validation_decision(invalid, before)["accepted"])


class RoundProtectionTests(unittest.TestCase):
    def exercise(self, after=None, before=None, stop_at=None, failure=None, second=False, corrupt=False,
                 missing_test=False, missing_baseline=False, corrupt_test=False):
        # 실제 원장·상태·train_round를 실행하고 GPU·네트워크·모델 연산 경계만 대체합니다.
        from datetime import date, timedelta
        events = []
        before = before or {"n": 144, "mae": 1.0, "mape": 0.01}
        after = after or {"n": 144, "mae": 0.9, "mape": 0.009}
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            stop = root / "중지요청"
            latest = root / "latest.json"
            save_json(latest, {"checkpoint": str(root / "old-model"), "cumulative_steps": 10, "sha256": {}})
            old_bytes = latest.read_bytes()
            days = [(date(2020, 1, 1) + timedelta(days=n)).isoformat() for n in range(200)]
            rows = [{"date": day, "close": 10, "open": 10} for day in days]
            market = {"series": {"QQQM": rows}}
            split = {"train_start": days[0], "train_end": "2024-06-03", "test_end": "2026-11-06"}
            save_json(root / "split.json", split)
            save_json(root / "고정학습자료.json", market["series"])
            save_json(root / "기반모델.json", {"snapshot": "local-base", "revision": "fixed"})
            if not missing_baseline:
                save_json(root / "기준모델평가.json", {"metrics": []})
            test_path = root / "최초시험평가.json"
            if not missing_test:
                save_json(test_path, {"metrics": {"model": "최초 시험", "mae": 1, "mape": .1, "n": 1}})
            if corrupt_test:
                test_path.write_text("{", encoding="utf-8")
            test_bytes = test_path.read_bytes() if test_path.exists() else None
            save_json(root / "fx.json", {"rates": [{"date": "2024-01-02", "rate": 1300}], "rate": 1300})
            if corrupt:
                (root / "과적합방어_5차.json").write_text("{", encoding="utf-8")

            class Flags:
                def all(self): return True
                def any(self): return False

            class Array(list):
                def __le__(self, other): return Flags()

            numpy = types.SimpleNamespace(asarray=lambda values, **kwargs: Array(values), float32="float32",
                                          isfinite=lambda value: math.isfinite(value) if isinstance(value, (int, float)) else Flags())
            torch = types.SimpleNamespace(set_num_threads=lambda value: None, float32="float32", __version__="test",
                                          cuda=types.SimpleNamespace(is_available=lambda: False))

            class Pipeline:
                model = object()

                @classmethod
                def from_pretrained(cls, *args, **kwargs): return cls()

                def fit(self, **kwargs):
                    events.append(("fit", kwargs))
                    if failure == "fit": raise RuntimeError("제어된 학습 실패")
                    if stop_at == "fit": stop.write_text("중지", encoding="utf-8")
                    callback = kwargs["callbacks"][0]
                    callback.on_log(None, None, None, {"loss": math.nan if failure == "nonfinite" else .2})
                    state = types.SimpleNamespace(global_step=kwargs["num_steps"])
                    callback.on_step_end(None, state, types.SimpleNamespace(should_training_stop=False))
                    return Pipeline()

            evaluations = []

            def evaluate(*args, **kwargs):
                events.append(("evaluate", args[3]))
                index = len(evaluations)
                evaluations.append(index)
                if failure == "before" and index == 0: raise RuntimeError("제어된 학습 전 평가 실패")
                if failure == "after" and index == 1: raise RuntimeError("제어된 학습 후 평가 실패")
                if stop_at == "after" and index == 1: stop.write_text("중지", encoding="utf-8")
                details = [{"symbol": "QQQM", "date": str(n), "context_end": str(n - 1), "actual": 10,
                            "previous": 10, "prediction": 10} for n in range(144)]
                if failure == "different_keys" and index == 1:
                    details[0]["date"] = "changed"
                return (before if index == 0 else after), details

            def commit(source, steps, number):
                events.append(("commit", steps))
                if stop_at == "commit":
                    stop.write_text("중지", encoding="utf-8")
                    raise InterruptedError("포인터 발행 전 중지")
                pointer = {"checkpoint": "new", "saved_at": "now", "cumulative_steps": 10 + steps}
                save_json(latest, pointer)
                return pointer

            def publish(*args):
                events.append(("publish", None))
                if failure == "publish": raise OSError("검증 통과 포인터 발행 뒤 보고 실패")

            for name, value in (("PRIVATE", root), ("STOP", stop), ("SPLIT", root / "split.json"),
                                ("TRAINING", root / "train.json"), ("LATEST", latest)):
                stack.enter_context(patch.object(modeling, name, value))
            stack.enter_context(patch.object(runtime, "STATE", root / "state.json"))
            stack.enter_context(patch.object(runtime, "PRIVATE", root))
            stack.enter_context(patch("src.paper.FX", root / "fx.json"))
            stack.enter_context(patch.object(modeling, "load_config", return_value={"train_steps": 1000, "batch_size": 2, "context_length": 256}))
            stack.enter_context(patch.object(modeling, "load_market", return_value=market))
            stack.enter_context(patch.object(modeling, "prepare_training_corpus", return_value=(corpus(days), {"post_initial_test_observations": 0})))
            stack.enter_context(patch.object(modeling, "evaluate", side_effect=evaluate))
            if missing_baseline:
                stack.enter_context(patch.object(modeling, "evaluation_windows", return_value=([[10]], [10], [10], [])))
            stack.enter_context(patch.object(modeling, "commit_checkpoint", side_effect=commit))
            stack.enter_context(patch.object(modeling, "publish_forecasts", side_effect=publish))
            stack.enter_context(patch.dict("sys.modules", {"numpy": numpy, "torch": torch,
                "chronos": types.SimpleNamespace(Chronos2Pipeline=Pipeline),
                "huggingface_hub": types.SimpleNamespace(HfApi=object, snapshot_download=lambda *a, **k: ""),
                "transformers": types.SimpleNamespace(TrainerCallback=object)}))
            caught = None
            result = None
            try:
                result = modeling.train_round()
                if second:
                    events.clear()
                    old_bytes = latest.read_bytes()
                    result = modeling.train_round()
            except Exception as error:
                caught = error
            ledger = None if corrupt else read_json(root / "과적합방어_5차.json")
            state = runtime.load_state()
            state["fixture_test_exists"] = test_path.exists()
            state["fixture_test_unchanged"] = (test_path.read_bytes() if test_path.exists() else None) == test_bytes
            return events, caught, result, latest.read_bytes() == old_bytes, ledger, state

    def test_accepted_update_evaluates_before_and_after_fit_then_commits(self):
        events, error, result, unchanged, ledger, state = self.exercise()
        self.assertIsNone(error)
        self.assertEqual([event[0] for event in events], ["evaluate", "fit", "evaluate", "commit", "publish"])
        fit = next(value for name, value in events if name == "fit")
        self.assertEqual(fit["num_steps"], 25)
        self.assertEqual(fit["weight_decay"], .01)
        self.assertEqual(fit["max_grad_norm"], 1.0)
        self.assertEqual(fit["learning_rate"], 1e-6)
        self.assertFalse(unchanged)
        self.assertEqual(ledger["attempts"][0]["status"], "accepted")

    def test_worse_and_invalid_metrics_preserve_latest_and_publish_nothing(self):
        for after in ({"n": 144, "mae": 2, "mape": .01}, {"n": 144, "mae": .5, "mape": .02},
                      {"n": 0, "mae": .5, "mape": .001}, {"n": 144, "mae": math.nan, "mape": .001}):
            with self.subTest(after=after):
                events, error, result, unchanged, ledger, state = self.exercise(after=after)
                self.assertTrue(unchanged)
                self.assertFalse(any(name in ("commit", "publish") for name, _ in events))
                self.assertEqual(ledger["attempts"][0]["status"], "rejected")

    def test_eval_fit_failure_nonfinite_and_stop_never_change_latest(self):
        for options in ({"failure": "before"}, {"failure": "after"}, {"failure": "fit"},
                        {"failure": "nonfinite"}, {"stop_at": "fit"}, {"stop_at": "after"}):
            with self.subTest(options=options):
                events, error, result, unchanged, ledger, state = self.exercise(**options)
                self.assertIsNotNone(error)
                self.assertTrue(unchanged)
                self.assertFalse(any(name in ("commit", "publish") for name, _ in events))
                self.assertIn(ledger["attempts"][0]["status"], ("failed", "interrupted"))

    def test_repeated_round_waits_without_loading_or_fitting_model(self):
        events, error, result, unchanged, ledger, state = self.exercise(second=True)
        self.assertIsNone(error)
        self.assertEqual(events, [])
        self.assertTrue(unchanged)
        self.assertEqual(result["status"], "waiting_new_data")

    def test_corrupt_ledger_prevents_any_model_work_or_latest_change(self):
        events, error, result, unchanged, ledger, state = self.exercise(corrupt=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertTrue(unchanged)
        self.assertEqual(events, [])

    def test_validation_keys_change_rejects_without_touching_first_test(self):
        events, error, result, unchanged, ledger, state = self.exercise(failure="different_keys")
        self.assertIsNone(error)
        self.assertTrue(unchanged)
        self.assertEqual([value for name, value in events if name == "evaluate"], ["validation", "validation"])
        self.assertEqual(ledger["attempts"][0]["status"], "rejected")

    def test_stop_after_gate_before_pointer_is_reported_as_interrupted(self):
        events, error, result, unchanged, ledger, state = self.exercise(stop_at="commit")
        self.assertIsInstance(error, InterruptedError)
        self.assertTrue(unchanged)
        self.assertEqual(state["training"]["guard"]["status"], "interrupted")

    def test_error_after_validated_pointer_publish_reports_committed_weights(self):
        events, error, result, unchanged, ledger, state = self.exercise(failure="publish")
        self.assertIsInstance(error, OSError)
        self.assertFalse(unchanged)
        self.assertEqual(state["training"]["guard"]["status"], "post_commit_error")
        self.assertEqual(ledger["attempts"][0]["status"], "accepted")

    def test_missing_test_and_baseline_never_evaluate_test_or_gate_commit(self):
        events, error, result, unchanged, ledger, state = self.exercise(missing_test=True, missing_baseline=True)
        self.assertIsNone(error)
        self.assertEqual([value for name, value in events if name == "evaluate"], ["validation", "validation", "validation"])
        self.assertFalse(unchanged)
        self.assertEqual(ledger["attempts"][0]["status"], "accepted")
        self.assertFalse(state["fixture_test_exists"])
        self.assertEqual(state["validation"]["first_test_status"], "unmeasured")

    def test_missing_or_corrupt_test_record_cannot_block_accepted_update(self):
        for options in ({"missing_test": True}, {"corrupt_test": True}):
            with self.subTest(options=options):
                events, error, result, unchanged, ledger, state = self.exercise(**options)
                self.assertIsNone(error)
                self.assertFalse(unchanged)
                self.assertEqual([value for name, value in events if name == "evaluate"], ["validation", "validation"])
                self.assertTrue(state["fixture_test_unchanged"])
                self.assertEqual(ledger["attempts"][0]["status"], "accepted")

    def test_existing_initial_test_is_read_for_display_without_modification(self):
        events, error, result, unchanged, ledger, state = self.exercise()
        self.assertIsNone(error)
        self.assertTrue(state["fixture_test_unchanged"])
        self.assertEqual(state["validation"]["first_test_status"], "recorded")
        self.assertTrue(any(entry.get("model") == "최초 시험" for entry in state["validation"]["metrics"]))


class LoopAndCommitTests(unittest.TestCase):
    def test_waiting_still_reconciles_account_and_uses_cooperative_delay(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            attempts = []

            def training():
                attempts.append(1)
                if len(attempts) > 1:
                    raise InterruptedError("대기 없는 반복을 검사에서 종료합니다")
                return {"status": "waiting_new_data", "wait_seconds": 300}

            with patch.object(run_pipeline, "STOP", Path(folder) / "중지"), \
                 patch.object(run_pipeline, "load_market", return_value={"series": {"QQQM": []}}), \
                 patch.object(run_pipeline, "reconcile_forward", side_effect=lambda *a: calls.append("forward")), \
                 patch.object(run_pipeline, "reconcile_paper", side_effect=lambda *a: calls.append("paper")), \
                 patch.object(modeling, "refresh_live_forecasts"), \
                 patch.object(run_pipeline, "train_round", side_effect=training), \
                 patch.object(run_pipeline, "update_state"), patch.object(run_pipeline, "log"), \
                 patch.object(run_pipeline, "wait_cooperatively", side_effect=lambda seconds: calls.append(seconds) or False):
                run_pipeline.run_loop("start", {}, True, float("inf"), 0)
            self.assertEqual(calls, ["forward", "paper", 300])

    def test_single_writer_lock_is_held_before_reserving_training_dates(self):
        from contextlib import nullcontext
        guard = importlib.import_module("src.overfit_guard")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stop = root / "중지"
            reservations = []

            def reserve():
                with self.assertRaisesRegex(RuntimeError, "다른 수집"):
                    with runtime.worker_lock():
                        self.fail("동일 원장의 동시 writer가 허용되었습니다")
                reservations.append(guard.reserve_training(root / "원장.json", corpus(["2024-01-02", "2024-01-03"]), 1000, 2))

            with patch.object(runtime, "PRIVATE", root), patch.object(runtime, "LOCK", root / "작업잠금"), \
                 patch.object(run_pipeline, "STOP", stop), patch.object(run_pipeline, "load_config", return_value={}), \
                 patch.object(run_pipeline, "load_market", return_value={"series": {"QQQM": []}}), \
                 patch.object(run_pipeline, "reconcile_forward"), patch.object(run_pipeline, "reconcile_paper"), \
                 patch.object(modeling, "refresh_live_forecasts"), \
                 patch.object(run_pipeline, "train_round", side_effect=reserve), \
                 patch.object(run_pipeline, "update_state"), patch.object(run_pipeline, "market_monitor", return_value=nullcontext()):
                run_pipeline.run("train")
            self.assertEqual(len(reservations), 1)
            self.assertEqual(reservations[0]["status"], "reserved")

    def test_stop_seen_during_file_checks_or_before_pointer_publish_preserves_latest(self):
        import os
        for phase in ("file_check", "pointer"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                stop = root / "중지"
                source = root / "학습모델"
                source.mkdir()
                (source / "model.safetensors").write_bytes("검사용 모델".encode("utf-8"))
                (source / "config.json").write_text("{}", encoding="utf-8")
                latest = root / "checkpoints" / "latest.json"
                save_json(latest, {"checkpoint": "old", "cumulative_steps": 50})
                before = latest.read_bytes()

                class Handle:
                    def __enter__(self): return self
                    def __exit__(self, *args): return False
                    def keys(self): return ["weight"]
                    def get_tensor(self, key):
                        if phase == "file_check": stop.write_text("중지", encoding="utf-8")
                        return object()

                real_replace = os.replace

                def replace(src, dst):
                    real_replace(src, dst)
                    if Path(src) == source and phase == "pointer": stop.write_text("중지", encoding="utf-8")

                finite = types.SimpleNamespace(all=lambda: types.SimpleNamespace(item=lambda: True))
                with patch.object(modeling, "STOP", stop), patch.object(modeling, "LATEST", latest), \
                     patch.dict("sys.modules", {"safetensors": types.SimpleNamespace(safe_open=lambda *a, **k: Handle()),
                                                 "torch": types.SimpleNamespace(isfinite=lambda value: finite)}), \
                     patch.object(modeling.os, "replace", side_effect=replace):
                    with self.assertRaises(InterruptedError):
                        modeling.commit_checkpoint(source, 25, 1)
                self.assertEqual(latest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

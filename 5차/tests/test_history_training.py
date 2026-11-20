"""전체 과거자료와 동결 평가 경계의 학습 입력 회귀 검사입니다."""

from pathlib import Path
import importlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from src import market
from src.core import read_json


SPLIT = {"train_start": "2020-10-13", "train_end": "2024-06-03",
         "validation_start": "2024-06-04", "validation_end": "2025-08-21",
         "test_start": "2025-08-22", "test_end": "2026-11-06"}
AS_OF = "2026-11-20T00:00:00+00:00"


def row(day, close=10):
    return {"date": day, "open": close, "close": close}


class HistoryTrainingTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("src.training_data"), "전체 학습자료 구성 모듈이 필요합니다")
        return importlib.import_module("src.training_data")

    def test_fetch_requests_history_before_2000(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"chart": {"result": [{"timestamp": []}]}}).encode()

        with patch.object(market, "urlopen", return_value=Response()):
            _, url = market.fetch_chart("XLK")
        self.assertEqual(parse_qs(urlparse(url).query)["period1"], ["0"])

    def test_all_symbols_history_fx_and_global_embargo_with_separate_segments(self):
        module = self.module()
        rows = [row("1999-01-04"), row("2020-01-02"), row("2024-06-03"),
                row("2024-06-04"), row("2025-08-22"), row("2026-11-06"),
                row("2026-11-09"), row("2026-11-10")]
        symbols = ("QQQM", "SCHG", "SPYM", "VTI", "VEA", "VWO", "BND", "IEF", "TLT", "SGOV", "GLD", "XLK", "XLF", "XLV", "XLE")
        frozen = {"SCHG": [row("2020-10-13", 7), row("2024-06-03", 8)]}
        fx = {"rates": [{"date": r["date"], "rate": 1300} for r in rows]}
        corpus = module.build_training_corpus({s: rows for s in symbols}, frozen, {}, fx, SPLIT, AS_OF)
        self.assertEqual(len(corpus["segments"]), 32)
        schg = next(s for s in corpus["segments"] if s["symbol"] == "SCHG" and s["period"] == "historical")
        self.assertEqual(schg["rows"], [row("1999-01-04"), row("2020-01-02"), row("2020-10-13", 7), row("2024-06-03", 8)])
        for segment in corpus["segments"]:
            dates = [r["date"] for r in segment["rows"]]
            self.assertTrue(all(d <= "2024-06-03" for d in dates) or all(d > "2026-11-06" for d in dates))
        fx_segments = [s for s in corpus["segments"] if s["symbol"] == "KRW=X"]
        self.assertEqual([s["kind"] for s in fx_segments], ["fx", "fx"])
        self.assertEqual(corpus["summary"]["post_initial_test_observations"], 32)

    def test_previous_values_win_and_new_dates_merge_without_duplicate_rows(self):
        module = self.module()
        base = {"XLK": [row("1999-01-04"), row("2024-06-03")]}
        old = module.build_training_corpus(base, {}, {}, {}, SPLIT, AS_OF)
        changed = {"XLK": [row("1998-12-31", 3), row("1999-01-04", 99), row("2024-06-03"), row("2024-06-03")]}
        corpus = module.build_training_corpus(changed, {}, {}, {}, SPLIT, AS_OF, previous=old)
        self.assertEqual(corpus["segments"][0]["rows"], [row("1998-12-31", 3), row("1999-01-04"), row("2024-06-03")])
        self.assertEqual(corpus["summary"]["new_observations"], 1)
        self.assertEqual(corpus["summary"]["preserved_revision_observations"], 1)

    def test_conflicting_same_day_invalid_and_future_fail_without_silent_fill(self):
        module = self.module()
        for rows in ([row("2024-01-02"), row("2024-01-02", 99)],
                     [row("2024-01-02", float("nan"))], [row("2024-01-02", 0)],
                     [row("2026-11-21")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                module.build_training_corpus({"XLK": rows}, {}, {}, {}, SPLIT, AS_OF)

    def test_singleton_new_segment_is_pending_then_becomes_trainable(self):
        module = self.module()
        initial = module.build_training_corpus({"XLK": [row("2026-11-09")]}, {}, {}, {}, SPLIT, AS_OF)
        self.assertEqual(initial["summary"]["train_observations"], 0)
        self.assertEqual(initial["summary"]["pending_observations"], 1)
        corpus = module.build_training_corpus({"XLK": [row("2026-11-10")]}, {}, {}, {}, SPLIT, AS_OF, previous=initial)
        self.assertEqual(corpus["summary"]["train_observations"], 2)
        self.assertEqual(corpus["summary"]["pending_observations"], 0)

    def test_snapshot_reference_is_content_addressed_and_detects_tampering(self):
        module = self.module()
        value = {"series": {"XLK": [row("1999-01-04"), row("2024-06-03")]}}
        with tempfile.TemporaryDirectory() as folder:
            reference = module.save_snapshot(Path(folder), "원본", value)
            again = module.save_snapshot(Path(folder), "원본", value)
            self.assertEqual(reference, again)
            self.assertEqual(read_json(Path(folder) / reference["path"]), value)
            self.assertEqual(len(list(Path(folder).rglob("*.json"))), 1)
            (Path(folder) / reference["path"]).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "해시"):
                module.save_snapshot(Path(folder), "원본", value)

    def test_prepared_corpus_reuses_snapshot_and_preserves_prior_source(self):
        module = self.module()
        original = {"series": {"XLK": [row("1999-01-04"), row("2024-06-03")]}}
        changed = {"series": {"XLK": [row("1999-01-04", 99), row("2024-06-03"), row("2026-11-09"), row("2026-11-10")]}}
        with tempfile.TemporaryDirectory() as folder:
            _, first = module.prepare_training_corpus(folder, original, {}, {}, {}, SPLIT, AS_OF)
            corpus, second = module.prepare_training_corpus(folder, changed, {}, {}, {}, SPLIT, AS_OF)
            _, third = module.prepare_training_corpus(folder, changed, {}, {}, {}, SPLIT, AS_OF)
            self.assertEqual(corpus["segments"][0]["rows"][0]["close"], 10)
            self.assertEqual(second["new_observations"], 2)
            self.assertEqual(third["new_observations"], 0)
            self.assertEqual(second["corpus"], third["corpus"])
            self.assertEqual(read_json(Path(folder) / first["source"]["path"])["market"], original)
            with self.assertRaisesRegex(RuntimeError, "고정 분할"):
                module.prepare_training_corpus(folder, changed, {}, {}, {}, {**SPLIT, "train_end": "2024-06-02"}, AS_OF)

    def test_interrupted_snapshot_write_leaves_no_partial_content_addressed_file(self):
        module = self.module()
        real_open, real_fdopen = Path.open, os.fdopen

        class InterruptedWrite:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def write(self, value):
                self.stream.write(value[:5])
                raise OSError("검사용 쓰기 중단")

        def open_path(path, mode="r", *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            return InterruptedWrite(stream) if mode == "xb" else stream

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Path, "open", open_path), patch.object(os, "fdopen", side_effect=lambda *a, **k: InterruptedWrite(real_fdopen(*a, **k))):
                with self.assertRaisesRegex(OSError, "쓰기 중단"):
                    module.save_snapshot(folder, "원본", {"test": [1, 2, 3]})
            self.assertEqual([p for p in Path(folder).rglob("*") if p.is_file()], [])
            reference = module.save_snapshot(folder, "원본", {"test": [1, 2, 3]})
            self.assertEqual(read_json(Path(folder) / reference["path"]), {"test": [1, 2, 3]})

    def test_previous_unknown_segment_metadata_is_rejected_without_silent_loss(self):
        module = self.module()
        for mutation in ({"period": "unknown"}, {"kind": "unknown"}, {"rows": {"date": "2026-11-09"}}):
            with self.subTest(mutation=mutation):
                previous = module.build_training_corpus({"XLK": [row("2026-11-09"), row("2026-11-10")]},
                                                        {}, {}, {}, SPLIT, AS_OF)
                previous["segments"][0].update(mutation)
                with self.assertRaisesRegex(RuntimeError, "보존 학습자료"):
                    module.build_training_corpus({}, {}, {}, {}, SPLIT, AS_OF, previous=previous)

if __name__ == "__main__":
    unittest.main()

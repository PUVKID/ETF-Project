"""고정 평가 구간을 제외한 전체 과거·신규 학습자료를 보존합니다."""

from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .core import read_json, save_json, session_completed


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def save_snapshot(root, category, value):
    """내용이 같은 자료는 한 번만 저장하고 기존 파일의 무결성을 확인합니다."""
    root = Path(root)
    encoded = _encoded(value)
    digest = hashlib.sha256(encoded).hexdigest()
    relative = Path("자료스냅샷_4차") / category / (digest + ".json")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("보존 자료 스냅샷 해시가 일치하지 않습니다")
    else:
        # 완성된 임시 파일만 이동하여 쓰기 실패가 영구 해시 파일을 오염시키지 않게 합니다.
        descriptor, temporary = tempfile.mkstemp(prefix=".스냅샷-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {"path": relative.as_posix(), "sha256": digest, "bytes": len(encoded)}


def _rows(rows, as_of, field="close"):
    result = {}
    for original in rows:
        row = dict(original)
        day = row["date"]
        if date.fromisoformat(day).isoformat() != day or not session_completed(day, as_of):
            raise ValueError(f"{day}: 미래·미완결 또는 잘못된 날짜의 학습자료입니다")
        value = float(row[field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{day}: 유효하지 않은 학습 관측값입니다")
        row[field] = value
        if "open" in row and (not math.isfinite(float(row["open"])) or float(row["open"]) <= 0):
            raise ValueError(f"{day}: 유효하지 않은 시가입니다")
        if day in result and result[day] != row:
            raise ValueError(f"{day}: 중복 날짜의 값이 다릅니다")
        result[day] = row
    return result


def build_training_corpus(market_series, frozen, extension, fx, split, as_of=None, previous=None):
    """검증·시험 사이를 잇지 않고 종목·환율별 두 독립 구간을 구성합니다."""
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    train_end, test_end = split["train_end"], split["test_end"]
    if train_end >= test_end:
        raise ValueError("고정 학습·시험 경계의 순서가 잘못되었습니다")
    if previous is not None:
        if not isinstance(previous, dict) or previous.get("schema_version") != 1 or not isinstance(previous.get("segments"), list):
            raise RuntimeError("4차 보존 학습자료의 스키마가 유효하지 않습니다")
        if previous.get("split") != split:
            raise RuntimeError("4차 보존 학습자료의 고정 분할이 변경되었습니다")
    previous_rows = {}
    for segment in (previous or {}).get("segments", []):
        if (not isinstance(segment, dict) or segment.get("period") not in ("historical", "post_test")
                or segment.get("kind") not in ("etf", "fx")
                or not isinstance(segment.get("symbol"), str) or not segment["symbol"]
                or (segment["kind"] == "fx") != (segment["symbol"] == "KRW=X")
                or not isinstance(segment.get("rows"), list) or not segment["rows"]
                or any(not isinstance(row, dict) for row in segment["rows"])):
            raise RuntimeError("4차 보존 학습자료의 구간 메타데이터가 유효하지 않습니다")
        key = (segment["symbol"], segment["period"])
        if key in previous_rows:
            raise RuntimeError("보존 학습자료의 구간이 중복되었습니다")
        try:
            values = _rows(segment["rows"], as_of)
        except (ValueError, KeyError, TypeError) as error:
            raise RuntimeError("4차 보존 학습자료의 관측값이 유효하지 않습니다") from error
        if any(not (day <= train_end if key[1] == "historical" else day > test_end) for day in values):
            raise RuntimeError("보존 학습자료가 고정 평가 제외 구간을 침범했습니다")
        previous_rows[key] = values
    sources = {symbol: ("etf", _rows(rows, as_of)) for symbol, rows in market_series.items()}
    rates = _rows(fx.get("rates", []), as_of, "rate")
    if rates:
        sources["KRW=X"] = ("fx", {day: {"date": day, "close": row["rate"]} for day, row in rates.items()})
    # 최초 고정값과 3차 확장값은 같은 날짜의 후속 조정값보다 우선합니다.
    for source in (extension, frozen):
        for symbol, rows in source.items():
            known = sources.setdefault(symbol, ("etf", {}))[1]
            known.update(_rows(rows, as_of))
    for symbol, _ in previous_rows:
        sources.setdefault(symbol, ("fx" if symbol == "KRW=X" else "etf", {}))
    segments, new_count, revisions = [], 0, 0
    for symbol, (kind, rows) in sorted(sources.items()):
        for period in ("historical", "post_test"):
            selected = {day: value for day, value in rows.items()
                        if (day <= train_end if period == "historical" else day > test_end)}
            old = previous_rows.get((symbol, period), {})
            revisions += sum(day in old and old[day] != value for day, value in selected.items())
            new_count += len(selected.keys() - old.keys())
            selected.update(old)
            if selected:
                ordered = [selected[day] for day in sorted(selected)]
                segments.append({"symbol": symbol, "kind": kind, "period": period, "rows": ordered,
                                 "eligible": len(ordered) >= 2})
    eligible = [segment for segment in segments if segment["eligible"]]
    summary = {"symbols": sorted({segment["symbol"] for segment in segments}),
               "etf_symbols": sorted({segment["symbol"] for segment in segments if segment["kind"] == "etf"}),
               "fx_symbols": sorted({segment["symbol"] for segment in segments if segment["kind"] == "fx"}),
               "series_count": len(eligible), "min_past": 1, "prediction_length": 1,
               "train_observations": sum(len(segment["rows"]) for segment in eligible),
               "pending_observations": sum(len(segment["rows"]) for segment in segments if not segment["eligible"]),
               "new_observations": new_count, "preserved_revision_observations": revisions,
               "post_initial_test_observations": sum(len(segment["rows"]) for segment in eligible if segment["period"] == "post_test"),
               "fx_role": "독립 단변량 학습 시계열; ETF 공변량 아님"}
    summary["segments"] = [{key: segment[key] for key in ("symbol", "kind", "period", "eligible")}
                           | {"rows": len(segment["rows"]), "start": segment["rows"][0]["date"],
                              "end": segment["rows"][-1]["date"]} for segment in segments]
    return {"schema_version": 1, "split": dict(split), "segments": segments, "summary": summary}


def prepare_training_corpus(root, market, frozen, extension, fx, split, as_of=None):
    """실제 입력 스냅샷과 누적 포인터를 저장하고 회차에는 참조만 전달합니다."""
    root = Path(root)
    pointer_path = root / "학습자료포인터_4차.json"
    pointer = read_json(pointer_path)
    previous = None
    if pointer:
        path = root / pointer["corpus"]["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != pointer["corpus"]["sha256"]:
            raise RuntimeError("4차 보존 학습자료 해시가 일치하지 않습니다")
        previous = read_json(path)
    corpus = build_training_corpus(market["series"], frozen, extension, fx, split, as_of, previous)
    frozen_payload = {key: corpus[key] for key in ("schema_version", "split", "segments")}
    reference = save_snapshot(root, "학습입력", frozen_payload)
    source = save_snapshot(root, "입력출처", {"market": market, "fx": fx,
                                            "original_frozen": frozen, "legacy_extension": extension})
    manifest = {"corpus": reference, "source": source, "as_of": as_of or datetime.now(timezone.utc).isoformat(),
                **corpus["summary"], "excluded_period": {"after": split["train_end"], "through": split["test_end"]},
                "revision_policy": "최초 고정자료·3차 확장자료 및 4차 최초 보존값 우선; 새 날짜만 추가"}
    save_json(pointer_path, manifest)
    return corpus, manifest

"""사전학습 모델의 실제 추가 학습과 고정 시간순 비교입니다."""

from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from .core import common_dates, forecast_metrics, make_split, now_iso, read_json, save_json
from .market import load_market
from .runtime import PRIVATE, STOP, load_config, load_state, log, record_failure, update_state
from .universe import UNIVERSE_HASH
from .training_data import prepare_training_corpus
from .overfit_guard import MONITOR_SCOPE, finish_attempt, reserve_training, validation_decision


MODEL_ID = "amazon/chronos-2"
SPLIT = PRIVATE / "고정분할.json"
TRAINING = PRIVATE / "학습기록.json"
LATEST = PRIVATE / "체크포인트" / "latest.json"


def build_training_extension(market_series, frozen, train_end):
    # 새 ETF도 최초 공통 검증·시험 날짜보다 앞선 구간만 동결해 교차 자산 시간 누출을 막습니다.
    return {symbol: [dict(row) for row in rows if row["date"] <= train_end]
            for symbol, rows in market_series.items() if symbol not in frozen
            and len([row for row in rows if row["date"] <= train_end]) >= 2}


def prepare_environment():
    cache = PRIVATE / "캐시"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def evaluation_windows(series, split, segment, context_length, limit=48):
    start = split[f"{segment}_start"]
    end = split[f"{segment}_end"]
    inputs, actual, previous, keys = [], [], [], []
    for symbol, rows in sorted(series.items()):
        available = [index for index, row in enumerate(rows) if start <= row["date"] <= end and index >= 32]
        # 일정 간격 표본이며 미래 가격이나 성과를 기준으로 표본을 고르지 않습니다.
        if len(available) > limit:
            available = sorted({available[int(index * (len(available) - 1) / (limit - 1))] for index in range(limit)})
        for index in available:
            inputs.append([row["close"] for row in rows[max(0, index - context_length):index]])
            actual.append(rows[index]["close"])
            previous.append(rows[index - 1]["close"])
            keys.append({"symbol": symbol, "date": rows[index]["date"], "context_end": rows[index - 1]["date"]})
    if not inputs:
        raise RuntimeError("시간순 평가 표본이 부족합니다")
    return inputs, actual, previous, keys


def predict_values(pipeline, inputs, context_length, batch_size=4):
    import numpy as np
    predictions = pipeline.predict([np.asarray(values, dtype=np.float32) for values in inputs],
                                   prediction_length=1, context_length=context_length, batch_size=batch_size)
    median = min(range(len(pipeline.quantiles)), key=lambda index: abs(pipeline.quantiles[index] - 0.5))
    return [float(prediction[0, median, 0].detach().cpu()) for prediction in predictions]


def evaluate(pipeline, series, split, segment, context_length, label):
    inputs, actual, previous, keys = evaluation_windows(series, split, segment, context_length)
    predictions = predict_values(pipeline, inputs, context_length)
    metrics = {"model": label, "split": segment, **forecast_metrics(actual, predictions, previous)}
    return metrics, [{**key, "actual": truth, "previous": old, "prediction": value}
                     for key, truth, old, value in zip(keys, actual, previous, predictions)]


def validation_basis(details, expected_count):
    basis = [{key: row[key] for key in ("symbol", "date", "context_end", "actual", "previous")} for row in details]
    if len(basis) != expected_count or len({(row["symbol"], row["date"], row["context_end"]) for row in basis}) != expected_count:
        raise ValueError("고정 검증의 표본 수·식별자가 일치하지 않습니다")
    return basis


def refresh_live_forecasts():
    """채택된 저장본의 최신 예측을 학습 예약·검증 채택과 독립적으로 발행합니다."""
    if STOP.exists():
        raise InterruptedError("운영 예측 전 중지 요청을 확인했습니다")
    latest = read_json(LATEST)
    if latest is None:
        return {"status": "waiting_checkpoint"}
    config = load_config()
    context_length = int(config.get("context_length", 256))
    if not 32 <= context_length <= 512:
        raise ValueError("운영 예측 문맥 제한을 벗어났습니다")
    series = load_market()["series"]
    if not series or any(not rows for rows in series.values()):
        raise ValueError("운영 예측에 필요한 완결 일봉이 없습니다")
    signature = hashlib.sha256(json.dumps({"checkpoint": latest, "universe": UNIVERSE_HASH,
        "context_length": context_length,
        "series": {symbol: rows[-context_length:] for symbol, rows in sorted(series.items())}},
        sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    journal = PRIVATE / "운영예측_6차.json"
    previous = read_json(journal, {})
    if previous.get("signature") == signature:
        return {"status": "unchanged", "created_at": previous["created_at"]}
    update_state(forecast={"status": "loading", "message": "채택된 운영 모델로 최신 완결 일봉을 예측합니다"})
    _publish_saved_forecasts(latest, series, context_length, config)
    if STOP.exists():
        raise InterruptedError("운영 예측 완료 기록 전 중지 요청을 확인했습니다")
    result = {"status": "ready", "signature": signature, "created_at": now_iso(),
              "checkpoint_saved_at": latest.get("saved_at"),
              "dates": {symbol: rows[-1]["date"] for symbol, rows in sorted(series.items())}}
    save_json(journal, result)
    update_state(forecast=result)
    return result


def _publish_saved_forecasts(latest, series, context_length, config):
    """해시를 확인한 운영 저장본만 읽고 학습 없이 예측합니다."""
    source = Path(latest["checkpoint"])
    expected_files = latest.get("sha256", {})
    if not expected_files or not any(name.endswith(".safetensors") for name in expected_files):
        raise ValueError("운영 저장본의 가중치 해시 기록이 없습니다")
    for name, expected in expected_files.items():
        if Path(name).name != name:
            raise ValueError("운영 저장본 파일 이름이 유효하지 않습니다")
        digest = hashlib.sha256()
        with (source / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError("운영 예측 저장본의 SHA-256이 기록과 일치하지 않습니다")
    prepare_environment()
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        fraction = float(config.get("gpu_memory_fraction", 0.25))
        if not 0.05 <= fraction <= 0.3:
            raise ValueError("GPU 메모리 제한은 5~30% 이내여야 합니다")
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    if STOP.exists():
        raise InterruptedError("운영 모델 로드 전 중지 요청을 확인했습니다")
    pipeline = Chronos2Pipeline.from_pretrained(str(source), device_map=device,
                                               torch_dtype=torch.float32, local_files_only=True)
    try:
        pipeline.model.eval()
        with torch.inference_mode():
            publish_forecasts(pipeline, series, context_length, "Chronos-2 채택 저장본 · 독립 운영 예측")
    finally:
        del pipeline
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def train_round():
    if STOP.exists():
        raise InterruptedError("학습 라이브러리를 불러오기 전 중지 요청을 확인했습니다")
    config = load_config()
    market = load_market()
    series = market["series"]
    dates = common_dates(series)
    if len(dates) < 200:
        raise RuntimeError("모델 학습에는 최소 200개 공통 완결 일봉이 필요합니다")
    split = read_json(SPLIT)
    if split is None:
        split = make_split(dates)
        split["created_at"] = now_iso()
        save_json(SPLIT, split)
    # 후속 수집으로 수정된 과거 조정값이 모델 비교에 섞이지 않도록 최초 학습 데이터도 동결합니다.
    frozen_path = PRIVATE / "고정학습자료.json"
    frozen = read_json(frozen_path)
    if frozen is None:
        frozen = {symbol: [row for row in rows if split["train_start"] <= row["date"] <= split["test_end"]]
                  for symbol, rows in series.items()}
        save_json(frozen_path, frozen)
    series = frozen
    extension_path = PRIVATE / "확장학습자료_3차.json"
    extension = read_json(extension_path)
    if extension is None:
        extension = {"train_end": split["train_end"], "created_at": now_iso(),
                     "series": build_training_extension(market["series"], frozen, split["train_end"])}
        save_json(extension_path, extension)
    missing = build_training_extension(market["series"], {**frozen, **extension["series"]}, split["train_end"])
    if missing:
        extension["series"].update(missing)
        save_json(extension_path, extension)
    if extension["train_end"] != split["train_end"] or any(
            row["date"] > split["train_end"] for rows in extension["series"].values() for row in rows):
        raise RuntimeError("확장 학습자료가 최초 고정 학습 기간을 벗어났습니다")
    from .paper import FX
    fx_training = read_json(FX, {})
    if not fx_training.get("rates"):
        raise RuntimeError("환율 독립 학습 시계열을 확보하기 전 학습을 시작할 수 없습니다")
    corpus, training_data = prepare_training_corpus(PRIVATE, market, frozen, extension["series"],
                                                   fx_training, split, now_iso())
    update_state(training={"training_data": training_data})
    context_length = int(config.get("context_length", 256))
    if not 32 <= context_length <= 512:
        raise ValueError("학습 문맥 제한을 벗어났습니다")
    guard_path = PRIVATE / "과적합방어_5차.json"
    try:
        guard = reserve_training(guard_path, corpus, int(config.get("train_steps", 20)),
                                 int(config.get("batch_size", 2)))
    except Exception:
        update_state(training={"status": "guard_blocked", "guard": {"status": "blocked", "budget_steps": 0,
                     "reason": "과적합 방어 원장 또는 예산 오류로 학습을 차단했습니다. 원문을 확인해야 합니다.",
                     "monitor_scope": MONITOR_SCOPE}})
        raise
    update_state(training={"guard": guard})
    if guard["status"] == "waiting_new_data":
        update_state(training={"status": "waiting_new_data", "step": 0, "total_steps": 0},
                     job={"message": guard["reason"]})
        return guard
    pointer_before = LATEST.read_bytes() if LATEST.exists() else None
    try:
        return _train_reserved(config, market, series, frozen, extension, split, corpus, training_data, guard)
    except BaseException as error:
        # 예약은 fit 전부터 소비됩니다. 중단·실패 뒤 같은 관측을 다시 학습하지 않습니다.
        ledger = read_json(guard_path)
        attempt = next(entry for entry in ledger["attempts"] if entry["attempt_id"] == guard["attempt_id"])
        outcome = "interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed"
        if attempt["status"] == "reserved":
            finish_attempt(guard_path, guard["attempt_id"], outcome, {"error": str(error)})
        pointer_after = LATEST.read_bytes() if LATEST.exists() else None
        published = pointer_after != pointer_before
        update_state(training={"guard": {**guard, "status": "post_commit_error" if published else outcome,
                     "reason": "검증을 통과한 운영 가중치 발행 뒤 후속 처리에서 중단·오류가 발생했습니다. 발행된 가중치는 유지합니다."
                     if published else "운영 포인터 발행 전 중단·실패로 기존 가중치를 유지합니다. 예약 날짜는 소비된 상태로 유지합니다."}})
        raise


def _train_reserved(config, market, series, frozen, extension, split, corpus, training_data, guard):
    prepare_environment()
    update_state(training={"status": "loading"}, job={"message": "학습 라이브러리와 마지막 저장 모델을 불러오고 있습니다"})
    import numpy as np
    import torch
    from chronos import Chronos2Pipeline
    from huggingface_hub import HfApi, snapshot_download
    from transformers import TrainerCallback
    if STOP.exists():
        raise InterruptedError("학습 라이브러리를 불러온 뒤 중지 요청을 확인했습니다")
    record = read_json(TRAINING, {"rounds": [], "last_round": 0})
    number = record.get("last_round", 0) + 1
    record["last_round"] = number
    save_json(TRAINING, record)
    latest = read_json(LATEST, {})
    run_dir = PRIVATE / "학습회차" / f"{number:04d}차"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = now_iso()
    total_steps = guard["budget_steps"]
    context_length = int(config.get("context_length", 256))
    if not 1 <= total_steps <= 25 or not 32 <= context_length <= 512:
        raise ValueError("초기 운영의 학습 단계/문맥 제한을 벗어났습니다")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        fraction = float(config.get("gpu_memory_fraction", 0.25))
        if not 0.05 <= fraction <= 0.3:
            raise ValueError("GPU 메모리 제한은 5~30% 이내여야 합니다")
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        torch.cuda.empty_cache()
    update_state(training={"status": "downloading", "step": 0, "total_steps": total_steps,
                          "round": number, "loss": None, "device": device,
                          "started_at": started, "finished_at": None,
                          "reused_data": number > 1, "split": split},
                 job={"message": "사전학습 모델과 고정 시간순 자료를 준비하고 있습니다"})
    model_record = PRIVATE / "기반모델.json"
    base_info = read_json(model_record)
    if base_info is None:
        info = HfApi().model_info(MODEL_ID)
        revision = info.sha
        snapshot = snapshot_download(MODEL_ID, revision=revision,
                                     local_dir=str(PRIVATE / "기반모델"),
                                     allow_patterns=["*.json", "*.safetensors", "LICENSE", "NOTICE", "README.md"])
        base_info = {"model": MODEL_ID, "revision": revision, "snapshot": snapshot,
                     "downloaded_at": now_iso(), "pretraining_time_boundary": "확인 불가"}
        save_json(model_record, base_info)
    if STOP.exists():
        raise InterruptedError("모델 준비 후 중지 요청을 확인했습니다")
    log(f"Chronos-2 실제 로드: {device}, 회차 {number}, {total_steps}단계")
    resume_source = latest.get("checkpoint")
    baseline_path = PRIVATE / "기준모델평가.json"
    baselines = read_json(baseline_path)
    base_pipeline = None
    if baselines is None:
        base_pipeline = Chronos2Pipeline.from_pretrained(base_info["snapshot"], device_map=device, torch_dtype=torch.float32)
        baselines = {"metrics": [], "details": {}, "created_at": now_iso(), "revision": base_info["revision"]}
        # 학습 루프는 검증 기준만 만듭니다. 시험 계산은 별도 독립 평가에서 수행합니다.
        for segment in ("validation",):
            metrics, details = evaluate(base_pipeline, series, split, segment, context_length, "사전학습 그대로")
            inputs, actual, previous, _ = evaluation_windows(series, split, segment, context_length)
            naive = {"model": "무변화 예측", "split": segment, **forecast_metrics(actual, previous, previous)}
            baselines["metrics"].extend([metrics, naive])
            baselines["details"][segment] = details
        save_json(baseline_path, baselines)
    if resume_source:
        for filename, expected in latest.get("sha256", {}).items():
            digest = hashlib.sha256()
            with (Path(resume_source) / filename).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise RuntimeError("이전 체크포인트의 SHA-256이 저장 기록과 일치하지 않습니다")
        del base_pipeline
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        pipeline = Chronos2Pipeline.from_pretrained(resume_source, device_map=device, torch_dtype=torch.float32)
    else:
        pipeline = base_pipeline or Chronos2Pipeline.from_pretrained(base_info["snapshot"], device_map=device, torch_dtype=torch.float32)

    if STOP.exists():
        raise InterruptedError("학습 전 검증 전에 중지 요청을 확인했습니다")
    update_state(training={"status": "evaluating"}, job={"message": "운영 가중치의 학습 전 고정 검증 오차를 확인하고 있습니다"})
    before_metrics, before_details = evaluate(pipeline, series, split, "validation", context_length, "계속 학습 · 변경 전")
    if not validation_decision(before_metrics, before_metrics)["accepted"]:
        raise ValueError("학습 전 고정 검증의 유한한 양수 표본 지표가 없어 학습을 차단했습니다")
    before_basis = validation_basis(before_details, before_metrics["n"])
    save_json(run_dir / "학습전검증예측.json", before_details)
    if STOP.exists():
        raise InterruptedError("학습 전 검증 뒤 중지 요청을 확인했습니다")

    class Progress(TrainerCallback):
        def __init__(self):
            self.last_publish = 0.0
            self.loss = None

        def on_step_end(self, args, state, control, **kwargs):
            stopped = STOP.exists()
            current = time.monotonic()
            if current - self.last_publish >= 1 or state.global_step >= total_steps or stopped:
                section = {"status": "training", "step": state.global_step,
                           "cumulative_steps": latest.get("cumulative_steps", 0) + state.global_step}
                if self.loss is not None:
                    section["loss"] = self.loss
                update_state(training=section, job={"message": f"추가 학습 {state.global_step}/{total_steps}단계"})
                self.last_publish = current
            if stopped:
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                value = float(logs["loss"])
                if not np.isfinite(value):
                    raise FloatingPointError("유한하지 않은 학습 손실로 저장을 중단했습니다")
                # 유한값 검사는 로그마다 유지하고 전체 상태 파일 발행은 step callback의 1Hz로 제한합니다.
                self.loss = value

        def on_train_end(self, args, state, control, **kwargs):
            section = {"step": state.global_step,
                       "cumulative_steps": latest.get("cumulative_steps", 0) + state.global_step}
            if self.loss is not None:
                section["loss"] = self.loss
            update_state(training=section)

    train_inputs = [np.asarray([row["close"] for row in segment["rows"]], dtype=np.float32)
                    for segment in corpus["segments"] if segment["eligible"]]
    if not train_inputs or any(not np.isfinite(values).all() or (values <= 0).any() for values in train_inputs):
        raise ValueError("실제 모델 입력의 유한한 양의 float32 학습값을 확인하지 못했습니다")
    new_observations = training_data["post_initial_test_observations"]
    digest = hashlib.sha256(json.dumps(series, sort_keys=True).encode()).hexdigest()
    manifest = {"round": number, "started_at": started, "model": MODEL_ID, "revision": base_info["revision"],
                "initial_checkpoint": resume_source or base_info["snapshot"], "split": split,
                "frozen_data_sha256": digest, "new_training_data": new_observations > 0,
                "training_data": training_data,
                "guard": guard, "before_validation": before_metrics,
                "post_initial_test_observations": new_observations,
                "expansion_training_symbols": sorted(extension["series"]),
                "expansion_train_end": extension["train_end"],
                "monitoring_symbols": sorted(frozen), "monitoring_scope": "기존 최초3종목 고정구간; 확장15종목 평가지표와 혼합하지 않음",
                "train_observations": sum(len(values) for values in train_inputs), "steps": total_steps,
                "batch_size": int(config.get("batch_size", 2)), "context_length": context_length,
                "device": device, "gpu_memory_fraction": config.get("gpu_memory_fraction", 0.25),
                "torch": torch.__version__, "method": "full parameter fine-tuning, AdamW, learning_rate=1e-6",
                "weight_decay": 0.01, "max_grad_norm": 1.0,
                "optimizer_resume": False, "resume_note": "직전 저장본의 가중치를 이어 사용하며 optimizer 상태는 회차마다 새로 생성"}
    save_json(run_dir / "실행명세.json", manifest)
    update_state(training={"status": "training", "resumed": bool(resume_source),
                          "cumulative_steps": latest.get("cumulative_steps", 0)})
    tuned = pipeline.fit(inputs=train_inputs, prediction_length=1, min_past=1,
                         context_length=context_length, learning_rate=1e-6, num_steps=total_steps,
                         batch_size=int(config.get("batch_size", 2)), output_dir=run_dir,
                         finetuned_ckpt_name="학습모델", callbacks=[Progress()],
                         logging_steps=20, logging_nan_inf_filter=False, disable_tqdm=True,
                         optim="adamw_torch", weight_decay=0.01, max_grad_norm=1.0, bf16=False, tf32=False,
                         save_strategy="no", report_to="none", seed=42 + number)
    actual_steps = load_state()["training"]["step"]
    if type(actual_steps) is not int or not 1 <= actual_steps <= total_steps:
        raise RuntimeError("예약된 예산 안의 실제 학습 단계가 확인되지 않았습니다")
    if tuned.model is None:
        raise RuntimeError("추가 학습 모델이 없습니다")
    if STOP.exists():
        save_json(run_dir / "중지기록.json", {"at": now_iso(), "checkpoint_updated": False, "steps": actual_steps})
        raise InterruptedError("중지 요청으로 학습 중 가중치를 운영 저장본에 반영하지 않았습니다")
    update_state(training={"status": "evaluating"}, validation={"status": "running"},
                 job={"message": "같은 모델의 변경 전후 검증 오차를 비교하고 있습니다 · 운영 저장본 보호"})
    monitor_metrics, monitor_details = evaluate(tuned, series, split, "validation", context_length, "계속 학습 · 현재")
    save_json(run_dir / "검증예측.json", monitor_details)
    if STOP.exists():
        raise InterruptedError("학습 후 검증 뒤 중지 요청으로 운영 저장본 반영을 중단했습니다")
    decision = validation_decision(before_metrics, monitor_metrics)
    try:
        if validation_basis(monitor_details, monitor_metrics["n"]) != before_basis:
            decision.update(accepted=False, reason="학습 전후 고정 검증 표본의 날짜·자산·정답이 달라 저장을 차단했습니다")
    except (ValueError, KeyError, TypeError):
        decision.update(accepted=False, reason="학습 후 고정 검증 표본 식별자를 확인하지 못해 저장을 차단했습니다")
    save_json(run_dir / "과적합방어결과.json", {**decision, "before": before_metrics,
              "after": monitor_metrics if validation_decision(monitor_metrics, monitor_metrics)["accepted"] else None})
    if not decision["accepted"]:
        finish_attempt(PRIVATE / "과적합방어_5차.json", guard["attempt_id"], "rejected", decision)
        outcome = {**guard, **decision, "status": "rejected"}
        update_state(training={"status": "guard_rejected", "guard": outcome,
                     "cumulative_steps": latest.get("cumulative_steps", 0), "finished_at": now_iso()},
                     validation={"status": "guard_rejected"}, job={"message": decision["reason"] + " · 기존 저장 가중치 유지"})
        return outcome
    metrics = list(baselines["metrics"]) + [monitor_metrics]
    if STOP.exists():
        raise InterruptedError("운영 체크포인트 반영 전 중지 요청을 확인했습니다")
    # 검증 허용을 먼저 영속 기록합니다. 이 기록 자체가 체크포인트 저장 성공은 아닙니다.
    finish_attempt(PRIVATE / "과적합방어_5차.json", guard["attempt_id"], "accepted", decision)
    pointer = commit_checkpoint(run_dir / "학습모델", actual_steps, number)
    update_state(training={"checkpoint": "latest", "checkpoint_saved_at": pointer["saved_at"],
                          "cumulative_steps": pointer["cumulative_steps"], "guard": {**guard, **decision, "status": "accepted"}})
    # 시험 자료는 학습·저장 판단에서 완전히 분리합니다. 기존 결과는 발행 뒤 표시용으로만 읽습니다.
    first_test_status = "unmeasured"
    first_test_note = "기존 최초 시험 기록이 없습니다. 학습 루프에서는 시험을 계산하거나 기록을 새로 만들지 않습니다."
    try:
        initial_test = read_json(PRIVATE / "최초시험평가.json")
        if initial_test is not None:
            historical_metrics = initial_test["metrics"]
            if (not isinstance(historical_metrics, dict)
                    or any(type(historical_metrics.get(key)) not in (int, float) for key in ("mae", "mape", "n"))
                    or any(isinstance(value, (int, float)) and not np.isfinite(value)
                           for value in historical_metrics.values())):
                raise ValueError("기존 최초 시험 표시 지표가 유효하지 않습니다")
            metrics.append(historical_metrics)
            first_test_status = "recorded"
            first_test_note = "기존 최초 시험 기록을 표시용으로 읽었습니다. 이번 학습의 저장 조건에 사용하지 않았습니다."
    except (OSError, ValueError, KeyError, TypeError) as error:
        record_failure(error)
        first_test_status = "unavailable"
        first_test_note = "기존 최초 시험 기록을 표시할 수 없습니다. 원본을 보존하며 학습 저장 판단에는 사용하지 않았습니다."
    from .paper import FX
    fx = read_json(FX)
    if fx:
        metrics = [{**entry, "mae_krw": entry["mae"] * fx["rate"], "mae_currency": "USD 원 오차를 현재 시장환율로 환산한 참고값"}
                   for entry in metrics]
    result = {**manifest, "finished_at": now_iso(), "checkpoint": "latest",
              "metrics": monitor_metrics, "cumulative_steps": pointer["cumulative_steps"],
              "loss": load_state()["training"].get("loss")}
    record["rounds"].append(result)
    save_json(run_dir / "학습결과.json", result)
    save_json(TRAINING, record)
    update_state(training={"status": "completed", "finished_at": result["finished_at"],
                          "checkpoint": "latest", "step": actual_steps,
                          "model": MODEL_ID, "revision": base_info["revision"]},
                 validation={"status": "ready", "metrics": metrics,
                             "first_test_only": True, "split": split,
                             "first_test_status": first_test_status, "first_test_note": first_test_note,
                             "coverage_symbols": sorted(frozen),
                             "coverage_note": "최초3종목 고정 검증·시험만; 확장15종목 독립 평가로 표시하지 않음",
                             "independent_forward_observations": load_state().get("forward", {}).get("n", 0)})
    # 동일 모델의 마지막 가중치로 실제 신규 관측 전 전망을 기록합니다.
    del pipeline
    gc.collect()
    if STOP.exists():
        raise InterruptedError("예측·주문 생성 전 중지 요청을 확인했습니다. 직전 학습 저장본은 보존했습니다")
    publish_forecasts(tuned, market["series"], context_length, "Chronos-2 계속 학습 latest")
    del tuned
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    log(f"{number}차 추가 학습 저장 · 누적 {pointer['cumulative_steps']}단계 · 같은 고정 자료 재사용={number > 1}")
    return result


def commit_checkpoint(source, steps, round_number):
    """완성 파일 검사를 마친 뒤 포인터를 원자 저장하고 직전 복구본 하나를 남깁니다."""
    from safetensors import safe_open
    import torch
    source = Path(source)
    files = list(source.glob("*.safetensors"))
    if not files or not (source / "config.json").is_file():
        raise RuntimeError("완성된 safetensors/config 체크포인트가 없습니다")
    hashes = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if not keys:
                raise RuntimeError("체크포인트 텐서가 비어 있습니다")
            for key in keys:
                tensor = handle.get_tensor(key)
                finite = bool(torch.isfinite(tensor).all().item())
                del tensor
                if not finite:
                    raise FloatingPointError(f"저장 전 체크포인트의 {key}에 유한하지 않은 값이 있습니다")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    previous = read_json(LATEST, {})
    store = LATEST.parent.resolve()
    store.mkdir(parents=True, exist_ok=True)
    destination = store / ("저장-" + uuid.uuid4().hex)
    if STOP.exists():
        raise InterruptedError("파일 검사 뒤 중지를 확인하여 운영 포인터를 유지했습니다")
    os.replace(source, destination)
    pointer = {"checkpoint": str(destination), "previous_checkpoint": previous.get("checkpoint"),
               "saved_at": now_iso(), "round": round_number,
               "cumulative_steps": previous.get("cumulative_steps", 0) + steps, "sha256": hashes}
    if STOP.exists():
        raise InterruptedError("원자 포인터 발행 직전 중지를 확인하여 운영 포인터를 유지했습니다")
    save_json(LATEST, pointer)
    obsolete = previous.get("previous_checkpoint")
    if obsolete:
        target = Path(obsolete).resolve()
        if target.parent != store or not target.name.startswith("저장-"):
            raise RuntimeError("복구본 정리 경로가 체크포인트 전용 폴더를 벗어났습니다")
        shutil.rmtree(target)
    return pointer


def publish_forecasts(pipeline, series, context_length, label):
    if STOP.exists():
        raise InterruptedError("예측 준비 전 중지 요청을 확인했습니다")
    symbols = sorted(series)
    values = predict_values(pipeline, [[row["close"] for row in series[symbol][-context_length:]] for symbol in symbols], context_length)
    if STOP.exists():
        raise InterruptedError("예측 계산 중 중지를 확인하여 새 신호와 거래를 발행하지 않습니다")
    created_at = now_iso()
    forecasts = [{"symbol": symbol, "as_of": series[symbol][-1]["date"], "horizon_sessions": 1,
                  "adjusted_price": value, "last_adjusted_price": series[symbol][-1]["close"],
                  "model": label, "created_at": created_at, "policy_version": 3,
                  "universe_hash": UNIVERSE_HASH} for symbol, value in zip(symbols, values)]
    from .paper import FX
    fx = read_json(FX)
    if fx:
        for forecast in forecasts:
            forecast["adjusted_price_krw"] = forecast["adjusted_price"] * fx["rate"]
            forecast["last_adjusted_price_krw"] = forecast["last_adjusted_price"] * fx["rate"]
    ledger = read_json(PRIVATE / "전진예측원장.json", [])
    for forecast in forecasts:
        # 같은 날짜 예측을 유리한 후속 값으로 덮어쓰지 않고 최초 등록만 전진 평가합니다.
        if not any(row["symbol"] == forecast["symbol"] and row["as_of"] == forecast["as_of"]
                   and row.get("universe_hash") == UNIVERSE_HASH for row in ledger):
            ledger.append({**forecast, "actual": None, "target_date": None})
    if STOP.exists():
        raise InterruptedError("예측 원장 저장 전 중지 요청을 확인했습니다")
    save_json(PRIVATE / "전진예측원장.json", ledger)
    signals = read_json(PRIVATE / "모의신호.json", [])
    origin = min(row["as_of"] for row in forecasts)
    from .autotrading import select_allocation
    from .paper import ACCOUNT
    if not any(row["as_of"] == origin and row.get("policy_version") == 3 and row.get("universe_hash") == UNIVERSE_HASH for row in signals):
        # 기존 세 종목의 최초 신호는 보존하며 3차 후보 범위의 하루 최초 전망을 별도 고정합니다.
        account = read_json(ACCOUNT, {})
        equity = account.get("current_balance_krw", 1000000) / fx["rate"] if fx else 0
        allocation = select_allocation(forecasts, equity, load_config()["autotrading"])
        signals.append({"id": "v3:" + created_at, "as_of": origin, "created_at": created_at,
                        **allocation, "model": label, "policy": allocation["name"]})
        if STOP.exists():
            raise InterruptedError("모의 신호 저장 전 중지 요청을 확인했습니다")
        save_json(PRIVATE / "모의신호.json", signals)
    update_state(forecasts=forecasts)
    reconcile_forward(series)
    from .paper import reconcile_paper
    if STOP.exists():
        raise InterruptedError("계좌 재조정 전 중지 요청을 확인했습니다")
    reconcile_paper(series)


def reconcile_forward(series):
    ledger_path = PRIVATE / "전진예측원장.json"
    ledger = read_json(ledger_path, [])
    for entry in ledger:
        if entry["actual"] is not None:
            continue
        future = [row for row in series.get(entry["symbol"], []) if row["date"] > entry["as_of"]]
        if future:
            from datetime import time as clock_time
            from zoneinfo import ZoneInfo
            closing = datetime.combine(datetime.fromisoformat(future[0]["date"]).date(), clock_time(16), ZoneInfo("America/New_York"))
            if datetime.fromisoformat(entry["created_at"]) >= closing:
                entry["excluded_reason"] = "신호 생성 전에 이미 끝난 첫 목표 봉이므로 전진 성과에서 제외"
                continue
            entry["actual"] = future[0]["close"]
            entry["target_date"] = future[0]["date"]
            entry["resolved_at"] = now_iso()
    save_json(ledger_path, ledger)
    completed = [row for row in ledger if row["actual"] is not None and row.get("universe_hash") == UNIVERSE_HASH]
    previous = [row for row in ledger if row["actual"] is not None and row.get("universe_hash") is None]
    forward = {"status": "ready" if completed else "pending", "n": len(completed),
               "message": "3차 후보 범위에서 신호 생성 뒤 새 완결 일봉을 기다리고 있습니다",
               "universe_hash": UNIVERSE_HASH, "coverage_symbols": sorted({r["symbol"] for r in completed}),
               "previous_scope": {"n": len(previous), "coverage_symbols": sorted({r["symbol"] for r in previous}),
                                  "note": "기존3종목 최초 전진 기록, 새 후보 범위 수치와 합산하지 않음"}}
    if completed:
        forward.update(forecast_metrics([row["actual"] for row in completed],
                                       [row["adjusted_price"] for row in completed],
                                       [row["last_adjusted_price"] for row in completed]))
    update_state(forward=forward)
    return forward

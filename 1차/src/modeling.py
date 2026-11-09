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
from .runtime import PRIVATE, STOP, load_config, load_state, log, update_state


MODEL_ID = "amazon/chronos-2"
SPLIT = PRIVATE / "고정분할.json"
TRAINING = PRIVATE / "학습기록.json"
LATEST = PRIVATE / "체크포인트" / "latest.json"


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


def train_round():
    if STOP.exists():
        raise InterruptedError("학습 라이브러리를 불러오기 전 중지 요청을 확인했습니다")
    prepare_environment()
    update_state(training={"status": "loading"}, job={"message": "학습 라이브러리와 마지막 저장 모델을 불러오고 있습니다"})
    import numpy as np
    import torch
    from chronos import Chronos2Pipeline
    from huggingface_hub import HfApi, snapshot_download
    from transformers import TrainerCallback
    if STOP.exists():
        raise InterruptedError("학습 라이브러리를 불러온 뒤 중지 요청을 확인했습니다")

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
    record = read_json(TRAINING, {"rounds": [], "last_round": 0})
    number = record.get("last_round", 0) + 1
    record["last_round"] = number
    save_json(TRAINING, record)
    latest = read_json(LATEST, {})
    run_dir = PRIVATE / "학습회차" / f"{number:04d}차"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = now_iso()
    total_steps = int(config.get("train_steps", 20))
    context_length = int(config.get("context_length", 256))
    if not 1 <= total_steps <= 1000 or not 32 <= context_length <= 512:
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
        for segment in ("validation", "test"):
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

    train_inputs = [np.asarray([row["close"] for row in rows if row["date"] <= split["train_end"]], dtype=np.float32)
                    for rows in series.values()]
    new_observations = 0
    for rows in market["series"].values():
        new_values = [row["close"] for row in rows if row["date"] > split["test_end"]]
        # 초기 시험 이후 새 구간은 별도 시계열로 학습하여 제외 구간을 가로질러 붙이지 않습니다.
        if len(new_values) >= 2:
            train_inputs.append(np.asarray(new_values, dtype=np.float32))
            new_observations += len(new_values)
    digest = hashlib.sha256(json.dumps(series, sort_keys=True).encode()).hexdigest()
    manifest = {"round": number, "started_at": started, "model": MODEL_ID, "revision": base_info["revision"],
                "initial_checkpoint": resume_source or base_info["snapshot"], "split": split,
                "frozen_data_sha256": digest, "new_training_data": new_observations > 0,
                "post_initial_test_observations": new_observations,
                "train_observations": sum(len(values) for values in train_inputs), "steps": total_steps,
                "batch_size": int(config.get("batch_size", 2)), "context_length": context_length,
                "device": device, "gpu_memory_fraction": config.get("gpu_memory_fraction", 0.25),
                "torch": torch.__version__, "method": "full parameter fine-tuning, AdamW, learning_rate=1e-6",
                "optimizer_resume": False, "resume_note": "직전 저장본의 가중치를 이어 사용하며 optimizer 상태는 회차마다 새로 생성"}
    save_json(run_dir / "실행명세.json", manifest)
    update_state(training={"status": "training", "resumed": bool(resume_source),
                          "cumulative_steps": latest.get("cumulative_steps", 0)})
    tuned = pipeline.fit(inputs=train_inputs, prediction_length=1,
                         context_length=context_length, learning_rate=1e-6, num_steps=total_steps,
                         batch_size=int(config.get("batch_size", 2)), output_dir=run_dir,
                         finetuned_ckpt_name="학습모델", callbacks=[Progress()],
                         logging_steps=20, logging_nan_inf_filter=False, disable_tqdm=True,
                         optim="adamw_torch", bf16=False, tf32=False,
                         save_strategy="no", report_to="none", seed=42 + number)
    actual_steps = load_state()["training"]["step"]
    if tuned.model is None:
        raise RuntimeError("추가 학습 모델이 없습니다")
    pointer = commit_checkpoint(run_dir / "학습모델", actual_steps, number)
    update_state(training={"checkpoint": "latest", "checkpoint_saved_at": pointer["saved_at"],
                          "cumulative_steps": pointer["cumulative_steps"]})
    if STOP.exists():
        save_json(run_dir / "중지기록.json", {"at": now_iso(), "checkpoint": "latest", "steps": actual_steps})
        raise InterruptedError("현재 단계까지 체크포인트를 저장하고 중지했습니다")
    update_state(training={"status": "evaluating"}, validation={"status": "running"},
                 job={"message": "같은 모델의 예측 오차를 기록하고 있습니다 · 모델 선택 절차 없음"})
    monitor_metrics, monitor_details = evaluate(tuned, series, split, "validation", context_length, "계속 학습 · 현재")
    save_json(run_dir / "검증예측.json", monitor_details)
    metrics = list(baselines["metrics"]) + [monitor_metrics]
    # 최초 시험은 기록으로만 유지하며 계속 학습 여부를 결정하지 않습니다.
    if not (PRIVATE / "최초시험평가.json").exists():
        test_metrics, test_details = evaluate(tuned, series, split, "test", context_length, "추가 학습 · 최초 시험")
        save_json(PRIVATE / "최초시험평가.json", {"metrics": test_metrics, "details": test_details})
    initial_test = read_json(PRIVATE / "최초시험평가.json")
    if initial_test:
        metrics.append(initial_test["metrics"])
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
                             "independent_forward_observations": load_state().get("forward", {}).get("n", 0)})
    # 동일 모델의 마지막 가중치로 실제 신규 관측 전 전망을 기록합니다.
    del pipeline
    gc.collect()
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
    os.replace(source, destination)
    pointer = {"checkpoint": str(destination), "previous_checkpoint": previous.get("checkpoint"),
               "saved_at": now_iso(), "round": round_number,
               "cumulative_steps": previous.get("cumulative_steps", 0) + steps, "sha256": hashes}
    save_json(LATEST, pointer)
    obsolete = previous.get("previous_checkpoint")
    if obsolete:
        target = Path(obsolete).resolve()
        if target.parent != store or not target.name.startswith("저장-"):
            raise RuntimeError("복구본 정리 경로가 체크포인트 전용 폴더를 벗어났습니다")
        shutil.rmtree(target)
    return pointer


def publish_forecasts(pipeline, series, context_length, label):
    symbols = sorted(series)
    values = predict_values(pipeline, [[row["close"] for row in series[symbol][-context_length:]] for symbol in symbols], context_length)
    created_at = now_iso()
    forecasts = [{"symbol": symbol, "as_of": series[symbol][-1]["date"], "horizon_sessions": 1,
                  "adjusted_price": value, "last_adjusted_price": series[symbol][-1]["close"],
                  "model": label, "created_at": created_at} for symbol, value in zip(symbols, values)]
    from .paper import FX
    fx = read_json(FX)
    if fx:
        for forecast in forecasts:
            forecast["adjusted_price_krw"] = forecast["adjusted_price"] * fx["rate"]
            forecast["last_adjusted_price_krw"] = forecast["last_adjusted_price"] * fx["rate"]
    ledger = read_json(PRIVATE / "전진예측원장.json", [])
    for forecast in forecasts:
        # 같은 날짜 예측을 유리한 후속 값으로 덮어쓰지 않고 최초 등록만 전진 평가합니다.
        if not any(row["symbol"] == forecast["symbol"] and row["as_of"] == forecast["as_of"] for row in ledger):
            ledger.append({**forecast, "actual": None, "target_date": None})
    save_json(PRIVATE / "전진예측원장.json", ledger)
    signals = read_json(PRIVATE / "모의신호.json", [])
    origin = min(row["as_of"] for row in forecasts)
    if not any(row["as_of"] == origin for row in signals):
        # 하루 최초 latest 전망만 모의 신호로 고정해 반복 학습 결과의 유리한 소급 선택을 막습니다.
        scores = {row["symbol"]: max(0.5, min(1.5, 1 + 20 * (row["adjusted_price"] / row["last_adjusted_price"] - 1)))
                  for row in forecasts}
        total = sum(scores.values())
        signals.append({"id": created_at, "as_of": origin, "created_at": created_at,
                        "weights": {symbol: score / total for symbol, score in scores.items()},
                        "model": label, "policy": "양의 제한 점수를 정규화한 소수점 매매 · 하루 최초 신호"})
        save_json(PRIVATE / "모의신호.json", signals)
    update_state(forecasts=forecasts)
    reconcile_forward(series)
    from .paper import reconcile_paper
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
    completed = [row for row in ledger if row["actual"] is not None]
    forward = {"status": "ready" if completed else "pending", "n": len(completed),
               "message": "운용 시작 후 새 완결 일봉을 기다리고 있습니다"}
    if completed:
        forward.update(forecast_metrics([row["actual"] for row in completed],
                                       [row["adjusted_price"] for row in completed],
                                       [row["last_adjusted_price"] for row in completed]))
    update_state(forward=forward)

"""
train.py
========

Назначение
----------
Этот скрипт обучает YOLOv8 (Ultralytics) на подготовленном датасете RTSD
в формате YOLO (см. `data/yolo/dataset.yaml`).

Что делает скрипт:
- запускает обучение предобученной модели `yolov8n.pt` на 50 эпох;
- сохраняет результаты обучения в `models/experiment1/` (графики, логи, веса);
- копирует лучшую модель `best.pt` в `models/model.pt`;
- сохраняет краткую информацию о модели и метриках в `models/model_info.txt`.

Скрипт intentionally простой и "линейный": без CLI, параметры — вверху файла.
"""

from __future__ import annotations

import csv
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch
    from ultralytics import YOLO
except ModuleNotFoundError as e:
    print(
        "[ОШИБКА] Не найдена зависимость: "
        f"{e}.\n"
        "Установите зависимости проекта командой:\n"
        "  pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    raise SystemExit(1)

# ----------------------------
# Настройки (меняйте при необходимости)
# ----------------------------

DATASET_YAML = Path("data/yolo/dataset.yaml")

MODEL_PRETRAINED = "yolov8n.pt"  # скачивается автоматически при первом запуске

EPOCHS = 50
BATCH = 16
IMGSZ = 800
OPTIMIZER = "AdamW"
LR0 = 0.001
AUGMENT = True
COS_LR = True
CLS = 1.5
MIXUP = 0.2
DEGREES=5.0
SCALE = 0.5

PROJECT_DIR = Path("models")
EXPERIMENT_NAME = "experiment1"

FINAL_MODEL_PATH = PROJECT_DIR / "model.pt"
MODEL_INFO_PATH = PROJECT_DIR / "model_info.txt"


def _die(message: str, exit_code: int = 1) -> None:
    """Печатает сообщение об ошибке и завершает программу."""
    print(f"[ОШИБКА] {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _ensure_dir(path: Path) -> None:
    """Создаёт папку, если её ещё нет."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _die(f"Не удалось создать папку `{path}`: {e}")


def _safe_get(d: Any, key: str) -> Any:
    """Безопасно достаёт значение из dict-похожего объекта."""
    try:
        return d.get(key)
    except Exception:
        return None


def try_extract_metrics_from_train_result(train_result: Any) -> dict[str, Any]:
    """
    Пытается извлечь метрики напрямую из объекта, который вернул Ultralytics.

    В разных версиях Ultralytics API мог немного меняться, поэтому делаем
    аккуратные проверки и не падаем, если ключей нет.
    """
    metrics: dict[str, Any] = {}

    # Часто встречается `.results_dict`
    if hasattr(train_result, "results_dict"):
        rd = getattr(train_result, "results_dict", None)
        if isinstance(rd, dict):
            metrics.update(rd)

    # Иногда результат — просто dict
    if isinstance(train_result, dict):
        metrics.update(train_result)

    return metrics


def try_extract_map50_from_results_csv(run_dir: Path) -> float | None:
    """
    Fallback: если не удалось достать метрики из train_result,
    попробуем прочитать `results.csv` в папке эксперимента.

    Обычно файл лежит так:
      models/experiment1/results.csv
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except OSError:
        return None

    if not rows:
        return None

    # Берём последнюю строку (последняя эпоха).
    last = rows[-1]
    v = last.get("metrics/mAP50(B)")
    if v is None:
        return None

    try:
        return float(v)
    except ValueError:
        return None


def get_ultralytics_save_dir(train_result: Any, fallback_run_dir: Path) -> Path:
    """
    Пытается определить реальную папку, куда Ultralytics сохранил эксперимент.

    Почему это важно:
    - Ultralytics может автоматически переименовать `name` (например, experiment1 -> experiment15),
      если такая папка уже существует;
    - иногда путь оказывается не `models/experiment1`, а, например, `runs/detect/models/experiment15`.

    Поэтому нельзя жёстко полагаться только на `PROJECT_DIR / EXPERIMENT_NAME`.
    """
    # 1) Самый надёжный вариант — train_result.save_dir (есть во многих версиях Ultralytics)
    if hasattr(train_result, "save_dir"):
        sd = getattr(train_result, "save_dir")
        if sd:
            try:
                p = Path(sd)
                if p.exists():
                    return p
            except Exception:
                pass

    # 2) Иногда путь лежит в model.trainer.save_dir, но сюда мы его напрямую не передаём.
    # Поэтому оставим только fallback.
    return fallback_run_dir


def find_best_checkpoint(save_dir: Path) -> Path | None:
    """
    Ищет `weights/best.pt`:
    - сначала в `save_dir/weights/best.pt`;
    - затем (fallback) ищет самый свежий `**/weights/best.pt` в `save_dir` и в `runs/`.
    """
    direct = save_dir / "weights" / "best.pt"
    if direct.exists():
        return direct

    # Иногда Ultralytics мог сохранить не совсем туда, куда ожидаем.
    # Для robustness попробуем найти самый свежий best.pt.
    candidates: list[Path] = []
    if save_dir.exists():
        candidates.extend(save_dir.rglob("weights/best.pt"))

    runs_dir = Path("runs")
    if runs_dir.exists():
        candidates.extend(runs_dir.rglob("weights/best.pt"))

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    print("[INFO] === YOLOv8: обучение на RTSD ===")

    # Проверяем, что датасет подготовлен
    if not DATASET_YAML.exists():
        _die(
            f"Не найден файл `{DATASET_YAML}`.\n"
            "Сначала запустите `python src/prepare_data.py`."
        )

    _ensure_dir(PROJECT_DIR)

    # Создаём модель
    print(f"[INFO] Загружаем предобученную модель: {MODEL_PRETRAINED}")
    model = YOLO(MODEL_PRETRAINED)

    # Запускаем обучение.
    # Важно: `project` и `name` задают папку для артефактов: models/experiment1/
    print("[INFO] Старт обучения...")
    train_result = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        optimizer=OPTIMIZER,
        lr0=LR0,
        augment=AUGMENT,
        project=str(PROJECT_DIR),
        name=EXPERIMENT_NAME,
        save=True,
        plots=True,
        cls=CLS,
        cos_lr=COS_LR,
        mixup=MIXUP,
        degrees=DEGREES,
        scale=SCALE,
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    )
    print("[INFO] Обучение завершено.")

    # Важно: Ultralytics может сохранить результаты не ровно в `models/experiment1`.
    # Получим реальный save_dir (или используем fallback) и уже относительно него найдём best.pt.
    fallback_run_dir = PROJECT_DIR / EXPERIMENT_NAME
    save_dir = get_ultralytics_save_dir(train_result, fallback_run_dir=fallback_run_dir)
    print(f"[INFO] Папка эксперимента (save_dir): {save_dir}")

    best_path = find_best_checkpoint(save_dir)

    if best_path is None or not best_path.exists():
        _die(
            "Не найден файл лучшей модели `best.pt`.\n"
            "Проверьте, куда Ultralytics сохранил эксперимент (обычно `runs/` или `models/`)."
        )

    # Копируем best.pt в models/model.pt — удобно для отчёта/демо
    try:
        FINAL_MODEL_PATH.write_bytes(best_path.read_bytes())
    except OSError as e:
        _die(f"Не удалось скопировать `{best_path}` -> `{FINAL_MODEL_PATH}`: {e}")

    print(f"[INFO] Лучшая модель сохранена: {FINAL_MODEL_PATH}")

    # Метрики: сначала пробуем из train_result, затем fallback через results.csv
    metrics = try_extract_metrics_from_train_result(train_result)
    map50 = (
        _safe_get(metrics, "metrics/mAP50(B)")
        or _safe_get(metrics, "metrics/mAP50")
        or _safe_get(metrics, "map50")
    )

    # Если map50 не нашли (или оно не число), пробуем results.csv
    map50_float: float | None = None
    if map50 is not None:
        try:
            map50_float = float(map50)
        except (TypeError, ValueError):
            map50_float = None

    if map50_float is None:
        map50_float = try_extract_map50_from_results_csv(save_dir)

    # Пишем model_info.txt (полезно для отчёта)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_lines = [
        "YOLOv8 training summary (RTSD)",
        f"datetime: {now}",
        f"dataset_yaml: {DATASET_YAML.as_posix()}",
        f"pretrained: {MODEL_PRETRAINED}",
        f"epochs: {EPOCHS}",
        f"batch: {BATCH}",
        f"imgsz: {IMGSZ}",
        f"optimizer: {OPTIMIZER}",
        f"lr0: {LR0}",
        f"augment: {AUGMENT}",
        f"project_dir: {PROJECT_DIR.as_posix()}",
        f"experiment_name: {EXPERIMENT_NAME}",
        f"save_dir: {save_dir.as_posix()}",
        f"best_checkpoint: {best_path.as_posix()}",
        f"final_model: {FINAL_MODEL_PATH.as_posix()}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
    ]

    if map50_float is not None:
        info_lines.append(f"metrics/mAP50(B): {map50_float:.6f}")
    else:
        info_lines.append("metrics/mAP50(B): (не удалось извлечь автоматически)")


    # На случай если удалось достать ещё какие-то метрики — допишем их в конец.
    # (Сильно не расписываем, чтобы файл был короткий и понятный.)
    interesting_keys = [
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
    ]
    for k in interesting_keys:
        v = _safe_get(metrics, k)
        if v is None:
            continue
        try:
            info_lines.append(f"{k}: {float(v):.6f}")
        except (TypeError, ValueError):
            info_lines.append(f"{k}: {v}")

    try:
        MODEL_INFO_PATH.write_text("\n".join(info_lines) + "\n", encoding="utf-8")
    except OSError as e:
        _die(f"Не удалось записать `{MODEL_INFO_PATH}`: {e}")

    print(f"[INFO] Информация о модели сохранена: {MODEL_INFO_PATH}")
    print(f"[INFO] Графики/логи эксперимента лежат в: {save_dir}")


if __name__ == "__main__":
    main()

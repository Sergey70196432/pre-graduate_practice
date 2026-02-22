"""
prepare_data.py
===============

Назначение
----------
Этот скрипт берёт исходные данные GTSDB из `data/raw/` и подготавливает их
в формате, удобном для обучения YOLO:

- копирует изображения `*.ppm` в `data/yolo/images/`
- создаёт файлы разметки `data/yolo/labels/*.txt` в формате YOLO
- случайно делит изображения на train/val в пропорции 600/300
- создаёт списки `train.txt` и `val.txt` (пути к изображениям)
- создаёт конфиг `dataset.yaml` для Ultralytics YOLO

Формат входной разметки (`gt.txt`)
----------------------------------
Каждая строка:
    filename;x1;y1;x2;y2;class_id

Пример:
    00000.ppm;774;411;815;446;1

Важно про class_id:
-------------------
В разных источниках GTSDB встречаются class_id от 1 до 43.
Для YOLO удобнее использовать 0..42, поэтому скрипт автоматически проверяет
диапазон class_id и при необходимости делает сдвиг -1.

Скрипт специально сделан «линейным» и понятным: параметры меняются вверху файла.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict

import cv2
import yaml

# ----------------------------
# Настройки (меняйте при необходимости)
# ----------------------------

RAW_DIR = Path("data/raw/FullIJCNN2013")
YOLO_DIR = Path("data/yolo")

IMAGES_DIR = YOLO_DIR / "images"
LABELS_DIR = YOLO_DIR / "labels"

GT_PATH = RAW_DIR / "gt.txt"

NC = 43  # число классов в датасете
CLASS_NAMES = [str(i) for i in range(NC)]  # имена классов (упрощённо)

TRAIN_SIZE = 600
VAL_SIZE = 300
RANDOM_SEED = 42

# Сколько знаков после запятой писать в координатах YOLO
YOLO_FLOAT_PRECISION = 6

# Важно: Ultralytics YOLO не поддерживает .ppm как входные изображения,
# поэтому при подготовке датасета конвертируем .ppm -> .jpg.
OUTPUT_IMAGE_EXT = ".jpg"
JPEG_QUALITY = 95


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


@dataclass(frozen=True)
class GtBox:
    """Одна bbox-аннотация из gt.txt."""

    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int


def read_gt_file(gt_path: Path) -> DefaultDict[str, list[GtBox]]:
    """
    Читает `gt.txt` и группирует боксы по имени файла изображения.

    Возвращает словарь:
      filename -> список GtBox
    """
    if not gt_path.exists():
        _die(
            f"Не найден `{gt_path}`.\n"
            "Сначала запустите `python src/download_data.py`."
        )

    boxes_by_file: DefaultDict[str, list[GtBox]] = defaultdict(list)

    try:
        # `gt.txt` обычно ASCII/UTF-8, но на всякий случай включаем errors='replace'
        lines = gt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        _die(f"Не удалось прочитать `{gt_path}`: {e}")

    for line_idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split(";")
        if len(parts) != 6:
            _die(
                f"Некорректная строка в gt.txt (строка {line_idx}):\n"
                f"{line}\n"
                "Ожидалось 6 полей, разделённых ';'."
            )

        filename, x1, y1, x2, y2, class_id = parts
        try:
            box = GtBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                class_id=int(class_id),
            )
        except ValueError:
            _die(
                f"Некорректные числа в gt.txt (строка {line_idx}):\n{line}"
            )

        boxes_by_file[filename].append(box)

    if not boxes_by_file:
        _die("Файл gt.txt прочитан, но аннотаций не найдено (пусто).")

    return boxes_by_file


def detect_class_shift(boxes_by_file: dict[str, list[GtBox]]) -> int:
    """
    Определяет, нужно ли сдвигать class_id.

    Логика:
    - если минимальный class_id == 1 и максимальный == 43, то сдвигаем на -1
    - иначе считаем, что уже 0..42
    """
    all_ids: list[int] = []
    for boxes in boxes_by_file.values():
        for b in boxes:
            all_ids.append(b.class_id)

    min_id = min(all_ids)
    max_id = max(all_ids)

    print(f"[INFO] Диапазон class_id в gt.txt: min={min_id}, max={max_id}")

    if min_id == 1 and max_id == NC:
        print("[INFO] Похоже, class_id в gt.txt начинается с 1. Сдвигаем на -1 → 0..42.")
        return -1

    if min_id == 0 and max_id == NC - 1:
        print("[INFO] class_id уже в диапазоне 0..42. Сдвиг не нужен.")
        return 0

    # Неизвестный вариант — лучше сообщить пользователю, чтобы он заметил.
    print(
        "[WARN] Неожиданный диапазон class_id. Я НЕ делаю автоматический сдвиг.\n"
        "       Проверьте gt.txt и при необходимости поправьте логику в detect_class_shift()."
    )
    return 0


def clamp(v: int, vmin: int, vmax: int) -> int:
    """Ограничивает v в диапазон [vmin, vmax]."""
    return max(vmin, min(vmax, v))


def box_to_yolo(
    box: GtBox, img_w: int, img_h: int, class_shift: int
) -> tuple[int, float, float, float, float] | None:
    """
    Переводит bbox из (x1,y1,x2,y2) в YOLO-формат:
      class_id x_center y_center width height (все координаты 0..1)

    Возвращает None, если bbox после приведения некорректен.
    """
    # Сдвигаем class_id при необходимости (1..43 -> 0..42)
    class_id = box.class_id + class_shift

    # Минимальные sanity-checks
    if img_w <= 0 or img_h <= 0:
        return None

    # В gt.txt x2/y2 обычно уже "правый/нижний" край.
    # На всякий случай нормализуем порядок и прижмём к размерам изображения.
    x1 = min(box.x1, box.x2)
    x2 = max(box.x1, box.x2)
    y1 = min(box.y1, box.y2)
    y2 = max(box.y1, box.y2)

    # Прижимаем к границам кадра (важно: координаты пиксельные)
    x1 = clamp(x1, 0, img_w - 1)
    x2 = clamp(x2, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    y2 = clamp(y2, 0, img_h - 1)

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Ещё одна защита от совсем странных значений
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return None
    if not (0.0 < w_norm <= 1.0 and 0.0 < h_norm <= 1.0):
        return None

    return class_id, x_center, y_center, w_norm, h_norm


def write_yolo_label_file(label_path: Path, yolo_lines: list[str]) -> None:
    """Пишет `.txt` разметки YOLO (или пустой файл, если объектов нет)."""
    try:
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
    except OSError as e:
        _die(f"Не удалось записать `{label_path}`: {e}")


def main() -> None:
    print("[INFO] === GTSDB: подготовка датасета в формате YOLO ===")

    # Проверим, что исходные данные действительно на месте
    if not RAW_DIR.exists():
        _die(
            f"Не найдена папка `{RAW_DIR}`.\n"
            "Сначала запустите `python src/download_data.py`."
        )

    _ensure_dir(YOLO_DIR)
    _ensure_dir(IMAGES_DIR)
    _ensure_dir(LABELS_DIR)

    boxes_by_file = read_gt_file(GT_PATH)
    class_shift = detect_class_shift(boxes_by_file)

    # Собираем список изображений из data/raw/*.ppm
    raw_images = sorted(RAW_DIR.glob("*.ppm"))
    if not raw_images:
        _die(f"В `{RAW_DIR}` не найдено изображений `*.ppm`.")

    print(f"[INFO] Найдено изображений в raw: {len(raw_images)}")

    # Подготовим генератор формата для чисел (чтобы было красиво/стабильно)
    float_fmt = f"{{:.{YOLO_FLOAT_PRECISION}f}}"

    # Статистика для отчёта / sanity-check
    written_labels = 0
    skipped_boxes = 0
    used_class_ids: list[int] = []

    for raw_img_path in raw_images:
        filename = raw_img_path.name

        # 1) Читаем исходное изображение (.ppm), чтобы узнать размеры и сохранить в .jpg
        img = cv2.imread(str(raw_img_path), cv2.IMREAD_COLOR)
        if img is None:
            _die(
                f"OpenCV не смог прочитать изображение `{raw_img_path}`.\n"
                "Проверьте, что установлен `opencv-python` и файл не повреждён."
            )
        img_h, img_w = img.shape[:2]

        # 2) Сохраняем изображение в data/yolo/images/ в поддерживаемом формате (.jpg)
        dst_img_path = IMAGES_DIR / f"{raw_img_path.stem}{OUTPUT_IMAGE_EXT}"
        ok = cv2.imwrite(
            str(dst_img_path),
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
        )
        if not ok:
            _die(f"Не удалось записать изображение `{dst_img_path}` (cv2.imwrite вернул False).")

        # 3) Генерируем строки разметки YOLO для текущего файла (.txt)
        yolo_lines: list[str] = []
        for box in boxes_by_file.get(filename, []):
            yolo = box_to_yolo(box, img_w=img_w, img_h=img_h, class_shift=class_shift)
            if yolo is None:
                skipped_boxes += 1
                continue

            class_id, x_center, y_center, w_norm, h_norm = yolo
            used_class_ids.append(class_id)

            # В YOLO формат: class x_center y_center width height
            yolo_lines.append(
                " ".join(
                    [
                        str(class_id),
                        float_fmt.format(x_center),
                        float_fmt.format(y_center),
                        float_fmt.format(w_norm),
                        float_fmt.format(h_norm),
                    ]
                )
            )

        # 4) Пишем label-файл (даже если пустой)
        label_path = LABELS_DIR / f"{raw_img_path.stem}.txt"
        write_yolo_label_file(label_path, yolo_lines)
        written_labels += 1

    if used_class_ids:
        print(
            f"[INFO] class_id после сдвига: min={min(used_class_ids)}, max={max(used_class_ids)}"
        )
        if min(used_class_ids) < 0 or max(used_class_ids) >= NC:
            _die(
                "После нормализации class_id вышел за диапазон 0..42.\n"
                "Проверьте `gt.txt` и логику сдвига классов (detect_class_shift)."
            )
    else:
        print("[WARN] Не найдено ни одной валидной bbox (все были пропущены).")

    if skipped_boxes > 0:
        print(f"[WARN] Пропущено bbox из-за некорректных координат: {skipped_boxes}")

    print(f"[INFO] Создано label-файлов: {written_labels}")

    # ----------------------------
    # Split train/val (600/300)
    # ----------------------------
    rel_image_paths = [
        Path("images") / f"{p.stem}{OUTPUT_IMAGE_EXT}"
        for p in raw_images
    ]

    random.seed(RANDOM_SEED)
    random.shuffle(rel_image_paths)

    required_total = TRAIN_SIZE + VAL_SIZE
    if len(rel_image_paths) < required_total:
        print(
            f"[WARN] Ожидалось {required_total} изображений для split 600/300, "
            f"но найдено {len(rel_image_paths)}. Split будет сделан по доступным данным."
        )

    if len(rel_image_paths) > required_total:
        print(
            f"[WARN] Найдено больше {required_total} изображений ({len(rel_image_paths)}). "
            "Для соответствия отчёту возьмём первые 900 после перемешивания."
        )
        rel_image_paths = rel_image_paths[:required_total]

    train_count = min(TRAIN_SIZE, len(rel_image_paths))
    val_count = max(0, min(VAL_SIZE, len(rel_image_paths) - train_count))

    train_list = rel_image_paths[:train_count]
    val_list = rel_image_paths[train_count : train_count + val_count]

    train_txt = YOLO_DIR / "train.txt"
    val_txt = YOLO_DIR / "val.txt"

    try:
        # Важно для Ultralytics:
        # Если строки в train.txt/val.txt начинаются с "./", то Ultralytics автоматически
        # преобразует их в абсолютные пути относительно папки, где лежит сам список
        # (т.е. относительно `data/yolo/`).
        #
        # Если писать просто "images/00000.jpg", то Ultralytics будет трактовать путь
        # относительно текущей рабочей директории (корня проекта), и обучение упадёт.
        train_txt.write_text(
            "\n".join(f"./{p.as_posix()}" for p in train_list) + "\n",
            encoding="utf-8",
        )
        val_txt.write_text(
            "\n".join(f"./{p.as_posix()}" for p in val_list) + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        _die(f"Не удалось записать train/val списки: {e}")

    print(f"[INFO] Split: train={len(train_list)}, val={len(val_list)}")

    # ----------------------------
    # dataset.yaml
    # ----------------------------
    dataset_yaml_path = YOLO_DIR / "dataset.yaml"

    dataset_cfg = {
        "path": "./data/yolo",
        "train": "train.txt",
        "val": "val.txt",
        "nc": NC,
        "names": CLASS_NAMES,
    }

    try:
        dataset_yaml_path.write_text(
            yaml.safe_dump(dataset_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except OSError as e:
        _die(f"Не удалось записать `{dataset_yaml_path}`: {e}")

    # Небольшой вывод для sanity-check (удобно вставлять в отчёт)
    if used_class_ids:
        print(
            "[INFO] Пример (1 строка) YOLO-разметки: "
            f"{(LABELS_DIR / (raw_images[0].stem + '.txt')).name}"
        )
        try:
            sample = (
                (LABELS_DIR / f"{raw_images[0].stem}.txt")
                .read_text(encoding="utf-8")
                .strip()
                .splitlines()
            )
            if sample:
                print(f"       {sample[0]}")
            else:
                print("       (пусто — в этом изображении нет объектов)")
        except OSError:
            pass

    print("[INFO] Готово. Следующий шаг: запустите `python src/train.py`.")


if __name__ == "__main__":
    main()

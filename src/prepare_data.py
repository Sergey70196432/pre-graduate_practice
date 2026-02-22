"""
prepare_data.py
===============

Назначение
----------
Этот скрипт берёт исходные данные RTSD из `data/raw/` и подготавливает их
в формате, удобном для обучения YOLO:

- копирует изображения `*.jpg/*.png` в `data/yolo/images/` (имена сохраняются)
- создаёт файлы разметки `data/yolo/labels/*.txt` в формате YOLO
- формирует `train.txt` и `val.txt` на основе `train_filenames.txt/test_filenames.txt`
  (и включает только изображения, у которых есть хотя бы одна аннотация)
- создаёт конфиг `dataset.yaml` для Ultralytics YOLO

Структура RTSD после `download_data.py`
--------------------------------------
Ожидается:
  data/raw/rtsd-d3-frames/ — изображения (все в одной директории)
  data/raw/rtsd-d3-gt/
    train_filenames.txt
    test_filenames.txt
    <group_1>/train_gt.csv + test_gt.csv
    <group_2>/train_gt.csv + test_gt.csv
    ...

CSV-формат (допущение)
----------------------
Ожидаем поля:
  filename, x1, y1, x2, y2, class_name
Разделитель обычно "," (иногда ";"), первая строка может быть заголовком.
Если реальный формат отличается, смотрите функции `_parse_csv_header/_parse_row`.

Скрипт специально сделан «линейным» и понятным: параметры меняются вверху файла.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import shutil
import csv

try:
    import cv2
    import yaml
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

RAW_DIR = Path("data/raw")
FRAMES_DIR = RAW_DIR / "rtsd-d3-frames"
GT_DIR = RAW_DIR / "rtsd-d3-gt"
YOLO_DIR = Path("data/yolo")

IMAGES_DIR = YOLO_DIR / "images"
LABELS_DIR = YOLO_DIR / "labels"

TRAIN_FILENAMES_PATH = GT_DIR / "train_filenames.txt"
TEST_FILENAMES_PATH = GT_DIR / "test_filenames.txt"

# Ожидаемое число классов в RTSD (по описанию — 106).
# Если по факту окажется меньше/больше, скрипт НЕ упадёт, но выведет предупреждение.
EXPECTED_NC = 106

# Сколько знаков после запятой писать в координатах YOLO
YOLO_FLOAT_PRECISION = 6


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
class RtBox:
    """Одна bbox-аннотация из RTSD CSV."""

    filename: str
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        _die(f"Не удалось прочитать `{path}`: {e}")


def _read_filename_list(path: Path) -> set[str]:
    """
    Читает `train_filenames.txt` / `test_filenames.txt`.
    В файле ожидаются имена изображений (например, frame_000001.jpg), по одному на строку.
    """
    if not path.exists():
        _die(
            f"Не найден `{path}`.\n"
            "Сначала запустите `python src/download_data.py`."
        )
    names: set[str] = set()
    for line in _read_lines(path):
        line = line.strip()
        if not line:
            continue
        names.add(line)
    return names


def _list_image_files(dir_path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    try:
        return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts])
    except OSError as e:
        _die(f"Не удалось прочитать папку `{dir_path}`: {e}")


def _build_image_index(frames_dir: Path) -> dict[str, Path]:
    """
    Строит индекс: filename -> path.
    Нужен, чтобы быстро проверять наличие файла и находить его на диске.
    """
    idx: dict[str, Path] = {}
    for p in _list_image_files(frames_dir):
        idx[p.name] = p
    return idx


def _copy_images_to_yolo(images: Iterable[Path], images_dir: Path) -> None:
    _ensure_dir(images_dir)
    copied = 0
    for p in images:
        dst = images_dir / p.name
        try:
            shutil.copy2(p, dst)
        except OSError as e:
            _die(f"Не удалось скопировать `{p}` -> `{dst}`: {e}")
        copied += 1
    print(f"[INFO] Скопировано изображений в `{images_dir}`: {copied}")


def _sniff_csv_dialect(csv_path: Path) -> csv.Dialect:
    """
    Пытается определить разделитель CSV (обычно ',' или ';').
    Если не вышло — используем запятую.
    """
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            sample = f.read(4096)
    except OSError as e:
        _die(f"Не удалось открыть CSV `{csv_path}`: {e}")

    try:
        return csv.Sniffer().sniff(sample, delimiters=";,")
    except csv.Error:
        # Fallback: запятая
        class _Comma(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            escapechar = None
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _Comma()


def _looks_like_header_row(row: list[str]) -> bool:
    joined = " ".join(c.strip().lower() for c in row if c is not None)
    return "filename" in joined and ("x1" in joined or "x_min" in joined or "xmin" in joined)


def _parse_csv_header(header: list[str]) -> dict[str, int]:
    """
    Если CSV содержит заголовок — пытаемся сопоставить имена колонок с индексами.
    Поддерживаем несколько вариантов названий (на случай отличий в исходнике).
    """
    norm = [h.strip().lower() for h in header]

    def _find(*candidates: str) -> int | None:
        for c in candidates:
            if c in norm:
                return norm.index(c)
        return None

    idx = {
        "filename": _find("filename", "file", "image", "img", "name"),
        "x1": _find("x1", "xmin", "x_min", "left"),
        "y1": _find("y1", "ymin", "y_min", "top"),
        "x2": _find("x2", "xmax", "x_max", "right"),
        "y2": _find("y2", "ymax", "y_max", "bottom"),
        "class": _find("class_name", "class", "label", "class_id"),
    }

    if any(v is None for v in idx.values()):
        # Если заголовок есть, но мы не смогли распознать — лучше явно предупредить.
        missing = [k for k, v in idx.items() if v is None]
        _die(
            "Не удалось распознать колонки CSV по заголовку.\n"
            f"Файл: {header}\n"
            f"Не найдены колонки: {missing}\n"
            "Проверьте формат CSV и при необходимости поправьте `_parse_csv_header()`."
        )

    return {k: int(v) for k, v in idx.items() if v is not None}


def _parse_row(row: list[str], col_idx: dict[str, int] | None) -> RtBox | None:
    """
    Преобразует строку CSV в RtBox.

    Допущение:
    - Если нет заголовка, ожидаем 6 полей:
      filename, x1, y1, x2, y2, class_name
    """
    try:
        if col_idx is None:
            if len(row) < 6:
                return None
            filename = row[0].strip()
            x1, y1, x2, y2 = row[1:5]
            class_name = row[5].strip()
        else:
            filename = row[col_idx["filename"]].strip()
            x1 = row[col_idx["x1"]]
            y1 = row[col_idx["y1"]]
            x2 = row[col_idx["x2"]]
            y2 = row[col_idx["y2"]]
            class_name = row[col_idx["class"]].strip()

        if not filename or not class_name:
            return None

        return RtBox(
            filename=filename,
            x1=int(float(x1)),
            y1=int(float(y1)),
            x2=int(float(x2)),
            y2=int(float(y2)),
            class_name=class_name,
        )
    except (ValueError, IndexError):
        return None


def iter_boxes_from_csv(csv_path: Path) -> Iterable[RtBox]:
    """
    Итерируется по RtBox из одного CSV.
    Делает минимум предположений о разделителе и заголовке.
    """
    if not csv_path.exists():
        _die(f"CSV не найден: {csv_path}")

    dialect = _sniff_csv_dialect(csv_path)

    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, dialect)
            first_row: list[str] | None = None
            try:
                first_row = next(reader)
            except StopIteration:
                return

            col_idx: dict[str, int] | None = None
            if first_row is not None and _looks_like_header_row(first_row):
                col_idx = _parse_csv_header(first_row)
            else:
                box = _parse_row(first_row, col_idx=None)
                if box is not None:
                    yield box

            for row in reader:
                if not row:
                    continue
                box = _parse_row(row, col_idx=col_idx)
                if box is None:
                    continue
                yield box
    except OSError as e:
        _die(f"Не удалось прочитать CSV `{csv_path}`: {e}")


def clamp(v: int, vmin: int, vmax: int) -> int:
    """Ограничивает v в диапазон [vmin, vmax]."""
    return max(vmin, min(vmax, v))


def box_to_yolo(
    box: RtBox, img_w: int, img_h: int, class_id: int
) -> tuple[int, float, float, float, float] | None:
    """
    Переводит bbox из (x1,y1,x2,y2) в YOLO-формат:
      class_id x_center y_center width height (все координаты 0..1)

    Возвращает None, если bbox после приведения некорректен.
    """
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
    print("[INFO] === RTSD: подготовка датасета в формате YOLO ===")

    # Проверим, что исходные данные действительно на месте
    if not RAW_DIR.exists() or not FRAMES_DIR.exists() or not GT_DIR.exists():
        _die(
            f"Не найдены исходные данные RTSD в `{RAW_DIR}`.\n"
            "Сначала запустите `python src/download_data.py`."
        )

    _ensure_dir(YOLO_DIR)
    _ensure_dir(IMAGES_DIR)
    _ensure_dir(LABELS_DIR)

    # ----------------------------
    # Шаг 1: изображения -> data/yolo/images (копируем как есть)
    # ----------------------------
    image_index = _build_image_index(FRAMES_DIR)
    if not image_index:
        _die(f"В `{FRAMES_DIR}` не найдено изображений `*.jpg/*.png`.")

    print(f"[INFO] Найдено изображений в raw frames: {len(image_index)}")
    _copy_images_to_yolo(image_index.values(), IMAGES_DIR)

    # ----------------------------
    # Шаг 2: читаем списки train/test
    # ----------------------------
    train_files = _read_filename_list(TRAIN_FILENAMES_PATH)
    test_files = _read_filename_list(TEST_FILENAMES_PATH)
    print(f"[INFO] train_filenames: {len(train_files)}, test_filenames: {len(test_files)}")

    # ----------------------------
    # Шаг 3: читаем все CSV и строим mapping классов
    # ----------------------------
    csv_paths = sorted(GT_DIR.glob("*/train_gt.csv")) + sorted(GT_DIR.glob("*/test_gt.csv"))
    if not csv_paths:
        _die(f"В `{GT_DIR}` не найдено файлов `*/train_gt.csv` / `*/test_gt.csv`.")

    # Первый проход: собираем уникальные class_name
    all_class_names: set[str] = set()
    total_rows = 0
    bad_rows = 0
    for csv_path in csv_paths:
        for box in iter_boxes_from_csv(csv_path):
            total_rows += 1
            if not box.class_name:
                bad_rows += 1
                continue
            all_class_names.add(box.class_name)

    if not all_class_names:
        _die("Не удалось собрать ни одного имени класса из CSV. Проверьте формат исходных данных.")

    class_names_sorted = sorted(all_class_names)
    class_to_id = {name: i for i, name in enumerate(class_names_sorted)}

    if len(class_names_sorted) != EXPECTED_NC:
        print(
            f"[WARN] Ожидалось классов: {EXPECTED_NC}, найдено: {len(class_names_sorted)}.\n"
            "       Скрипт продолжит работу, но проверьте корректность разметки/файлов."
        )

    # Сохраняем classes.txt (по одному имени класса на строку)
    classes_txt_path = YOLO_DIR / "classes.txt"
    try:
        classes_txt_path.write_text("\n".join(class_names_sorted) + "\n", encoding="utf-8")
    except OSError as e:
        _die(f"Не удалось записать `{classes_txt_path}`: {e}")
    print(f"[INFO] Сохранён список классов: {classes_txt_path} (n={len(class_names_sorted)})")

    # Второй проход: генерируем YOLO labels
    float_fmt = f"{{:.{YOLO_FLOAT_PRECISION}f}}"
    image_size_cache: dict[str, tuple[int, int]] = {}  # filename -> (w, h)
    yolo_lines_by_file: dict[str, list[str]] = {}
    skipped_boxes = 0
    missing_images = 0

    def _get_image_size(filename: str) -> tuple[int, int] | None:
        if filename in image_size_cache:
            return image_size_cache[filename]

        p = image_index.get(filename)
        if p is None:
            return None

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        image_size_cache[filename] = (w, h)
        return (w, h)

    for csv_path in csv_paths:
        for box in iter_boxes_from_csv(csv_path):
            class_id = class_to_id.get(box.class_name)
            if class_id is None:
                skipped_boxes += 1
                continue

            size = _get_image_size(box.filename)
            if size is None:
                missing_images += 1
                continue
            img_w, img_h = size

            yolo = box_to_yolo(box, img_w=img_w, img_h=img_h, class_id=class_id)
            if yolo is None:
                skipped_boxes += 1
                continue

            class_id2, x_center, y_center, w_norm, h_norm = yolo
            line = " ".join(
                [
                    str(class_id2),
                    float_fmt.format(x_center),
                    float_fmt.format(y_center),
                    float_fmt.format(w_norm),
                    float_fmt.format(h_norm),
                ]
            )
            yolo_lines_by_file.setdefault(box.filename, []).append(line)

    if missing_images > 0:
        print(
            f"[WARN] Не найдены изображения для некоторых аннотаций: {missing_images}.\n"
            "       Проверьте соответствие имён файлов в CSV и в `data/raw/rtsd-d3-frames/`."
        )
    if skipped_boxes > 0:
        print(f"[WARN] Пропущено bbox из-за некорректных строк/координат: {skipped_boxes}")

    if not yolo_lines_by_file:
        _die("Не удалось сформировать ни одного label-файла (нет валидных аннотаций).")

    written_labels = 0
    for filename, lines in yolo_lines_by_file.items():
        label_path = LABELS_DIR / f"{Path(filename).stem}.txt"
        write_yolo_label_file(label_path, lines)
        written_labels += 1

    print(f"[INFO] Создано label-файлов: {written_labels}")

    # ----------------------------
    # Шаг 4: train.txt / val.txt по train/test спискам
    # ----------------------------
    # В RTSD test_filenames.txt будем трактовать как val.
    # Важно: включаем только те изображения, у которых есть хотя бы одна аннотация.
    annotated_files = set(yolo_lines_by_file.keys())

    train_annotated = sorted([fn for fn in annotated_files if fn in train_files])
    val_annotated = sorted([fn for fn in annotated_files if fn in test_files])

    if not train_annotated:
        print(
            "[WARN] Не найдено ни одного train-изображения с аннотациями.\n"
            "       Возможные причины: несовпадение имён файлов или пустая разметка."
        )
    if not val_annotated:
        print(
            "[WARN] Не найдено ни одного val-изображения с аннотациями.\n"
            "       Возможные причины: несовпадение имён файлов или пустая разметка."
        )

    train_list = [Path("images") / fn for fn in train_annotated]
    val_list = [Path("images") / fn for fn in val_annotated]

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
    # Шаг 5: dataset.yaml
    # ----------------------------
    dataset_yaml_path = YOLO_DIR / "dataset.yaml"

    dataset_cfg = {
        "path": "./data/yolo",
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(class_names_sorted),
        "names": class_names_sorted,
    }

    try:
        dataset_yaml_path.write_text(
            yaml.safe_dump(dataset_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except OSError as e:
        _die(f"Не удалось записать `{dataset_yaml_path}`: {e}")

    # Небольшой вывод для sanity-check (удобно вставлять в отчёт)
    try:
        any_file = next(iter(train_annotated or val_annotated))
        sample_label = LABELS_DIR / f"{Path(any_file).stem}.txt"
        print(f"[INFO] Пример label-файла: {sample_label.name}")
        sample = sample_label.read_text(encoding="utf-8").strip().splitlines()
        if sample:
            print(f"       {sample[0]}")
    except StopIteration:
        pass
    except OSError:
        pass

    print("[INFO] Готово. Следующий шаг: запустите `python src/train.py`.")


if __name__ == "__main__":
    main()

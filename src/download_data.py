"""
download_data.py
================

Назначение
----------
Этот скрипт скачивает датасет дорожных знаков GTSDB (архив FullIJCNN2013.zip),
распаковывает его в папку `data/raw/` и делает простые проверки, что данные
действительно появились (файл `gt.txt` и изображения `*.ppm`).

Скрипт intentionally сделан максимально простым и "линейным" — без CLI,
параметры задаются в верхней части файла как константы.

Формат архива (важное)
----------------------
Внутри архива есть:
- 900 изображений-сцен в корне архива (формат .ppm) — именно они нужны для детекции.
- файл `gt.txt` в корне архива — разметка bbox:
  filename;x1;y1;x2;y2;class_id
- папки классов (43 шт.) для задачи классификации — их можно игнорировать.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ----------------------------
# Настройки (меняйте при необходимости)
# ----------------------------

# Прямая ссылка на архив GTSDB (FullIJCNN2013.zip)
DATASET_URL = (
    "https://sid.erda.dk/public/archives/"
    "ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"
)

# Куда скачивать и распаковывать исходные данные
RAW_DIR = Path("data/raw")

# Имя zip-файла в RAW_DIR (можно оставить как есть)
ARCHIVE_PATH = RAW_DIR / "FullIJCNN2013.zip"

# Удалять ли архив после успешной распаковки
DELETE_ARCHIVE_AFTER_EXTRACT = True

# Таймаут сети (сек.)
REQUEST_TIMEOUT = 30

# Размер чанка при потоковой загрузке (байт)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


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


def download_with_progress(url: str, dst_path: Path) -> None:
    """
    Скачивает файл по URL в `dst_path` потоково.

    Почему так:
    - архив достаточно большой, поэтому скачиваем стримингом;
    - `tqdm` показывает прогресс по Content-Length (если сервер его отдаёт).
    """
    print(f"[INFO] Скачиваем архив: {url}")

    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()

            total = int(r.headers.get("Content-Length", 0))
            tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")

            # Если остался старый ".part" от предыдущей попытки — удалим, чтобы не путаться.
            if tmp_path.exists():
                tmp_path.unlink()

            with open(tmp_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Download",
            ) as pbar:
                for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))

            # Атомарно переименовываем .part -> .zip
            tmp_path.replace(dst_path)

    except requests.exceptions.RequestException as e:
        _die(f"Ошибка сети при скачивании: {e}")
    except OSError as e:
        _die(f"Ошибка записи файла `{dst_path}`: {e}")

    print(f"[INFO] Архив сохранён: {dst_path}")


def extract_zip(zip_path: Path, dst_dir: Path) -> None:
    """
    Распаковывает zip-архив в `dst_dir`.

    Примечание:
    - для учебного проекта используем стандартный `zipfile`;
    - распаковка идёт поверх текущей папки (если файлы уже есть — будут перезаписаны).
    """
    print(f"[INFO] Распаковываем архив в `{dst_dir}`...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dst_dir)
    except zipfile.BadZipFile:
        _die("Скачанный архив повреждён или это не zip-файл.")
    except OSError as e:
        _die(f"Ошибка при распаковке: {e}")
    print("[INFO] Распаковка завершена.")


def validate_raw_dataset(raw_dir: Path) -> None:
    """
    Проверяет, что после распаковки датасет выглядит корректно:
    - существует `gt.txt`
    - найдено много `*.ppm` (ожидается 900)
    """
    gt_path = raw_dir / "gt.txt"
    if not gt_path.exists():
        _die(f"После распаковки не найден файл разметки `{gt_path}`.")

    ppm_files = sorted(raw_dir.glob("*.ppm"))
    if not ppm_files:
        _die(
            "После распаковки не найдено ни одного изображения `*.ppm` в `data/raw/`.\n"
            "Проверьте, что архив распаковался в правильную папку."
        )

    print(f"[INFO] Найден файл разметки: {gt_path}")
    print(f"[INFO] Найдено изображений `*.ppm`: {len(ppm_files)} (ожидается около 900)")


def main() -> None:
    print("[INFO] === GTSDB: скачивание и распаковка ===")

    _ensure_dir(RAW_DIR)

    # Если архив уже скачан — повторно не качаем (удобно при повторных запусках).
    if not ARCHIVE_PATH.exists():
        download_with_progress(DATASET_URL, ARCHIVE_PATH)
    else:
        print(f"[INFO] Архив уже существует, пропускаем скачивание: {ARCHIVE_PATH}")

    extract_zip(ARCHIVE_PATH, RAW_DIR)
    validate_raw_dataset(RAW_DIR)

    if DELETE_ARCHIVE_AFTER_EXTRACT:
        try:
            ARCHIVE_PATH.unlink(missing_ok=True)  # Python 3.8+: missing_ok
        except TypeError:
            # На случай очень старого Python (маловероятно) — fallback.
            if ARCHIVE_PATH.exists():
                ARCHIVE_PATH.unlink()
        except OSError as e:
            # Не критично: данные уже распакованы.
            print(f"[WARN] Не удалось удалить архив `{ARCHIVE_PATH}`: {e}")
        else:
            print(f"[INFO] Архив удалён: {ARCHIVE_PATH}")

    print("[INFO] Готово. Следующий шаг: запустите `python src/prepare_data.py`.")


if __name__ == "__main__":
    # Запускаем основную функцию.
    main()

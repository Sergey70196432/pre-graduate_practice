"""
download_data.py
================

Назначение
----------
Этот скрипт скачивает датасет дорожных знаков RTSD (Russian Traffic Sign Images Dataset)
для задачи детекции:
- архив с изображениями (frames): `rtsd-d3-frames.tar.lzma`
- архив с аннотациями (gt): `rtsd-d3-gt.tar.lzma`

Далее скрипт распаковывает архивы в `data/raw/` и приводит структуру к виду:
  data/raw/frames/  — все изображения (*.jpg/*.png)
  data/raw/gt/      — аннотации (train/test списки + CSV по подпапкам)

Скрипт intentionally сделан максимально простым и "линейным" — без CLI,
параметры задаются в верхней части файла как константы.

Важно
------
1) Архивы RTSD имеют формат `.tar.lzma`, поэтому используем `lzma` + `tarfile`
   (стандартная библиотека).
2) Точная структура внутри архива может отличаться (иногда создаётся папка
   `rtsd-d3-frames/` или `rtsd-d3-gt/`). Поэтому после распаковки мы
   "нормализуем" структуру в `data/raw/frames` и `data/raw/gt`.

Примечание по практике:
-----------------------
Иногда сервер отдаёт архивы с gzip-сжатием (по сигнатуре `1f 8b 08`), хотя имя файла
имеет расширение `.tar.lzma`. Поэтому распаковка сделана "robust":
сначала пробуем `tarfile.open(..., mode='r:*')` (авто-детект gzip/xz/bz2),
и только потом — fallback через `lzma.open()` для случая "сырого" LZMA.
"""

from __future__ import annotations

import sys
import tarfile
import lzma
import shutil
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
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

# Прямые ссылки на архивы RTSD (детекция, D3)
FRAMES_URL = "https://s31vla.storage.yandex.net/rdisk/0c4a9c1ed425fd395fb5ea8e216d60b4db935bcf67b46bfc186119113aff191f/699b4041/MvNI3RYFtn4ndA7GfO2FzkKyIPEWkC0O4pjKvxJJhiaa91IJByiXWn_-PVXVxLKGea6GlO8aPMuOxt3qBY0OZw==?uid=0&filename=rtsd-d3-frames.tar.lzma&disposition=attachment&hash=uUmSztHomJj/j%2B%2BjiITRWA%2BtUAmS8HYC03qfu%2BzOlVw%3D%3A/detection/rtsd-d3-frames.tar.lzma&limit=0&content_type=application%2Fx-gzip&owner_uid=14067337&fsize=1328352301&hid=821e78d788d89665c94974429a862db5&media_type=compressed&tknv=v3&ts=64b6d312fd240&s=83fdea098364e32e45e314be6a9780a9bac3a832a0eb70b47d4d3424e1a034df&pb=U2FsdGVkX1_dVgauKDiJINkG-mn0c6pqiwlbnrzWt8TZwpcrgbUoP9MSW-VpjMPD4C-FOJukw3n8SNqRYzSMIkn7KUud_F5cPYouJ2QFqi4"
GT_URL = "https://s62vla.storage.yandex.net/rdisk/72f840260d0b7aeeb31809548eb1b14f75ab85e0038075b5cc2f76238f9a37db/699b4070/i6WfKJFXr9n8BY1KoxBqJrHRNfpDQp_TLwdnZHQBTL9H8G-6HxaxTZI0lGRhvmzTyeSe4LxB7iyj0jJWxKqHuQ==?uid=0&filename=rtsd-d3-gt.tar.lzma&disposition=attachment&hash=uUmSztHomJj/j%2B%2BjiITRWA%2BtUAmS8HYC03qfu%2BzOlVw%3D%3A/detection/rtsd-d3-gt.tar.lzma&limit=0&content_type=application%2Fx-gzip&owner_uid=14067337&fsize=253262&hid=49bad8777c7cd6e0d8f962d18dbd3d8d&media_type=compressed&tknv=v3&ts=64b6d33fcfc00&s=33e995d5eb4ec9883e27690a2620ef0e28873eae4eb1fa06db2bbc86ddcc2022&pb=U2FsdGVkX1_gRBJZhfSN1XU1kaXFNi2sBuj84HRero02bJykkq8AeJJuZ7TzUEI1U_epXZ9Zf-Qy1hDYHblwsyV9FwzwJbqXHjC3h_h-aF8"

# Куда скачивать и распаковывать исходные данные
RAW_DIR = Path("data/raw")

# Куда в итоге должны попасть данные
FRAMES_DIR = RAW_DIR / "rtsd-d3-frames"
GT_DIR = RAW_DIR / "rtsd-d3-gt"

# Имена архивов в RAW_DIR
FRAMES_ARCHIVE_PATH = RAW_DIR / "rtsd-d3-frames.tar.lzma"
GT_ARCHIVE_PATH = RAW_DIR / "rtsd-d3-gt.tar.lzma"

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

            # Атомарно переименовываем .part -> итоговый файл
            tmp_path.replace(dst_path)

    except requests.exceptions.RequestException as e:
        _die(f"Ошибка сети при скачивании: {e}")
    except OSError as e:
        _die(f"Ошибка записи файла `{dst_path}`: {e}")

    print(f"[INFO] Архив сохранён: {dst_path}")


def _is_within_directory(directory: Path, target: Path) -> bool:
    """
    Защита от path traversal при распаковке tar.
    """
    try:
        directory_resolved = directory.resolve()
        target_resolved = target.resolve()
        target_resolved.relative_to(directory_resolved)
        return True
    except Exception:
        return False


def _safe_extract_tar(tar: tarfile.TarFile, dst_dir: Path) -> None:
    """
    Безопасная распаковка tar: запрещаем выход за пределы `dst_dir`.
    """
    for member in tar.getmembers():
        member_path = dst_dir / member.name
        if not _is_within_directory(dst_dir, member_path):
            _die(
                "Обнаружена попытка path traversal при распаковке tar: "
                f"member={member.name}"
            )
    tar.extractall(dst_dir)


def extract_tar_archive(archive_path: Path, dst_dir: Path) -> None:
    """
    Распаковывает tar-архив в `dst_dir`, даже если он дополнительно сжат.

    Примечание:
    - сначала пробуем `tarfile.open(archive_path, mode='r:*')`:
      tarfile сам определит gzip/bz2/xz по сигнатуре;
    - если это не сработало (например, "сырой" `.lzma`), пробуем `lzma.open` + `tarfile.open(fileobj=...)`;
    - распаковка идёт поверх текущей папки (если файлы уже есть — могут быть перезаписаны).
    """
    if not archive_path.exists():
        _die(f"Архив не найден: {archive_path}")

    print(f"[INFO] Распаковываем архив: {archive_path.name}")

    # 1) Основной путь: tarfile сам определяет сжатие (gzip/bz2/xz)
    try:
        with tarfile.open(archive_path, mode="r:*") as tar:
            _safe_extract_tar(tar, dst_dir)
        print("[INFO] Распаковка завершена.")
        return
    except tarfile.TarError:
        pass
    except OSError as e:
        _die(f"Ошибка при распаковке: {e}")

    # 2) Fallback: "сырой" LZMA поверх tar
    try:
        with lzma.open(archive_path, "rb") as f:
            with tarfile.open(fileobj=f, mode="r:*") as tar:
                _safe_extract_tar(tar, dst_dir)
    except (lzma.LZMAError, tarfile.TarError):
        _die("Скачанный архив повреждён или это не tar-архив (gzip/xz/bz2/lzma).")
    except OSError as e:
        _die(f"Ошибка при распаковке: {e}")

    print("[INFO] Распаковка завершена.")


def _list_images(dir_path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    try:
        return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts])
    except OSError:
        return []


def _list_images_recursive(dir_path: Path) -> list[Path]:
    """
    Рекурсивно собирает изображения (jpg/png) внутри `dir_path`.

    Зачем это нужно:
    - в некоторых версиях/зеркалах RTSD кадры лежат не в одной папке, а, например,
      в `rtsd-d3-frames/train/` и `rtsd-d3-frames/test/`.
    """
    exts = {".jpg", ".jpeg", ".png"}
    try:
        return sorted([p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    except OSError:
        return []


def _find_dir_with_images(raw_dir: Path) -> Path | None:
    """
    Пытается найти папку, где лежат изображения после распаковки.

    Допущение:
    - чаще всего изображения лежат либо прямо в `data/raw/`, либо в одной папке
      уровня `data/raw/<something>/`.
    """
    if _list_images(raw_dir):
        return raw_dir

    try:
        children = [p for p in raw_dir.iterdir() if p.is_dir()]
    except OSError:
        return None

    candidates: list[tuple[int, Path]] = []
    for d in children:
        # 1) сначала быстрый вариант: изображения прямо в этой папке
        imgs = _list_images(d)
        if imgs:
            candidates.append((len(imgs), d))
            continue

        # 2) fallback: изображения глубже (например train/test)
        imgs2 = _list_images_recursive(d)
        if imgs2:
            candidates.append((len(imgs2), d))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _find_gt_root(raw_dir: Path) -> Path | None:
    """
    Пытается найти папку, где лежат `train_filenames.txt`/`test_filenames.txt`
    и подпапки с `*_gt.csv` после распаковки.
    """
    if (raw_dir / "train_filenames.txt").exists() and (raw_dir / "test_filenames.txt").exists():
        return raw_dir

    try:
        children = [p for p in raw_dir.iterdir() if p.is_dir()]
    except OSError:
        return None

    for d in children:
        if (d / "train_filenames.txt").exists() and (d / "test_filenames.txt").exists():
            return d

    return None


def _move_all_images_to_frames(src_dir: Path, frames_dir: Path) -> None:
    """
    Перемещает все изображения из `src_dir` в `frames_dir`.

    Важное допущение:
    - по описанию RTSD кадры могут лежать в одной папке, но на практике иногда
      встречается структура с подпапками `train/` и `test/`.
      Поэтому здесь собираем изображения рекурсивно.
    """
    _ensure_dir(frames_dir)
    imgs = _list_images_recursive(src_dir)
    if not imgs:
        return

    moved = 0
    collisions = 0
    for p in imgs:
        dst = frames_dir / p.name
        if dst.exists():
            # На всякий случай защищаемся от коллизий имён.
            # Обычно их быть не должно, но если есть — не перезаписываем молча.
            if dst.stat().st_size == p.stat().st_size:
                # Похоже на дубликат — можно просто удалить источник.
                try:
                    p.unlink()
                except OSError:
                    pass
                collisions += 1
                continue

            collisions += 1
            # Переименуем, добавив суффикс.
            stem = Path(p.name).stem
            suf = Path(p.name).suffix
            k = 1
            while True:
                candidate = frames_dir / f"{stem}__dup{k}{suf}"
                if not candidate.exists():
                    dst = candidate
                    break
                k += 1

        try:
            p.replace(dst)
        except OSError as e:
            _die(f"Не удалось переместить `{p}` -> `{dst}`: {e}")
        moved += 1

    print(f"[INFO] Перемещено изображений в `{frames_dir}`: {moved}")
    if collisions > 0:
        print(f"[WARN] Обнаружены коллизии имён изображений: {collisions}")


def _move_gt_to_dir(src_gt_root: Path, gt_dir: Path) -> None:
    """
    Переносит содержимое GT-архива в `data/raw/gt/`.
    """
    _ensure_dir(gt_dir)

    # Если `gt_dir` уже содержит нужные файлы — ничего не делаем.
    if (gt_dir / "train_filenames.txt").exists() and (gt_dir / "test_filenames.txt").exists():
        return

    try:
        for p in src_gt_root.iterdir():
            dst = gt_dir / p.name
            if dst.exists():
                # Удаляем старое, чтобы move был детерминированным
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            p.replace(dst)
    except OSError as e:
        _die(f"Не удалось перенести GT-данные в `{gt_dir}`: {e}")


def normalize_raw_structure(raw_dir: Path) -> None:
    """
    Приводит структуру `data/raw/` к виду:
      data/raw/rtsd-d3-frames/ — изображения
      data/raw/rtsd-d3-gt/     — аннотации
    """
    # 1) Frames
    if not _list_images(FRAMES_DIR):
        src_frames = _find_dir_with_images(raw_dir)
        if src_frames is None:
            _die(
                "После распаковки не удалось найти папку с изображениями RTSD.\n"
                "Проверьте содержимое `data/raw/`."
            )
        _move_all_images_to_frames(src_frames, FRAMES_DIR)

    # 2) GT
    if not (GT_DIR / "train_filenames.txt").exists():
        src_gt_root = _find_gt_root(raw_dir)
        if src_gt_root is None:
            _die(
                "После распаковки не удалось найти папку с аннотациями (train_filenames.txt/test_filenames.txt).\n"
                "Проверьте содержимое `data/raw/`."
            )
        _move_gt_to_dir(src_gt_root, GT_DIR)


def validate_raw_dataset(raw_dir: Path) -> None:
    """
    Проверяет, что после распаковки датасет выглядит корректно:
    - в `data/raw/rtsd-d3-frames/` есть изображения `*.jpg/*.png`
    - в `data/raw/rtsd-d3-gt/` есть `train_filenames.txt`, `test_filenames.txt`
    - в `data/raw/rtsd-d3-gt/*/` есть `train_gt.csv`/`test_gt.csv`
    """
    frames = _list_images(FRAMES_DIR)
    if not frames:
        _die(
            "После распаковки не найдено ни одного изображения `*.jpg/*.png` в `data/raw/rtsd-d3-frames/`.\n"
            "Проверьте, что архив распаковался в правильную папку."
        )

    train_list = GT_DIR / "train_filenames.txt"
    test_list = GT_DIR / "test_filenames.txt"
    if not train_list.exists() or not test_list.exists():
        _die(
            "После распаковки не найдены списки train/test в `data/raw/rtsd-d3-gt/`.\n"
            "Ожидались файлы `train_filenames.txt` и `test_filenames.txt`."
        )

    csv_files = sorted(GT_DIR.glob("*/train_gt.csv")) + sorted(GT_DIR.glob("*/test_gt.csv"))
    if not csv_files:
        _die(
            "После распаковки не найдены CSV-файлы разметки в `data/raw/rtsd-d3-gt/*/`.\n"
            "Ожидались `*/train_gt.csv` и `*/test_gt.csv`."
        )

    print(f"[INFO] Найдено изображений: {len(frames)} (в `data/raw/rtsd-d3-frames/`)")
    print(f"[INFO] Найдено CSV разметки: {len(csv_files)} (в `data/raw/rtsd-d3-gt/*/`)")


def main() -> None:
    print("[INFO] === RTSD: скачивание и распаковка ===")

    _ensure_dir(RAW_DIR)
    _ensure_dir(FRAMES_DIR)
    _ensure_dir(GT_DIR)

    # Если архивы уже скачаны — повторно не качаем (удобно при повторных запусках).
    if not FRAMES_ARCHIVE_PATH.exists():
        download_with_progress(FRAMES_URL, FRAMES_ARCHIVE_PATH)
    else:
        print(f"[INFO] Архив уже существует, пропускаем скачивание: {FRAMES_ARCHIVE_PATH}")

    if not GT_ARCHIVE_PATH.exists():
        download_with_progress(GT_URL, GT_ARCHIVE_PATH)
    else:
        print(f"[INFO] Архив уже существует, пропускаем скачивание: {GT_ARCHIVE_PATH}")

    # Распаковка: сначала frames, затем gt
    extract_tar_archive(FRAMES_ARCHIVE_PATH, RAW_DIR)
    extract_tar_archive(GT_ARCHIVE_PATH, RAW_DIR)

    # Нормализуем структуру
    normalize_raw_structure(RAW_DIR)

    validate_raw_dataset(RAW_DIR)

    if DELETE_ARCHIVE_AFTER_EXTRACT:
        for p in [FRAMES_ARCHIVE_PATH, GT_ARCHIVE_PATH]:
            try:
                p.unlink(missing_ok=True)  # Python 3.8+: missing_ok
            except TypeError:
                if p.exists():
                    p.unlink()
            except OSError as e:
                print(f"[WARN] Не удалось удалить архив `{p}`: {e}")
            else:
                print(f"[INFO] Архив удалён: {p}")

    print("[INFO] Готово. Следующий шаг: запустите `python src/prepare_data.py`.")


if __name__ == "__main__":
    # Запускаем основную функцию.
    main()

"""
Модуль обработки видео в отдельном потоке.

Здесь находится класс VideoProcessor, который:
- открывает видеофайл через OpenCV (cv2.VideoCapture);
- читает кадры в цикле;
- запускает детекцию на каждом кадре через модель YOLOv8;
- отправляет результаты (кадр + список детекций) в очередь queue.Queue.

Важно:
GUI (tkinter) работает в главном потоке и НЕ должен зависать.
Поэтому обработка видео вынесена в отдельный поток (threading).
"""

from __future__ import annotations

import threading
import time
from typing import Any, List, Tuple

import cv2


# Тип одной детекции (для полного лога):
# ((x1, y1, x2, y2), label, conf, time_sec, frame_index)
Detection = Tuple[Tuple[float, float, float, float], str, float, float, int]


class VideoProcessor:
    """
    Класс для чтения и обработки видео в отдельном потоке.
    """

    def __init__(self, model: Any, video_path: str, result_queue: Any, run_id: int) -> None:
        # Сохраняем входные данные.
        self.model = model
        self.video_path = video_path
        self.result_queue = result_queue
        self.run_id = int(run_id)

        # Инициализируем атрибуты.
        self.cap = None
        self.running = False
        self.thread = None
        self.fps = None
        self.frame_count = 0

        # Данные о видео (для трек-бара).
        self.total_frames: int | None = None

        # Управление из GUI: перемотка и пауза/продолжение.
        self._lock = threading.Lock()
        self._seek_time_sec: float | None = None
        self._paused: bool = False

    def start_processing(self) -> None:
        """
        Запускает обработку видео в отдельном потоке.
        """
        self.running = True
        self.thread = threading.Thread(target=self._process, daemon=True)
        self.thread.start()

    def request_seek(self, time_sec: float) -> None:
        """
        Запрос перемотки видео на указанное время (в секундах).

        Важно:
        - Метод вызывается из GUI (главный поток).
        - Реальная перемотка выполняется внутри потока обработки (в _process),
          чтобы не было гонок вокруг cv2.VideoCapture.
        """
        with self._lock:
            self._seek_time_sec = max(0.0, float(time_sec))

    def set_paused(self, paused: bool) -> None:
        """
        Ставит обработку на паузу или продолжает.
        """
        with self._lock:
            self._paused = bool(paused)

    def _process(self) -> None:
        """
        Внутренний метод обработки видео.

        Он выполняется в отдельном потоке и общается с GUI через очередь:
        Формат сообщений (все сообщения содержат run_id, чтобы GUI мог игнорировать старые запуски):
        - (type, payload, run_id)

        Где type:
        - 'frame'     : payload=(frame, detections)
        - 'finished'  : payload=None
        - 'stopped'   : payload=None
        - 'error'     : payload=message
        - 'status'    : payload=message
        - 'meta'      : payload=dict(fps, total_frames, duration_sec)
        - 'progress'  : payload=dict(time_sec, frame_index)
        """
        try:
            # Сообщаем GUI, что поток реально стартовал.
            self._put("status", "Поток обработки запущен. Открываем видео...")
            self.cap = cv2.VideoCapture(self.video_path)

            # Если видео не открылось — отправляем ошибку в очередь и выходим.
            if not self.cap.isOpened():
                self._put("error", f"Не удалось открыть видеофайл: {self.video_path}")
                return

            self._put("status", "Видео открыто. Читаем параметры...")

            # FPS нужен для вычисления времени (в секундах) для лога.
            # Иногда CAP_PROP_FPS возвращает 0 — тогда ставим разумное значение по умолчанию.
            fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0.0:
                fps = 25.0
            self.fps = fps

            # Достаём общее число кадров (может быть 0 для некоторых форматов).
            tf = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.total_frames = tf if tf > 0 else None
            duration = (self.total_frames / self.fps) if (self.total_frames and self.fps) else None

            # Отправляем метаданные (нужно для трек-бара).
            self._put(
                "meta",
                {
                    "fps": self.fps,
                    "total_frames": self.total_frames,
                    "duration_sec": duration,
                },
            )

            self.frame_count = 0
            self._put("status", f"FPS: {self.fps:.2f}. Начинаем обработку кадров...")

            # ------------------------------------------------------------
            # Ограничение скорости воспроизведения (real-time playback)
            # ------------------------------------------------------------
            # На мощной видеокарте детекция может идти быстрее номинального FPS файла.
            # Тогда "видео" в GUI выглядит ускоренным, потому что мы выдаём кадры быстрее реального времени.
            #
            # Решение: синхронизируем выдачу кадров по времени видео:
            #   target_wall_time = base_wall + (video_time - base_video_time)
            #
            # base_* сбрасываем при перемотке и корректируем при паузе.
            base_wall = time.monotonic()
            base_video_time = 0.0
            last_paused = False
            paused_started_at: float | None = None

            # Основной цикл чтения кадров.
            while self.running:
                # 1) Считываем команды управления от GUI (seek / pause).
                seek_time: float | None = None
                paused: bool = False
                with self._lock:
                    if self._seek_time_sec is not None:
                        seek_time = self._seek_time_sec
                        self._seek_time_sec = None
                    paused = self._paused

                if seek_time is not None:
                    # Переводим время в номер кадра.
                    target_frame = int(seek_time * self.fps)
                    if target_frame < 0:
                        target_frame = 0
                    # Если известно общее число кадров — ограничим сверху.
                    if self.total_frames is not None:
                        target_frame = min(target_frame, max(0, self.total_frames - 1))

                    # Перематываем.
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    self.frame_count = target_frame
                    self._put("status", f"Перемотка: {_sec_to_mmss(seek_time)}")

                    # Сбрасываем синхронизацию воспроизведения после перемотки.
                    base_wall = time.monotonic()
                    base_video_time = self.frame_count / self.fps

                    # Если мы на паузе — покажем кадр после перемотки сразу (без детекции),
                    # чтобы пользователь видел результат перемотки.
                    if paused:
                        ok_preview, preview = self.cap.read()
                        if ok_preview:
                            self.frame_count += 1
                            t_preview = self.frame_count / self.fps
                            self._put("frame", (preview, [], self.frame_count, t_preview))
                            self._put("progress", {"time_sec": t_preview, "frame_index": self.frame_count})

                # Обработка переходов пауза/продолжить для корректной синхронизации времени.
                if paused and not last_paused:
                    paused_started_at = time.monotonic()
                if (not paused) and last_paused and paused_started_at is not None:
                    # Сдвигаем базовую "стеночную" точку на длительность паузы,
                    # чтобы после продолжения не было скачка скорости.
                    base_wall += time.monotonic() - paused_started_at
                    paused_started_at = None
                last_paused = paused

                # 2) Если пауза включена — не читаем кадры, но остаёмся “живыми”.
                if paused:
                    time.sleep(0.05)
                    continue

                ok, frame = self.cap.read()
                if not ok:
                    # Кадров больше нет (конец видео) или ошибка чтения.
                    break

                self.frame_count += 1
                current_time = self.frame_count / self.fps

                detections: List[Detection] = []

                # Первую картинку можно отправить в GUI даже без детекций,
                # чтобы пользователь сразу увидел, что видео читается.
                # Это также помогает понять, что "ничего не происходит" из-за долгого инференса,
                # а не из-за проблем с чтением видео.
                if self.frame_count == 1:
                    self._put("frame", (frame, [], self.frame_count, current_time))
                    self._put("status", "Первый кадр получен. Выполняем детекцию...")

                # Запускаем модель на кадре.
                # В ultralytics результат обычно возвращается списком (по одному элементу на изображение).
                try:
                    results = self.model.predict(source=frame, verbose=False)
                except Exception:
                    # На всякий случай пробуем альтернативный вызов (иногда используют self.model(frame)).
                    results = self.model(frame)

                if results and len(results) > 0:
                    r0 = results[0]
                    boxes = getattr(r0, "boxes", None)

                    if boxes is not None and len(boxes) > 0:
                        # Достаём координаты, классы и уверенности.
                        # Обычно это torch.Tensor; мы приводим к обычным Python-значениям.
                        xyxy = boxes.xyxy
                        cls = boxes.cls
                        conf = boxes.conf

                        # Имена классов — в model.names (словарь id -> label).
                        # Если по какой-то причине нет, используем "class_{id}".
                        names = getattr(self.model, "names", None) or {}

                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = xyxy[i].tolist()
                            class_id = int(cls[i].item()) if hasattr(cls[i], "item") else int(cls[i])
                            confidence = float(conf[i].item()) if hasattr(conf[i], "item") else float(conf[i])
                            label = names.get(class_id, f"class_{class_id}")

                             # Если уверенность ниже 0.5, пропускаем.
                            if confidence < 0.5:
                                continue

                            # Для "полного лога" НЕ отбрасываем детекции по порогу уверенности.
                            detections.append(((x1, y1, x2, y2), label, confidence))

                # Отправляем кадр и детекции в GUI через очередь.
                # Важно: кадр отправляем как есть (BGR), рисование будет в GUI.
                self._put("frame", (frame, detections, self.frame_count, current_time))

                # Периодически обновляем статус, чтобы было видно, что работа идёт.
                if self.frame_count % 30 == 0:
                    self._put("status", f"Обработано кадров: {self.frame_count}")

                # Обновление прогресса для трек-бара (не слишком часто).
                if self.frame_count % 5 == 0:
                    self._put("progress", {"time_sec": current_time, "frame_index": self.frame_count})

                # Синхронизация по времени видео (если обрабатываем быстрее, чем FPS).
                target_wall = base_wall + (current_time - base_video_time)
                # Спим маленькими порциями, чтобы быстрее реагировать на Stop/Pause/Seek.
                while self.running:
                    with self._lock:
                        if self._paused:
                            break
                    sleep_s = target_wall - time.monotonic()
                    if sleep_s <= 0:
                        break
                    time.sleep(min(0.05, sleep_s))

            # Если вышли из цикла — значит видео закончилось или мы остановили обработку.
            if self.running:
                self._put("finished", None)
            else:
                self._put("stopped", None)

        except Exception as exc:  # noqa: BLE001
            self._put("error", f"Ошибка при обработке видео: {exc}")
        finally:
            # Освобождаем ресурсы OpenCV.
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass

    def stop_processing(self) -> None:
        """
        Останавливает обработку и ждёт завершения потока.
        """
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _put(self, msg_type: str, payload: Any) -> None:
        """
        Отправка сообщений в GUI с run_id.
        """
        self.result_queue.put((msg_type, payload, self.run_id))


def _sec_to_mmss(time_sec: float) -> str:
    """
    Вспомогательная функция для форматирования времени в MM:SS.
    Используется в статусе перемотки.
    """
    total_seconds = int(round(time_sec))
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"


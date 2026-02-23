"""
Графический интерфейс (GUI) приложения для детекции дорожных знаков в видео.
"""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import time
import os
import shutil
import threading
from datetime import datetime
from typing import Any, List, Tuple

import cv2
from PIL import Image, ImageTk

from .video_processor import VideoProcessor


def _format_time_mmss(time_sec: float) -> str:
    """
    Преобразует время в секундах в строку вида MM:SS.
    """
    total_seconds = int(round(time_sec))
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"


def _format_time_mmss_ms(time_sec: float) -> str:
    """
    Преобразует время в секундах в строку вида MM:SS.mmm (миллисекунды).
    """
    if time_sec < 0:
        time_sec = 0.0
    mm = int(time_sec // 60)
    ss = int(time_sec % 60)
    ms = int(round((time_sec - int(time_sec)) * 1000.0))
    # Защита от редкого случая, когда из-за округления получается 1000 мс.
    if ms >= 1000:
        ms = 0
        ss += 1
        if ss >= 60:
            ss = 0
            mm += 1
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


class App:
    """
    Основной класс GUI-приложения.
    """

    def __init__(self, root: tk.Tk, model: Any) -> None:
        self.root = root
        self.model = model

        # Очередь для результатов от VideoProcessor (кадры/ошибки/событие завершения).
        self.result_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

        # Данные текущей сессии.
        self.video_processor: VideoProcessor | None = None
        self.video_path: str | None = None
        self._processing_started_at: float | None = None
        # Полный лог пишем в файл, а в UI показываем аккуратно (порциями).
        self._ui_max_log_lines_per_frame = 200
        # Чтобы не дублировать одинаковые status-сообщения в статус-строке.
        self._last_status_message: str | None = None
        # Метаданные видео для трек-бара.
        self._video_duration_sec: float | None = None
        self._video_fps: float | None = None
        self._updating_seek_scale = False
        self._seek_user_dragging = False
        self._paused = False
        # Идентификатор текущего запуска обработки.
        # Нужен, чтобы игнорировать сообщения из очереди от старых потоков.
        self._run_id = 0

        # Асинхронная запись полного лога в файл.
        self._log_queue: "queue.Queue[str | None]" = queue.Queue()
        self._log_thread: threading.Thread | None = None
        self._log_file_path: str | None = None
        self._log_writer_running: bool = False

        # Настройки окна.
        self.root.title("Детекция дорожных знаков (YOLOv8)")
        self.root.geometry("960x720")
        # Минимальный размер окна, чтобы элементы не "схлопывались" и были видны.
        self.root.minsize(900, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Минималистичная тёмная палитра.
        self._bg = "#1f1f1f"
        self._panel_bg = "#252525"
        self._fg = "#eaeaea"

        self.root.configure(bg=self._bg)

        # Для ttk-виджетов (спиннер) используем простой кроссплатформенный стиль.
        try:
            style = ttk.Style()
            style.theme_use("clam")
        except Exception:
            pass

        # Используем grid: 0 видео, 1 кнопки, 2 трек-бар, 3 лог, 4 статус.
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=20)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=3)
        self.root.grid_rowconfigure(4, weight=0)

        # ----------------------------
        # Элементы интерфейса (вертикальная компоновка)
        # ----------------------------

        # 1) Область видео.
        video_frame = tk.Frame(self.root, bg="black", bd=1, relief=tk.SUNKEN)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(video_frame, bg="#000000", highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        # Текст-заглушка в области видео.
        self._video_text_id = self.video_canvas.create_text(
            10,
            10,
            anchor="nw",
            fill="#ffffff",
            font=("Helvetica", 16, "bold"),
            text="Видео не выбрано.\nНажмите «Выбрать видео».",
        )

        # Идентификатор изображения на Canvas (будем обновлять его при каждом кадре).
        self._canvas_image_id: int | None = None

        # 2) Панель кнопок.
        btn_frame = tk.Frame(self.root, bg=self._bg)
        btn_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        btn_frame.grid_columnconfigure(4, weight=1)

        # Кнопка выбора видео.
        self.btn_select = tk.Button(btn_frame, text="Выбрать видео", command=self.select_video)
        self.btn_select.grid(row=0, column=0, padx=(0, 8), pady=5, sticky="w")

        # Кнопка запуска обработки.
        self.btn_start = tk.Button(btn_frame, text="Запуск", command=self.start_processing, state=tk.DISABLED)
        self.btn_start.grid(row=0, column=1, padx=(0, 8), pady=5, sticky="w")

        # Кнопка-переключатель: "Стоп" (пауза) / "Продолжить".
        self.btn_stop = tk.Button(btn_frame, text="Стоп", command=self.toggle_pause, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=2, padx=(0, 8), pady=5, sticky="w")

        # Кнопка сохранения лога.
        self.btn_save = tk.Button(btn_frame, text="Сохранить лог", command=self.save_log)
        self.btn_save.grid(row=0, column=3, padx=(0, 8), pady=5, sticky="w")

        # Небольшая подсказка справа (показывает путь к видео).
        self.video_hint = tk.Label(btn_frame, text="Видео: не выбрано", bg=self._bg, fg=self._fg, anchor="w")
        self.video_hint.grid(row=0, column=4, padx=(6, 0), pady=5, sticky="ew")

        # 2.1) Трек-бар (ползунок) для перемотки по времени.
        seek_frame = tk.Frame(self.root, bg=self._bg)
        seek_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
        seek_frame.grid_columnconfigure(1, weight=1)

        self.seek_left_label = tk.Label(seek_frame, text="00:00", bg=self._bg, fg=self._fg)
        self.seek_left_label.grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = tk.Scale(
            seek_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            showvalue=False,
            variable=self.seek_var,
            command=self.on_seek_changed,
            bg=self._bg,
            fg=self._fg,
            highlightthickness=0,
            troughcolor="#3a3a3a",
            activebackground="#5a5a5a",
        )
        self.seek_scale.grid(row=0, column=1, sticky="ew")

        self.seek_right_label = tk.Label(seek_frame, text="00:00", bg=self._bg, fg=self._fg)
        self.seek_right_label.grid(row=0, column=2, sticky="e", padx=(8, 0))

        self.seek_scale.bind("<ButtonPress-1>", self._on_seek_press)
        self.seek_scale.bind("<ButtonRelease-1>", self._on_seek_release)

        # 3) Текстовое поле лога с прокруткой.
        log_frame = tk.Frame(self.root, bg=self._panel_bg, bd=1, relief=tk.SUNKEN)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 6))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=8,
            bg="#111111",
            fg=self._fg,
            insertbackground=self._fg,
            relief=tk.FLAT,
            borderwidth=0,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.log_text.insert(tk.END, "Лог распознавания будет выводиться здесь.\n")
        self.log_text.see(tk.END)

        # 4) Строка статуса.
        status_frame = tk.Frame(self.root, bg=self._bg)
        status_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        status_frame.grid_columnconfigure(1, weight=1)

        # Спиннер (лоадер). В Tkinter это удобно делать через ttk.Progressbar в режиме indeterminate.
        self.spinner = ttk.Progressbar(status_frame, mode="indeterminate", length=140)
        self.spinner.grid(row=0, column=0, padx=(0, 10), pady=2, sticky="w")

        self.status_label = tk.Label(status_frame, text="Готово. Выберите видео.", anchor="w", bg=self._bg, fg=self._fg)
        self.status_label.grid(row=0, column=1, sticky="ew")

        # Чтобы изображение не исчезало (Tkinter требует хранить ссылку на PhotoImage).
        self._tk_image: ImageTk.PhotoImage | None = None

        # Запускаем периодическое обновление (проверка очереди и обновление GUI).
        self.update_frame()

    def _set_processing_ui(self, processing: bool) -> None:
        """
        Включает/выключает элементы интерфейса, связанные с обработкой.

        processing=True:
          - блокируем кнопку "Запуск"
          - запускаем спиннер
        processing=False:
          - останавливаем спиннер
          - возвращаем кнопку "Запуск" (если видео выбрано)
        """
        if processing:
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_stop.config(text="Стоп")
            self._paused = False
            # Чем меньше число — тем быстрее "крутится" спиннер.
            self.spinner.start(10)
        else:
            self.spinner.stop()
            self.btn_start.config(state=tk.NORMAL if self.video_path else tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_stop.config(text="Стоп")
            self._paused = False

    def select_video(self) -> None:
        """
        Открывает диалог выбора видеофайла.
        Поддерживаемые форматы: mp4, avi, mov.
        """
        file_path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[
                ("Видео файлы", "*.mp4 *.avi *.mov"),
                ("MP4", "*.mp4"),
                ("AVI", "*.avi"),
                ("MOV", "*.mov"),
                ("Все файлы", "*.*"),
            ],
        )

        if file_path:
            # Если во время выбора нового видео уже идёт обработка,
            # корректно остановим текущий поток (иначе он может продолжать слать сообщения в очередь).
            if self.video_processor is not None:
                self.video_processor.stop_processing()
                self.video_processor = None
                self._set_processing_ui(False)
                self._stop_log_writer()

            # При смене видео сбрасываем паузу и очищаем очередь сообщений.
            # Также увеличим run_id, чтобы любые "долетевшие" сообщения от старого запуска игнорировались.
            self._run_id += 1
            self._paused = False
            self._clear_result_queue()
            # Если логгер был запущен — остановим (при выборе нового видео должен начаться новый лог-файл).
            self._stop_log_writer()

            self.video_path = file_path
            self.btn_start.config(state=tk.NORMAL)
            self.video_hint.config(text=f"Видео: {self.video_path}")
            self.status_label.config(text="Видео выбрано. Загружаем превью...")
            self.log_text.insert(tk.END, f"Выбрано видео: {self.video_path}\n")
            self.log_text.see(tk.END)

            # Сбрасываем трек-бар и состояние метаданных.
            self._reset_seek_ui()

            # Загружаем и показываем превью (первый кадр) сразу после выбора видео.
            self._show_preview(self.video_path)
            self.status_label.config(text="Видео выбрано. Нажмите «Запуск».")

    def start_processing(self) -> None:
        """
        Запускает обработку видео: создаёт VideoProcessor и запускает поток.
        """
        if not self.video_path:
            messagebox.showwarning("Внимание", "Сначала выберите видеофайл.")
            return

        # Если раньше уже была обработка, на всякий случай останавливаем её.
        if self.video_processor is not None:
            self.video_processor.stop_processing()
            self.video_processor = None
            self._stop_log_writer()

        # Новый запуск — увеличиваем run_id и очищаем очередь (чтобы старые события не ломали UI).
        self._run_id += 1
        self._paused = False
        self._clear_result_queue()

        # Очищаем лог в интерфейсе.
        self._last_status_message = None
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "Запуск обработки...\n")
        self.log_text.see(tk.END)

        # Запускаем асинхронную запись полного лога в файл.
        self._start_log_writer()

        # Создаём обработчик видео и запускаем.
        self.video_processor = VideoProcessor(self.model, self.video_path, self.result_queue, run_id=self._run_id)
        self.video_processor.start_processing()

        # Включаем "режим обработки" (спиннер + блокировка кнопки).
        self._set_processing_ui(True)
        self._processing_started_at = time.time()
        self.status_label.config(text="Обработка... (ожидаем первый кадр)")
        # Принудительно даём Tkinter отрисовать изменения сразу (чтобы спиннер/статус появились моментально).
        self.root.update_idletasks()

    def toggle_pause(self) -> None:
        """
        Переключатель "Стоп/Продолжить":
        - "Стоп" = пауза (остановка обработки без закрытия видео)
        - "Продолжить" = возобновление
        """
        if self.video_processor is None:
            return

        self._paused = not self._paused
        self.video_processor.set_paused(self._paused)

        if self._paused:
            self.btn_stop.config(text="Продолжить")
            self.spinner.stop()
            self.status_label.config(text="Пауза.")
        else:
            self.btn_stop.config(text="Стоп")
            self.spinner.start(10)
            self.status_label.config(text="Обработка...")

    def update_frame(self) -> None:
        """
        Периодически вызывается через after() и обновляет GUI.

        Логика:
        - забираем элементы из очереди без блокировки;
        - если пришёл кадр — рисуем bbox и обновляем окно;
        - если обработка завершена — обновляем статус и разблокируем кнопку "Запуск";
        - если ошибка — показываем messagebox и возвращаем интерфейс в нормальное состояние.
        """
        # Чтобы интерфейс был отзывчивым, НЕ пытаемся перерисовать 100 кадров за один тик.
        # Идея простая: берём из очереди всё, но отображаем только ПОСЛЕДНИЙ доступный кадр.
        # Это нормально для видео: пользователю важнее "актуальная картинка", чем каждый кадр подряд.
        last_frame_payload: Any = None
        finished_received = False
        stopped_received = False
        error_message: str | None = None
        last_status_message: str | None = None
        meta_payload: Any = None
        progress_payload: Any = None

        try:
            while True:
                item = self.result_queue.get_nowait()
                # Ожидаемый формат: (type, payload, run_id).
                # Если формат не совпал — игнорируем сообщение.
                if not (isinstance(item, tuple) and len(item) == 3):
                    continue

                msg_type, payload, run_id = item
                if int(run_id) != int(self._run_id):
                    # Это сообщение от старого запуска — игнорируем.
                    continue

                if msg_type == "frame":
                    last_frame_payload = payload
                elif msg_type == "finished":
                    finished_received = True
                elif msg_type == "stopped":
                    stopped_received = True
                elif msg_type == "error":
                    error_message = str(payload)
                elif msg_type == "status":
                    last_status_message = str(payload)
                elif msg_type == "meta":
                    meta_payload = payload
                elif msg_type == "progress":
                    progress_payload = payload

        except queue.Empty:
            pass

        # Обрабатываем события в правильном порядке: ошибка > кадр > завершение.
        if error_message is not None:
            self.status_label.config(text="Ошибка.")
            self._set_processing_ui(False)
            messagebox.showerror("Ошибка", error_message)
            self._stop_log_writer()

        # Служебный статус из потока (чтобы было понятно, что происходит).
        if last_status_message is not None and error_message is None:
            # Не выключаем спиннер: это именно промежуточные сообщения.
            self.status_label.config(text=last_status_message)
            self._last_status_message = last_status_message

        # Метаданные видео (для диапазона трек-бара).
        if meta_payload is not None and error_message is None:
            try:
                fps = meta_payload.get("fps")
                if fps is not None and float(fps) > 0:
                    self._video_fps = float(fps)

                duration = meta_payload.get("duration_sec")
                if duration is not None and duration > 0:
                    self._video_duration_sec = float(duration)
                    self._set_seek_range(self._video_duration_sec)
                    self.seek_right_label.config(text=_format_time_mmss(self._video_duration_sec))
            except Exception:
                pass

        # Время ожидания для обновления GUI в зависимости от FPS видео.
        wait_time = int(1000 / (self._video_fps or 25))

        # Прогресс (обновляем трек-бар).
        if progress_payload is not None and error_message is None:
            try:
                # Чтобы не было расхождений, считаем время от fps и номера кадра, если возможно.
                if self._video_fps and progress_payload.get("frame_index") is not None:
                    frame_i = int(progress_payload.get("frame_index"))
                    tsec = frame_i / float(self._video_fps)
                else:
                    tsec = float(progress_payload.get("time_sec", 0.0))
                self._update_seek_position(tsec)
            except Exception:
                pass

        if last_frame_payload is not None:
            # Ожидаемый формат payload для кадра: (frame, detections, frame_index, time_sec).
            if not (isinstance(last_frame_payload, tuple) and len(last_frame_payload) == 4):
                # Если формат не совпал — игнорируем.
                self.root.after(wait_time, self.update_frame)
                return

            frame, detections, frame_index, time_sec = last_frame_payload
            self._handle_frame(frame, detections, frame_index=int(frame_index), time_sec=float(time_sec))
            # Как только пришёл первый кадр, считаем, что процесс "живой".
            self._processing_started_at = None

        if finished_received:
            self.status_label.config(text="Готово.")
            self._set_processing_ui(False)
            self._processing_started_at = None
            self.video_processor = None
            self._stop_log_writer()

        if stopped_received:
            self.status_label.config(text="Остановлено.")
            self._set_processing_ui(False)
            self._processing_started_at = None
            self.video_processor = None
            self._stop_log_writer()

        # Если после старта долго нет кадров/ошибок — обновим статус подсказкой.
        if (
            self._processing_started_at is not None
            and error_message is None
            and last_frame_payload is None
            and not finished_received
        ):
            elapsed = time.time() - self._processing_started_at
            if elapsed > 2.0:
                self.status_label.config(text="Обработка... первый кадр может обрабатываться несколько секунд")

        # Планируем следующий вызов с учетом FPS видео.
        self.root.after(wait_time, self.update_frame)

    def _clear_result_queue(self) -> None:
        """
        Очищает очередь результатов (если там остались сообщения от прошлого запуска).
        """
        try:
            while True:
                self.result_queue.get_nowait()
        except queue.Empty:
            pass

    def _reset_seek_ui(self) -> None:
        """
        Сбрасывает трек-бар в 0 и очищает информацию о длительности.
        Вызывается при выборе нового видео.
        """
        self._video_duration_sec = None
        self._video_fps = None
        self._updating_seek_scale = True
        try:
            self.seek_scale.config(to=1.0)
            self.seek_var.set(0.0)
            self.seek_left_label.config(text="00:00")
            self.seek_right_label.config(text="00:00")
        finally:
            self._updating_seek_scale = False

    def _show_preview(self, video_path: str) -> None:
        """
        Показывает превью видео: читает первый кадр и выводит его на Canvas.
        Также пытается определить длительность (если доступны fps и frame_count).
        """
        # Обновим заглушку (чтобы пользователь видел, что идёт загрузка).
        try:
            self.video_canvas.itemconfig(self._video_text_id, state="normal", text="Загрузка превью...")
        except Exception:
            pass

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                self.status_label.config(text="Не удалось открыть видео для превью.")
                self.video_canvas.itemconfig(self._video_text_id, state="normal", text="Не удалось открыть видео.")
                return

            # Пытаемся получить длительность.
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
                self._video_duration_sec = float(duration)
                self._set_seek_range(self._video_duration_sec)
                self.seek_right_label.config(text=_format_time_mmss(self._video_duration_sec))

            ok, frame = cap.read()
            if not ok or frame is None:
                self.video_canvas.itemconfig(self._video_text_id, state="normal", text="Не удалось получить превью.")
                return

            # Показываем кадр без детекций/лога.
            self._display_frame_on_canvas(frame)

        finally:
            try:
                cap.release()
            except Exception:
                pass

    def _display_frame_on_canvas(self, frame_bgr: Any) -> None:
        """
        Отображает BGR-кадр на Canvas (без логирования и без bbox).
        """
        img = frame_bgr

        # Подгоняем по размеру Canvas (быстро через OpenCV).
        h, w = img.shape[:2]
        target_w = int(self.video_canvas.winfo_width())
        target_h = int(self.video_canvas.winfo_height())

        if target_w < 50 or target_h < 50:
            target_w = 900
            target_h = 500

        scale = min(target_w / float(w), target_h / float(h))
        if scale > 1.0:
            scale = 1.0
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self._tk_image = ImageTk.PhotoImage(pil_img)

        if self._canvas_image_id is None:
            self._canvas_image_id = self.video_canvas.create_image(0, 0, anchor="nw", image=self._tk_image)
        else:
            self.video_canvas.itemconfig(self._canvas_image_id, image=self._tk_image)

        # Скрываем текст-заглушку.
        try:
            self.video_canvas.itemconfig(self._video_text_id, state="hidden")
        except Exception:
            pass

    def _set_seek_range(self, duration_sec: float) -> None:
        if duration_sec and duration_sec > 0:
            self.seek_scale.config(to=duration_sec)

    def _update_seek_position(self, time_sec: float) -> None:
        if self._seek_user_dragging:
            return
        self._updating_seek_scale = True
        try:
            self.seek_var.set(max(0.0, time_sec))
            self.seek_left_label.config(text=_format_time_mmss(time_sec))
        finally:
            self._updating_seek_scale = False

    def _on_seek_press(self, _event: Any) -> None:
        self._seek_user_dragging = True

    def _on_seek_release(self, _event: Any) -> None:
        self._seek_user_dragging = False
        self._perform_seek()

    def on_seek_changed(self, _value: str) -> None:
        if self._updating_seek_scale:
            return
        v = float(self.seek_var.get())
        self.seek_left_label.config(text=_format_time_mmss(v))

    def _perform_seek(self) -> None:
        if self.video_processor is None:
            return
        self.video_processor.request_seek(float(self.seek_var.get()))

    def _handle_frame(
        self,
        frame: Any,
        detections: List[Any],
        frame_index: int | None = None,
        time_sec: float | None = None,
    ) -> None:
        """
        Обрабатывает одно сообщение с кадром:
        - рисует bbox и подписи;
        - обновляет картинку в Label;
        - добавляет строки в текстовый лог.
        """
        # Рисуем на копии кадра, чтобы не менять исходные данные (на практике это безопаснее).
        img = frame.copy()

        # Для UI будем добавлять ограниченное число строк (чтобы интерфейс не зависал),
        # а полный лог пишем в файл асинхронно.
        ui_log_lines: List[str] = []

        for det in detections:
            # det: ((x1,y1,x2,y2), label, conf)
            (x1, y1, x2, y2), label, conf = det

            # Координаты переводим в int для OpenCV.
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))

            # Прямоугольник.
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

            # Текст (класс + уверенность).
            conf_percent = float(conf) * 100.0
            text = f"{label} {conf_percent}%"

            # Подпись рисуем чуть выше верхней границы bbox.
            text_pos = (p1[0], max(0, p1[1] - 10))
            cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Полный лог: кадр + точное время + bbox + класс + уверенность.
            # Время считаем через fps и номер кадра, чтобы исключить расхождения.
            if self._video_fps and float(self._video_fps) > 0:
                time_exact_sec = int(frame_index) / float(self._video_fps)
            else:
                time_exact_sec = float(time_sec)

            time_str = _format_time_mmss_ms(time_exact_sec)
            line = (
                f"{int(frame_index)};"
                f"{time_exact_sec:.3f};"
                f"{time_str};"
                f"{p1[0]};{p1[1]};{p2[0]};{p2[1]};"
                f"{label};"
                f"{float(conf):.6f};"
                f"{conf_percent:.1f}"
            )
            # Полный лог пишем асинхронно в файл (в память не копим).
            if self._log_writer_running:
                self._log_queue.put(line)

            # В UI выводим человекочитаемую строку (и bbox тоже, чтобы было наглядно).
            if len(ui_log_lines) < self._ui_max_log_lines_per_frame:
                ui_log_lines.append(
                    f"[{time_str}] frame={frame_index} {label} ({conf_percent:.1f}%) bbox={p1 + p2}"
                )

        if ui_log_lines:
            self.log_text.insert(tk.END, "\n".join(ui_log_lines) + "\n")
            self.log_text.see(tk.END)

        # Важно для производительности:
        # PIL.Image.resize на каждом кадре может быть очень медленным и "подвешивать" интерфейс.
        # Поэтому уменьшаем изображение быстрее через OpenCV (cv2.resize) ДО конвертации в PIL.
        h, w = img.shape[:2]
        target_w = int(self.video_canvas.winfo_width())
        target_h = int(self.video_canvas.winfo_height())

        # Пока окно только создалось, winfo_width/height могут быть маленькими (1-2 px).
        # Тогда используем запасной ограничитель по ширине.
        if target_w < 50 or target_h < 50:
            target_w = 900
            target_h = 500

        scale = min(target_w / float(w), target_h / float(h))
        # Увеличивать картинку нет смысла (будет мыльно и дороже), ограничим scale сверху.
        if scale > 1.0:
            scale = 1.0

        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Конвертация BGR (OpenCV) -> RGB (PIL/Tkinter).
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        self._tk_image = ImageTk.PhotoImage(pil_img)

        # Рисуем изображение на Canvas.
        # При первом кадре создаём объект, дальше — только обновляем его.
        if self._canvas_image_id is None:
            self._canvas_image_id = self.video_canvas.create_image(0, 0, anchor="nw", image=self._tk_image)
        else:
            self.video_canvas.itemconfig(self._canvas_image_id, image=self._tk_image)

        # Убираем текст-заглушку, когда появилось изображение.
        try:
            self.video_canvas.itemconfig(self._video_text_id, state="hidden")
        except Exception:
            pass

    def save_log(self) -> None:
        """
        Сохраняет лог распознаваний в текстовый файл .txt.
        """
        if not self._log_file_path or not os.path.exists(self._log_file_path):
            messagebox.showinfo("Информация", "Лог-файл ещё не создан. Сначала запустите обработку видео.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Сохранить лог",
            defaultextension=".txt",
            initialfile=os.path.basename(self._log_file_path),
            filetypes=[("Текстовый файл", "*.txt"), ("Все файлы", "*.*")],
        )

        if not save_path:
            return

        # Копирование большого файла может быть долгим — делаем в отдельном потоке.
        src_path = self._log_file_path
        self.status_label.config(text="Сохраняем лог...")

        def _copy_worker() -> None:
            try:
                shutil.copyfile(src_path, save_path)
                self.root.after(0, lambda: self.status_label.config(text=f"Лог сохранён: {save_path}"))
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Не удалось сохранить лог: {exc}"))

        threading.Thread(target=_copy_worker, daemon=True).start()

    def on_close(self) -> None:
        """
        Корректное закрытие приложения:
        - останавливаем поток обработки (если он запущен);
        - закрываем окно.
        """
        try:
            if self.video_processor is not None:
                self.video_processor.stop_processing()
            self._stop_log_writer()
        finally:
            self.root.destroy()

    def _start_log_writer(self) -> None:
        """
        Запускает асинхронную запись полного лога в файл:
        logs/<video_filename>_<YYYYMMDD_HHMMSS>.txt
        """
        if not self.video_path:
            return

        # На всякий случай остановим предыдущий writer.
        self._stop_log_writer()

        os.makedirs("logs", exist_ok=True)
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        # Упрощаем имя файла: только буквы/цифры/_/-
        safe_base = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file_path = os.path.join("logs", f"{safe_base}_{ts}.txt")

        # Новый поток — новая очередь.
        self._log_queue = queue.Queue()
        self._log_writer_running = True

        header = "frame_index;time_sec;time_mmss_ms;x1;y1;x2;y2;class;conf;conf_percent"

        def _writer(path: str) -> None:
            try:
                # buffering=1 => построчная буферизация (файл актуален "на текущий момент")
                with open(path, "w", encoding="utf-8", buffering=1) as f:
                    f.write(header + "\n")
                    while True:
                        item = self._log_queue.get()
                        if item is None:
                            break
                        f.write(item + "\n")
            finally:
                self._log_writer_running = False

        self._log_thread = threading.Thread(target=_writer, args=(self._log_file_path,), daemon=True)
        self._log_thread.start()

    def _stop_log_writer(self) -> None:
        """
        Останавливает поток записи лога (если он запущен).
        Файл остаётся на диске и может быть "Сохранён" (скопирован) в любое место.
        """
        if self._log_thread is None:
            return
        try:
            # Сигнал завершения.
            self._log_queue.put(None)
            self._log_thread.join(timeout=2.0)
        finally:
            self._log_thread = None
            self._log_writer_running = False


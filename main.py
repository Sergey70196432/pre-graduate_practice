"""
Точка входа в приложение (GUI).

Запуск:
    python main.py

Программа:
1) Загружает обученную модель YOLOv8 из файла (см. src/model_loader.py)
2) Открывает окно tkinter и запускает GUI (см. src/gui.py)
"""

from __future__ import annotations

import tkinter as tk

from src.gui import App
from src.model_loader import load_model


def main() -> None:
    # Загружаем модель (при ошибке load_model сам выведет сообщение и завершит программу).
    model = load_model("models/model.pt")

    # Создаём окно и запускаем приложение.
    root = tk.Tk()
    App(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()


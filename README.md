## Детекция дорожных знаков (YOLOv8) — инструкция

Проект включает:
- скрипты для **скачивания датасета RTSD**, **подготовки данных под YOLO** и **обучения модели**
- **GUI-приложение** (Tkinter) для детекции знаков на видео с сохранением лога

## Установка зависимостей

В корне проекта:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

На Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Если на Windows есть NVIDIA GPU и хотите CUDA-сборку PyTorch, можно поставить `torch/torchvision` из `win-torch.requirements.txt`, а остальные зависимости — из `requirements.txt`.

## Запуск скриптов (download_data → prepare_data → train)

Все команды запускаются **из корня проекта**.

### 1) Скачать и распаковать RTSD (`download_data`)

```bash
python src/download_data.py
```

Что произойдёт:
- скачает архивы RTSD и распакует их в `data/raw/`
- приведёт структуру к виду:
  - `data/raw/rtsd-d3-frames/` — изображения
  - `data/raw/rtsd-d3-gt/` — аннотации

### 2) Подготовить датасет в формате YOLO (`prepare_data`)

```bash
python src/prepare_data.py
```

Что произойдёт:
- скопирует изображения в `data/yolo/images/`
- создаст разметку YOLO в `data/yolo/labels/`
- сформирует `data/yolo/train.txt` и `data/yolo/val.txt`
- создаст `data/yolo/dataset.yaml`

### 3) Обучить модель (`train`)

```bash
python src/train.py
```

Что произойдёт:
- запустит обучение Ultralytics YOLOv8 на `data/yolo/dataset.yaml`
- сохранит артефакты эксперимента в `models/` (и/или `runs/`, если Ultralytics переименует папку)
- скопирует лучшую модель `best.pt` в **`models/model.pt`** (это файл, который использует GUI)
- запишет сводку в `models/model_info.txt`

## Запуск приложения (`main.py`)

GUI использует модель из **`models/model.pt`**. Этот файл появляется после `python src/train.py`.

Запуск:

```bash
python main.py
```

## Как пользоваться приложением (GUI)

- **Выбрать видео**: нажмите «Выбрать видео» и укажите файл (`.mp4`, `.avi`, `.mov`)
- **Запуск детекции**: нажмите «Запуск»
- **Пауза/продолжить**: кнопка «Стоп» переключается в «Продолжить»
- **Перемотка**: используйте ползунок времени (трек-бар)
- **Логи**:
  - во время обработки полный лог пишется в `logs/` (файл создаётся автоматически)
  - кнопка «Сохранить лог» позволяет сохранить (скопировать) лог в выбранное место

Формат строк полного лога:
`frame_index;time_sec;time_mmss_ms;x1;y1;x2;y2;class;conf;conf_percent`

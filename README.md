# Детектор Людей

Реальное обнаружение людей на видео с использованием конвейера на основе YOLO.

## Структура проекта

```
human_det_test/
├── model/
│   ├── main.py           # Точка входа для обработки видео
│   ├── model.py          # Реализация конвейера YOLOPipeline
│   ├── tools.py          # Вспомогательные функции для отображения результатов
│   └── new_human.jpg     # Пример изображения
├── requirements.txt      # Зависимости Python
├── README.md             # Документация проекта
└── .gitignore            # Файлы и директории, игнорируемые Git
```

## Требования

- Python 3.6 или выше
- pip (установщик пакетов Python)

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone <repository-url>
   cd human_det_test
   ```

2. Установите зависимости в новую вертуальную среду:
   ```bash
   python3 -m venv .venv
   
   . .venv/bin/activate # for linux
   ./.venv/Scripts/activate # fow windows

   pip install -r requirements.txt
   ```

## Использование

Запустите основной скрипт для обработки видео и обнаружения людей:

```bash
python model/main.py /путь/до/файла_видео.mp4
```

- **Вход:** Путь к видеофайлу, который необходимо обработать.
- **Выход:** В корневой директории проекта будет создан файл `new_video.mp4` с рамками, наложенными на обнаруженных людей.

Пример:
```bash
python model/main.py sample_videos/test.mp4
```

## Зависимости

- jupyterlab
- seaborn
- requests
- pandas
- torch
- torchvision
- opencv-python


## Лицензия

Проект распространяется под лицензией MIT.

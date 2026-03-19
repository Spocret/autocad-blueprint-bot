# Blueprint Bot

Telegram-бот для обработки архитектурных чертежей.

## Возможности

- Распознавание планов этажей из фото
- Определение стен, дверей, окон, лестниц, размеров
- Генерация SVG и DXF файлов
- Соответствие ГОСТ
- Интерактивное уточнение нераспознанных элементов

## Установка

### 1. Клонирование
```bash
cd blueprint_bot
```

### 2. Создание виртуального окружения
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Настройка .env
Скопируй `.env.example` в `.env` и заполни:
```
BOT_TOKEN=токен_от_BotFather
GEMINI_API_KEY=ключ_от_aistudio.google.com
DATABASE_PATH=blueprint_bot.db
LOG_LEVEL=INFO
```

Получите API ключ на aistudio.google.com

### 5. Запуск
```bash
python main.py
```

## Использование

1. Найди бота в Telegram
2. Отправь команду `/start`
3. Отправь фото чертежа
4. Следуй инструкциям бота

## Команды

- `/start` — начало работы
- `/new` — новый чертёж
- `/cancel` — отмена
- `/help` — помощь

## Структура проекта

```
blueprint_bot/
├── main.py              # точка входа
├── config.py            # конфигурация
├── handlers/            # обработчики команд
│   ├── start.py         # /start, /help, /cancel, /new
│   ├── blueprint.py     # обработка чертежей (основной флоу)
│   └── correction.py    # исправление элементов
├── services/            # сервисы
│   ├── image_processor.py  # OpenCV обработка
│   ├── ai_recognizer.py    # AI распознавание (Google Gemini API)
│   ├── svg_generator.py    # генерация SVG
│   └── dxf_generator.py    # генерация DXF
├── models/
│   └── database.py      # SQLite база данных
└── utils/
    └── gost_styles.py   # стили ГОСТ
```

## Диагностика

Перед первым запуском (или при проблемах с API) выполни диагностику окружения:

```bash
python diagnose.py
```

Скрипт проверяет:

- наличие `.env` и обязательных переменных (`BOT_TOKEN`, `GEMINI_API_KEY`)
- подключение к Telegram API (`getMe`)
- доступность каждой модели Gemini (`gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro`)

Пример вывода:

```
✅ BOT_TOKEN: OK (bot: @myblueprint_bot)
✅ GEMINI_API_KEY: задан
✅ gemini-2.5-flash: доступна (ответ за 1.3с)
✅ gemini-2.5-flash-lite: доступна (ответ за 0.9с)
❌ gemini-2.5-pro: 403 permission denied — модель недоступна для вашего ключа

✅ Рекомендуемая модель: gemini-2.5-flash
   Укажите в .env: GEMINI_MODEL=gemini-2.5-flash
```

Зависимости скрипта уже входят в `requirements.txt` (`python-dotenv`, `google-generativeai`).
Дополнительно нужен `requests`:

```bash
pip install requests
```

## Технологии

- Python 3.11+
- aiogram 3.x
- Google Gemini API (gemini-2.5-flash-preview-04-17)
- OpenCV
- ezdxf
- svgwrite
- aiosqlite

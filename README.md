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
OPENROUTER_API_KEY=ключ_от_openrouter.ai
DATABASE_PATH=blueprint_bot.db
LOG_LEVEL=INFO
```

Получите API ключ на [openrouter.ai/keys](https://openrouter.ai/keys)

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
│   ├── ai_recognizer.py    # AI распознавание (OpenRouter API)
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

- наличие `.env` и обязательных переменных (`BOT_TOKEN`, `OPENROUTER_API_KEY`)
- подключение к Telegram API (`getMe`)
- доступность моделей через OpenRouter API

Пример вывода:

```
✅ BOT_TOKEN: OK (bot: @myblueprint_bot)
✅ OPENROUTER_API_KEY: задан
✅ google/gemini-2.0-flash-exp:free: доступна (ответ за 1.2с)
❌ meta-llama/llama-3.2-11b-vision-instruct:free: 429 rate limit

✅ Рекомендуемая модель: google/gemini-2.0-flash-exp:free
   Укажите в .env: OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free
```

Зависимости скрипта уже входят в `requirements.txt` (`python-dotenv`, `openai`).
Дополнительно нужен `requests`:

```bash
pip install requests
```

## Технологии

- Python 3.11+
- aiogram 3.x
- OpenRouter API (google/gemini-2.0-flash-exp:free)
- OpenCV
- ezdxf
- svgwrite
- aiosqlite

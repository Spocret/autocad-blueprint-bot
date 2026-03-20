"""
Диагностический скрипт для Blueprint Bot.
Проверяет доступность Telegram API, OpenRouter API и конфигурации .env.

Запуск: python diagnose.py
"""

import os
import sys
import time

OK  = "✅"
ERR = "❌"
WRN = "⚠️ "

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELS_TO_CHECK = [
    "google/gemma-3-27b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-3-12b-it:free",
]

MINIMAL_PROMPT = "Ответь одним словом: готов"


def load_env() -> dict[str, str | None]:
    """Загружает .env и возвращает значения обязательных переменных."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")

    if not os.path.isfile(env_path):
        return {"_env_exists": False, "BOT_TOKEN": None, "OPENROUTER_API_KEY": None}

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
    except ImportError:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

    return {
        "_env_exists": True,
        "BOT_TOKEN": os.getenv("BOT_TOKEN"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    }


def check_env(env: dict) -> tuple[bool, str | None, str | None]:
    """
    Проверяет наличие .env и обязательных переменных.
    Возвращает (all_ok, bot_token, openrouter_key).
    """
    print("\n── Конфигурация .env ──────────────────────────────────────────")

    all_ok = True

    if not env.get("_env_exists"):
        print(f"{ERR} .env файл не найден (ожидается рядом с diagnose.py)")
        return False, None, None

    print(f"{OK} .env файл найден")

    bot_token = env.get("BOT_TOKEN")
    openrouter_key = env.get("OPENROUTER_API_KEY")

    if bot_token:
        masked = bot_token[:8] + "..." + bot_token[-4:]
        print(f"{OK} BOT_TOKEN: задан ({masked})")
    else:
        print(f"{ERR} BOT_TOKEN: не задан")
        all_ok = False

    if openrouter_key:
        masked = openrouter_key[:8] + "..." + openrouter_key[-4:]
        print(f"{OK} OPENROUTER_API_KEY: задан ({masked})")
    else:
        print(f"{ERR} OPENROUTER_API_KEY: не задан")
        all_ok = False

    return all_ok, bot_token, openrouter_key


def check_telegram(bot_token: str | None) -> bool:
    """Проверяет связь с Telegram API через метод getMe."""
    print("\n── Telegram API ───────────────────────────────────────────────")

    if not bot_token:
        print(f"{ERR} BOT_TOKEN отсутствует — проверка пропущена")
        return False

    try:
        import requests  # type: ignore
    except ImportError:
        print(f"{WRN} Пакет 'requests' не установлен — pip install requests")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/getMe"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except requests.exceptions.ConnectionError:
        print(f"{ERR} BOT_TOKEN: нет соединения с api.telegram.org")
        return False
    except requests.exceptions.Timeout:
        print(f"{ERR} BOT_TOKEN: таймаут подключения к Telegram API")
        return False
    except Exception as exc:
        print(f"{ERR} BOT_TOKEN: неожиданная ошибка — {exc}")
        return False

    if resp.status_code == 200 and data.get("ok"):
        username = data["result"].get("username", "unknown")
        print(f"{OK} BOT_TOKEN: OK (bot: @{username})")
        return True
    elif resp.status_code == 401:
        print(f"{ERR} BOT_TOKEN: 401 Unauthorized — токен недействителен")
    else:
        desc = data.get("description", resp.text[:120])
        print(f"{ERR} BOT_TOKEN: HTTP {resp.status_code} — {desc}")
    return False


def check_openrouter_models(openrouter_key: str | None) -> dict[str, dict]:
    """
    Перебирает список моделей OpenRouter и проверяет каждую минимальным запросом.
    Возвращает словарь {model_name: {"ok": bool, "detail": str}}.
    """
    print("\n── OpenRouter API ─────────────────────────────────────────────")

    results: dict[str, dict] = {}

    if not openrouter_key:
        print(f"{ERR} OPENROUTER_API_KEY отсутствует — проверка пропущена")
        for m in MODELS_TO_CHECK:
            results[m] = {"ok": False, "detail": "нет API ключа"}
        return results

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print(f"{WRN} Пакет 'openai' не установлен — pip install openai")
        for m in MODELS_TO_CHECK:
            results[m] = {"ok": False, "detail": "пакет не установлен"}
        return results

    client = OpenAI(api_key=openrouter_key, base_url=OPENROUTER_BASE_URL)

    for model_name in MODELS_TO_CHECK:
        try:
            start = time.monotonic()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": MINIMAL_PROMPT}],
                max_tokens=16,
                temperature=0.0,
            )
            elapsed = time.monotonic() - start
            _ = response.choices[0].message.content
            detail = f"доступна (ответ за {elapsed:.1f}с)"
            results[model_name] = {"ok": True, "detail": detail}
            print(f"{OK} {model_name}: {detail}")
        except Exception as exc:
            exc_str = str(exc)
            detail = _classify_error(exc_str)
            results[model_name] = {"ok": False, "detail": detail}
            print(f"{ERR} {model_name}: {detail}")

        time.sleep(1.5)

    return results


def _classify_error(exc_str: str) -> str:
    """Преобразует строку исключения в читаемое сообщение об ошибке."""
    s = exc_str.lower()
    if "403" in exc_str or "permission" in s or "forbidden" in s:
        return "403 permission denied — модель недоступна для вашего ключа"
    if "401" in exc_str or "api_key" in s or ("invalid" in s and "key" in s):
        return "401 invalid API key — проверьте OPENROUTER_API_KEY"
    if "404" in exc_str or "not found" in s:
        return "404 модель не найдена (возможно, имя устарело)"
    if "429" in exc_str or "quota" in s or "rate limit" in s or "resource exhausted" in s:
        return "429 rate limit — попробуйте позже"
    if "500" in exc_str or "internal" in s:
        return "500 внутренняя ошибка сервера"
    if "connection" in s or "timeout" in s or "network" in s:
        return "нет соединения с openrouter.ai"
    short = exc_str.strip().replace("\n", " ")
    return short[:120] if len(short) > 120 else short


def print_summary(
    env_ok: bool,
    tg_ok: bool,
    model_results: dict[str, dict],
) -> None:
    """Выводит итоговую таблицу и рекомендацию."""
    print("\n══════════════════════════════════════════════════════════════")
    print("  ИТОГ ДИАГНОСТИКИ")
    print("══════════════════════════════════════════════════════════════")

    print(f"  .env / переменные : {OK if env_ok else ERR}")
    print(f"  Telegram Bot API  : {OK if tg_ok  else ERR}")

    available = [m for m, r in model_results.items() if r["ok"]]
    unavailable = [m for m, r in model_results.items() if not r["ok"]]

    if available:
        print(f"  Доступные модели  : {', '.join(available)}")
    if unavailable:
        print(f"  Недоступные модели: {', '.join(unavailable)}")

    print()

    if not env_ok:
        print(f"{ERR} Заполните .env файл — без него бот не запустится.")
    elif not tg_ok:
        print(f"{ERR} Исправьте BOT_TOKEN в .env.")
    elif not available:
        print(f"{ERR} Ни одна модель OpenRouter недоступна.")
        print("   Проверьте OPENROUTER_API_KEY на openrouter.ai/keys")
    else:
        recommended = available[0]
        print(f"{OK} Рекомендуемая модель: {recommended}")
        print(f"   Укажите в .env: OPENROUTER_MODEL={recommended}")

    print()


def main() -> int:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          Blueprint Bot — Диагностика окружения              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    env = load_env()
    env_ok, bot_token, openrouter_key = check_env(env)
    tg_ok = check_telegram(bot_token)
    model_results = check_openrouter_models(openrouter_key)

    print_summary(env_ok, tg_ok, model_results)

    all_ok = env_ok and tg_ok and any(r["ok"] for r in model_results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

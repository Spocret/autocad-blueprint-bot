"""
Сервис AI-распознавания архитектурных чертежей через OpenRouter API.
"""

import asyncio
import base64
import io
import json
import logging
import re
from typing import Optional

from openai import AsyncOpenAI
from PIL import Image

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    CONFIDENCE_THRESHOLD,
)

# Бесплатные vision-модели на OpenRouter (в порядке приоритета)
FALLBACK_MODELS = [
    OPENROUTER_MODEL,
    "google/gemma-3-27b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-3-12b-it:free",
]

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Ошибка при работе с AI-сервисом."""
    pass


class AIRecognizer:
    """Распознаёт архитектурные чертежи через OpenRouter API."""

    def __init__(self):
        """Инициализация клиента OpenRouter."""
        if not OPENROUTER_API_KEY:
            raise AIServiceError("OPENROUTER_API_KEY не задан в переменных окружения.")
        try:
            self._client = AsyncOpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
            )
            self._active_model_name = OPENROUTER_MODEL
            logger.info("AIRecognizer инициализирован: модель=%s", OPENROUTER_MODEL)
        except Exception as exc:
            logger.exception("Ошибка инициализации AIRecognizer: %s", exc)
            raise AIServiceError(f"Не удалось инициализировать AIRecognizer: {exc}") from exc

    async def recognize(self, image_bytes: bytes, scale: Optional[str] = None) -> dict:
        """
        Главный метод распознавания архитектурного чертежа.

        Аргументы:
            image_bytes: байты обработанного изображения
            scale: масштаб чертежа, если уже определён (например "1:100"), или None

        Возвращает:
            Словарь с распознанными элементами чертежа.
        """
        if not image_bytes:
            raise AIServiceError("Получены пустые байты изображения.")

        # 1. Конвертируем bytes в JPEG и кодируем в base64
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.debug("Изображение загружено через PIL, размер: %s", pil_image.size)
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=90)
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.exception("Ошибка подготовки изображения: %s", exc)
            raise AIServiceError(f"Не удалось подготовить изображение: {exc}") from exc

        # 2. Формируем prompt
        prompt_text = self._build_prompt(scale)

        # 3. Вызываем OpenRouter API с fallback по моделям
        response_text = await self._call_with_fallback(image_b64, prompt_text)

        if not response_text or not response_text.strip():
            raise AIServiceError("OpenRouter вернул пустой ответ.")

        logger.debug("Сырой ответ модели (первые 500 символов): %s", response_text[:500])

        # 4. Парсим JSON из ответа
        data = self._parse_response(response_text)

        # 5. Если масштаб был передан явно — переписываем поле в ответе
        if scale and not data.get("scale"):
            data["scale"] = scale

        # 6. Определяем элементы с низким confidence
        data["low_confidence_elements"] = self._find_low_confidence(data)

        logger.info(
            "Распознавание завершено: %d помещений, %d стен, %d дверей, %d окон, "
            "%d размерных линий, %d лестниц, %d меток, %d рукописных пометок, "
            "%d элементов с низким confidence.",
            len(data.get("rooms", [])),
            len(data.get("walls", [])),
            len(data.get("doors", [])),
            len(data.get("windows", [])),
            len(data.get("dimensions", [])),
            len(data.get("stairs", [])),
            len(data.get("labels", [])),
            len(data.get("handwritten_notes", [])),
            len(data.get("low_confidence_elements", [])),
        )

        return data

    async def _call_with_fallback(
        self,
        image_b64: str,
        prompt_text: str,
    ) -> str:
        """
        Перебирает список моделей, для каждой делает до 3 попыток при rate limit.
        """
        errors: dict[str, str] = {}

        for model_name in dict.fromkeys(FALLBACK_MODELS):
            result = await self._call_with_retry(model_name, image_b64, prompt_text)
            if result is not None:
                self._active_model_name = model_name
                return result
            errors[model_name] = "см. лог выше"

        error_details = "; ".join(f"{m}: недоступна" for m in errors)
        logger.error("Все модели OpenRouter недоступны. %s", error_details)
        raise AIServiceError(f"Все модели OpenRouter недоступны.\n{error_details}")

    async def _call_with_retry(
        self,
        model_name: str,
        image_b64: str,
        prompt_text: str,
        max_retries: int = 3,
        base_delay: float = 10.0,
    ) -> str | None:
        """
        Вызывает одну модель с retry при rate limit (429).
        При других ошибках (404, 403) сразу возвращает None для перехода к следующей.
        """
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Отправка запроса к OpenRouter '%s' (попытка %d/%d)...",
                    model_name, attempt, max_retries,
                )
                response = await self._client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt_text,
                                },
                            ],
                        }
                    ],
                    temperature=0.1,
                    max_tokens=8192,
                )
                logger.info("Ответ от OpenRouter получен успешно (модель: %s).", model_name)
                return response.choices[0].message.content or ""

            except Exception as exc:
                exc_str = str(exc)
                is_quota = (
                    "429" in exc_str
                    or "rate limit" in exc_str.lower()
                    or "resource exhausted" in exc_str.lower()
                )
                is_unavailable = (
                    "404" in exc_str
                    or "No endpoints" in exc_str
                    or "403" in exc_str
                )

                if is_unavailable:
                    logger.warning("Модель '%s' недоступна: %s. Пробую следующую...", model_name, exc_str[:120])
                    return None
                elif is_quota and attempt < max_retries:
                    delay = base_delay * attempt
                    logger.warning(
                        "Квота '%s' исчерпана (попытка %d/%d). Жду %.0f сек...",
                        model_name, attempt, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Ошибка OpenRouter API '%s': %s", model_name, exc_str)
                    return None

        return None

    def _build_prompt(self, scale: Optional[str]) -> str:
        """
        Составляет детальный prompt для распознавания российских строительных чертежей по ГОСТ.
        """
        scale_instruction = (
            f"Масштаб чертежа уже известен: {scale}. Используй его при вычислении реальных размеров."
            if scale
            else (
                "Масштаб чертежа: найди надпись вида 'Масштаб 1:X' или '1:X' на чертеже и запиши "
                "в поле scale. Если не найдено — оставь null."
            )
        )

        few_shot_example = """Вот пример правильного результата распознавания российского строительного чертежа подвала. Используй точно такую же структуру.

{
  "floor": "Подвал",
  "scale": "1:100",
  "rooms": [
    {"id": "1", "area": 16.1, "dimensions": {"w": 6.53, "h": 2.46}, "center_x_percent": 12.5, "center_y_percent": 30.2, "confidence": 0.95},
    {"id": "2", "area": 13.0, "dimensions": {"w": 6.53, "h": 2.0}, "center_x_percent": 28.4, "center_y_percent": 30.2, "confidence": 0.93},
    {"id": "3", "area": 10.4, "dimensions": {"w": 4.04, "h": 2.56}, "center_x_percent": 44.1, "center_y_percent": 28.0, "confidence": 0.91},
    {"id": "12", "area": 6.8, "dimensions": {"w": 3.93, "h": 1.74}, "center_x_percent": 70.0, "center_y_percent": 55.0, "confidence": 0.88}
  ],
  "walls": [
    {"type": "load_bearing", "thickness_mm": 500, "start": {"x": 5.0, "y": 10.0}, "end": {"x": 95.0, "y": 10.0}, "confidence": 0.97},
    {"type": "partition", "thickness_mm": 150, "start": {"x": 30.0, "y": 10.0}, "end": {"x": 30.0, "y": 60.0}, "confidence": 0.92}
  ],
  "doors": [
    {"position_x_percent": 15.5, "position_y_percent": 35.0, "width_mm": 900, "opening_direction": "inward_left", "confidence": 0.90},
    {"position_x_percent": 42.0, "position_y_percent": 28.5, "width_mm": 800, "opening_direction": "inward_right", "confidence": 0.88}
  ],
  "windows": [
    {"position_x_percent": 8.0, "position_y_percent": 10.0, "width_mm": 1200, "confidence": 0.92},
    {"position_x_percent": 88.0, "position_y_percent": 10.0, "width_mm": 1200, "confidence": 0.91}
  ],
  "stairs": [
    {"type": "маршевая", "position_x_percent": 50.0, "position_y_percent": 12.0, "steps": 10}
  ],
  "dimensions": [
    {"value": 2.88, "unit": "m", "confidence": 0.96},
    {"value": 2.85, "unit": "m", "confidence": 0.95},
    {"value": 3.90, "unit": "m", "confidence": 0.94}
  ],
  "unclear_elements": [
    {"id": "u1", "type": "label", "description": "нечитаемая надпись в левом нижнем углу", "position_x_percent": 3.0, "position_y_percent": 92.0}
  ]
}"""

        return f"""Ты — эксперт по распознаванию российских строительных чертежей поэтажных планов по ГОСТ.

{few_shot_example}

## КОНТЕКСТ: ОСОБЕННОСТИ РОССИЙСКИХ ЧЕРТЕЖЕЙ

Обязательно учитывай при распознавании:
- Красные или розовые цифры — номер помещения и его площадь в формате "номер/площадьм²" (например "12/6.8")
- Числа рядом со стенами и размерные линии — размеры в метрах
- ТОЛСТЫЕ линии (визуально жирные) — несущие стены толщиной 400–600 мм
- ТОНКИЕ линии — перегородки толщиной 100–200 мм
- Дуга у дверного проёма — дверь, направление дуги показывает сторону открывания
- Три параллельные линии в проёме — окно
- Сетка параллельных линий со ступенями — лестница
- Надпись "Масштаб 1:X" или просто "1:X" — масштаб чертежа
- Название этажа пишется крупно в верхней части чертежа (ПОДВАЛ, 1 ЭТАЖ, МАНСАРДА и т.д.)

## ДВУХЭТАПНОЕ РАСПОЗНАВАНИЕ

Этап 1 — сначала найди название этажа и масштаб.
{scale_instruction}

Этап 2 — распознай все элементы чертежа:
- Каждое помещение с его номером и площадью из красных/розовых цифр
- Все несущие стены и перегородки
- Все двери с направлением открывания
- Все окна
- Лестницы
- Размерные линии с числами

## РАЗБИВКА ИЗОБРАЖЕНИЯ

Если чертёж содержит много помещений (более 10), мысленно раздели изображение на 4 части:
верхний-левый, верхний-правый, нижний-левый, нижний-правый.
Внимательно проанализируй каждую часть отдельно, затем объедини результаты.
Это поможет не пропустить мелкие элементы.

## КООРДИНАТЫ

Все позиции элементов указывай в ПРОЦЕНТАХ от размеров изображения (от 0.0 до 100.0):
- center_x_percent, center_y_percent для помещений
- start/end x,y для стен
- position_x_percent, position_y_percent для дверей, окон, лестниц
Это важно — НЕ в пикселях, а в процентах.

## УВЕРЕННОСТЬ

Для каждого элемента с уверенностью ниже 0.7 — обязательно добавляй в unclear_elements.

## ФОРМАТ ОТВЕТА

Верни СТРОГО валидный JSON без какого-либо текста до или после него, без markdown-блоков (```).

{{
  "floor": "название этажа с чертежа или null",
  "scale": "{scale if scale else 'найденный масштаб или null'}",
  "rooms": [
    {{
      "id": "номер с чертежа",
      "area": площадь_число_или_null,
      "dimensions": {{"w": ширина_м, "h": высота_м}},
      "center_x_percent": процент_от_ширины,
      "center_y_percent": процент_от_высоты,
      "confidence": 0.95
    }}
  ],
  "walls": [
    {{
      "type": "load_bearing или partition",
      "thickness_mm": толщина,
      "start": {{"x": процент, "y": процент}},
      "end": {{"x": процент, "y": процент}},
      "confidence": 0.95
    }}
  ],
  "doors": [
    {{
      "position_x_percent": процент,
      "position_y_percent": процент,
      "width_mm": ширина,
      "opening_direction": "inward_left или inward_right или outward_left или outward_right",
      "confidence": 0.90
    }}
  ],
  "windows": [
    {{
      "position_x_percent": процент,
      "position_y_percent": процент,
      "width_mm": ширина,
      "confidence": 0.90
    }}
  ],
  "stairs": [
    {{
      "type": "маршевая или винтовая",
      "position_x_percent": процент,
      "position_y_percent": процент,
      "steps": количество_ступеней_или_null
    }}
  ],
  "dimensions": [
    {{
      "value": число,
      "unit": "m",
      "confidence": 0.95
    }}
  ],
  "unclear_elements": [
    {{
      "id": "u1",
      "type": "dimension или label или wall или door",
      "description": "описание проблемы распознавания",
      "position_x_percent": процент,
      "position_y_percent": процент
    }}
  ]
}}

ВАЖНО: верни ТОЛЬКО валидный JSON. Никакого текста до или после. Никаких ``` блоков.
"""

    def _parse_response(self, response_text: str) -> dict:
        """
        Извлекает и парсит JSON из ответа модели.
        """
        text = response_text.strip()

        # Попытка 1: убрать markdown-блоки ```json ... ``` или ``` ... ```
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if markdown_match:
            text = markdown_match.group(1).strip()
            logger.debug("Извлечён JSON из markdown-блока.")

        # Попытка 2: найти первый { и последний } — взять подстроку
        if not text.startswith("{"):
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                text = text[brace_start : brace_end + 1]
                logger.debug("Извлечён JSON по фигурным скобкам.")

        # Попытка 3: прямой парсинг
        try:
            data = json.loads(text)
            logger.debug("JSON успешно распарсен.")
            return data
        except json.JSONDecodeError as exc:
            logger.warning("Первичный парсинг JSON не удался: %s. Пробуем исправить...", exc)

        # Попытка 4: убрать trailing commas (распространённая ошибка моделей)
        try:
            fixed = re.sub(r",\s*([}\]])", r"\1", text)
            data = json.loads(fixed)
            logger.debug("JSON успешно распарсен после удаления trailing commas.")
            return data
        except json.JSONDecodeError as exc:
            logger.error("Не удалось распарсить JSON после исправления: %s", exc)
            logger.error("Проблемный текст (первые 1000 символов): %s", text[:1000])
            raise AIServiceError(
                f"Не удалось распарсить JSON ответ модели: {exc}. "
                f"Начало ответа: {text[:200]}"
            ) from exc

    def _find_low_confidence(self, data: dict) -> list[dict]:
        """
        Находит все элементы чертежа с confidence ниже порогового значения.
        """
        low_confidence: list[dict] = []

        element_groups = {
            "walls": "wall",
            "doors": "door",
            "windows": "window",
            "dimensions": "dimension",
            "stairs": "stair",
            "labels": "label",
            "handwritten_notes": "handwritten_note",
            "rooms": "room",
        }

        for group_key, element_type in element_groups.items():
            elements = data.get(group_key, [])
            if not isinstance(elements, list):
                continue

            for element in elements:
                if not isinstance(element, dict):
                    continue

                confidence = element.get("confidence")
                if confidence is None:
                    continue

                try:
                    conf_value = float(confidence)
                except (ValueError, TypeError):
                    continue

                if conf_value < CONFIDENCE_THRESHOLD:
                    element_id = element.get("id", "unknown")

                    if element_type == "handwritten_note":
                        reason = "рукописный текст"
                    elif conf_value < 0.3:
                        reason = "элемент едва различим на чертеже"
                    elif conf_value < 0.5:
                        reason = "низкое качество изображения или частичное перекрытие"
                    else:
                        reason = "неоднозначная интерпретация элемента"

                    bbox = self._estimate_bbox(element, element_type)

                    low_confidence.append(
                        {
                            "element_id": element_id,
                            "element_type": element_type,
                            "bbox": bbox,
                            "reason": reason,
                        }
                    )
                    logger.debug(
                        "Элемент с низким confidence: id=%s, type=%s, confidence=%.2f",
                        element_id,
                        element_type,
                        conf_value,
                    )

        return low_confidence

    def _estimate_bbox(self, element: dict, element_type: str) -> dict:
        """
        Вычисляет приблизительный bounding box элемента по его координатам.
        """
        try:
            if element_type == "wall":
                x1 = int(element.get("x1", 0))
                y1 = int(element.get("y1", 0))
                x2 = int(element.get("x2", 0))
                y2 = int(element.get("y2", 0))
                x = min(x1, x2)
                y = min(y1, y2)
                width = max(abs(x2 - x1), 10)
                height = max(abs(y2 - y1), 10)
                return {"x": x, "y": y, "width": width, "height": height}

            elif element_type == "dimension":
                x1 = int(element.get("x1", 0))
                y1 = int(element.get("y1", 0))
                x2 = int(element.get("x2", 0))
                y2 = int(element.get("y2", 0))
                x = min(x1, x2)
                y = min(y1, y2)
                width = max(abs(x2 - x1), 20)
                height = max(abs(y2 - y1), 20)
                return {"x": x - 5, "y": y - 5, "width": width + 10, "height": height + 10}

            elif element_type in ("door", "window"):
                x = int(element.get("x", 0))
                y = int(element.get("y", 0))
                w = int(element.get("width", 100))
                return {"x": x - w // 2, "y": y - 20, "width": w, "height": 40}

            elif element_type in ("label", "handwritten_note"):
                x = int(element.get("x", 0))
                y = int(element.get("y", 0))
                text = str(element.get("text", ""))
                approx_width = max(len(text) * 8, 30)
                return {"x": x - 5, "y": y - 5, "width": approx_width, "height": 20}

            elif element_type == "room":
                cx = int(element.get("center_x", 0))
                cy = int(element.get("center_y", 0))
                return {"x": cx - 50, "y": cy - 30, "width": 100, "height": 60}

            elif element_type == "stair":
                x = int(element.get("x", 0))
                y = int(element.get("y", 0))
                w = int(element.get("width", 100))
                h = int(element.get("height", 200))
                return {"x": x, "y": y, "width": w, "height": h}

        except (ValueError, TypeError) as exc:
            logger.debug("Не удалось вычислить bbox для элемента: %s", exc)

        return {"x": 0, "y": 0, "width": 50, "height": 50}


# Глобальный экземпляр для использования в хендлерах
ai_recognizer = AIRecognizer()

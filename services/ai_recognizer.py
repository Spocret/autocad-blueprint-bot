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
        return """You are an expert at reading Russian architectural floor plan drawings (GOST standard).

This is a TOP-DOWN VIEW floor plan drawing.
You are looking at the building from above.
All walls, rooms, doors and windows are shown as a 2D top-down projection.

STEP 1 - Find the scale:
Look for text like "Масштаб 1:X" or "М 1:X".
Extract the scale number X.

STEP 2 - Find building boundaries:
Find the 4 outer corners of the entire building.
Set bottom-left corner as point zero.
X grows to the right, Y grows upward.
All coordinates must be in real millimeters calculated using the scale.

STEP 3 - Recognize walls:
Find every wall segment in the drawing.
Thick lines are load bearing walls, real thickness 380 to 600 millimeters.
Thin lines are partition walls, real thickness 100 to 200 millimeters.
Return start point, end point and thickness for each wall in millimeters.

STEP 4 - Recognize rooms:
Find every room on the plan.
Red or pink numbers show room number and area in format "number/area" for example "1/12.6" means room 1 with area 12.6 square meters.
Return room number, area, position and dimensions in millimeters for each room.

STEP 5 - Recognize doors:
Find every door opening.
A door is shown as an arc near a wall opening.
The arc shows the direction the door opens.
Return position, width and opening direction for each door in millimeters.

STEP 6 - Recognize windows:
Find every window opening.
A window is shown as three parallel lines inside a wall opening.
Return position and width for each window in millimeters.

STEP 7 - Recognize stairs:
Find every staircase.
Stairs are shown as a grid of parallel lines.
Return position, dimensions and number of steps.

STEP 8 - Recognize dimensions:
Find all dimension numbers next to walls.
These are sizes in meters.
Return each dimension value converted to millimeters.

STEP 9 - Self check:
Calculate total building area from boundaries.
Calculate sum of all room areas.
If difference is more than 20 percent — recheck room boundaries and fix errors.

STEP 10 - Unclear elements:
Any element you are not confident about must be added to unclear list.
Describe what it is and where it is located, write description in Russian.

Return result as valid JSON only.
No explanations, no markdown, no extra text.
Just the JSON object."""

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

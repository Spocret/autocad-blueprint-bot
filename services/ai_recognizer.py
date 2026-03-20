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
            f"Масштаб чертежа уже известен: {scale}. Используй его при вычислении реальных размеров в миллиметрах."
            if scale
            else (
                "Масштаб чертежа: найди надпись вида 'Масштаб 1:X' или '1:X' на чертеже и запиши "
                "в поле scale. Если не найдено — оставь null. "
                "Все координаты и размеры рассчитывай с учётом найденного масштаба и возвращай в миллиметрах."
            )
        )

        few_shot_example = """Вот пример правильного результата распознавания.
Здание 25000 x 19000 мм, масштаб 1:100. Начало координат — левый нижний угол здания.

{
  "floor": "1 ЭТАЖ",
  "scale": "1:100",
  "building_bounds": {
    "min_x": 0, "min_y": 0, "max_x": 25000, "max_y": 19000,
    "width_mm": 25000, "height_mm": 19000
  },
  "rooms": [
    {"id": "1", "area": 12.4, "x": 0, "y": 14690, "width": 2880, "height": 4310, "confidence": 0.95},
    {"id": "2", "area": 10.8, "x": 2880, "y": 14690, "width": 2850, "height": 4310, "confidence": 0.93},
    {"id": "3", "area": 38.5, "x": 5730, "y": 9500, "width": 9000, "height": 4280, "confidence": 0.91}
  ],
  "walls": [
    {"type": "load_bearing", "thickness_mm": 500, "start_x": 0, "start_y": 0, "end_x": 25000, "end_y": 0, "confidence": 0.97},
    {"type": "load_bearing", "thickness_mm": 500, "start_x": 0, "start_y": 0, "end_x": 0, "end_y": 19000, "confidence": 0.97},
    {"type": "partition", "thickness_mm": 150, "start_x": 2880, "start_y": 14690, "end_x": 2880, "end_y": 19000, "confidence": 0.92}
  ],
  "doors": [
    {"x": 1200, "y": 14690, "width_mm": 900, "opening_direction": "inward_left", "confidence": 0.90},
    {"x": 3500, "y": 14690, "width_mm": 800, "opening_direction": "inward_right", "confidence": 0.88}
  ],
  "windows": [
    {"x": 500, "y": 19000, "width_mm": 1200, "confidence": 0.92},
    {"x": 3200, "y": 19000, "width_mm": 1200, "confidence": 0.91}
  ],
  "stairs": [
    {"type": "маршевая", "x": 12000, "y": 9500, "width": 3000, "height": 4000, "steps": 10}
  ],
  "dimensions": [
    {"value": 2880, "unit": "mm", "confidence": 0.96},
    {"value": 2850, "unit": "mm", "confidence": 0.95},
    {"value": 9000, "unit": "mm", "confidence": 0.94}
  ],
  "unclear_elements": [
    {"id": "u1", "type": "label", "description": "нечитаемая надпись", "x": 500, "y": 18500}
  ]
}"""

        return f"""Ты — эксперт по распознаванию российских строительных чертежей поэтажных планов по ГОСТ.

{few_shot_example}

## КОНТЕКСТ: ОСОБЕННОСТИ РОССИЙСКИХ ЧЕРТЕЖЕЙ

Обязательно учитывай при распознавании:
- Красные или розовые цифры — номер помещения и его площадь в формате "номер/площадьм²" (например "12/6.8")
- Числа рядом со стенами и размерные линии — размеры в метрах или миллиметрах
- ТОЛСТЫЕ линии (визуально жирные) — несущие стены толщиной 400–600 мм
- ТОНКИЕ линии — перегородки толщиной 100–200 мм
- Дуга у дверного проёма — дверь, направление дуги показывает сторону открывания
- Три параллельные линии в проёме — окно
- Сетка параллельных линий со ступенями — лестница
- Надпись "Масштаб 1:X" или просто "1:X" — масштаб чертежа
- Название этажа пишется крупно в верхней части чертежа (ПОДВАЛ, 1 ЭТАЖ, МАНСАРДА и т.д.)

## СИСТЕМА КООРДИНАТ (КАК В AUTOCAD)

КРИТИЧЕСКИ ВАЖНО — используй следующую систему координат:
- Начало координат (0, 0) — это ЛЕВЫЙ НИЖНИЙ угол всего здания
- Ось X направлена ВПРАВО (X растёт слева направо)
- Ось Y направлена ВВЕРХ (Y растёт снизу вверх)
- Все координаты возвращать строго в МИЛЛИМЕТРАХ с учётом масштаба чертежа
- НЕ в процентах, НЕ в пикселях, НЕ в метрах — только в МИЛЛИМЕТРАХ

## ТРЁХЭТАПНОЕ РАСПОЗНАВАНИЕ

Этап 1 — найди название этажа и масштаб.
{scale_instruction}

Этап 2 — определи якорные точки (систему координат):
Найди 4 угловые точки контура всего здания:
  - левый нижний угол → это точка (0, 0)
  - правый нижний угол → это точка (ширина_здания_мм, 0)
  - левый верхний угол → это точка (0, высота_здания_мм)
  - правый верхний угол → это точка (ширина_здания_мм, высота_здания_мм)
Запиши эти значения в поле building_bounds.
ВСЕ остальные координаты считай относительно этих якорных точек.

Этап 3 — распознай все элементы чертежа в абсолютных координатах (мм):
- Каждое помещение: левый нижний угол (x, y), ширина и высота в мм
- Все несущие стены и перегородки: начало (start_x, start_y) и конец (end_x, end_y) в мм
- Все двери: точка на стене (x, y) в мм плюс ширина проёма
- Все окна: точка на стене (x, y) в мм плюс ширина проёма
- Лестницы
- Размерные линии с числами

## РАЗБИВКА ИЗОБРАЖЕНИЯ

Если чертёж содержит много помещений (более 10), мысленно раздели изображение на 4 части:
верхний-левый, верхний-правый, нижний-левый, нижний-правый.
Внимательно проанализируй каждую часть отдельно, затем объедини результаты.
Это поможет не пропустить мелкие элементы.

## САМОПРОВЕРКА (ОБЯЗАТЕЛЬНО)

После распознавания всех помещений выполни проверку:
1. Вычисли сумму площадей всех помещений: S_сумма = сумма (width × height) для каждой комнаты
2. Вычисли общую площадь здания: S_здание = building_bounds.width_mm × building_bounds.height_mm
3. Если |S_сумма - S_здание| / S_здание > 0.20 (расхождение больше 20%) — это ошибка.
   В таком случае пересмотри координаты помещений и пересчитай. Исправь результат перед возвратом.
4. Запиши результат проверки в поле self_check.

## УВЕРЕННОСТЬ

Для каждого элемента с уверенностью ниже 0.7 — обязательно добавляй в unclear_elements.

## ФОРМАТ ОТВЕТА

Верни СТРОГО валидный JSON без какого-либо текста до или после него, без markdown-блоков (```).

{{
  "floor": "название этажа с чертежа или null",
  "scale": "{scale if scale else 'найденный масштаб или null'}",
  "building_bounds": {{
    "min_x": 0,
    "min_y": 0,
    "max_x": ширина_здания_мм,
    "max_y": высота_здания_мм,
    "width_mm": ширина_здания_мм,
    "height_mm": высота_здания_мм
  }},
  "self_check": {{
    "rooms_area_sum_mm2": сумма_площадей_помещений,
    "building_area_mm2": площадь_здания,
    "discrepancy_percent": процент_расхождения,
    "passed": true_или_false
  }},
  "rooms": [
    {{
      "id": "номер с чертежа",
      "area": площадь_м2_или_null,
      "x": левый_нижний_угол_x_мм,
      "y": левый_нижний_угол_y_мм,
      "width": ширина_мм,
      "height": высота_мм,
      "confidence": 0.95
    }}
  ],
  "walls": [
    {{
      "type": "load_bearing или partition",
      "thickness_mm": толщина,
      "start_x": начало_x_мм,
      "start_y": начало_y_мм,
      "end_x": конец_x_мм,
      "end_y": конец_y_мм,
      "confidence": 0.95
    }}
  ],
  "doors": [
    {{
      "x": позиция_x_мм,
      "y": позиция_y_мм,
      "width_mm": ширина_проёма,
      "opening_direction": "inward_left или inward_right или outward_left или outward_right",
      "confidence": 0.90
    }}
  ],
  "windows": [
    {{
      "x": позиция_x_мм,
      "y": позиция_y_мм,
      "width_mm": ширина_проёма,
      "confidence": 0.90
    }}
  ],
  "stairs": [
    {{
      "type": "маршевая или винтовая",
      "x": левый_нижний_угол_x_мм,
      "y": левый_нижний_угол_y_мм,
      "width": ширина_мм,
      "height": высота_мм,
      "steps": количество_ступеней_или_null
    }}
  ],
  "dimensions": [
    {{
      "value": число_мм,
      "unit": "mm",
      "confidence": 0.95
    }}
  ],
  "unclear_elements": [
    {{
      "id": "u1",
      "type": "dimension или label или wall или door",
      "description": "описание проблемы распознавания",
      "x": позиция_x_мм,
      "y": позиция_y_мм
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

"""
Сервис AI-распознавания архитектурных чертежей через Google Gemini API.
"""

import asyncio
import io
import json
import logging
import re
from typing import Optional

import google.generativeai as genai
from PIL import Image

from config import GEMINI_API_KEY, GEMINI_MODEL, CONFIDENCE_THRESHOLD

GEMINI_FALLBACK_MODELS = [
    GEMINI_MODEL,
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Ошибка при работе с AI-сервисом."""
    pass


class AIRecognizer:
    """Распознаёт архитектурные чертежи с помощью Google Gemini Vision."""

    def __init__(self):
        """Инициализация клиента Google Gemini."""
        if not GEMINI_API_KEY:
            raise AIServiceError("GEMINI_API_KEY не задан в переменных окружения.")
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            self._active_model_name = GEMINI_MODEL
            logger.info("AIRecognizer инициализирован: модель=%s", GEMINI_MODEL)
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

        # 1. Конвертируем bytes в PIL.Image для передачи в Gemini
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.debug("Изображение загружено через PIL, размер: %s", pil_image.size)
        except Exception as exc:
            logger.exception("Ошибка загрузки изображения через PIL: %s", exc)
            raise AIServiceError(f"Не удалось загрузить изображение: {exc}") from exc

        # 2. Формируем prompt
        prompt_text = self._build_prompt(scale)

        # 3. Вызываем Gemini API с изображением (с fallback по моделям)
        response = None
        errors: dict[str, str] = {}

        for model_name in dict.fromkeys(GEMINI_FALLBACK_MODELS):
            response = await self._call_with_retry(model_name, pil_image, prompt_text, errors)
            if response is not None:
                self._active_model_name = model_name
                logger.info("Ответ от Gemini получен успешно (модель: %s).", model_name)
                break

        if response is None:
            error_details = "; ".join(f"{m}: {e}" for m, e in errors.items())
            logger.error("Все модели Gemini недоступны. Детали: %s", error_details)
            raise AIServiceError(
                f"Все модели Gemini недоступны.\n{error_details}"
            )

        # 4. Извлекаем текст ответа
        try:
            response_text = response.text
        except Exception as exc:
            logger.error("Не удалось получить текст из ответа Gemini: %s", exc)
            raise AIServiceError("Ответ Gemini пустой или имеет неожиданный формат.") from exc

        if not response_text or not response_text.strip():
            raise AIServiceError("Gemini вернул пустой ответ.")

        logger.debug("Сырой ответ модели (первые 500 символов): %s", response_text[:500])

        # 5. Парсим JSON из ответа
        data = self._parse_response(response_text)

        # 6. Если масштаб был передан явно — переписываем поле в ответе
        if scale and not data.get("scale"):
            data["scale"] = scale

        # 7. Определяем элементы с низким confidence и добавляем в low_confidence_elements
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

    async def _call_with_retry(
        self,
        model_name: str,
        pil_image,
        prompt_text: str,
        errors: dict,
        max_retries: int = 3,
        base_delay: float = 10.0,
    ):
        """
        Вызывает Gemini API с retry при ошибке квоты (429).
        При других ошибках сразу записывает в errors и возвращает None.
        """
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Отправка запроса к Gemini '%s' (попытка %d/%d)...",
                    model_name, attempt, max_retries,
                )
                model = genai.GenerativeModel(model_name)
                response = await model.generate_content_async(
                    [pil_image, prompt_text],
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=8192,
                    ),
                )
                return response
            except Exception as exc:
                exc_str = str(exc)
                is_quota = "429" in exc_str or "quota" in exc_str.lower() or "resource exhausted" in exc_str.lower()

                if is_quota and attempt < max_retries:
                    delay = base_delay * attempt
                    logger.warning(
                        "Квота Gemini '%s' исчерпана (попытка %d/%d). Жду %.0f сек...",
                        model_name, attempt, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    errors[model_name] = exc_str
                    logger.warning("Модель '%s' недоступна: %s", model_name, exc_str)
                    return None
        return None

    def _build_prompt(self, scale: Optional[str]) -> str:
        """
        Составляет детальный prompt для мультимодальной модели.

        Аргументы:
            scale: масштаб чертежа, если известен, иначе None.

        Возвращает:
            Строку с инструкцией для модели.
        """
        scale_instruction = (
            f"Масштаб чертежа: {scale}. Используй его при вычислении реальных размеров."
            if scale
            else (
                "Масштаб чертежа не известен. "
                "Попробуй определить его из надписей на чертеже (например '1:100', '1:50'). "
                "Запиши найденный масштаб в поле 'scale'. "
                "Если масштаб не обнаружен, оставь 'scale' равным null."
            )
        )

        return f"""Ты — специализированная система распознавания архитектурных чертежей (планов этажей).
На изображении представлен архитектурный чертёж или план помещения.

{scale_instruction}

Твоя задача — тщательно проанализировать чертёж и вернуть результат СТРОГО в виде JSON без каких-либо пояснений, комментариев или текста за пределами JSON.

Распознай следующие элементы:

1. **Стены** (walls): несущие (load_bearing) и перегородки (partition). Определи координаты начала и конца каждой стены в пикселях, толщину в миллиметрах реального размера.
2. **Двери** (doors): определи позицию (центр проёма), ширину в миллиметрах, угол поворота, направление открывания (left/right/inward/outward).
3. **Окна** (windows): определи позицию (центр проёма), ширину в миллиметрах.
4. **Размерные линии** (dimensions): цифровые размеры с единицами измерения (mm/cm/m), координаты начала и конца линии в пикселях.
5. **Подписи/метки** (labels): текстовые надписи, цифры, обозначения помещений.
6. **Лестницы** (stairs): координаты, размеры, количество ступеней, направление подъёма (up/down).
7. **Помещения** (rooms): название (если указано), площадь в кв. метрах (если можно вычислить), координаты центра в пикселях.
8. **Рукописные пометки** (handwritten_notes): рукописный текст, аннотации, нарисованные вручную символы.

Для каждого элемента обязательно укажи поле `confidence` — степень уверенности распознавания от 0.0 до 1.0:
- 0.9–1.0: элемент чётко виден и однозначно идентифицирован
- 0.7–0.9: элемент виден, но есть небольшая неопределённость
- 0.5–0.7: элемент частично виден или неоднозначен
- ниже 0.5: элемент едва различим, предположительное распознавание

Все координаты (x, y, x1, y1, x2, y2, center_x, center_y) указывай в пикселях относительно переданного изображения.
Размеры архитектурных элементов (thickness, width, height) указывай в миллиметрах реального размера (используй масштаб).

Верни результат в следующем формате JSON:
{{
  "scale": "1:100",
  "total_area": 45.5,
  "rooms": [
    {{
      "id": "room_1",
      "name": "Кухня",
      "area": 12.5,
      "center_x": 250,
      "center_y": 300
    }}
  ],
  "walls": [
    {{
      "id": "wall_1",
      "type": "load_bearing",
      "x1": 100, "y1": 100,
      "x2": 500, "y2": 100,
      "thickness": 300,
      "confidence": 0.95
    }}
  ],
  "doors": [
    {{
      "id": "door_1",
      "x": 200, "y": 100,
      "width": 900,
      "angle": 0,
      "swing_direction": "left",
      "confidence": 0.9
    }}
  ],
  "windows": [
    {{
      "id": "window_1",
      "x": 300, "y": 100,
      "width": 1500,
      "confidence": 0.85
    }}
  ],
  "dimensions": [
    {{
      "id": "dim_1",
      "value": 3500,
      "unit": "mm",
      "x1": 100, "y1": 50,
      "x2": 500, "y2": 50,
      "confidence": 0.92
    }}
  ],
  "stairs": [
    {{
      "id": "stair_1",
      "x": 400, "y": 200,
      "width": 1200,
      "height": 2400,
      "steps_count": 12,
      "direction": "up",
      "confidence": 0.88
    }}
  ],
  "labels": [
    {{
      "id": "label_1",
      "text": "3500",
      "x": 300, "y": 80,
      "confidence": 0.95
    }}
  ],
  "handwritten_notes": [
    {{
      "id": "note_1",
      "text": "несущая",
      "x": 150, "y": 120,
      "confidence": 0.6
    }}
  ]
}}

ВАЖНО: верни ТОЛЬКО валидный JSON. Никакого текста до или после JSON. Никаких markdown-блоков (```). Только чистый JSON.
"""

    def _parse_response(self, response_text: str) -> dict:
        """
        Извлекает и парсит JSON из ответа модели.

        Обрабатывает случаи, когда модель оборачивает JSON в markdown-блоки (```json ... ```)
        или добавляет посторонний текст.

        Аргументы:
            response_text: сырой текст ответа модели.

        Возвращает:
            Распарсенный словарь.
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

        Аргументы:
            data: словарь с распознанными элементами.

        Возвращает:
            Список словарей с описанием элементов с низким confidence.
        """
        low_confidence: list[dict] = []

        # Типы элементов и их человекочитаемые названия
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

                    # Определяем причину низкого confidence
                    if element_type == "handwritten_note":
                        reason = "рукописный текст"
                    elif conf_value < 0.3:
                        reason = "элемент едва различим на чертеже"
                    elif conf_value < 0.5:
                        reason = "низкое качество изображения или частичное перекрытие"
                    else:
                        reason = "неоднозначная интерпретация элемента"

                    # Вычисляем приблизительный bounding box
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

        Аргументы:
            element: словарь с полями элемента.
            element_type: тип элемента (wall, door, window и т.д.).

        Возвращает:
            Словарь с полями x, y, width, height в пикселях.
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

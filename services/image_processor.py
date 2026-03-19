"""
Модуль обработки изображений архитектурных чертежей.
Выполняет коррекцию перспективы, улучшение контраста,
бинаризацию и определение масштаба.
"""

import asyncio
import io
import logging
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    """Результат обработки изображения чертежа."""

    image_bytes: bytes                    # обработанное изображение в bytes (JPEG)
    scale: Optional[str]                  # обнаруженный масштаб, например "1:100"
    scale_pixels_per_mm: Optional[float]  # пикселей на мм
    width_px: int                         # ширина обработанного изображения
    height_px: int                        # высота обработанного изображения
    original_width_px: int                # исходная ширина
    original_height_px: int               # исходная высота
    is_valid: bool                        # True если изображение похоже на чертёж
    quality_issues: list[str] = field(default_factory=list)  # список проблем


class ImageProcessor:
    """Обработчик изображений архитектурных чертежей."""

    # Минимальное допустимое разрешение
    MIN_WIDTH = 800
    MIN_HEIGHT = 600

    # Максимальный размер после выравнивания перспективы (по длинной стороне)
    MAX_LONG_SIDE = 4096

    # Padding для кропов
    CROP_PADDING = 20

    # Паттерны масштаба: "1:100", "М 1:100", "M1:50", "м1:200" и т.д.
    SCALE_PATTERN = re.compile(
        r'[мМmM]?\s*1\s*:\s*(25|50|100|200|250|500|1000|2000|5000)',
        re.IGNORECASE,
    )

    # Стандартные масштабы и размеры рамки штампа в мм (ширина × высота)
    # По ГОСТ 2.104 штамп имеет ширину 185 мм
    STAMP_WIDTH_MM = 185.0

    async def process(self, image_bytes: bytes) -> ProcessedImage:
        """
        Главный метод — полная обработка изображения чертежа.
        Запускает: валидацию, коррекцию перспективы, улучшение контраста,
        бинаризацию и определение масштаба.
        """
        loop = asyncio.get_event_loop()

        try:
            # Декодируем bytes → numpy array через PIL (поддерживает больше форматов)
            img = await loop.run_in_executor(None, self._bytes_to_ndarray, image_bytes)
        except Exception as exc:
            logger.exception("Ошибка декодирования изображения: %s", exc)
            raise ValueError(f"Не удалось декодировать изображение: {exc}") from exc

        original_h, original_w = img.shape[:2]
        logger.info("Загружено изображение %dx%d", original_w, original_h)

        # --- Шаг 1: Валидация ---
        try:
            is_valid, quality_issues = await loop.run_in_executor(
                None, self._validate, img
            )
        except Exception as exc:
            logger.exception("Ошибка валидации: %s", exc)
            is_valid, quality_issues = False, [f"Ошибка валидации: {exc}"]

        # --- Шаг 2: Коррекция перспективы ---
        try:
            img = await loop.run_in_executor(None, self._correct_perspective, img)
        except Exception as exc:
            logger.warning("Коррекция перспективы пропущена: %s", exc)

        # --- Шаг 3: Улучшение контраста ---
        try:
            img = await loop.run_in_executor(None, self._enhance_contrast, img)
        except Exception as exc:
            logger.warning("Улучшение контраста пропущено: %s", exc)

        # --- Шаг 4: Бинаризация ---
        try:
            img = await loop.run_in_executor(None, self._binarize, img)
        except Exception as exc:
            logger.warning("Бинаризация пропущена: %s", exc)

        # --- Шаг 5: Определение масштаба ---
        scale: Optional[str] = None
        scale_pixels_per_mm: Optional[float] = None
        try:
            scale, scale_pixels_per_mm = await loop.run_in_executor(
                None, self._detect_scale, img
            )
        except Exception as exc:
            logger.warning("Определение масштаба не удалось: %s", exc)

        # Кодируем результат в JPEG bytes
        try:
            result_bytes = await loop.run_in_executor(
                None, self._ndarray_to_jpeg_bytes, img
            )
        except Exception as exc:
            logger.exception("Ошибка кодирования результата: %s", exc)
            raise RuntimeError(f"Не удалось закодировать обработанное изображение: {exc}") from exc

        processed_h, processed_w = img.shape[:2]

        logger.info(
            "Обработка завершена: %dx%d → %dx%d, масштаб=%s, валидность=%s",
            original_w, original_h,
            processed_w, processed_h,
            scale, is_valid,
        )

        return ProcessedImage(
            image_bytes=result_bytes,
            scale=scale,
            scale_pixels_per_mm=scale_pixels_per_mm,
            width_px=processed_w,
            height_px=processed_h,
            original_width_px=original_w,
            original_height_px=original_h,
            is_valid=is_valid,
            quality_issues=quality_issues,
        )

    # ------------------------------------------------------------------
    # Вспомогательные методы кодирования / декодирования
    # ------------------------------------------------------------------

    @staticmethod
    def _bytes_to_ndarray(image_bytes: bytes) -> np.ndarray:
        """Конвертирует bytes в BGR numpy array через PIL."""
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(pil_image, dtype=np.uint8)
        # PIL даёт RGB, OpenCV работает с BGR
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _ndarray_to_jpeg_bytes(img: np.ndarray, quality: int = 92) -> bytes:
        """Кодирует BGR numpy array в JPEG bytes."""
        success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("cv2.imencode вернул False")
        return buf.tobytes()

    # ------------------------------------------------------------------
    # Основные шаги обработки
    # ------------------------------------------------------------------

    def _validate(self, img: np.ndarray) -> tuple[bool, list[str]]:
        """
        Проверяет качество и пригодность изображения.
        Возвращает (is_valid, список_проблем).
        """
        issues: list[str] = []
        h, w = img.shape[:2]

        # Проверка минимального разрешения
        if w < self.MIN_WIDTH or h < self.MIN_HEIGHT:
            issues.append(
                f"Низкое разрешение ({w}x{h}), минимум {self.MIN_WIDTH}x{self.MIN_HEIGHT}"
            )

        # Проверка яркости (слишком тёмное / слишком светлое)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
            if mean_brightness < 40:
                issues.append("Слишком тёмное изображение")
            elif mean_brightness > 220:
                issues.append("Слишком светлое изображение (возможно, пустой лист)")
        except Exception as exc:
            logger.warning("Не удалось проверить яркость: %s", exc)

        # Проверка наличия прямых линий (признак чертежа)
        has_lines = False
        try:
            gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray_blurred, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=80,
                minLineLength=max(50, min(w, h) // 10),
                maxLineGap=10,
            )
            if lines is not None and len(lines) >= 5:
                has_lines = True
        except Exception as exc:
            logger.warning("Не удалось проверить наличие линий: %s", exc)

        if not has_lines:
            issues.append("Не обнаружено характерных линий чертежа")

        is_valid = (
            w >= self.MIN_WIDTH
            and h >= self.MIN_HEIGHT
            and has_lines
        )

        return is_valid, issues

    def _correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """
        Выравнивает перспективу: ищет большой прямоугольный контур листа бумаги
        и применяет перспективное преобразование.
        Если лист не найден — возвращает исходное изображение.
        """
        h, w = img.shape[:2]

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blurred, 30, 100)

            # Дилатация для соединения разрывов контура
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=2)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return img

            # Ищем наибольший контур, похожий на прямоугольник
            best_quad = None
            best_area = 0
            min_area = w * h * 0.10  # контур должен занимать хотя бы 10% площади

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4 and area > best_area:
                    best_area = area
                    best_quad = approx

            if best_quad is None:
                logger.debug("Прямоугольный контур листа не найден, перспектива не корректируется")
                return img

            # Упорядочиваем углы: верхний-левый, верхний-правый, нижний-правый, нижний-левый
            pts = best_quad.reshape(4, 2).astype(np.float32)
            pts = self._order_points(pts)

            # Вычисляем размеры выходного изображения
            tl, tr, br, bl = pts
            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            height_left = np.linalg.norm(bl - tl)
            height_right = np.linalg.norm(br - tr)

            dst_w = int(max(width_top, width_bottom))
            dst_h = int(max(height_left, height_right))

            # Масштабируем по максимальной стороне
            long_side = max(dst_w, dst_h)
            if long_side > self.MAX_LONG_SIDE:
                scale_factor = self.MAX_LONG_SIDE / long_side
                dst_w = int(dst_w * scale_factor)
                dst_h = int(dst_h * scale_factor)

            dst_pts = np.array(
                [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
                dtype=np.float32,
            )

            M = cv2.getPerspectiveTransform(pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (dst_w, dst_h))

            logger.debug(
                "Перспектива скорректирована: %dx%d → %dx%d", w, h, dst_w, dst_h
            )
            return warped

        except Exception as exc:
            logger.warning("Ошибка коррекции перспективы: %s", exc)
            return img

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Упорядочивает 4 точки контура:
        [верхний-левый, верхний-правый, нижний-правый, нижний-левый].
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # верхний-левый (min x+y)
        rect[2] = pts[np.argmax(s)]   # нижний-правый (max x+y)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # верхний-правый (min y-x)
        rect[3] = pts[np.argmax(diff)]  # нижний-левый  (max y-x)
        return rect

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Улучшает контрастность изображения через CLAHE в LAB-пространстве.
        clipLimit=2.0, tileGridSize=(8,8).
        """
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)

            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            logger.debug("Контраст улучшен через CLAHE")
            return result

        except Exception as exc:
            logger.warning("Ошибка улучшения контраста: %s", exc)
            return img

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """
        Адаптивная бинаризация:
        - конвертация в grayscale
        - adaptiveThreshold (GAUSSIAN_C, blockSize=11, C=2)
        - морфологическое закрытие для удаления шума
        - возврат в BGR (3 канала)
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            binary = cv2.adaptiveThreshold(
                gray,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=11,
                C=2,
            )

            # Морфологическое закрытие для удаления мелкого шума
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Конвертируем обратно в BGR (3 канала)
            result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

            logger.debug("Бинаризация выполнена")
            return result

        except Exception as exc:
            logger.warning("Ошибка бинаризации: %s", exc)
            return img

    def _detect_scale(self, img: np.ndarray) -> tuple[Optional[str], Optional[float]]:
        """
        Определяет масштаб чертежа через анализ характерных регионов.

        Стратегия (без OCR):
        1. Анализируем нижний правый и нижний левый углы (область штампа по ГОСТ).
        2. В этих регионах ищем прямоугольные блоки текста через морфологические операции.
        3. В найденных блоках пробуем сопоставить пиксельный паттерн с шаблонами масштабов
           при помощи анализа контуров цифр (грубый подход без OCR).
        4. Если явный паттерн не найден — возвращаем (None, None).

        Примечание: точное распознавание текста возможно только с OCR (pytesseract).
        Данная реализация намеренно возвращает (None, None) в большинстве случаев,
        чтобы передать задачу AI-распознавателю на следующем этапе.
        """
        try:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Регионы вероятного расположения штампа (нижний правый / нижний левый)
            stamp_regions = [
                # нижний правый угол — наиболее распространённое место штампа
                gray[int(h * 0.75):h, int(w * 0.55):w],
                # нижний левый угол
                gray[int(h * 0.75):h, 0:int(w * 0.45)],
                # весь нижний пояс
                gray[int(h * 0.88):h, :],
            ]

            for region in stamp_regions:
                result = self._search_scale_in_region(region)
                if result is not None:
                    scale_str, denominator = result
                    # Вычисляем pixels/mm на основе ширины штампа
                    # Ширина штампа по ГОСТ 2.104 — 185 мм
                    region_w = region.shape[1]
                    # Грубая оценка: весь регион ≈ ширина штампа
                    pixels_per_mm = region_w / self.STAMP_WIDTH_MM if region_w > 0 else None
                    logger.info("Масштаб обнаружен: %s", scale_str)
                    return scale_str, pixels_per_mm

        except Exception as exc:
            logger.warning("Ошибка определения масштаба: %s", exc)

        return None, None

    def _search_scale_in_region(
        self, region: np.ndarray
    ) -> Optional[tuple[str, int]]:
        """
        Ищет паттерн масштаба в заданном регионе через анализ текстовых блоков.
        Возвращает (scale_str, denominator) или None.

        Использует морфологическое выделение текстовых строк и анализ
        горизонтальных сегментов для поиска характерного соотношения "1:N".
        """
        try:
            if region.size == 0:
                return None

            rh, rw = region.shape[:2]

            # Порог с инверсией (текст — тёмный на светлом фоне)
            _, thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Морфологическое расширение для соединения символов в блоки
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(rw * 0.04) + 3, 3))
            connected = cv2.dilate(thresh, kernel_h, iterations=2)

            contours, _ = cv2.findContours(
                connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Перебираем текстовые блоки по убыванию площади
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                x, y, bw, bh = cv2.boundingRect(cnt)

                # Отфильтровываем слишком маленькие и слишком большие блоки
                if bw < rw * 0.05 or bh < 4 or bw > rw * 0.95:
                    continue
                if bh > rh * 0.5:
                    continue

                # Извлекаем блок с небольшим запасом
                pad = 2
                y1 = max(0, y - pad)
                y2 = min(rh, y + bh + pad)
                x1 = max(0, x - pad)
                x2 = min(rw, x + bw + pad)
                block = thresh[y1:y2, x1:x2]

                # Анализируем горизонтальный профиль блока для поиска "1:N"
                scale_result = self._analyze_block_for_scale(block, bw, bh)
                if scale_result is not None:
                    return scale_result

        except Exception as exc:
            logger.debug("Ошибка поиска масштаба в регионе: %s", exc)

        return None

    def _analyze_block_for_scale(
        self, block: np.ndarray, bw: int, bh: int
    ) -> Optional[tuple[str, int]]:
        """
        Анализирует бинарный блок на предмет паттерна масштаба "1:N".

        Логика: ищет характерное распределение вертикальных полос заполнения,
        типичное для записи "1:100", "1:50" и т.д.
        Это приближённый эвристический метод.
        """
        try:
            if block.size == 0 or bw < 10 or bh < 5:
                return None

            # Вертикальный профиль заполнения (сколько белых пикселей в каждом столбце)
            col_profile = block.sum(axis=0) / 255.0
            total_cols = len(col_profile)

            if total_cols < 10:
                return None

            # Нормализуем профиль
            max_val = col_profile.max()
            if max_val == 0:
                return None
            norm_profile = col_profile / max_val

            # Ищем характерную структуру: высокий пик (цифра "1"), провал (пробел),
            # высокий пик (двоеточие ":"), провал, серия пиков (число N)
            # Упрощённо: ищем наличие хотя бы 3 отдельных группы заполненных столбцов
            threshold_fill = 0.2
            in_group = False
            groups: list[tuple[int, int]] = []
            group_start = 0

            for i, val in enumerate(norm_profile):
                if val > threshold_fill and not in_group:
                    in_group = True
                    group_start = i
                elif val <= threshold_fill and in_group:
                    in_group = False
                    groups.append((group_start, i - 1))

            if in_group:
                groups.append((group_start, total_cols - 1))

            # Ожидаем минимум 3 группы для паттерна "1 : N"
            if len(groups) < 3:
                return None

            # Оцениваем соотношение ширин групп для определения числа после ":"
            # Первая группа — "1" (узкая), вторая — ":" (очень узкая), остальные — число
            first_w = groups[0][1] - groups[0][0] + 1
            last_groups_w = sum(g[1] - g[0] + 1 for g in groups[2:])

            if first_w == 0:
                return None

            ratio = last_groups_w / first_w

            # Сопоставляем соотношение с известными масштабами
            # "50" ≈ 2 символа, "100" ≈ 3 символа → ширина ~в 2-3 раза больше "1"
            candidate_denominators = [25, 50, 100, 200, 250, 500, 1000, 2000, 5000]

            # Грубая эвристика по длине числа
            if ratio < 1.5:
                candidates = [25, 50]
            elif ratio < 2.5:
                candidates = [50, 100]
            elif ratio < 3.5:
                candidates = [100, 200, 250]
            elif ratio < 5.0:
                candidates = [500, 200, 250]
            else:
                candidates = [1000, 2000, 5000]

            if not candidates:
                return None

            # Выбираем наиболее вероятный масштаб (первый кандидат)
            denominator = candidates[0]
            scale_str = f"1:{denominator}"
            return scale_str, denominator

        except Exception as exc:
            logger.debug("Ошибка анализа блока: %s", exc)

        return None

    # ------------------------------------------------------------------
    # Вспомогательный метод для кропов
    # ------------------------------------------------------------------

    async def extract_crop(self, image_bytes: bytes, bbox: dict) -> bytes:
        """
        Вырезает прямоугольный участок изображения с padding 20px.

        bbox: {"x": int, "y": int, "width": int, "height": int} — в пикселях.
        Возвращает bytes (JPEG).
        """
        loop = asyncio.get_event_loop()

        try:
            img = await loop.run_in_executor(None, self._bytes_to_ndarray, image_bytes)
        except Exception as exc:
            logger.exception("extract_crop: ошибка декодирования: %s", exc)
            raise ValueError(f"Не удалось декодировать изображение: {exc}") from exc

        try:
            h, w = img.shape[:2]
            pad = self.CROP_PADDING

            x = int(bbox.get("x", 0))
            y = int(bbox.get("y", 0))
            bw = int(bbox.get("width", w))
            bh = int(bbox.get("height", h))

            # Применяем padding с ограничением границ изображения
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                raise ValueError("Пустой кроп после применения bbox")

            result_bytes = await loop.run_in_executor(
                None, self._ndarray_to_jpeg_bytes, crop
            )
            logger.debug("Кроп извлечён: bbox=%s, результат %dx%d", bbox, x2 - x1, y2 - y1)
            return result_bytes

        except Exception as exc:
            logger.exception("extract_crop: ошибка извлечения кропа: %s", exc)
            raise RuntimeError(f"Не удалось извлечь кроп: {exc}") from exc


# Глобальный экземпляр
image_processor = ImageProcessor()

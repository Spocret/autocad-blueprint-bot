import asyncio
import logging
import math
import os
from typing import Optional

import ezdxf
from ezdxf import colors
from ezdxf.enums import TextEntityAlignment

logger = logging.getLogger(__name__)

# Имена слоёв
LAYER_WALLS_LOAD = "WALLS_LOAD"
LAYER_WALLS_PART = "WALLS_PART"
LAYER_DOORS = "DOORS"
LAYER_WINDOWS = "WINDOWS"
LAYER_DIMENSIONS = "DIMENSIONS"
LAYER_TEXT = "TEXT"
LAYER_STAIRS = "STAIRS"

# Высота текста в мм (в пространстве модели)
TEXT_HEIGHT_MM = 200.0


class DXFGenerator:
    """Генератор DXF чертежей из распознанных данных планировки."""

    async def generate(self, data: dict, output_path: str) -> str:
        """Сгенерировать DXF файл из данных распознавания и сохранить по указанному пути."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._generate_sync, data, output_path)
            return result
        except Exception as e:
            logger.error(f"Ошибка генерации DXF: {e}", exc_info=True)
            raise

    def _generate_sync(self, data: dict, output_path: str) -> str:
        """Синхронная генерация DXF документа."""
        try:
            doc = ezdxf.new(dxfversion="R2010")
            doc.units = ezdxf.units.MM

            self._setup_layers(doc)

            msp = doc.modelspace()

            scale_str = data.get("scale", "1:100")
            scale = self._get_scale_factor(scale_str)
            logger.info(f"Масштаб чертежа: {scale_str}, коэффициент = {scale}")

            walls = data.get("walls", [])
            if walls:
                self._draw_walls(msp, walls, scale)
                logger.info(f"Нарисовано стен: {len(walls)}")

            doors = data.get("doors", [])
            if doors:
                self._draw_doors(msp, doors, scale)
                logger.info(f"Нарисовано дверей: {len(doors)}")

            windows = data.get("windows", [])
            if windows:
                self._draw_windows(msp, windows, scale)
                logger.info(f"Нарисовано окон: {len(windows)}")

            dimensions = data.get("dimensions", [])
            if dimensions:
                self._draw_dimensions(msp, dimensions, scale)
                logger.info(f"Нарисовано размерных линий: {len(dimensions)}")

            stairs = data.get("stairs", [])
            if stairs:
                self._draw_stairs(msp, stairs, scale)
                logger.info(f"Нарисовано лестниц: {len(stairs)}")

            labels = data.get("labels", [])
            if labels:
                self._draw_labels(msp, labels, scale)
                logger.info(f"Нарисовано подписей: {len(labels)}")

            rooms = data.get("rooms", [])
            if rooms:
                self._draw_rooms(msp, rooms, scale)
                logger.info(f"Нарисовано помещений: {len(rooms)}")

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            doc.saveas(output_path)
            logger.info(f"DXF сохранён: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Ошибка в _generate_sync: {e}", exc_info=True)
            raise

    def _setup_layers(self, doc) -> None:
        """Создать все слои с нужными цветами и толщинами линий."""
        try:
            layers_config = [
                # (имя, цвет ACI, lineweight в сотых мм)
                (LAYER_WALLS_LOAD, 7,  80),
                (LAYER_WALLS_PART, 8,  35),
                (LAYER_DOORS,      2,  25),
                (LAYER_WINDOWS,    5,  25),
                (LAYER_DIMENSIONS, 3,  18),
                (LAYER_TEXT,       7,  18),
                (LAYER_STAIRS,     6,  25),
            ]
            for name, color, lw in layers_config:
                layer = doc.layers.new(name=name)
                layer.color = color
                layer.lineweight = lw
            logger.debug("Слои DXF созданы")
        except Exception as e:
            logger.error(f"Ошибка создания слоёв: {e}", exc_info=True)
            raise

    def _get_scale_factor(self, scale_str: str) -> float:
        """Вернуть числовой коэффициент из строки масштаба вида '1:100'."""
        try:
            parts = str(scale_str).strip().split(":")
            if len(parts) == 2:
                numerator = float(parts[0])
                denominator = float(parts[1])
                if numerator > 0:
                    return denominator / numerator
            logger.warning(f"Неизвестный формат масштаба '{scale_str}', используется 1.0")
            return 1.0
        except Exception as e:
            logger.error(f"Ошибка парсинга масштаба '{scale_str}': {e}")
            return 1.0

    # ------------------------------------------------------------------
    # Стены
    # ------------------------------------------------------------------

    def _draw_walls(self, msp, walls: list, scale: float) -> None:
        """Нарисовать стены как прямоугольные LWPOLYLINE с учётом толщины."""
        for wall in walls:
            try:
                x1 = wall["x1"] * scale
                y1 = wall["y1"] * scale
                x2 = wall["x2"] * scale
                y2 = wall["y2"] * scale
                thickness = wall.get("thickness", 200) * scale / scale  # уже в пикселях → мм через scale

                # Перевод пиксельной толщины в мм: thickness в данных в мм (относительно реального масштаба)
                # thickness в JSON задана в мм реального размера, поэтому не умножаем на scale
                half_t = wall.get("thickness", 200) / 2.0

                wall_type = wall.get("type", "partition")
                layer = LAYER_WALLS_LOAD if wall_type == "load_bearing" else LAYER_WALLS_PART

                # Направление стены
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length == 0:
                    continue

                # Единичный вектор перпендикуляра
                nx = -dy / length
                ny = dx / length

                # Угловые точки прямоугольника стены
                points = [
                    (x1 + nx * half_t, y1 + ny * half_t),
                    (x2 + nx * half_t, y2 + ny * half_t),
                    (x2 - nx * half_t, y2 - ny * half_t),
                    (x1 - nx * half_t, y1 - ny * half_t),
                ]

                polyline = msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})
                _ = polyline  # явно не используем, но линия добавлена в msp

            except Exception as e:
                logger.warning(f"Ошибка рисования стены {wall.get('id', '?')}: {e}")

    # ------------------------------------------------------------------
    # Двери
    # ------------------------------------------------------------------

    def _draw_doors(self, msp, doors: list, scale: float) -> None:
        """Нарисовать двери: полотно (LINE) + дуга открывания (ARC)."""
        for door in doors:
            try:
                x = door["x"] * scale
                y = door["y"] * scale
                width = door.get("width", 900)          # ширина в мм (реальная)
                angle_deg = door.get("angle", 0)        # угол поворота проёма
                swing = door.get("swing_direction", "left")

                # Полотно двери — горизонтальная линия от точки проёма
                angle_rad = math.radians(angle_deg)
                end_x = x + width * math.cos(angle_rad)
                end_y = y + width * math.sin(angle_rad)

                msp.add_line(
                    start=(x, y),
                    end=(end_x, end_y),
                    dxfattribs={"layer": LAYER_DOORS},
                )

                # Дуга открывания: центр — начало полотна, радиус = ширина двери
                if swing == "left":
                    arc_start = angle_deg
                    arc_end = angle_deg + 90
                else:
                    arc_start = angle_deg - 90
                    arc_end = angle_deg

                msp.add_arc(
                    center=(x, y),
                    radius=width,
                    start_angle=arc_start,
                    end_angle=arc_end,
                    dxfattribs={"layer": LAYER_DOORS},
                )

            except Exception as e:
                logger.warning(f"Ошибка рисования двери {door.get('id', '?')}: {e}")

    # ------------------------------------------------------------------
    # Окна
    # ------------------------------------------------------------------

    def _draw_windows(self, msp, windows: list, scale: float) -> None:
        """Нарисовать окна тремя параллельными линиями."""
        for window in windows:
            try:
                x = window["x"] * scale
                y = window["y"] * scale
                width = window.get("width", 1500)       # ширина в мм (реальная)

                # Глубина условного обозначения окна — 100 мм
                depth = 100.0

                # Три горизонтальные линии: снаружи, посередине, изнутри
                for offset in (0.0, depth / 2.0, depth):
                    msp.add_line(
                        start=(x, y + offset),
                        end=(x + width, y + offset),
                        dxfattribs={"layer": LAYER_WINDOWS},
                    )

                # Боковые ограничения проёма
                msp.add_line(start=(x, y), end=(x, y + depth), dxfattribs={"layer": LAYER_WINDOWS})
                msp.add_line(start=(x + width, y), end=(x + width, y + depth), dxfattribs={"layer": LAYER_WINDOWS})

            except Exception as e:
                logger.warning(f"Ошибка рисования окна {window.get('id', '?')}: {e}")

    # ------------------------------------------------------------------
    # Размерные линии
    # ------------------------------------------------------------------

    def _draw_dimensions(self, msp, dimensions: list, scale: float) -> None:
        """Нарисовать линейные размеры через DIMLINEAR."""
        for dim in dimensions:
            try:
                x1 = dim["x1"] * scale
                y1 = dim["y1"] * scale
                x2 = dim["x2"] * scale
                y2 = dim["y2"] * scale

                # Точка размерной линии (смещение от базовой)
                # Берём середину между y1 и y2, слегка смещённую
                mid_y = (y1 + y2) / 2.0
                defpoint = (x2, mid_y)

                dim_entity = msp.add_linear_dim(
                    base=(x1, y1),          # начало выносной линии 1
                    p1=(x1, y1),
                    p2=(x2, y2),
                    dxfattribs={"layer": LAYER_DIMENSIONS},
                )
                dim_entity.render()

            except Exception as e:
                logger.warning(f"Ошибка рисования размера {dim.get('id', '?')}: {e}")

    # ------------------------------------------------------------------
    # Лестницы
    # ------------------------------------------------------------------

    def _draw_stairs(self, msp, stairs: list, scale: float) -> None:
        """Нарисовать лестницы горизонтальными ступенями."""
        for stair in stairs:
            try:
                x = stair["x"] * scale
                y = stair["y"] * scale
                width = stair.get("width", 1200)        # ширина в мм
                height = stair.get("height", 2400)      # длина марша в мм
                steps = stair.get("steps_count", 12)
                direction = stair.get("direction", "up")

                if steps <= 0:
                    steps = 1

                step_height = height / steps

                # Контур лестничного марша
                msp.add_lwpolyline(
                    [(x, y), (x + width, y), (x + width, y + height), (x, y + height)],
                    close=True,
                    dxfattribs={"layer": LAYER_STAIRS},
                )

                # Горизонтальные линии ступеней
                for i in range(1, steps):
                    step_y = y + i * step_height
                    msp.add_line(
                        start=(x, step_y),
                        end=(x + width, step_y),
                        dxfattribs={"layer": LAYER_STAIRS},
                    )

                # Стрелка направления движения (диагональная линия)
                arrow_x = x + width / 2.0
                if direction == "up":
                    msp.add_line(
                        start=(arrow_x, y + step_height),
                        end=(arrow_x, y + height - step_height),
                        dxfattribs={"layer": LAYER_STAIRS},
                    )
                else:
                    msp.add_line(
                        start=(arrow_x, y + height - step_height),
                        end=(arrow_x, y + step_height),
                        dxfattribs={"layer": LAYER_STAIRS},
                    )

            except Exception as e:
                logger.warning(f"Ошибка рисования лестницы {stair.get('id', '?')}: {e}")

    # ------------------------------------------------------------------
    # Подписи
    # ------------------------------------------------------------------

    def _draw_labels(self, msp, labels: list, scale: float) -> None:
        """Нарисовать текстовые подписи через MTEXT."""
        for label in labels:
            try:
                x = label["x"] * scale
                y = label["y"] * scale
                text = str(label.get("text", ""))

                msp.add_mtext(
                    text,
                    dxfattribs={
                        "layer": LAYER_TEXT,
                        "char_height": TEXT_HEIGHT_MM,
                        "insert": (x, y),
                    },
                )

            except Exception as e:
                logger.warning(f"Ошибка рисования подписи {label.get('id', '?')}: {e}")

    # ------------------------------------------------------------------
    # Помещения
    # ------------------------------------------------------------------

    def _draw_rooms(self, msp, rooms: list, scale: float) -> None:
        """Нарисовать нумерацию и названия помещений через MTEXT."""
        for room in rooms:
            try:
                cx = room.get("center_x", 0) * scale
                cy = room.get("center_y", 0) * scale
                name = room.get("name", "")
                area_raw = room.get("area")
                area = float(area_raw) if area_raw is not None else 0.0

                # Формат: «Кухня\n12.5 м²»
                text = f"{name}\\P{area:.1f} м²" if name else f"{area:.1f} м²"

                msp.add_mtext(
                    text,
                    dxfattribs={
                        "layer": LAYER_TEXT,
                        "char_height": TEXT_HEIGHT_MM,
                        "insert": (cx, cy),
                    },
                )

            except Exception as e:
                logger.warning(f"Ошибка рисования помещения {room.get('id', '?')}: {e}")


dxf_generator = DXFGenerator()

"""
Сервис генерации SVG чертежей из данных AI-распознавания.
"""

import asyncio
import logging
import math
import os
from datetime import datetime

import svgwrite

logger = logging.getLogger(__name__)


class SVGGenerator:
    """Генератор SVG-чертежей из структурированных данных."""

    SVG_WIDTH = 1000
    SVG_HEIGHT = 700

    # Стили стен
    WALL_LOAD_BEARING_STROKE = "#000000"
    WALL_LOAD_BEARING_FILL = "#000000"
    WALL_LOAD_BEARING_WIDTH = "0.8px"

    WALL_PARTITION_STROKE = "#666666"
    WALL_PARTITION_FILL = "#666666"
    WALL_PARTITION_WIDTH = "0.4px"

    # Стили размерных линий
    DIM_STROKE = "#000000"
    DIM_WIDTH = "0.25px"

    # Элементы с низкой уверенностью
    LOW_CONF_COLOR = "#ff0000"

    # Шрифт
    FONT_FAMILY = "Arial"
    FONT_SIZE = 12

    # Штамп
    STAMP_WIDTH = 185
    STAMP_HEIGHT = 55

    async def generate(self, data: dict, output_path: str) -> tuple[str, str]:
        """
        Сгенерировать SVG из данных AI-распознавания.

        Возвращает кортеж (svg_content, output_path).
        """
        try:
            logger.info("Начало генерации SVG, output_path=%s", output_path)

            scale_str = data.get("scale", "1:100")
            rooms = data.get("rooms", [])
            walls = data.get("walls", [])
            doors = data.get("doors", [])
            windows = data.get("windows", [])
            dimensions = data.get("dimensions", [])
            stairs = data.get("stairs", [])
            labels = data.get("labels", [])
            low_conf_elements = data.get("low_confidence_elements", [])

            # Идентификаторы элементов с низкой уверенностью
            low_conf_ids: set = {
                el.get("element_id", "") for el in low_conf_elements
            }

            w = self.SVG_WIDTH
            h = self.SVG_HEIGHT

            dwg = svgwrite.Drawing(
                filename=output_path,
                size=(f"{w}px", f"{h}px"),
                profile="full",
            )

            # Белый фон
            dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="#ffffff"))

            # Маркер стрелки для размерных линий
            self._add_arrow_marker(dwg)

            # --- Слои ---
            g_walls_lb = dwg.g(id="walls_load_bearing")
            g_walls_pt = dwg.g(id="walls_partitions")
            g_doors = dwg.g(id="doors")
            g_windows = dwg.g(id="windows")
            g_stairs = dwg.g(id="stairs")
            g_dimensions = dwg.g(id="dimensions")
            g_labels = dwg.g(id="labels")
            g_rooms = dwg.g(id="rooms")
            g_frame = dwg.g(id="frame")

            # Рисуем каждый слой
            self._draw_walls(dwg, g_walls_lb, g_walls_pt, walls)
            self._draw_doors(dwg, g_doors, doors)
            self._draw_windows(dwg, g_windows, windows)
            self._draw_stairs(dwg, g_stairs, stairs)
            self._draw_dimensions(dwg, g_dimensions, dimensions)
            self._draw_labels(dwg, g_labels, labels, low_conf_ids)
            self._draw_rooms(dwg, g_rooms, rooms)
            self._draw_frame(dwg, g_frame, scale_str, w, h)

            # Добавляем слои в документ в нужном порядке
            dwg.add(g_walls_lb)
            dwg.add(g_walls_pt)
            dwg.add(g_doors)
            dwg.add(g_windows)
            dwg.add(g_stairs)
            dwg.add(g_dimensions)
            dwg.add(g_labels)
            dwg.add(g_rooms)
            dwg.add(g_frame)

            # Сохраняем файл
            await asyncio.to_thread(self._save_drawing, dwg, output_path)

            svg_content = dwg.tostring()
            logger.info("SVG успешно сгенерирован, размер=%d байт", len(svg_content))
            return svg_content, output_path

        except Exception as exc:
            logger.exception("Ошибка генерации SVG: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Вспомогательный метод сохранения (вызывается в потоке)
    # ------------------------------------------------------------------

    def _save_drawing(self, dwg: svgwrite.Drawing, output_path: str) -> None:
        """Сохранить SVG-документ на диск."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dwg.saveas(output_path)
            logger.debug("SVG сохранён: %s", output_path)
        except Exception as exc:
            logger.exception("Ошибка сохранения SVG: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Маркер стрелки
    # ------------------------------------------------------------------

    def _add_arrow_marker(self, dwg: svgwrite.Drawing) -> None:
        """Добавить маркер стрелки для размерных линий (в обоих направлениях)."""
        try:
            defs = dwg.defs

            # Стрелка «вперёд»
            marker = dwg.marker(
                id="arrow",
                insert=(10, 5),
                size=(10, 10),
                orient="auto",
                markerUnits="strokeWidth",
            )
            marker.add(
                dwg.path(d="M 0 0 L 10 5 L 0 10 z", fill=self.DIM_STROKE)
            )
            defs.add(marker)

            # Стрелка «назад»
            marker_rev = dwg.marker(
                id="arrow_rev",
                insert=(0, 5),
                size=(10, 10),
                orient="auto",
                markerUnits="strokeWidth",
            )
            marker_rev.add(
                dwg.path(d="M 10 0 L 0 5 L 10 10 z", fill=self.DIM_STROKE)
            )
            defs.add(marker_rev)

        except Exception as exc:
            logger.warning("Не удалось добавить маркер стрелки: %s", exc)

    # ------------------------------------------------------------------
    # Стены
    # ------------------------------------------------------------------

    def _draw_walls(
        self,
        dwg: svgwrite.Drawing,
        group_lb: svgwrite.container.Group,
        group_pt: svgwrite.container.Group,
        walls: list,
    ) -> None:
        """Нарисовать стены: несущие и перегородки."""
        try:
            for wall in walls:
                try:
                    wid = wall.get("id", "")
                    wall_type = wall.get("type", "partition")
                    x1 = float(wall.get("x1", 0))
                    y1 = float(wall.get("y1", 0))
                    x2 = float(wall.get("x2", 0))
                    y2 = float(wall.get("y2", 0))
                    thickness_mm = float(wall.get("thickness", 150))
                    confidence = float(wall.get("confidence", 1.0))

                    # Толщина стены в пикселях (масштаб ~1px = 10mm)
                    thickness_px = max(thickness_mm / 10.0, 1.5)

                    is_load_bearing = wall_type == "load_bearing"

                    stroke_color = (
                        self.WALL_LOAD_BEARING_STROKE
                        if is_load_bearing
                        else self.WALL_PARTITION_STROKE
                    )
                    stroke_width = (
                        self.WALL_LOAD_BEARING_WIDTH
                        if is_load_bearing
                        else self.WALL_PARTITION_WIDTH
                    )
                    fill_color = (
                        self.WALL_LOAD_BEARING_FILL
                        if is_load_bearing
                        else self.WALL_PARTITION_FILL
                    )

                    # Низкая уверенность → красный контур
                    if confidence < 0.7:
                        stroke_color = self.LOW_CONF_COLOR

                    line = dwg.line(
                        start=(x1, y1),
                        end=(x2, y2),
                        stroke=stroke_color,
                        stroke_width=thickness_px,
                        stroke_linecap="square",
                        id=wid,
                    )

                    if is_load_bearing:
                        group_lb.add(line)
                    else:
                        group_pt.add(line)

                except Exception as exc:
                    logger.warning("Ошибка отрисовки стены %s: %s", wall.get("id"), exc)

        except Exception as exc:
            logger.exception("Ошибка в _draw_walls: %s", exc)

    # ------------------------------------------------------------------
    # Двери
    # ------------------------------------------------------------------

    def _draw_doors(
        self, dwg: svgwrite.Drawing, group: svgwrite.container.Group, doors: list
    ) -> None:
        """Нарисовать двери: полотно двери + дуга открывания."""
        try:
            for door in doors:
                try:
                    did = door.get("id", "")
                    x = float(door.get("x", 0))
                    y = float(door.get("y", 0))
                    width_mm = float(door.get("width", 900))
                    angle_deg = float(door.get("angle", 0))
                    swing = door.get("swing_direction", "left")
                    confidence = float(door.get("confidence", 1.0))

                    # Ширина двери в пикселях
                    door_w = width_mm / 10.0

                    stroke = "#333333" if confidence >= 0.7 else self.LOW_CONF_COLOR

                    # Группа с трансформацией поворота
                    g = dwg.g(
                        id=did,
                        transform=f"rotate({angle_deg},{x},{y})",
                    )

                    # Линия полотна двери
                    g.add(
                        dwg.line(
                            start=(x, y),
                            end=(x + door_w, y),
                            stroke=stroke,
                            stroke_width="1px",
                            stroke_linecap="round",
                        )
                    )

                    # Дуга открывания
                    # Для левой навески: дуга от (x, y) по часовой
                    # Для правой: от (x + door_w, y) против часовой
                    if swing == "left":
                        arc_cx, arc_cy = x, y
                        arc_x_end = x
                        arc_y_end = y + door_w
                        sweep = 0
                    else:
                        arc_cx, arc_cy = x + door_w, y
                        arc_x_end = x + door_w
                        arc_y_end = y + door_w
                        sweep = 1

                    d = (
                        f"M {arc_cx} {arc_cy} "
                        f"A {door_w} {door_w} 0 0 {sweep} {arc_x_end} {arc_y_end}"
                    )
                    g.add(
                        dwg.path(
                            d=d,
                            fill="none",
                            stroke=stroke,
                            stroke_width="0.5px",
                            stroke_dasharray="4,2",
                        )
                    )

                    group.add(g)

                except Exception as exc:
                    logger.warning("Ошибка отрисовки двери %s: %s", door.get("id"), exc)

        except Exception as exc:
            logger.exception("Ошибка в _draw_doors: %s", exc)

    # ------------------------------------------------------------------
    # Окна
    # ------------------------------------------------------------------

    def _draw_windows(
        self, dwg: svgwrite.Drawing, group: svgwrite.container.Group, windows: list
    ) -> None:
        """Нарисовать окна: три параллельные линии в проёме."""
        try:
            for window in windows:
                try:
                    wid = window.get("id", "")
                    x = float(window.get("x", 0))
                    y = float(window.get("y", 0))
                    width_mm = float(window.get("width", 1500))
                    confidence = float(window.get("confidence", 1.0))

                    win_w = width_mm / 10.0
                    stroke = "#0000aa" if confidence >= 0.7 else self.LOW_CONF_COLOR

                    g = dwg.g(id=wid)

                    # Три параллельные горизонтальные линии
                    offsets = [-3, 0, 3]
                    for off in offsets:
                        g.add(
                            dwg.line(
                                start=(x, y + off),
                                end=(x + win_w, y + off),
                                stroke=stroke,
                                stroke_width="0.8px",
                            )
                        )

                    group.add(g)

                except Exception as exc:
                    logger.warning("Ошибка отрисовки окна %s: %s", window.get("id"), exc)

        except Exception as exc:
            logger.exception("Ошибка в _draw_windows: %s", exc)

    # ------------------------------------------------------------------
    # Лестницы
    # ------------------------------------------------------------------

    def _draw_stairs(
        self, dwg: svgwrite.Drawing, group: svgwrite.container.Group, stairs: list
    ) -> None:
        """Нарисовать лестницы: ступени + стрелка направления."""
        try:
            for stair in stairs:
                try:
                    sid = stair.get("id", "")
                    x = float(stair.get("x", 0))
                    y = float(stair.get("y", 0))
                    width_mm = float(stair.get("width", 1200))
                    height_mm = float(stair.get("height", 2400))
                    steps_count = int(stair.get("steps_count", 10))
                    direction = stair.get("direction", "up")
                    confidence = float(stair.get("confidence", 1.0))

                    stair_w = width_mm / 10.0
                    stair_h = height_mm / 10.0

                    stroke = "#555555" if confidence >= 0.7 else self.LOW_CONF_COLOR

                    g = dwg.g(id=sid)

                    # Внешний контур лестничного блока
                    g.add(
                        dwg.rect(
                            insert=(x, y),
                            size=(stair_w, stair_h),
                            fill="none",
                            stroke=stroke,
                            stroke_width="0.8px",
                        )
                    )

                    # Горизонтальные линии ступеней
                    if steps_count > 0:
                        step_h = stair_h / steps_count
                        for i in range(1, steps_count):
                            sy = y + i * step_h
                            g.add(
                                dwg.line(
                                    start=(x, sy),
                                    end=(x + stair_w, sy),
                                    stroke=stroke,
                                    stroke_width="0.5px",
                                )
                            )

                    # Стрелка направления движения
                    mid_x = x + stair_w / 2
                    if direction == "up":
                        arrow_y1 = y + stair_h - 5
                        arrow_y2 = y + 5
                    else:
                        arrow_y1 = y + 5
                        arrow_y2 = y + stair_h - 5

                    g.add(
                        dwg.line(
                            start=(mid_x, arrow_y1),
                            end=(mid_x, arrow_y2),
                            stroke="#0000aa",
                            stroke_width="1px",
                            **{"marker-end": "url(#arrow)"},
                        )
                    )

                    group.add(g)

                except Exception as exc:
                    logger.warning(
                        "Ошибка отрисовки лестницы %s: %s", stair.get("id"), exc
                    )

        except Exception as exc:
            logger.exception("Ошибка в _draw_stairs: %s", exc)

    # ------------------------------------------------------------------
    # Размерные линии
    # ------------------------------------------------------------------

    def _draw_dimensions(
        self, dwg: svgwrite.Drawing, group: svgwrite.container.Group, dimensions: list
    ) -> None:
        """Нарисовать размерные линии со стрелками и текстом."""
        try:
            for dim in dimensions:
                try:
                    did = dim.get("id", "")
                    value = dim.get("value", "")
                    unit = dim.get("unit", "mm")
                    x1 = float(dim.get("x1", 0))
                    y1 = float(dim.get("y1", 0))
                    x2 = float(dim.get("x2", 0))
                    y2 = float(dim.get("y2", 0))
                    confidence = float(dim.get("confidence", 1.0))

                    stroke = self.DIM_STROKE if confidence >= 0.7 else self.LOW_CONF_COLOR

                    g = dwg.g(id=did)

                    # Основная линия со стрелками на концах
                    g.add(
                        dwg.line(
                            start=(x1, y1),
                            end=(x2, y2),
                            stroke=stroke,
                            stroke_width=self.DIM_WIDTH,
                            **{
                                "marker-start": "url(#arrow_rev)",
                                "marker-end": "url(#arrow)",
                            },
                        )
                    )

                    # Выносные линии (перпендикуляры в точках)
                    dx = x2 - x1
                    dy = y2 - y1
                    length = math.hypot(dx, dy)
                    if length > 0:
                        # Единичный перпендикуляр
                        perp_x = -dy / length * 8
                        perp_y = dx / length * 8

                        for px, py in [(x1, y1), (x2, y2)]:
                            g.add(
                                dwg.line(
                                    start=(px - perp_x, py - perp_y),
                                    end=(px + perp_x, py + perp_y),
                                    stroke=stroke,
                                    stroke_width=self.DIM_WIDTH,
                                )
                            )

                    # Текст значения над линией
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2

                    # Смещение текста перпендикулярно линии
                    if length > 0:
                        text_offset_x = -dy / length * 10
                        text_offset_y = dx / length * 10
                    else:
                        text_offset_x, text_offset_y = 0, -10

                    label = f"{value} {unit}" if unit else str(value)

                    g.add(
                        dwg.text(
                            label,
                            insert=(mid_x + text_offset_x, mid_y + text_offset_y),
                            fill=stroke,
                            font_family=self.FONT_FAMILY,
                            font_size=f"{self.FONT_SIZE * 0.8}px",
                            text_anchor="middle",
                            dominant_baseline="middle",
                        )
                    )

                    group.add(g)

                except Exception as exc:
                    logger.warning(
                        "Ошибка отрисовки размера %s: %s", dim.get("id"), exc
                    )

        except Exception as exc:
            logger.exception("Ошибка в _draw_dimensions: %s", exc)

    # ------------------------------------------------------------------
    # Подписи
    # ------------------------------------------------------------------

    def _draw_labels(
        self,
        dwg: svgwrite.Drawing,
        group: svgwrite.container.Group,
        labels: list,
        low_conf_ids: set,
    ) -> None:
        """Нарисовать текстовые подписи; красным — если низкая уверенность."""
        try:
            for label in labels:
                try:
                    lid = label.get("id", "")
                    text = label.get("text", "")
                    x = float(label.get("x", 0))
                    y = float(label.get("y", 0))
                    confidence = float(label.get("confidence", 1.0))

                    is_low_conf = lid in low_conf_ids or confidence < 0.7
                    fill = self.LOW_CONF_COLOR if is_low_conf else "#000000"

                    t = dwg.text(
                        text,
                        insert=(x, y),
                        fill=fill,
                        font_family=self.FONT_FAMILY,
                        font_size=f"{self.FONT_SIZE}px",
                        id=lid,
                    )
                    group.add(t)

                except Exception as exc:
                    logger.warning(
                        "Ошибка отрисовки подписи %s: %s", label.get("id"), exc
                    )

        except Exception as exc:
            logger.exception("Ошибка в _draw_labels: %s", exc)

    # ------------------------------------------------------------------
    # Нумерация помещений
    # ------------------------------------------------------------------

    def _draw_rooms(
        self, dwg: svgwrite.Drawing, group: svgwrite.container.Group, rooms: list
    ) -> None:
        """Нарисовать нумерацию помещений: имя + площадь по центру."""
        try:
            for room in rooms:
                try:
                    rid = room.get("id", "")
                    name = room.get("name", "")
                    area = room.get("area", "")
                    cx = float(room.get("center_x", 0))
                    cy = float(room.get("center_y", 0))

                    g = dwg.g(id=rid)

                    # Имя помещения
                    g.add(
                        dwg.text(
                            name,
                            insert=(cx, cy - 8),
                            fill="#000000",
                            font_family=self.FONT_FAMILY,
                            font_size=f"{self.FONT_SIZE}px",
                            font_weight="bold",
                            text_anchor="middle",
                            dominant_baseline="middle",
                        )
                    )

                    # Площадь помещения
                    area_text = f"{area} м²" if area != "" else ""
                    if area_text:
                        g.add(
                            dwg.text(
                                area_text,
                                insert=(cx, cy + 8),
                                fill="#444444",
                                font_family=self.FONT_FAMILY,
                                font_size=f"{self.FONT_SIZE * 0.9}px",
                                text_anchor="middle",
                                dominant_baseline="middle",
                            )
                        )

                    group.add(g)

                except Exception as exc:
                    logger.warning(
                        "Ошибка отрисовки помещения %s: %s", room.get("id"), exc
                    )

        except Exception as exc:
            logger.exception("Ошибка в _draw_rooms: %s", exc)

    # ------------------------------------------------------------------
    # Рамка и штамп
    # ------------------------------------------------------------------

    def _draw_frame(
        self,
        dwg: svgwrite.Drawing,
        group: svgwrite.container.Group,
        scale: str,
        width: int,
        height: int,
    ) -> None:
        """Нарисовать рамку листа и штамп в правом нижнем углу."""
        try:
            margin = 20
            stroke = "#000000"

            # Внешняя рамка листа
            group.add(
                dwg.rect(
                    insert=(margin, margin),
                    size=(width - margin * 2, height - margin * 2),
                    fill="none",
                    stroke=stroke,
                    stroke_width="1px",
                )
            )

            # Штамп в правом нижнем углу
            sw = self.STAMP_WIDTH
            sh = self.STAMP_HEIGHT
            sx = width - margin - sw
            sy = height - margin - sh

            # Фон штампа
            group.add(
                dwg.rect(
                    insert=(sx, sy),
                    size=(sw, sh),
                    fill="#ffffff",
                    stroke=stroke,
                    stroke_width="0.8px",
                )
            )

            # Строки штампа
            today = datetime.now().strftime("%d.%m.%Y")
            rows = [
                ("Наименование", "Архитектурный чертёж"),
                ("Масштаб", scale),
                ("Дата", today),
                ("Лист", "1"),
            ]

            row_h = sh / len(rows)
            label_col_w = sw * 0.45

            for i, (label_text, value_text) in enumerate(rows):
                ry = sy + i * row_h

                # Горизонтальная разделительная линия
                if i > 0:
                    group.add(
                        dwg.line(
                            start=(sx, ry),
                            end=(sx + sw, ry),
                            stroke=stroke,
                            stroke_width="0.5px",
                        )
                    )

                # Вертикальная разделительная линия (название / значение)
                group.add(
                    dwg.line(
                        start=(sx + label_col_w, sy),
                        end=(sx + label_col_w, sy + sh),
                        stroke=stroke,
                        stroke_width="0.5px",
                    )
                )

                text_y = ry + row_h / 2

                # Название поля
                group.add(
                    dwg.text(
                        label_text,
                        insert=(sx + 4, text_y),
                        fill=stroke,
                        font_family=self.FONT_FAMILY,
                        font_size="9px",
                        dominant_baseline="middle",
                    )
                )

                # Значение поля
                group.add(
                    dwg.text(
                        value_text,
                        insert=(sx + label_col_w + 4, text_y),
                        fill=stroke,
                        font_family=self.FONT_FAMILY,
                        font_size="9px",
                        dominant_baseline="middle",
                    )
                )

        except Exception as exc:
            logger.exception("Ошибка в _draw_frame: %s", exc)

# Стили согласно ГОСТ для архитектурных чертежей

# Толщины линий (в мм для печати)
LINE_WIDTHS = {
    "load_bearing_wall": 0.8,
    "partition_wall": 0.4,
    "dimension": 0.25,
    "door": 0.5,
    "window": 0.35,
    "stair": 0.35,
}

# Цвета (hex)
COLORS = {
    "load_bearing_wall": "#000000",
    "partition_wall": "#666666",
    "dimension": "#000000",
    "door": "#000000",
    "window": "#000000",
    "stair": "#000000",
    "label": "#000000",
    "low_confidence": "#ff0000",
    "background": "#ffffff",
    "frame": "#000000",
}

# Стандартные масштабы
STANDARD_SCALES = ["1:50", "1:100", "1:200", "1:500"]

# Стандартные размеры шрифтов (мм)
FONT_SIZES = {
    "room_name": 3.5,
    "room_area": 2.5,
    "dimension": 2.5,
    "label": 2.5,
    "stamp": 3.5,
}

# Параметры штампа (мм)
STAMP = {
    "width": 185,
    "height": 55,
}

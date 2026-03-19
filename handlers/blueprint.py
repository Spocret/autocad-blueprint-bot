"""
Основной флоу обработки чертежа:
  фото → обработка → уточнения → генерация SVG/DXF → подтверждение
"""

import io
import logging
import os
import re
import time

from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from models.database import db
from services.ai_recognizer import AIServiceError, ai_recognizer
from services.dxf_generator import dxf_generator
from services.image_processor import image_processor
from services.svg_generator import SVGGenerator

logger = logging.getLogger(__name__)

router = Router()

# ─────────────────────────────────────────
# FSM-состояния чертежа
# ─────────────────────────────────────────

class BlueprintStates(StatesGroup):
    WAITING_PHOTO = State()
    PROCESSING = State()
    WAITING_CLARIFICATION = State()
    WAITING_CORRECTION = State()
    DONE = State()


# ─────────────────────────────────────────
# Вспомогательные функции для клавиатур
# ─────────────────────────────────────────

def _scale_keyboard() -> InlineKeyboardMarkup:
    """Инлайн-кнопки выбора масштаба."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="1:50",  callback_data="scale:1:50"),
            InlineKeyboardButton(text="1:100", callback_data="scale:1:100"),
            InlineKeyboardButton(text="1:200", callback_data="scale:1:200"),
        ],
        [
            InlineKeyboardButton(text="✏️ Ввести вручную", callback_data="scale:manual"),
        ],
    ])


def _clarification_keyboard() -> InlineKeyboardMarkup:
    """Кнопки при уточнении нераспознанного элемента."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Пропустить",     callback_data="clarify:skip"),
            InlineKeyboardButton(text="Ввести значение", callback_data="clarify:input"),
        ],
    ])


def _result_keyboard() -> InlineKeyboardMarkup:
    """Кнопки после генерации результата."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Всё верно",       callback_data="result:confirm"),
            InlineKeyboardButton(text="✏️ Исправить",       callback_data="result:correct"),
            InlineKeyboardButton(text="🔄 Перегенерировать", callback_data="result:regenerate"),
        ],
    ])


# ─────────────────────────────────────────
# Шаг 1: приём фото
# ─────────────────────────────────────────

@router.message(F.photo)
async def photo_handler(message: Message, state: FSMContext, bot: Bot) -> None:
    """
    Принимает фото в любом состоянии (или без состояния).
    Скачивает байты и передаёт управление process_blueprint().
    """
    try:
        await message.answer("⏳ Обрабатываю чертёж...")
        await state.set_state(BlueprintStates.PROCESSING)

        # Скачиваем самую качественную версию фото
        photo = message.photo[-1]
        buf = io.BytesIO()
        await bot.download(photo, destination=buf)
        photo_bytes = buf.getvalue()

        await state.update_data(photo_bytes=photo_bytes)
        logger.info("Пользователь %s отправил фото (%d байт)", message.from_user.id, len(photo_bytes))

        await process_blueprint(message, state, bot)
    except Exception:
        logger.exception("Ошибка при получении фото от пользователя %s", message.from_user.id)
        await state.set_state(BlueprintStates.WAITING_PHOTO)
        await message.answer("❌ Произошла ошибка при получении фото. Попробуйте ещё раз.")


# ─────────────────────────────────────────
# Шаг 2: обработка изображения и распознавание
# ─────────────────────────────────────────

async def process_blueprint(message: Message, state: FSMContext, bot: Bot) -> None:
    """
    Основная логика обработки:
      1. Предобработка изображения
      2. AI-распознавание
      3. Уточнения или генерация файлов
    """
    try:
        data = await state.get_data()
        photo_bytes: bytes = data["photo_bytes"]

        # ── Предобработка ────────────────────────────────────────────────
        try:
            processed = await image_processor.process(photo_bytes)
        except Exception:
            logger.exception("Ошибка предобработки изображения")
            await message.answer("❌ Не удалось обработать изображение. Попробуйте другое фото.")
            await state.set_state(BlueprintStates.WAITING_PHOTO)
            return

        if not processed.is_valid:
            issues = ", ".join(processed.quality_issues) if processed.quality_issues else "неизвестные проблемы"
            await message.answer(
                f"❌ Не похоже на чертёж.\n"
                f"Качество: {issues}.\n\n"
                "Попробуйте другое фото."
            )
            await state.set_state(BlueprintStates.WAITING_PHOTO)
            return

        # ── Масштаб ──────────────────────────────────────────────────────
        scale = data.get("scale") or processed.scale
        if scale is None:
            await state.update_data(processed_image_bytes=processed.image_bytes)
            await state.set_state(BlueprintStates.WAITING_PHOTO)  # временно, ждём масштаб
            await state.update_data(awaiting_scale=True)
            await message.answer(
                "🔍 Не удалось определить масштаб чертежа.\n"
                "Выберите масштаб или введите вручную:",
                reply_markup=_scale_keyboard(),
            )
            return

        # ── AI-распознавание ─────────────────────────────────────────────
        try:
            recognized_data = await ai_recognizer.recognize(processed.image_bytes, scale)
        except AIServiceError as exc:
            logger.error("Ошибка AI-сервиса: %s", exc)
            error_hint = str(exc)
            if "API_KEY" in error_hint.upper() or "api key" in error_hint.lower():
                user_msg = "❌ Ошибка API-ключа Gemini. Проверьте GEMINI_API_KEY."
            elif "quota" in error_hint.lower() or "429" in error_hint:
                user_msg = "❌ Превышена квота Gemini API. Попробуйте позже."
            elif "blocked" in error_hint.lower() or "unavailable" in error_hint.lower():
                user_msg = "❌ Gemini API недоступен в вашем регионе или модель заблокирована."
            else:
                user_msg = f"❌ Ошибка сервиса распознавания. Попробуйте позже.\n<code>{error_hint[:200]}</code>"
            await message.answer(user_msg, parse_mode="HTML")
            await state.set_state(BlueprintStates.WAITING_PHOTO)
            return

        await state.update_data(
            recognized_data=recognized_data,
            processed_image_bytes=processed.image_bytes,
            scale=scale,
        )
        logger.info("AI распознал чертёж пользователя %s", message.from_user.id)

        # ── Уточнения или генерация ──────────────────────────────────────
        low_conf = recognized_data.get("low_confidence_elements", [])
        if low_conf:
            await handle_clarifications(message, state, bot, recognized_data)
        else:
            await generate_and_send(message, state, bot)

    except Exception:
        logger.exception("Непредвиденная ошибка в process_blueprint для пользователя %s", message.from_user.id)
        await state.set_state(BlueprintStates.WAITING_PHOTO)
        await message.answer("❌ Внутренняя ошибка. Попробуйте ещё раз или используйте /new.")


# ─────────────────────────────────────────
# Шаг 3: выбор масштаба через кнопки
# ─────────────────────────────────────────

@router.callback_query(F.data.startswith("scale:"))
async def scale_callback(callback: CallbackQuery, state: FSMContext, bot: Bot) -> None:
    """Обработка выбора масштаба через инлайн-кнопки."""
    try:
        await callback.answer()
        value = callback.data.split(":", 1)[1]  # "1:50", "1:100", "1:200" или "manual"

        if value == "manual":
            await state.update_data(awaiting_scale_manual=True)
            await callback.message.answer(
                "✏️ Введите масштаб в формате 1:N (например: 1:100):"
            )
            return

        await state.update_data(scale=value, awaiting_scale=False)
        logger.info("Пользователь %s выбрал масштаб %s", callback.from_user.id, value)
        await process_blueprint(callback.message, state, bot)
    except Exception:
        logger.exception("Ошибка в scale_callback для пользователя %s", callback.from_user.id)
        await callback.message.answer("❌ Ошибка при обработке масштаба.")


@router.message(BlueprintStates.WAITING_PHOTO)
async def scale_manual_input(message: Message, state: FSMContext, bot: Bot) -> None:
    """Принимает масштаб, введённый вручную в формате 1:N."""
    try:
        data = await state.get_data()
        if not data.get("awaiting_scale_manual"):
            # Если это не ручной ввод масштаба, сообщаем о режиме ожидания
            await message.answer("Отправьте фото чертежа или используйте /new.")
            return

        scale_text = message.text.strip() if message.text else ""
        if not re.fullmatch(r"1:\d+", scale_text):
            await message.answer(
                "⚠️ Неверный формат. Введите масштаб в виде 1:N (например: 1:100):"
            )
            return

        await state.update_data(scale=scale_text, awaiting_scale=False, awaiting_scale_manual=False)
        logger.info("Пользователь %s ввёл масштаб вручную: %s", message.from_user.id, scale_text)
        await process_blueprint(message, state, bot)
    except Exception:
        logger.exception("Ошибка в scale_manual_input для пользователя %s", message.from_user.id)
        await message.answer("❌ Ошибка при обработке масштаба.")


# ─────────────────────────────────────────
# Шаг 4: уточнение нераспознанных элементов
# ─────────────────────────────────────────

async def handle_clarifications(
    message: Message,
    state: FSMContext,
    bot: Bot,
    recognized_data: dict,
) -> None:
    """
    Последовательно запрашивает уточнения по элементам с низкой уверенностью.
    Сохраняет текущий индекс в FSM.
    """
    try:
        await state.set_state(BlueprintStates.WAITING_CLARIFICATION)
        await state.update_data(clarification_index=0, recognized_data=recognized_data)
        await _send_clarification_request(message, state, bot)
    except Exception:
        logger.exception("Ошибка в handle_clarifications")
        await generate_and_send(message, state, bot)


async def _send_clarification_request(message: Message, state: FSMContext, bot: Bot) -> None:
    """Отправляет запрос об одном нераспознанном элементе."""
    try:
        data = await state.get_data()
        recognized_data: dict = data.get("recognized_data", {})
        low_conf: list = recognized_data.get("low_confidence_elements", [])
        index: int = data.get("clarification_index", 0)

        if index >= len(low_conf):
            # Все уточнения получены — генерируем файлы
            await generate_and_send(message, state, bot)
            return

        element = low_conf[index]
        elem_type = element.get("element_type", "элемент")
        bbox = element.get("bbox", {})

        # Пробуем вырезать кроп — при ошибке продолжаем без него
        try:
            processed_image_bytes: bytes = data.get("processed_image_bytes", b"")
            crop_bytes = await image_processor.extract_crop(processed_image_bytes, bbox)
            await message.answer_photo(
                BufferedInputFile(crop_bytes, filename="crop.jpg"),
                caption=(
                    f"🔴 Не могу распознать элемент ({elem_type}).\n"
                    "Что здесь написано?\n"
                    "Отправьте значение или нажмите «Пропустить»"
                ),
                reply_markup=_clarification_keyboard(),
            )
        except Exception:
            logger.warning("Не удалось вырезать кроп для элемента %s", element.get("element_id"))
            await message.answer(
                f"🔴 Не могу распознать элемент ({elem_type}).\n"
                "Что здесь написано?\n"
                "Отправьте значение или нажмите «Пропустить»",
                reply_markup=_clarification_keyboard(),
            )
    except Exception:
        logger.exception("Ошибка при отправке запроса уточнения")
        await generate_and_send(message, state, bot)


@router.callback_query(BlueprintStates.WAITING_CLARIFICATION, F.data.startswith("clarify:"))
async def clarification_callback(callback: CallbackQuery, state: FSMContext, bot: Bot) -> None:
    """Обрабатывает нажатие «Пропустить» или «Ввести значение»."""
    try:
        await callback.answer()
        action = callback.data.split(":", 1)[1]

        if action == "skip":
            # Переходим к следующему элементу без изменений
            data = await state.get_data()
            await state.update_data(clarification_index=data.get("clarification_index", 0) + 1)
            await _send_clarification_request(callback.message, state, bot)

        elif action == "input":
            await callback.message.answer("✏️ Введите значение для этого элемента:")

    except Exception:
        logger.exception("Ошибка в clarification_callback")
        await callback.message.answer("❌ Ошибка. Продолжаем генерацию...")
        await generate_and_send(callback.message, state, bot)


@router.message(BlueprintStates.WAITING_CLARIFICATION)
async def clarification_text_response(message: Message, state: FSMContext, bot: Bot) -> None:
    """Принимает текстовый ответ пользователя при уточнении элемента."""
    try:
        data = await state.get_data()
        recognized_data: dict = data.get("recognized_data", {})
        low_conf: list = recognized_data.get("low_confidence_elements", [])
        index: int = data.get("clarification_index", 0)

        if index < len(low_conf):
            # Обновляем значение элемента
            low_conf[index]["clarified_value"] = message.text.strip()
            recognized_data["low_confidence_elements"] = low_conf
            await state.update_data(
                recognized_data=recognized_data,
                clarification_index=index + 1,
            )

        await _send_clarification_request(message, state, bot)
    except Exception:
        logger.exception("Ошибка в clarification_text_response")
        await message.answer("❌ Ошибка. Продолжаем генерацию...")
        await generate_and_send(message, state, bot)


# ─────────────────────────────────────────
# Шаг 5: генерация и отправка файлов
# ─────────────────────────────────────────

async def generate_and_send(message: Message, state: FSMContext, bot: Bot) -> None:
    """
    Генерирует SVG и DXF файлы и отправляет их пользователю
    вместе со сводным сообщением.
    """
    try:
        await state.set_state(BlueprintStates.DONE)

        data = await state.get_data()
        recognized_data: dict = data.get("recognized_data", {})
        user_id = message.from_user.id
        timestamp = int(time.time())

        # ── Создаём папку outputs/ ────────────────────────────────────────
        outputs_dir = os.path.join("outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        base_name = os.path.join(outputs_dir, f"blueprint_{user_id}_{timestamp}")
        svg_path = base_name + ".svg"
        dxf_path = base_name + ".dxf"

        # ── Генерация SVG ─────────────────────────────────────────────────
        try:
            svg_generator = SVGGenerator()
            svg_content, svg_path = await svg_generator.generate(recognized_data, svg_path)
        except Exception:
            logger.exception("Ошибка генерации SVG")
            svg_content = None
            svg_path = None

        # ── Генерация DXF ─────────────────────────────────────────────────
        try:
            dxf_path = await dxf_generator.generate(recognized_data, dxf_path)
        except Exception:
            logger.exception("Ошибка генерации DXF")
            dxf_path = None

        # ── Формируем текстовое сводное сообщение ─────────────────────────
        scale = recognized_data.get("scale") or data.get("scale") or "не определён"
        rooms: list = recognized_data.get("rooms", [])
        low_conf_ids: set = {
            el.get("element_id") for el in recognized_data.get("low_confidence_elements", [])
        }

        room_lines = []
        for room in rooms:
            name = room.get("name", "Помещение")
            area = room.get("area")
            room_id = room.get("id", "")
            area_str = f"{area} м²" if area is not None else "площадь не определена"
            marker = " 🔴" if room_id in low_conf_ids else ""
            room_lines.append(f"• {name} — {area_str}{marker}")

        rooms_text = "\n".join(room_lines) if room_lines else "Помещения не определены"

        summary = (
            f"📐 Чертёж обработан!\n\n"
            f"Помещения:\n{rooms_text}\n\n"
            f"Масштаб: {scale}"
        )

        await message.answer(summary, reply_markup=_result_keyboard())

        # ── Отправляем SVG как файл ───────────────────────────────────────
        if svg_content:
            try:
                svg_bytes = svg_content.encode("utf-8") if isinstance(svg_content, str) else svg_content
                await message.answer_document(
                    BufferedInputFile(svg_bytes, filename=f"blueprint_{timestamp}.svg"),
                    caption="📄 SVG-файл чертежа",
                )
            except Exception:
                logger.exception("Ошибка отправки SVG")

        # ── Отправляем DXF как файл ───────────────────────────────────────
        if dxf_path and os.path.exists(dxf_path):
            try:
                with open(dxf_path, "rb") as f:
                    dxf_bytes = f.read()
                await message.answer_document(
                    BufferedInputFile(dxf_bytes, filename=f"blueprint_{timestamp}.dxf"),
                    caption="📐 DXF-файл чертежа (AutoCAD)",
                )
            except Exception:
                logger.exception("Ошибка отправки DXF")

        logger.info("Файлы отправлены пользователю %s", user_id)

    except Exception:
        logger.exception("Непредвиденная ошибка в generate_and_send для пользователя %s", message.from_user.id)
        await message.answer("❌ Ошибка при генерации файлов. Используйте /new для повтора.")


# ─────────────────────────────────────────
# Шаг 6: действия с результатом
# ─────────────────────────────────────────

@router.callback_query(BlueprintStates.DONE, F.data == "result:confirm")
async def confirm_callback(callback: CallbackQuery, state: FSMContext) -> None:
    """Пользователь подтвердил результат."""
    try:
        await callback.answer()
        await state.clear()
        await callback.message.answer(
            "✅ Отлично! Файлы отправлены.\n"
            "Используйте /new для нового чертежа."
        )
        logger.info("Пользователь %s подтвердил результат", callback.from_user.id)
    except Exception:
        logger.exception("Ошибка в confirm_callback")


@router.callback_query(BlueprintStates.DONE, F.data == "result:regenerate")
async def regenerate_callback(callback: CallbackQuery, state: FSMContext, bot: Bot) -> None:
    """Перегенерация файлов с теми же данными."""
    try:
        await callback.answer("🔄 Перегенерирую...")
        logger.info("Пользователь %s запросил перегенерацию", callback.from_user.id)
        await generate_and_send(callback.message, state, bot)
    except Exception:
        logger.exception("Ошибка в regenerate_callback")
        await callback.message.answer("❌ Ошибка при перегенерации. Попробуйте /new.")

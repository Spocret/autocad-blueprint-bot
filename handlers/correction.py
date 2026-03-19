"""
Хендлеры исправления распознанных элементов чертежа.
Доступны из состояния WAITING_CORRECTION (после нажатия [✏️ Исправить]).
"""

import logging

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from handlers.blueprint import BlueprintStates, generate_and_send
from models.database import db

logger = logging.getLogger(__name__)

router = Router()


# ─────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────

def _elements_keyboard(elements: list[dict]) -> InlineKeyboardMarkup:
    """Строит клавиатуру со списком элементов для выбора."""
    buttons = []
    for i, el in enumerate(elements):
        label = _element_label(el, i)
        buttons.append([
            InlineKeyboardButton(
                text=label,
                callback_data=f"correct_elem:{i}",
            )
        ])
    buttons.append([
        InlineKeyboardButton(text="« Назад", callback_data="correct_elem:cancel"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _element_label(el: dict, index: int) -> str:
    """Формирует подпись элемента для кнопки."""
    elem_type = el.get("element_type", "элемент")
    current_val = el.get("clarified_value") or el.get("value") or "—"
    return f"{index + 1}. {elem_type}: {current_val}"


def _all_correctable_elements(recognized_data: dict) -> list[dict]:
    """
    Собирает все редактируемые элементы:
    rooms, labels и low_confidence_elements.
    """
    elements: list[dict] = []

    for room in recognized_data.get("rooms", []):
        elements.append({
            "source": "rooms",
            "id": room.get("id"),
            "element_type": room.get("name", "Помещение"),
            "value": room.get("area"),
        })

    for label in recognized_data.get("labels", []):
        elements.append({
            "source": "labels",
            "id": label.get("id"),
            "element_type": "Подпись",
            "value": label.get("text"),
        })

    for el in recognized_data.get("low_confidence_elements", []):
        elements.append({
            "source": "low_confidence",
            "id": el.get("element_id"),
            "element_type": el.get("element_type", "элемент"),
            "value": el.get("clarified_value") or el.get("value"),
        })

    return elements


def _format_elements_list(elements: list[dict]) -> str:
    """Формирует нумерованный текстовый список элементов."""
    if not elements:
        return "Нет элементов для исправления."
    lines = []
    for i, el in enumerate(elements, 1):
        label = f"{i}. {el.get('element_type', 'элемент')}: {el.get('value') or '—'}"
        lines.append(label)
    return "\n".join(lines)


# ─────────────────────────────────────────
# Хендлер 1: вход в режим исправления
# ─────────────────────────────────────────

@router.callback_query(F.data == "result:correct")
async def correct_callback(callback: CallbackQuery, state: FSMContext) -> None:
    """
    Активируется кнопкой [✏️ Исправить].
    Показывает список всех элементов с инлайн-кнопками.
    """
    try:
        await callback.answer()
        await state.set_state(BlueprintStates.WAITING_CORRECTION)

        data = await state.get_data()
        recognized_data: dict = data.get("recognized_data", {})
        elements = _all_correctable_elements(recognized_data)

        # Сохраняем список элементов в FSM для последующего редактирования
        await state.update_data(correctable_elements=elements)

        if not elements:
            await callback.message.answer("Нет элементов для исправления.")
            return

        text = "✏️ Выберите элемент для исправления:\n\n" + _format_elements_list(elements)
        await callback.message.answer(text, reply_markup=_elements_keyboard(elements))

        logger.info("Пользователь %s открыл режим исправления", callback.from_user.id)
    except Exception:
        logger.exception("Ошибка в correct_callback для пользователя %s", callback.from_user.id)
        await callback.message.answer("❌ Ошибка при открытии списка элементов.")


# ─────────────────────────────────────────
# Хендлер 2: выбор конкретного элемента
# ─────────────────────────────────────────

@router.callback_query(BlueprintStates.WAITING_CORRECTION, F.data.startswith("correct_elem:"))
async def select_element_callback(callback: CallbackQuery, state: FSMContext) -> None:
    """
    Пользователь нажал на конкретный элемент в списке.
    Показывает текущее значение и просит ввести новое.
    """
    try:
        await callback.answer()
        raw = callback.data.split(":", 1)[1]

        if raw == "cancel":
            await state.set_state(BlueprintStates.DONE)
            await callback.message.answer(
                "Исправление отменено. Используйте кнопки ниже для дальнейших действий."
            )
            return

        try:
            elem_index = int(raw)
        except ValueError:
            await callback.message.answer("❌ Неверный индекс элемента.")
            return

        data = await state.get_data()
        elements: list[dict] = data.get("correctable_elements", [])

        if elem_index < 0 or elem_index >= len(elements):
            await callback.message.answer("❌ Элемент не найден.")
            return

        element = elements[elem_index]
        current_value = element.get("value") or "не задано"
        elem_type = element.get("element_type", "элемент")

        await state.update_data(editing_element_index=elem_index)

        await callback.message.answer(
            f"📝 Элемент: {elem_type}\n"
            f"Текущее значение: {current_value}\n\n"
            "Введите новое значение:"
        )
        logger.info(
            "Пользователь %s выбрал элемент #%d для исправления",
            callback.from_user.id, elem_index,
        )
    except Exception:
        logger.exception("Ошибка в select_element_callback для пользователя %s", callback.from_user.id)
        await callback.message.answer("❌ Ошибка при выборе элемента.")


# ─────────────────────────────────────────
# Хендлер 3: приём исправленного значения
# ─────────────────────────────────────────

@router.message(BlueprintStates.WAITING_CORRECTION)
async def receive_correction(message: Message, state: FSMContext, bot) -> None:
    """
    Принимает новое значение от пользователя, обновляет элемент в FSM
    и сохраняет исправление в БД.
    """
    try:
        new_value = message.text.strip() if message.text else ""
        if not new_value:
            await message.answer("⚠️ Введите непустое значение.")
            return

        data = await state.get_data()
        elem_index: int = data.get("editing_element_index")
        elements: list[dict] = data.get("correctable_elements", [])
        recognized_data: dict = data.get("recognized_data", {})

        if elem_index is None or elem_index >= len(elements):
            await message.answer("❌ Элемент для исправления не найден. Начните снова.")
            return

        old_value = elements[elem_index].get("value")
        elem_id = elements[elem_index].get("id")
        elem_source = elements[elem_index].get("source")
        elem_type = elements[elem_index].get("element_type", "элемент")

        # ── Обновляем элемент в списке correctable_elements ──────────────
        elements[elem_index]["value"] = new_value

        # ── Применяем исправление к recognized_data ───────────────────────
        recognized_data = _apply_correction(recognized_data, elem_source, elem_id, new_value)

        await state.update_data(
            correctable_elements=elements,
            recognized_data=recognized_data,
            editing_element_index=None,
        )

        # ── Сохраняем в БД ────────────────────────────────────────────────
        try:
            await db.save_correction(
                user_id=message.from_user.id,
                element_id=elem_id,
                element_type=elem_type,
                old_value=old_value,
                new_value=new_value,
            )
        except Exception:
            logger.exception("Ошибка сохранения исправления в БД")

        await message.answer(
            f"✅ Исправлено: {elem_type} → {new_value}\n\n"
            "Нажмите [🔄 Перегенерировать] для обновления файлов.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🔄 Перегенерировать", callback_data="result:regenerate"),
                    InlineKeyboardButton(text="✏️ Ещё исправить",   callback_data="result:correct"),
                ],
            ]),
        )
        await state.set_state(BlueprintStates.DONE)

        logger.info(
            "Пользователь %s исправил %s: %s → %s",
            message.from_user.id, elem_type, old_value, new_value,
        )
    except Exception:
        logger.exception("Ошибка в receive_correction для пользователя %s", message.from_user.id)
        await message.answer("❌ Ошибка при сохранении исправления. Попробуйте снова.")


# ─────────────────────────────────────────
# Вспомогательная: применить исправление к recognized_data
# ─────────────────────────────────────────

def _apply_correction(recognized_data: dict, source: str, elem_id, new_value: str) -> dict:
    """
    Обновляет нужное поле в recognized_data по source и id элемента.
    """
    try:
        if source == "rooms":
            for room in recognized_data.get("rooms", []):
                if room.get("id") == elem_id:
                    # Пытаемся распарсить как число (площадь), иначе оставляем строку
                    try:
                        room["area"] = float(new_value.replace(",", "."))
                    except ValueError:
                        room["name"] = new_value
                    break

        elif source == "labels":
            for label in recognized_data.get("labels", []):
                if label.get("id") == elem_id:
                    label["text"] = new_value
                    break

        elif source == "low_confidence":
            for el in recognized_data.get("low_confidence_elements", []):
                if el.get("element_id") == elem_id:
                    el["clarified_value"] = new_value
                    break

    except Exception:
        logger.exception("Ошибка в _apply_correction (source=%s, id=%s)", source, elem_id)

    return recognized_data

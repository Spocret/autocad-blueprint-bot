"""
Хендлеры базовых команд: /start, /help, /cancel, /new
"""

import logging

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

logger = logging.getLogger(__name__)

router = Router()

WELCOME_TEXT = (
    "Привет! Я бот для обработки архитектурных чертежей.\n\n"
    "Я умею:\n"
    "• Распознавать планы этажей\n"
    "• Определять стены, двери, окна, лестницы\n"
    "• Генерировать SVG и DXF файлы\n"
    "• Соблюдать пропорции и масштаб\n\n"
    "Отправьте фото чертежа для начала работы.\n"
    "Используйте /new для нового чертежа."
)

HELP_TEXT = (
    "📋 Инструкция:\n"
    "1. Отправьте фото чертежа\n"
    "2. Я распознаю элементы\n"
    "3. Уточню непонятные места\n"
    "4. Отправлю готовые SVG и DXF файлы\n\n"
    "Команды:\n"
    "/new — начать новый чертёж\n"
    "/cancel — отменить обработку\n"
    "/help — эта справка"
)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    """Приветственное сообщение при первом запуске или команде /start."""
    try:
        await state.clear()
        await message.answer(WELCOME_TEXT)
        logger.info("Пользователь %s запустил бота", message.from_user.id)
    except Exception:
        logger.exception("Ошибка в обработчике /start для пользователя %s", message.from_user.id)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Справочное сообщение с инструкцией."""
    try:
        await message.answer(HELP_TEXT)
        logger.info("Пользователь %s запросил помощь", message.from_user.id)
    except Exception:
        logger.exception("Ошибка в обработчике /help для пользователя %s", message.from_user.id)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    """Отмена текущей операции с очисткой FSM-состояния."""
    try:
        current_state = await state.get_state()
        if current_state is None:
            await message.answer("Нечего отменять. Отправьте фото чертежа для начала.")
            return

        await state.clear()
        await message.answer(
            "❌ Обработка отменена.\n\n"
            "Отправьте фото чертежа или используйте /new для нового чертежа."
        )
        logger.info("Пользователь %s отменил операцию (состояние: %s)", message.from_user.id, current_state)
    except Exception:
        logger.exception("Ошибка в обработчике /cancel для пользователя %s", message.from_user.id)


@router.message(Command("new"))
async def cmd_new(message: Message, state: FSMContext) -> None:
    """Сброс состояния и начало работы с новым чертежом."""
    try:
        await state.clear()
        await message.answer(
            "🆕 Готов к новому чертежу!\n\n"
            "Отправьте фото архитектурного плана для начала обработки."
        )
        logger.info("Пользователь %s начал новый чертёж", message.from_user.id)
    except Exception:
        logger.exception("Ошибка в обработчике /new для пользователя %s", message.from_user.id)

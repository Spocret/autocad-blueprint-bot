import json
import logging
from datetime import datetime, timezone
from typing import Optional

from supabase import acreate_client, AsyncClient

from config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self._client: Optional[AsyncClient] = None

    async def init(self):
        """Инициализация подключения к Supabase"""
        try:
            self._client = await acreate_client(self.url, self.key)
            logger.info("Подключение к Supabase установлено")
        except Exception as e:
            logger.exception("Ошибка подключения к Supabase: %s", e)
            raise

    # ------------------------------------------------------------------
    # Сессии
    # ------------------------------------------------------------------

    async def create_session(self, user_id: int) -> int:
        """Создать новую сессию для пользователя, вернуть session_id"""
        try:
            result = await (
                self._client.table("sessions")
                .insert({"user_id": user_id})
                .execute()
            )
            session_id = result.data[0]["id"]
            logger.debug("Создана сессия %d для пользователя %d", session_id, user_id)
            return session_id
        except Exception as e:
            logger.exception("Ошибка создания сессии для пользователя %d: %s", user_id, e)
            raise

    async def get_session(self, user_id: int) -> Optional[dict]:
        """Получить последнюю активную сессию пользователя"""
        try:
            result = await (
                self._client.table("sessions")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.exception("Ошибка получения сессии пользователя %d: %s", user_id, e)
            raise

    async def update_session_state(self, session_id: int, state: str):
        """Обновить состояние FSM для указанной сессии"""
        try:
            now = datetime.now(timezone.utc).isoformat()
            await (
                self._client.table("sessions")
                .update({"state": state, "updated_at": now})
                .eq("id", session_id)
                .execute()
            )
            logger.debug("Сессия %d переведена в состояние '%s'", session_id, state)
        except Exception as e:
            logger.exception("Ошибка обновления состояния сессии %d: %s", session_id, e)
            raise

    # ------------------------------------------------------------------
    # Чертежи
    # ------------------------------------------------------------------

    async def create_blueprint(
        self, session_id: int, user_id: int, floor_number: int = 1
    ) -> int:
        """Создать запись чертежа, вернуть blueprint_id"""
        try:
            result = await (
                self._client.table("blueprints")
                .insert({
                    "session_id": session_id,
                    "user_id": user_id,
                    "floor_number": floor_number,
                })
                .execute()
            )
            blueprint_id = result.data[0]["id"]
            logger.debug(
                "Создан чертёж %d (сессия %d, пользователь %d, этаж %d)",
                blueprint_id, session_id, user_id, floor_number,
            )
            return blueprint_id
        except Exception as e:
            logger.exception("Ошибка создания чертежа: %s", e)
            raise

    async def update_blueprint(self, blueprint_id: int, **kwargs):
        """Обновить произвольные поля чертежа по blueprint_id"""
        if not kwargs:
            return
        try:
            for key, value in list(kwargs.items()):
                if isinstance(value, (dict, list)):
                    kwargs[key] = json.dumps(value, ensure_ascii=False)

            await (
                self._client.table("blueprints")
                .update(kwargs)
                .eq("id", blueprint_id)
                .execute()
            )
            logger.debug("Чертёж %d обновлён: %s", blueprint_id, list(kwargs.keys()))
        except Exception as e:
            logger.exception("Ошибка обновления чертежа %d: %s", blueprint_id, e)
            raise

    async def get_blueprint(self, blueprint_id: int) -> Optional[dict]:
        """Получить чертёж по id"""
        try:
            result = await (
                self._client.table("blueprints")
                .select("*")
                .eq("id", blueprint_id)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.exception("Ошибка получения чертежа %d: %s", blueprint_id, e)
            raise

    async def get_user_blueprints(self, user_id: int) -> list[dict]:
        """Получить все чертежи пользователя, отсортированные по дате создания"""
        try:
            result = await (
                self._client.table("blueprints")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.exception("Ошибка получения чертежей пользователя %d: %s", user_id, e)
            raise

    # ------------------------------------------------------------------
    # Элементы чертежа
    # ------------------------------------------------------------------

    async def add_element(
        self,
        blueprint_id: int,
        element_type: str,
        element_data: dict,
        confidence: float = 1.0,
    ) -> int:
        """Добавить элемент чертежа, вернуть element_id"""
        try:
            element_data_json = json.dumps(element_data, ensure_ascii=False)
            result = await (
                self._client.table("elements")
                .insert({
                    "blueprint_id": blueprint_id,
                    "element_type": element_type,
                    "element_data": element_data_json,
                    "confidence": confidence,
                })
                .execute()
            )
            element_id = result.data[0]["id"]
            logger.debug(
                "Добавлен элемент %d (тип '%s', чертёж %d, уверенность %.2f)",
                element_id, element_type, blueprint_id, confidence,
            )
            return element_id
        except Exception as e:
            logger.exception("Ошибка добавления элемента в чертёж %d: %s", blueprint_id, e)
            raise

    async def get_low_confidence_elements(
        self, blueprint_id: int, threshold: float = 0.7
    ) -> list[dict]:
        """Получить элементы чертежа с уверенностью ниже заданного порога"""
        try:
            result = await (
                self._client.table("elements")
                .select("*")
                .eq("blueprint_id", blueprint_id)
                .lt("confidence", threshold)
                .eq("is_confirmed", False)
                .order("confidence", desc=False)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.exception(
                "Ошибка получения элементов с низкой уверенностью для чертежа %d: %s",
                blueprint_id, e,
            )
            raise

    async def confirm_element(self, element_id: int):
        """Пометить элемент как подтверждённый пользователем"""
        try:
            await (
                self._client.table("elements")
                .update({"is_confirmed": True})
                .eq("id", element_id)
                .execute()
            )
            logger.debug("Элемент %d подтверждён", element_id)
        except Exception as e:
            logger.exception("Ошибка подтверждения элемента %d: %s", element_id, e)
            raise

    # ------------------------------------------------------------------
    # Исправления
    # ------------------------------------------------------------------

    async def save_correction(
        self,
        blueprint_id: int,
        element_id: int,
        original: str,
        corrected: str,
    ):
        """Сохранить исправление пользователя и обновить поле user_correction у элемента"""
        try:
            await (
                self._client.table("corrections")
                .insert({
                    "blueprint_id": blueprint_id,
                    "element_id": element_id,
                    "original_value": original,
                    "corrected_value": corrected,
                })
                .execute()
            )
            await (
                self._client.table("elements")
                .update({"user_correction": corrected, "is_confirmed": True})
                .eq("id", element_id)
                .execute()
            )
            logger.debug(
                "Исправление сохранено: элемент %d, чертёж %d", element_id, blueprint_id
            )
        except Exception as e:
            logger.exception(
                "Ошибка сохранения исправления для элемента %d: %s", element_id, e
            )
            raise

    # ------------------------------------------------------------------
    # Управление соединением
    # ------------------------------------------------------------------

    async def close(self):
        """Закрыть соединение с Supabase"""
        try:
            if self._client:
                await self._client.aclose()
                self._client = None
                logger.info("Соединение с Supabase закрыто")
        except Exception as e:
            logger.exception("Ошибка при закрытии соединения с Supabase: %s", e)


# Глобальный экземпляр базы данных
db = Database(SUPABASE_URL, SUPABASE_KEY)

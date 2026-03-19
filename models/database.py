import json
import logging
import aiosqlite
from typing import Optional

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def init(self):
        """Инициализация БД, создание таблиц если не существуют"""
        try:
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row

            await self._conn.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS sessions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    INTEGER NOT NULL,
                    state      TEXT    DEFAULT 'WAITING_PHOTO',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS blueprints (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id           INTEGER REFERENCES sessions(id),
                    user_id              INTEGER NOT NULL,
                    floor_number         INTEGER DEFAULT 1,
                    original_photo_path  TEXT,
                    processed_photo_path TEXT,
                    recognized_json      TEXT,
                    svg_path             TEXT,
                    dxf_path             TEXT,
                    scale                TEXT,
                    status               TEXT    DEFAULT 'pending',
                    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS elements (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    blueprint_id    INTEGER REFERENCES blueprints(id),
                    element_type    TEXT,
                    element_data    TEXT,
                    confidence      REAL    DEFAULT 1.0,
                    is_confirmed    INTEGER DEFAULT 0,
                    user_correction TEXT,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS corrections (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    blueprint_id    INTEGER REFERENCES blueprints(id),
                    element_id      INTEGER REFERENCES elements(id),
                    original_value  TEXT,
                    corrected_value TEXT,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await self._conn.commit()
            logger.info("База данных инициализирована: %s", self.db_path)
        except Exception as e:
            logger.exception("Ошибка инициализации БД: %s", e)
            raise

    # ------------------------------------------------------------------
    # Вспомогательный метод — преобразование aiosqlite.Row в dict
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row: Optional[aiosqlite.Row]) -> Optional[dict]:
        if row is None:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # Сессии
    # ------------------------------------------------------------------

    async def create_session(self, user_id: int) -> int:
        """Создать новую сессию для пользователя, вернуть session_id"""
        try:
            async with self._conn.execute(
                "INSERT INTO sessions (user_id) VALUES (?)",
                (user_id,),
            ) as cursor:
                await self._conn.commit()
                session_id = cursor.lastrowid
                logger.debug("Создана сессия %d для пользователя %d", session_id, user_id)
                return session_id
        except Exception as e:
            logger.exception("Ошибка создания сессии для пользователя %d: %s", user_id, e)
            raise

    async def get_session(self, user_id: int) -> Optional[dict]:
        """Получить последнюю активную сессию пользователя"""
        try:
            async with self._conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
                return self._row_to_dict(row)
        except Exception as e:
            logger.exception("Ошибка получения сессии пользователя %d: %s", user_id, e)
            raise

    async def update_session_state(self, session_id: int, state: str):
        """Обновить состояние FSM для указанной сессии"""
        try:
            await self._conn.execute(
                "UPDATE sessions SET state = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (state, session_id),
            )
            await self._conn.commit()
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
            async with self._conn.execute(
                "INSERT INTO blueprints (session_id, user_id, floor_number) VALUES (?, ?, ?)",
                (session_id, user_id, floor_number),
            ) as cursor:
                await self._conn.commit()
                blueprint_id = cursor.lastrowid
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
            # Сериализуем словари/списки в JSON-строки
            for key, value in list(kwargs.items()):
                if isinstance(value, (dict, list)):
                    kwargs[key] = json.dumps(value, ensure_ascii=False)

            set_clause = ", ".join(f"{col} = ?" for col in kwargs)
            values = list(kwargs.values()) + [blueprint_id]

            await self._conn.execute(
                f"UPDATE blueprints SET {set_clause} WHERE id = ?",
                values,
            )
            await self._conn.commit()
            logger.debug("Чертёж %d обновлён: %s", blueprint_id, list(kwargs.keys()))
        except Exception as e:
            logger.exception("Ошибка обновления чертежа %d: %s", blueprint_id, e)
            raise

    async def get_blueprint(self, blueprint_id: int) -> Optional[dict]:
        """Получить чертёж по id"""
        try:
            async with self._conn.execute(
                "SELECT * FROM blueprints WHERE id = ?",
                (blueprint_id,),
            ) as cursor:
                row = await cursor.fetchone()
                return self._row_to_dict(row)
        except Exception as e:
            logger.exception("Ошибка получения чертежа %d: %s", blueprint_id, e)
            raise

    async def get_user_blueprints(self, user_id: int) -> list[dict]:
        """Получить все чертежи пользователя, отсортированные по дате создания"""
        try:
            async with self._conn.execute(
                "SELECT * FROM blueprints WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
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
            async with self._conn.execute(
                """INSERT INTO elements (blueprint_id, element_type, element_data, confidence)
                   VALUES (?, ?, ?, ?)""",
                (blueprint_id, element_type, element_data_json, confidence),
            ) as cursor:
                await self._conn.commit()
                element_id = cursor.lastrowid
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
            async with self._conn.execute(
                """SELECT * FROM elements
                   WHERE blueprint_id = ? AND confidence < ? AND is_confirmed = 0
                   ORDER BY confidence ASC""",
                (blueprint_id, threshold),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.exception(
                "Ошибка получения элементов с низкой уверенностью для чертежа %d: %s",
                blueprint_id, e,
            )
            raise

    async def confirm_element(self, element_id: int):
        """Пометить элемент как подтверждённый пользователем"""
        try:
            await self._conn.execute(
                "UPDATE elements SET is_confirmed = 1 WHERE id = ?",
                (element_id,),
            )
            await self._conn.commit()
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
            await self._conn.execute(
                """INSERT INTO corrections (blueprint_id, element_id, original_value, corrected_value)
                   VALUES (?, ?, ?, ?)""",
                (blueprint_id, element_id, original, corrected),
            )
            # Сохраняем последнее исправление прямо в записи элемента для быстрого доступа
            await self._conn.execute(
                "UPDATE elements SET user_correction = ?, is_confirmed = 1 WHERE id = ?",
                (corrected, element_id),
            )
            await self._conn.commit()
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
        """Закрыть соединение с базой данных"""
        try:
            if self._conn:
                await self._conn.close()
                self._conn = None
                logger.info("Соединение с БД закрыто")
        except Exception as e:
            logger.exception("Ошибка при закрытии соединения с БД: %s", e)
            raise


# Глобальный экземпляр базы данных
db = Database(DATABASE_PATH)

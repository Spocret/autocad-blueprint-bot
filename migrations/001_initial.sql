-- Таблицы для blueprint_bot в Supabase (PostgreSQL)
-- Выполните этот скрипт в SQL-редакторе Supabase Dashboard

CREATE TABLE IF NOT EXISTS sessions (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT    NOT NULL,
    state      TEXT      NOT NULL DEFAULT 'WAITING_PHOTO',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS blueprints (
    id                   BIGSERIAL PRIMARY KEY,
    session_id           BIGINT REFERENCES sessions(id),
    user_id              BIGINT NOT NULL,
    floor_number         INTEGER DEFAULT 1,
    original_photo_path  TEXT,
    processed_photo_path TEXT,
    recognized_json      TEXT,
    svg_path             TEXT,
    dxf_path             TEXT,
    scale                TEXT,
    status               TEXT    DEFAULT 'pending',
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS elements (
    id              BIGSERIAL PRIMARY KEY,
    blueprint_id    BIGINT REFERENCES blueprints(id),
    element_type    TEXT,
    element_data    TEXT,
    confidence      DOUBLE PRECISION DEFAULT 1.0,
    is_confirmed    BOOLEAN DEFAULT FALSE,
    user_correction TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS corrections (
    id              BIGSERIAL PRIMARY KEY,
    blueprint_id    BIGINT REFERENCES blueprints(id),
    element_id      BIGINT REFERENCES elements(id),
    original_value  TEXT,
    corrected_value TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

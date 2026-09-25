CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS corpus (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    manifest text NOT NULL,
    profile jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id text PRIMARY KEY,
    title text NOT NULL,
    body text NOT NULL,
    sources jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text text NOT NULL,
    embedding vector NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);

CREATE TABLE IF NOT EXISTS chats (
    id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS turns (
    id uuid PRIMARY KEY,
    chat_id uuid NOT NULL REFERENCES chats(id),
    question text NOT NULL,
    status text NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
    response jsonb,
    error_code text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS turns_chat_idx ON turns(chat_id, created_at, id);

-- Migración inicial: Configurar pgvector
-- 
-- Ejecutar con: psql -U postgres -d fingerprint -f migrations/001_init_pgvector.sql

-- 1. Habilitar extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Las tablas se crean automáticamente vía SQLAlchemy, pero aquí está el schema de referencia

-- Tabla de vectores (creada por SQLAlchemy)
-- CREATE TABLE IF NOT EXISTS fingerprint_vectors (
--     id SERIAL PRIMARY KEY,
--     embedding vector(256) NOT NULL
-- );

-- 3. Crear índice IVFFlat para búsquedas rápidas
-- Nota: El índice se crea automáticamente en el código, pero puede hacerse manualmente:
--
-- Para desarrollo (<100K vectores):
-- CREATE INDEX IF NOT EXISTS fingerprint_vectors_embedding_idx 
-- ON fingerprint_vectors 
-- USING ivfflat (embedding vector_l2_ops)
-- WITH (lists = 100);
--
-- Para producción (100K-1M vectores):
-- CREATE INDEX IF NOT EXISTS fingerprint_vectors_embedding_idx 
-- ON fingerprint_vectors 
-- USING ivfflat (embedding vector_l2_ops)
-- WITH (lists = 1000);
--
-- Para gran escala (>1M vectores):
-- CREATE INDEX IF NOT EXISTS fingerprint_vectors_embedding_idx 
-- ON fingerprint_vectors 
-- USING ivfflat (embedding vector_l2_ops)
-- WITH (lists = 2000);

-- 4. Configuraciones recomendadas para performance
-- Aumentar shared_buffers si tienes RAM suficiente
-- ALTER SYSTEM SET shared_buffers = '4GB';
-- ALTER SYSTEM SET effective_cache_size = '12GB';
-- ALTER SYSTEM SET maintenance_work_mem = '1GB';
-- ALTER SYSTEM SET checkpoint_completion_target = 0.9;
-- ALTER SYSTEM SET wal_buffers = '16MB';
-- ALTER SYSTEM SET default_statistics_target = 100;
-- ALTER SYSTEM SET random_page_cost = 1.1;
-- ALTER SYSTEM SET effective_io_concurrency = 200;
-- 
-- Después de cambiar configuración:
-- SELECT pg_reload_conf();

-- 5. Verificar instalación
SELECT * FROM pg_extension WHERE extname = 'vector';

-- 6. Verificar versión de pgvector
SELECT vector_version();

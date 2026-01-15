-- Migración: Agregar columna image_path a la tabla fingerprints
-- Ejecutar con: psql -U postgres -d fingerprint -f migrations/002_add_image_path.sql

-- Agregar columna image_path si no existe
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'fingerprints' 
        AND column_name = 'image_path'
    ) THEN
        ALTER TABLE fingerprints ADD COLUMN image_path VARCHAR(500);
        RAISE NOTICE 'Columna image_path agregada exitosamente';
    ELSE
        RAISE NOTICE 'Columna image_path ya existe';
    END IF;
END $$;

-- Agregar columna minutiae_data si no existe
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'fingerprints' 
        AND column_name = 'minutiae_data'
    ) THEN
        ALTER TABLE fingerprints ADD COLUMN minutiae_data JSONB;
        RAISE NOTICE 'Columna minutiae_data agregada exitosamente';
    ELSE
        RAISE NOTICE 'Columna minutiae_data ya existe';
    END IF;
END $$;

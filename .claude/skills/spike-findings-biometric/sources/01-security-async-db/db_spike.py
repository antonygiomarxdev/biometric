import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, select

# 1. Base Declarativa Moderna (SQLAlchemy 2.0)
class Base(DeclarativeBase):
    pass

# 2. Modelo de ejemplo
class User(Base):
    __tablename__ = "spike_users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255))

# 3. Configuración del Engine Asíncrono
# Se usa asyncpg o psycopg (v3) en lugar de psycopg2
# Nota: Este spike usa sqlite+aiosqlite solo para demostración sin levantar un PG
DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def main():
    print("=== SPIKE: SQLAlchemy Asíncrono ===")
    
    # Crear tablas
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    # Insertar y consultar de forma 100% asíncrona
    async with AsyncSessionLocal() as session:
        # Insert
        new_user = User(username="forensic_admin", hashed_password="argon2_hash_here")
        session.add(new_user)
        await session.commit()
        
        # Select
        stmt = select(User).where(User.username == "forensic_admin")
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        print(f"\nUsuario recuperado de DB (Async): {user.username}")

if __name__ == "__main__":
    asyncio.run(main())

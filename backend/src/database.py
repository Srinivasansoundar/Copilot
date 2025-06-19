from sqlalchemy.ext.asyncio import create_async_engine,AsyncSession
from sqlalchemy.orm import sessionmaker

from src import config

from urllib.parse import quote_plus

DB_USERNAME = config.DB_USERNAME
DB_PASSWORD = quote_plus(config.DB_PASSWORD)  # encode special chars
DB_HOST = config.DB_HOST
DB_NAME = config.DB_NAME

SQLALCHEMY_DATABASE_URL = f"mysql+aiomysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL
)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autocommit=False, autoflush=False
)
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

env = os.getenv("ENVIRONMENT", "local")
if env == "production":
    DATABASE_URL = os.getenv("DATABASE_URL")
else:
    DATABASE_URL = os.getenv("DATABASE_LOCAL_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

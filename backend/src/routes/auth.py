from datetime import datetime, timedelta
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from dotenv import load_dotenv
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
import pandas as pd
import os
import json

from src.schemas.user import UserCreate, Token
from src.database import get_db
from src.models.user import User
from src.utils.auth import (
    authenticate_user,
    create_access_token,
    get_hashed_password,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Load environment variables
load_dotenv()

router = APIRouter(prefix='/api', tags=['auth_and_chat'])

# ---------------- LLM + SmartDataFrame Setup ----------------
DATASET_PATH = "C:\\Users\\swath\\Desktop\\Project\\eShipz\\Copilot\\backend\\data1.csv"
df = pd.read_csv(DATASET_PATH)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Missing GROQ_API_KEY in .env")

llm = ChatGroq(
    groq_api_key=api_key.strip(),
    model_name="llama3-70b-8192",
    temperature=0.2
)

smart_df = SmartDataframe(df, config={
    "llm": llm,
    "save_logs": False,
    "save_charts": False,
    "verbose": True
})

# ---------------- Register ----------------
@router.post("/auth/register")
async def register_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    if (await db.execute(select(User).filter(User.username == user.username))).scalars().first():
        raise HTTPException(status_code=400, detail="Username already registered")
    if (await db.execute(select(User).filter(User.email == user.email))).scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_hashed_password(user.password)
    db_user = User(email=user.email, username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

# ---------------- Login ----------------
@router.post("/auth/login", response_model=Token)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )

    redis = request.app.state.redis
    await redis.delete(f"user:{user.id}:conversation")
    await redis.hset(f"user:{user.id}:context", mapping={
        "session_start": datetime.utcnow().isoformat(),
        "total_queries": "0",
        "last_activity": datetime.utcnow().isoformat()
    })
    await redis.expire(f"user:{user.id}:context", 86400)
    return Token(access_token=access_token, token_type="Bearer")

# ---------------- Chat ----------------
@router.post("/chat/ask")
async def ask_bot(
    request: Request,
    payload: dict,
    current_user: User = Depends(get_current_user)  # Extracted from JWT
):
    redis = request.app.state.redis
    question = payload.get("query")

    if not question:
        return {"response": {"answer": "Missing question"}}

    user_id = str(current_user.id)
    conversation_key = f"user:{user_id}:conversation"
    context_key = f"user:{user_id}:context"

    # Store the user message first
    await redis.rpush(conversation_key, json.dumps({"role": "user", "message": question}))

    # -------- Fetch Last N Messages --------
    MAX_HISTORY_TURNS = 6  # 3 user-bot pairs
    history_raw = await redis.lrange(conversation_key, -MAX_HISTORY_TURNS, -1)
    history = [json.loads(msg) for msg in history_raw]

    # -------- Format History as Conversation --------
    formatted_history = ""
    for msg in history:
        role = msg.get("role", "user").capitalize()
        message = msg.get("message", "")
        formatted_history += f"{role}: {message}\n"

    # -------- Final Prompt --------
    prompt = (
        f"Conversation history:\n{formatted_history}\n"
        f"Current question: {question}"
    )

    # -------- Call LLM --------
    try:
        answer = smart_df.chat(prompt)
    except Exception as e:
        answer = f"Error during processing: {str(e)}"

    # Store bot response
    await redis.rpush(conversation_key, json.dumps({"role": "bot", "message": answer}))

    # Update context metadata
    await redis.hincrby(context_key, "total_queries", 1)
    await redis.hset(context_key, "last_activity", datetime.utcnow().isoformat())

    return {"response": {"answer": answer}}



# ---------------- Logout ----------------
@router.post("/auth/logout")
async def logout_user(request: Request, current_user: User = Depends(get_current_user)):
    redis = request.app.state.redis
    await redis.delete(f"user:{current_user.id}:conversation")
    await redis.delete(f"user:{current_user.id}:context")
    return {"message": "Successfully logged out"}

from datetime import timedelta,datetime
import logging
from fastapi import APIRouter,Depends,HTTPException,status,Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from src.schemas.user import UserResponse,UserCreate,Token
from src.database import get_db
from src.models.user import User
from src.utils.auth import (
    authenticate_user,
    create_access_token,
    get_hashed_password,
    get_current_user,
    clear_user_session,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
router=APIRouter(
    prefix='/api/auth',
    tags=['authentication']
)

@router.post("/register")
async def register_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    # logger.info(f"Registration attempt for username: {user.username}")
    
    # Check if username already exists
    result = await db.execute(select(User).filter(User.username == user.username))
    db_user_by_username = result.scalars().first()
    if db_user_by_username:
        # logger.warning(f"Registration failed: Username {user.username} already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # Check if email already exists
    result = await db.execute(select(User).filter(User.email == user.email))
    db_user_by_email = result.scalars().first()
    if db_user_by_email:
        # logger.warning(f"Registration failed: Email {user.email} already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    # logger.debug(f"Creating new user with username: {user.username}")
    hashed_password = get_hashed_password(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    # logger.info(f"User registered successfully: {user.username}")
    return db_user
@router.post("/login", response_model=Token)
async def login_for_access_token(
    request: Request,  # Add Request to access app.state
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id}, 
        expires_delta=access_token_expires
    )
    
    # Get Redis client from app state
    redis = request.app.state.redis
    
    # Initialize Redis conversation context
    await redis.delete(f"user:{user.id}:conversation")  # Clear old conversations
    await redis.hset(f"user:{user.id}:context", mapping={
        "session_start": datetime.utcnow().isoformat(),
        "total_queries": "0",
        "last_activity": datetime.utcnow().isoformat()
    })
    await redis.expire(f"user:{user.id}:context", 86400)  # 24 hour expiry
    
    return Token(access_token=access_token, token_type="Bearer")
@router.post("/logout")
async def logout_user(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    try:
        # Get Redis client from app state
        redis = request.app.state.redis
        
        # Clear user's Redis data
        await clear_user_session(redis, current_user.id)
        
        return {"message": "Successfully logged out"}
    
    except Exception as e:
        # logger.error(f"Logout error for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )
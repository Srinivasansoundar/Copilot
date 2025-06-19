from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import auth

from redis.asyncio import Redis
# CORS setup

import httpx
app=FastAPI(
    title="shipper copilot",
    version='v0',
    description="This is the conversational agent for multi user"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
async def startup_event():
    app.state.redis=Redis(host='localhost',port=6379)
    app.state.http_client=httpx.AsyncClient()
@app.on_event("shutdown")
async def shutdown_event():
    app.state.redis.close()
@app.get("/",tags=["health"])
async def health():
    return {'message':'Welcome to the Conversational agent'}
@app.get("/redis-health", tags=["health"])
async def redis_health():
    redis = app.state.redis
    if redis:
        try:
            if redis.ping():
                return {"status": "ok", "message": "Redis connected"}
        except Exception as e:
            return {"status": "error", "message": f"Redis error: {str(e)}"}
    return {"status": "error", "message": "Redis not connected"}
app.include_router(auth.router)

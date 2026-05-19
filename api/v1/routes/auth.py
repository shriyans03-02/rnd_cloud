from fastapi import APIRouter, Depends, HTTPException, Response, Request
from sqlalchemy.orm import Session
import secrets
import json
import base64
from datetime import timedelta

import redis.asyncio as aioredis

from app.db.session import get_db
from app.services.auth_service import AuthService
from app.schemas.auth import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail
from app.core.config import settings
from app.core.dependencies import get_current_user
from app.db.models.user import User
from app.core.redis import get_redis

router = APIRouter()

JWT_COOKIE_KEY       = settings.JWT_COOKIE_KEY
USER_INFO_COOKIE_KEY = settings.USER_INFO_COOKIE_KEY
REFRESH_COOKIE_KEY   = settings.REFRESH_COOKIE_KEY
TICKET_TTL           = settings.TICKET_TTL
REFRESH_TTL          = settings.REFRESH_TTL
IS_SECURE            = settings.SECURE_COOKIES


def _cookie_defaults(httponly: bool, max_age: int) -> dict:
    return dict(
        httponly=httponly,
        secure=True,        # ← hardcode True, not IS_SECURE from env
        samesite="none",    # ← was "strict", must be "none" for cross-origin
        max_age=max_age,
        path="/",
    )


def _set_auth_cookies(response: Response, user: User, access_token: str) -> None:
    """Set all three cookies: access_token, refresh_token, user_info."""
    # 1. HttpOnly JWT — 15 min
    response.set_cookie(
        key=JWT_COOKIE_KEY,
        value=access_token,
        **_cookie_defaults(httponly=True, max_age=60 * 15),
    )

    # 2. Refresh token — 7 days, HttpOnly
    refresh_token = secrets.token_urlsafe(48)
    response.set_cookie(
        key=REFRESH_COOKIE_KEY,
        value=refresh_token,
        **_cookie_defaults(httponly=True, max_age=REFRESH_TTL),
    )

    # 3. Plain user_info for UI state — base64 encoded JSON, 7 days
    user_info_raw = json.dumps({
        "id": user.id,
        "name": f"{user.first_name} {user.last_name}".strip(),
        "email": user.email,
        "role": user.role,
    })
    response.set_cookie(
        key=USER_INFO_COOKIE_KEY,
        value=base64.b64encode(user_info_raw.encode()).decode(),
        **_cookie_defaults(httponly=False, max_age=REFRESH_TTL),
    )

    return refresh_token  # caller stores in Redis


async def _store_refresh_token(refresh_token: str, user_id: int) -> None:
    try:
        r = get_redis()
        await r.setex(f"refresh:{refresh_token}", REFRESH_TTL, str(user_id))
    except Exception:
        # Redis down — login still works, silent refresh won't work
        # but basic JWT auth continues fine for 15 minutes
        pass


async def _revoke_refresh_token(refresh_token: str) -> None:
    try:
        r = get_redis()
        await r.delete(f"refresh:{refresh_token}")
    except Exception:
        pass


@router.post("/login")
async def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    try:
        user = AuthService.login(db, payload.email, password=payload.password)

        detail = ActivityDetail(
            action="login",
            entity="user",
            changes={},
            meta={
                "actor_id": user.id,
                "display_name": f"{user.first_name} {user.last_name}",
            },
        )
        ActivityLogService.log(
            db=db,
            actor_id=user.id,
            target_type=1,
            target_id=user.id,
            detail=detail,
        )
        db.commit()

        token_data = LoginResponse.from_user(user)
        refresh_token = _set_auth_cookies(response, user, token_data.access_token)
        await _store_refresh_token(refresh_token, user.id)

        # Return token + user in body for cross-origin environments
        return {
            "ok": True,
            "access_token": token_data.access_token,
            "user": {
                "id": user.id,
                "name": f"{user.first_name} {user.last_name}".strip(),
                "email": user.email,
                "role": user.role,
            },
        }

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/refresh")
async def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    refresh_token = request.cookies.get(REFRESH_COOKIE_KEY)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")

    try:
        r = get_redis()
        user_id_str = await r.get(f"refresh:{refresh_token}")
    except Exception:
        raise HTTPException(status_code=503, detail="Auth service temporarily unavailable")

    if not user_id_str:
        raise HTTPException(status_code=401, detail="Refresh token expired or invalid")

    user = db.query(User).filter(
        User.id == int(user_id_str),
        User.is_active == True,
    ).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    await _revoke_refresh_token(refresh_token)
    token_data = LoginResponse.from_user(user)
    new_refresh = _set_auth_cookies(response, user, token_data.access_token)
    await _store_refresh_token(new_refresh, user.id)

    # Also return new token in body
    return {
        "ok": True,
        "access_token": token_data.access_token,
    }


@router.post("/logout")
async def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    refresh_token = request.cookies.get(REFRESH_COOKIE_KEY)

    if refresh_token:
        try:
            r = get_redis()
            user_id_str = await r.get(f"refresh:{refresh_token}")

            if user_id_str:
                user = db.query(User).filter(User.id == int(user_id_str)).first()
                if user:
                    try:
                        detail = ActivityDetail(
                            action="logout",
                            entity="user",
                            changes={},
                            meta={
                                "actor_id": user.id,
                                "display_name": f"{user.first_name} {user.last_name}",
                            },
                        )
                        ActivityLogService.log(
                            db=db,
                            actor_id=user.id,
                            target_type=1,
                            target_id=user.id,
                            detail=detail,
                        )
                        db.commit()
                    except Exception:
                        pass  # never block logout due to logging failure

            await _revoke_refresh_token(refresh_token)

        except Exception:
            pass  # Redis down — skip logging and revocation, cookies still get cleared

    # Always runs regardless of Redis state
    response.delete_cookie(key=JWT_COOKIE_KEY, path="/")
    response.delete_cookie(key=REFRESH_COOKIE_KEY, path="/")
    response.delete_cookie(key=USER_INFO_COOKIE_KEY, path="/")
    return {"ok": True}


@router.post("/ws-ticket")
async def get_ws_ticket(current_user: User = Depends(get_current_user)):
    ticket = secrets.token_urlsafe(32)
    r = get_redis()
    await r.setex(f"ws_ticket:{ticket}", 30, str(current_user.id))
    return {"ticket": ticket}


@router.post("/register", response_model=RegisterResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = AuthService.register(db, payload)
        return user
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
from app.domains.user.schema import SignupRequest, SignupResponse, LoginRequest, LoginResponse
from app.domains.user.model import User
from app.domains.user.repository import create_user, get_user_by_email
from app.utils.auth import verify_password, create_jwt_token
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

def signup_user(request: SignupRequest, db: Session) -> SignupResponse:
    existing = get_user_by_email(request.email, db)
    if existing:
        raise HTTPException(status_code=400, detail="⚠이미 가입된 이메일입니다.")
    
    user = create_user(request, db)
    return SignupResponse(id=user.id, email=user.email, name=user.name)

def login_user(request: LoginRequest, db: Session) -> LoginResponse:
    user = get_user_by_email(request.email, db)
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="⚠인증 정보가 유효하지 않습니다.")
    
    token = create_jwt_token(user_id=user.id)
    return LoginResponse(access_token=token)
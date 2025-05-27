from fastapi import APIRouter, Depends, HTTPException, status
from app.domains.user.schema import SignupRequest, SignupResponse, LoginRequest, LoginResponse
from app.domains.user.service import signup_user, login_user
from sqlalchemy.orm import Session
from app.config.database import get_db

router = APIRouter()

@router.post("/signup", response_model=SignupResponse)
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    return signup_user(request, db)

@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    return login_user(request, db)
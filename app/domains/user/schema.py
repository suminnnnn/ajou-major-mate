from pydantic import BaseModel, EmailStr

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str

class SignupResponse(BaseModel):
    id: int
    email: EmailStr
    name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
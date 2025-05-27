from sqlalchemy.orm import Session
from app.domains.user.model import User
from app.domains.user.schema import SignupRequest
from app.config.database import get_db
from app.utils.auth import hash_password
from sqlalchemy.orm import Session
from langgraph_checkpoint_aws.saver import BedrockSessionSaver
import os

def get_user_by_email(email: str, db: Session):
    return db.query(User).filter(User.email == email).first()

def create_user(request: SignupRequest, db: Session):
    
    saver = BedrockSessionSaver(region_name=os.getenv("AWS_REGION"))
    session = saver.session_client.create_session()
    session_id = session.session_id
    
    new_user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        name=request.name,
        bedrock_session_id=session_id
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

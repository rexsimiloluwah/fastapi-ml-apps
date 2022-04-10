from fastapi import HTTPException, APIRouter 
from pydantic import ValidationError, BaseModel
from .chat import get_response

router = APIRouter()

class ChatbotSchema(BaseModel):
    message: str 

@router.post("/predict", response_model=dict)
async def predict(payload:ChatbotSchema):
    message = payload.message
    try: 
        return {
            "response": get_response(message)
        }
    except ValidationError as err: 
        raise HTTPException(status_code=400, detail=err)
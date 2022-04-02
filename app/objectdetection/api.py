from fastapi import APIRouter, HTTPException, File, UploadFile
from .utils import SSDObjectDetection, read_image_file

router = APIRouter()


@router.post("/predict", response_model=dict)
async def predict(file: UploadFile = File(...), save_image: bool = False):
    extension = file.filename.split(".")[-1]
    if not extension in ["jpg", "png", "jpeg"]:
        raise HTTPException(
            detail={
                "status": False,
                "error": "Only image files are supported i.e .jpg, .jpeg, .png",
            },
            status_code=422,
        )

    image = read_image_file(await file.read())
    detector = SSDObjectDetection(save_image=save_image)
    try:
        detector._load_model()
        response = detector.response(image)
        print("Object detection response",response)
        return {"status": True, "data": response}
    except Exception as e:
        return {"status": False, "error": str(e)}

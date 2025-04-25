from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def test_validate():
    return {"message": "Validate route working"}

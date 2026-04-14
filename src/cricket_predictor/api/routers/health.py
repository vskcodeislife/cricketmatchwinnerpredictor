from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter(tags=["health"])


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.head("/health")
@router.head("/")
def head_health() -> Response:
    return Response(status_code=200)

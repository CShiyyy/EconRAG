"""Admin endpoints — destructive operations gated behind explicit confirmation."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipeline.db.reset import hard_reset

router = APIRouter(prefix="/admin", tags=["admin"])


class ResetRequest(BaseModel):
    confirm: bool = False


@router.post("/reset")
def reset_system(body: ResetRequest):
    """Delete the database and LightRAG storage so the system can be re-initialized.

    Requires ``{ "confirm": true }`` in the request body to prevent accidental calls.
    After a successful reset, call ``POST /api/init`` to reinitialize.
    """
    if not body.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to proceed with hard reset")
    result = hard_reset()
    return {"status": "ok", **result}

from __future__ import annotations

from fastapi import HTTPException, Request

from api.state import ApplicationState


def get_state(request: Request) -> ApplicationState:
    state = getattr(request.app.state, "container", None)
    if not isinstance(state, ApplicationState):
        raise HTTPException(status_code=503, detail="Service is not ready.")
    return state

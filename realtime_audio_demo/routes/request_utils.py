import json
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


async def read_json_object(request: Request) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        return None, JSONResponse({"message": f"bad json: {exc}"}, status_code=400)

    if not isinstance(payload, dict):
        return None, JSONResponse({"message": "json body must be an object"}, status_code=400)

    return payload, None

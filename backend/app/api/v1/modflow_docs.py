"""MODFLOW variable definition serving API."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/modflow/definitions",
    tags=["modflow-docs"],
)

# Base path for definition JSON files
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "app" / "data" / "modflow_definitions"

# Try alternative path (when running from backend/)
if not DEFINITIONS_DIR.exists():
    DEFINITIONS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "modflow_definitions"


def _get_definitions_dir() -> Path:
    """Get the definitions directory path."""
    # Try multiple possible locations
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "data" / "modflow_definitions",
        Path(__file__).resolve().parent.parent / "data" / "modflow_definitions",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # Return first candidate even if it doesn't exist


MODEL_TYPE_MAP = {
    "mf6": "mf6",
    "mf2005": "mf2005",
    "mfnwt": "mf2005",  # NWT uses same package definitions as MF2005
    "mfusg": "mfusg",
}


@router.get("/{model_type}")
async def list_package_definitions(model_type: str):
    """List available package definitions for a model type."""
    mapped_type = MODEL_TYPE_MAP.get(model_type.lower())
    if not mapped_type:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

    defs_dir = _get_definitions_dir() / mapped_type
    if not defs_dir.exists():
        return {"model_type": mapped_type, "packages": []}

    packages = []
    for f in sorted(defs_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            packages.append({
                "name": data.get("package_name", f.stem.upper()),
                "description": data.get("description", ""),
                "file_extensions": data.get("file_extensions", []),
            })
        except Exception:
            packages.append({
                "name": f.stem.upper(),
                "description": "",
                "file_extensions": [],
            })

    return {"model_type": mapped_type, "packages": packages}


@router.get("/{model_type}/{package_name}")
async def get_package_definition(model_type: str, package_name: str):
    """Get the full definition for a specific package."""
    mapped_type = MODEL_TYPE_MAP.get(model_type.lower())
    if not mapped_type:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

    defs_dir = _get_definitions_dir() / mapped_type
    json_path = defs_dir / f"{package_name.lower()}.json"

    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No definition found for {package_name} in {mapped_type}",
        )

    try:
        data = json.loads(json_path.read_text())
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading definition: {e}")

"""API v1 router aggregating all endpoint routers."""

from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.projects import router as projects_router
from app.api.v1.upload import router as upload_router
from app.api.v1.visualization import router as visualization_router
from app.api.v1.simulation import router as simulation_router
from app.api.v1.results import router as results_router
from app.api.v1.observations import router as observations_router
from app.api.v1.zonebudget import router as zonebudget_router
from app.api.v1.zonebudget import zone_def_router
from app.api.v1.pest import router as pest_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(projects_router)
api_router.include_router(upload_router)
api_router.include_router(visualization_router)
api_router.include_router(simulation_router)
api_router.include_router(results_router)
api_router.include_router(observations_router)
api_router.include_router(zonebudget_router)
api_router.include_router(zone_def_router)
api_router.include_router(pest_router)

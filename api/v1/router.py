from fastapi import APIRouter
from app.api.v1.routes import (
    health,
    users,
    departments,
    site_hierarchy,
    site_locations,
    device_brands,
    cameras,
    nvrs,
    access_groups,
    members,
    site_location_access,
    member_access,
    embeddings,
    auth,
    tracking,
    reports,
    activity_logs,
    member_tracing,
    notification,
    playback_router,
    hls_proxy,
) 
v1_router = APIRouter(prefix="/v1") 
v1_router.include_router(health.router, tags=["Health"])
v1_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
v1_router.include_router(users.router, prefix="/users", tags=["Users"])
v1_router.include_router(departments.router, prefix="/departments", tags=["Departments"])
v1_router.include_router(site_hierarchy.router, prefix="/site_hierarchy", tags=["SiteHierarchy"])
v1_router.include_router(site_locations.router, prefix="/site_locations", tags=["SiteLocation"])
v1_router.include_router(device_brands.router, prefix="/device-brands", tags=["Device Brands"])
v1_router.include_router(cameras.router, prefix="/cameras", tags=["Cameras"])
v1_router.include_router(nvrs.router, prefix="/nvrs", tags=["NVRs"])
v1_router.include_router(access_groups.router, prefix="/access_groups", tags=["Access Groups"])
v1_router.include_router(members.router, prefix="/members", tags=["Members"])
v1_router.include_router(site_location_access.router, prefix="/site_location_access", tags=["Site Location Acess"])
v1_router.include_router(member_access.router, prefix="/member_access", tags=["Member Acess"])
v1_router.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"]) 
v1_router.include_router(tracking.router, prefix="/tracking", tags=["tracking"])  
v1_router.include_router(reports.router, prefix="/reports", tags=["reports"])  
v1_router.include_router(activity_logs.router, prefix="/activity_logs", tags=["Activity Logs"])
v1_router.include_router(member_tracing.router, prefix="/tracing", tags=["Member_tracing"])
v1_router.include_router(notification.router, prefix="/notification", tags=["notification"])
v1_router.include_router(playback_router.router, prefix="/video_playback", tags=["Video Playback"])
v1_router.include_router(hls_proxy.router, prefix="/hls", tags=["HLS Proxy"])
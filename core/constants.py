from pathlib import Path
import os
 
USER_ROLES = { 
    1: "View Admin", 
    2: "Primary Admin", 
    3: "Super Admin", 
}

MOVEMENT_TYPES = { 
    1: "Entry", 
    2: "Exit",  
}


TARGET_TYPE = {
    1: {"database": "users",           "entity": "user"},
    2: {"database": "departments",     "entity": "department"},
    3: {"database": "site_hierarchies","entity": "site_hierarchy"},
    4: {"database": "cameras",         "entity": "camera"},
    6: {"database": "members",         "entity": "member"},
    5: {"database": "access_groups",   "entity": "access_group"},
    9: {"database": "member_access",   "entity": "member_allocation"},
    8: {"database": "site_location_access", "entity": "grant_site_access"},
    10: {"database": "nvrs",           "entity": "nvr"},
}

NOTIFICATION_TYPE = {
    1:"UNAUTHORIZED_MEMBER",
    2:"UNKNOWN_PERSON"
}

NOTIFICATION_STATUS = {
    1: "UNREAD",
    2: "READ",
}

# --------------------------------
# DEVICE BRAND DEFAULTS / SEEDS
# --------------------------------

SUPPORTED_DEVICE_TYPES = ("camera", "nvr")
SUPPORTED_PLAYBACK_TIME_FORMATS = (
    "hikvision_utc",
    "hikvision_local",
    "cpplus_local",
    "iso_local",
)

PLAYBACK_TIME_FORMAT_LABELS = {
    "hikvision_utc": "Hikvision UTC (20260325T000000Z)",
    "hikvision_local": "Hikvision Local (20260325T000000)",
    "cpplus_local": "CP Plus / Dahua (2026_03_25_00_00_00)",
    "iso_local": "ISO Local (2026-03-25T00:00:00)",
}

DEFAULT_DEVICE_BRANDS = [
    {
        "name": "hikvision",
        "label": "Hikvision",
        "device_type": "camera",
        "live_rtsp_template": "rtsp://{rtsp_username}:{rtsp_password}@{ip_address}:{rtsp_port}/Streaming/channels/{rtsp_channel}",
        "playback_rtsp_template": None,
        "playback_time_format": None,
    },
    {
        "name": "cpplus",
        "label": "CP Plus",
        "device_type": "camera",
        "live_rtsp_template": "rtsp://{rtsp_username}:{rtsp_password}@{ip_address}:{rtsp_port}/cam/realmonitor?channel={rtsp_channel}&subtype={rtsp_subtype}",
        "playback_rtsp_template": None,
        "playback_time_format": None,
    },
    {
        "name": "dahua",
        "label": "Dahua",
        "device_type": "camera",
        "live_rtsp_template": "rtsp://{rtsp_username}:{rtsp_password}@{ip_address}:{rtsp_port}/cam/realmonitor?channel={rtsp_channel}&subtype={rtsp_subtype}",
        "playback_rtsp_template": None,
        "playback_time_format": None,
    },
    {
        "name": "generic",
        "label": "Generic",
        "device_type": "camera",
        "live_rtsp_template": "rtsp://{rtsp_username}:{rtsp_password}@{ip_address}:{rtsp_port}/{rtsp_channel}/{rtsp_subtype}",
        "playback_rtsp_template": None,
        "playback_time_format": None,
    },
    {
        "name": "hikvision",
        "label": "Hikvision",
        "device_type": "nvr",
        "live_rtsp_template": None,
        "playback_rtsp_template": "rtsp://{username}:{password}@{ip_address}:{port}/Streaming/tracks/{channel}01?starttime={starttime}&endtime={endtime}&streamkey={stream_key}",
        "playback_time_format": "hikvision_utc",
    },
    {
        "name": "cpplus",
        "label": "CP Plus",
        "device_type": "nvr",
        "live_rtsp_template": None,
        "playback_rtsp_template": "rtsp://{username}:{password}@{ip_address}:{port}/cam/playback?channel={channel}&starttime={starttime}&endtime={endtime}",
        "playback_time_format": "cpplus_local",
    },
    {
        "name": "dahua",
        "label": "Dahua",
        "device_type": "nvr",
        "live_rtsp_template": None,
        "playback_rtsp_template": "rtsp://{username}:{password}@{ip_address}:{port}/cam/playback?channel={channel}&starttime={starttime}&endtime={endtime}",
        "playback_time_format": "cpplus_local",
    },
    {
        "name": "generic",
        "label": "Generic",
        "device_type": "nvr",
        "live_rtsp_template": None,
        "playback_rtsp_template": "rtsp://{username}:{password}@{ip_address}:{port}/cam/playback?channel={channel}&starttime={starttime}&endtime={endtime}",
        "playback_time_format": "cpplus_local",
    },
]

# Backward-compatible fallback dictionaries. New code reads brand rows from
# device_brands first and uses these only when old rows have no brand_id.
SUPPORTED_BRANDS = tuple(sorted({row["name"] for row in DEFAULT_DEVICE_BRANDS}))
CAMERA_RTSP_URL_TEMPLATES = {
    row["name"]: row["live_rtsp_template"]
    for row in DEFAULT_DEVICE_BRANDS
    if row["device_type"] == "camera"
}
NVR_PLAYBACK_RTSP_TEMPLATES = {
    row["name"]: row["playback_rtsp_template"]
    for row in DEFAULT_DEVICE_BRANDS
    if row["device_type"] == "nvr"
}
NVR_PLAYBACK_TIME_FORMATS = {
    row["name"]: row["playback_time_format"]
    for row in DEFAULT_DEVICE_BRANDS
    if row["device_type"] == "nvr"
}

# --------------------------------
# EMBEDDING CONSTANTS
# --------------------------------

# --- EDIT / override with env vars ---
# Runtime live camera sources now come from the cameras DB table.
# Keep this empty so embedding extraction does not fall back to hard-coded streams.
RTSP_STREAMS = []


YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
DEVICE = os.getenv("DEVICE", "cuda:0")
CONF_THRES = float(os.getenv("CONF_THRES", 0.80))
IOU_THRES = float(os.getenv("IOU_THRES", 0.40))


EMB_CSV = os.getenv("EMB_CSV", "live_embeddings_multi.csv")
CROPS_ROOT = os.getenv("CROPS_ROOT", "crops")
EMB_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:1234@192.168.1.136:5432/m_vision",
)


# locks / simple runtime globals live inside services/embedding_service.py


# ensure crops dir exists by default
Path(CROPS_ROOT).mkdir(parents=True, exist_ok=True) 

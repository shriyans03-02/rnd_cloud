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
# EMBEDDING CONSTANTS
# --------------------------------

# --- EDIT / override with env vars ---
RTSP_STREAMS = [
   "rtsp://admin:rolex%40123@192.168.1.111:554/Streaming/channels/101"
    
]


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
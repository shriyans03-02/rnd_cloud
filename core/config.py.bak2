from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    DATABASE_URL: str

    # Your .env currently uses lowercase pipeline_args.
    pipeline_args: str = Field(default="")

    # RTSP config for CP Plus / Dahua live stream format:
    # rtsp://admin:admin%40123@10.10.43.251:554/cam/realmonitor?channel=1&subtype=0
    RTSP_USER: str = "admin"
    RTSP_PASS: str = "admin%40123"

    # Optional aliases supported by embedding_service.py.
    # If these are set, they override RTSP_USER / RTSP_PASS in embedding_service.py.
    RTSP_USERNAME: str = ""
    RTSP_PASSWORD: str = ""

    RTSP_PORT: str = "554"
    RTSP_PATH: str = "/cam/realmonitor"
    RTSP_SCHEME: str = "rtsp"
    RTSP_STREAM: str = ""

    RTSP_CHANNEL: str = "1"
    RTSP_SUBTYPE: str = "0"

    RTSP_URL_TEMPLATE: str = (
        "rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor"
        "?channel={channel}&subtype={subtype}"
    )

    FFMPEG_BIN: str = "ffmpeg"

    JWT_SECRET: str
    REDIS_URL: str
    SECURE_COOKIES: bool = True
    JWT_COOKIE_KEY: str = "access_token"
    USER_INFO_COOKIE_KEY: str = "user_info"
    REFRESH_COOKIE_KEY: str = "refresh_token"
    TICKET_TTL: int = 30
    REFRESH_TTL: int = 604800

    # NVR / CP Plus playback config
    NVR_IP: str = ""
    NVR_USER: str = ""
    NVR_PASS: str = ""
    NVR_PORT: str = "554"
    # Kept for legacy Hikvision deployments; CP Plus playback does not use it.
    NVR_STREAM_KEY: str = ""
    # CP Plus playback URL example:
    # rtsp://admin:Admin%40123@192.168.1.245:554/cam/playback?channel=1&starttime=2026_04_24_11_30_00&endtime=2026_04_24_12_00_00
    NVR_PLAYBACK_PATH: str = "/cam/playback"
    NVR_PLAYBACK_URL_TEMPLATE: str = (
        "rtsp://{username}:{password}@{ip}:{port}/cam/playback"
        "?channel={channel}&starttime={starttime}&endtime={endtime}"
    )
    # CP Plus site convention: the UI camera_id is the NVR channel number.
    # Set to "channel_map" only if you want CHANNEL_MAP camera_id:channel overrides.
    NVR_PLAYBACK_CHANNEL_SOURCE: str = "camera_id"
    # Used when timestamps arrive with UTC offsets/Z; naive timestamps are used as-is.
    NVR_PLAYBACK_TIMEZONE: str = "Asia/Kolkata"

    # MediaMTX
    MEDIAMTX_RTSP: str = "rtsp://localhost:8554"
    MEDIAMTX_INTERNAL: str = "http://localhost:8888"
    HLS_BASE_URL: str = ""
    # Generate mediamtx.yml from the cameras DB table at backend startup.
    # live/cam<ID> is created for every active camera; no hard-coded MediaMTX camera list required.
    MEDIAMTX_AUTOCONFIG: bool = True
    MEDIAMTX_CONFIG_PATH: str = "/root/mediamtx.yml"
    MEDIAMTX_PUBLIC_IP: str = "164.52.214.233"
    MEDIAMTX_SOURCE_ON_DEMAND: bool = True
    MEDIAMTX_CAMERA_RTSP_TRANSPORT: str = "tcp"
    MEDIAMTX_WRITE_QUEUE_SIZE: int = 256
    # Optional: let FastAPI start MediaMTX if port 8554 is not already open.
    MEDIAMTX_AUTOSTART: bool = True
    MEDIAMTX_BIN: str = "./mediamtx"
    MEDIAMTX_START_WAIT_SECONDS: float = 8.0

    # Optional NVDEC/H265 restream stage:
    # DB camera -> MediaMTX live/cam<ID> -> FFmpeg NVDEC restream -> MediaMTX ai/cam<ID>
    # Pipeline can then read ai/cam<ID>, which is lower-res clean H264.
    NVDEC_RESTREAM_ENABLED: bool = False
    NVDEC_RESTREAM_INPUT_CODEC: str = "hevc"  # hevc or h264
    NVDEC_RESTREAM_DECODER: str = "auto"      # auto, hevc_cuvid, h264_cuvid
    NVDEC_RESTREAM_ENCODER: str = "libx264"   # libx264 or h264_nvenc if available
    NVDEC_RESTREAM_WIDTH: int = 852
    NVDEC_RESTREAM_HEIGHT: int = 480
    NVDEC_RESTREAM_FPS: int = 20
    NVDEC_RESTREAM_BITRATE: str = "1200k"
    NVDEC_RESTREAM_BUFSIZE: str = "2400k"
    NVDEC_RESTREAM_GOP: int = 20
    NVDEC_RESTREAM_PRESET: str = "ultrafast"
    NVDEC_RESTREAM_USE_GPU_SCALE: bool = True
    NVDEC_RESTREAM_LOG_DIR: str = "logs/nvdec_restream"
    NVDEC_RESTREAM_INPUT_PREFIX: str = "live/cam"
    NVDEC_RESTREAM_OUTPUT_PREFIX: str = "ai/cam"
    NVDEC_RESTREAM_WARMUP_SECONDS: float = 5.0
    NVDEC_RESTREAM_RESTART_SECONDS: float = 2.0
    NVDEC_RESTREAM_MAX_CAMERAS: int = 0
    NVDEC_RESTREAM_CAMERA_IDS: str = ""

    # MediaMTX WebRTC / WHEP delivery for processed streams.
    # MEDIAMTX_WEBRTC_PUBLIC_BASE must be reachable by the browser.
    MEDIAMTX_WEBRTC_INTERNAL: str = "http://localhost:8889"
    MEDIAMTX_WEBRTC_PUBLIC_BASE: str = "http://localhost:8889"

    # Live processed-frame RTSP publisher. It publishes annotated frames to
    # MEDIAMTX_RTSP/<TRACKING_WEBRTC_PATH_PREFIX>{camera_id}; MediaMTX then
    # exposes the same path over WebRTC.
    TRACKING_WEBRTC_ENABLED: bool = True
    TRACKING_WEBRTC_PATH_PREFIX: str = "tracked/cam"
    TRACKING_WEBRTC_FPS: float = 20.0
    TRACKING_WEBRTC_GOP: int = 20
    TRACKING_WEBRTC_WIDTH: int = 852
    TRACKING_WEBRTC_HEIGHT: int = 480
    TRACKING_WEBRTC_CODEC: str = "auto"
    TRACKING_WEBRTC_BITRATE: str = "1800k"
    TRACKING_WEBRTC_BUFSIZE: str = "360k"
    TRACKING_WEBRTC_X264_PRESET: str = "ultrafast"
    TRACKING_WEBRTC_MODE: str = "hybrid"
    TRACKING_WEBRTC_OVERLAY_MAX_AGE_MS: int = 1500
    TRACKING_WEBRTC_DRAW_STATS: bool = True
    TRACKING_WEBRTC_LOG_DIR: str = "logs/ffmpeg_webrtc"
    TRACKING_WEBRTC_BOOTSTRAP_PLACEHOLDER: bool = True

    # Live unknown-box visibility.  Defaults keep live monitoring behavior unchanged:
    # known + unknown boxes are visible, with yellow Unknown labels.
    # To hide unknown boxes only in live streams, set TRACKING_HIDE_UNKNOWN=True.
    TRACKING_HIDE_UNKNOWN: bool = False
    TRACKING_SHOW_UNKNOWN_LABELS: bool = True

    # Startup resilience: camera table may be empty in a new DB, or RTSP cameras
    # may be temporarily unreachable while Tailscale/subnet routing is being fixed.
    # In that case FastAPI must still start so user/member/auth/admin APIs work.
    TRACKING_SOFT_START: bool = True
    TRACKING_ALLOW_EMPTY_SOURCES: bool = True
    TRACKING_FAIL_ON_NO_SOURCES: bool = False
    MEDIAMTX_KEEP_EXISTING_ON_EMPTY_DB: bool = True


    # Multi-process live AI scaling.
    # In production keep FastAPI API-only and run live AI via ai_worker.py.
    TRACKING_EXTERNAL_WORKERS: bool = False
    TRACKING_IN_API: bool = True
    AI_WORKER_CAMERA_LIMIT: int = 8
    AI_WORKER_REFRESH_SECONDS: int = 60
    AI_WORKER_WORKERS: int = 0
    AI_WORKER_LOG_DIR: str = "logs/ai_workers"

    # Embedding extraction UI preview. The backend publishes detected frames via
    # /api/v1/embeddings/preview and /api/v1/embeddings/preview.jpg instead of
    # opening a local cv2.imshow window on the server.
    EMBEDDING_PREVIEW_FPS: float = 8.0
    EMBEDDING_PREVIEW_JPEG_QUALITY: int = 80
    EMBEDDING_ENABLE_CV2_VIEWER: bool = False

    # Do not start one FFmpeg encoder per camera at backend startup.  Publishers
    # are started on demand when /v1/tracking/webrtc/{camera_id} is requested.
    # This is important for 12+ cameras on A100 because A100 has no NVENC;
    # libx264 encoding is CPU-bound.
    TRACKING_WEBRTC_AUTOSTART: bool = False
    TRACKING_WEBRTC_MAX_ACTIVE_PUBLISHERS: int = 0  # 0 = unlimited; set 4/6 if CPU is tight

    # Playback tracing stability settings.
    # Playback is on-demand and can run while the 12-camera live pipeline is already using CUDA.
    # Keep playback CPU-safe by default to avoid cross-framework CUDA/cudNN crashes.
    PLAYBACK_DEVICE: str = "cuda:0"
    PLAYBACK_YOLO_WEIGHTS: str = "yolov8n.pt"
    PLAYBACK_YOLO_IMGSZ: int = 512
    PLAYBACK_CONF: float = 0.30
    PLAYBACK_IOU: float = 0.45
    PLAYBACK_HALF: bool = True
    PLAYBACK_CUDNN_BENCHMARK: bool = True
    PLAYBACK_TRACKER_BACKEND: str = "iou"  # iou, bytetrack, deepsort, strongsort
    # Playback identity matching.  PLAYBACK_IDENTITY_MATCHING_ENABLED=True
    # means annotated playback uses DB face embeddings: member requests load only
    # the requested member; location requests load all active members.
    PLAYBACK_IDENTITY_MATCHING_ENABLED: bool = True
    PLAYBACK_FORCE_FACE_FOR_IDENTITY: bool = True
    PLAYBACK_USE_FACE: bool = True
    PLAYBACK_FACE_PROVIDER: str = "cpu"  # cpu is safest while live CUDA pipeline is running
    PLAYBACK_FACE_DET_SIZE: str = "640 640"
    PLAYBACK_FACE_EVERY_N: int = 10
    PLAYBACK_MEMBER_FACE_EVERY_N: int = 5
    PLAYBACK_LOCATION_FACE_EVERY_N: int = 10
    PLAYBACK_FACE_THRESH: float = 0.45
    PLAYBACK_FACE_GAP: float = 0.03
    PLAYBACK_FACE_STRONG_THRESH: float = 0.55
    PLAYBACK_MIN_FACE_DET_SCORE: float = 0.45
    PLAYBACK_MIN_FACE_PX: int = 18
    PLAYBACK_MIN_FACE_AREA_RATIO: float = 0.003
    PLAYBACK_FACE_IOU_LINK: float = 0.25
    PLAYBACK_VIDEO_FPS: float = 20.0
    # When True, playback behaves like a player: if AI cannot process every
    # frame, stale decoded frames are skipped so the clip does not run in slow
    # motion. The source is already clean H264 from playback_clean_*, so dropping
    # decoded frames here does not break HEVC/H264 reference chains.
    PLAYBACK_REALTIME_MODE: bool = True
    PLAYBACK_KEEP_ALL_FRAMES: bool = False
    PLAYBACK_QUEUE_SIZE: int = 2
    PLAYBACK_MAX_QUEUE_AGE_MS: int = 300
    PLAYBACK_MAX_DRAIN_PER_CYCLE: int = 512
    PLAYBACK_AI_READ_LATEST_ONLY: bool = True
    PLAYBACK_DECODE_LATEST_FRAME_ONLY: bool = True
    PLAYBACK_HIDE_UNKNOWN: bool = True
    PLAYBACK_STREAM_FREEZE_SECONDS: float = 300.0
    PLAYBACK_STREAM_OPEN_TIMEOUT_MS: int = 8000
    PLAYBACK_STREAM_READ_TIMEOUT_MS: int = 8000
    PLAYBACK_MAX_ACTIVE_SESSIONS: int = 1
    # Wait for actual decoded/AI frames before returning a playback session.
    # If CUDA playback AI does not produce frames, automatically restart only
    # the playback AI runner on CPU while keeping the clean restream/WebRTC path alive.
    PLAYBACK_WAIT_FOR_RAW_SECONDS: float = 30.0
    PLAYBACK_WAIT_FOR_AI_SECONDS: float = 20.0
    PLAYBACK_AI_FALLBACK_ON_TIMEOUT: bool = True
    PLAYBACK_AI_FALLBACK_DEVICE: str = "cpu"
    PLAYBACK_AI_ERROR_LOG_INTERVAL_SECONDS: float = 2.0
    # Playback publisher smoothing. Keep MediaMTX path alive but do not flash
    # black placeholder frames between decoded/annotated frames.
    PLAYBACK_HOLD_LAST_FRAME: bool = True
    PLAYBACK_PLACEHOLDER_BEFORE_FIRST_FRAME_ONLY: bool = True
    PLAYBACK_BOOTSTRAP_PLACEHOLDER: bool = True
    PLAYBACK_WEBRTC_MODE: str = "hybrid"
    PLAYBACK_WEBRTC_WIDTH: int = 1280
    PLAYBACK_WEBRTC_HEIGHT: int = 720
    PLAYBACK_WEBRTC_FPS: float = 20.0
    PLAYBACK_VIDEO_FPS: float = 20.0
    PLAYBACK_WEBRTC_GOP: int = 40
    PLAYBACK_WEBRTC_CODEC: str = "auto"
    PLAYBACK_WEBRTC_BITRATE: str = "6000k"
    PLAYBACK_WEBRTC_BUFSIZE: str = "12000k"
    PLAYBACK_WEBRTC_PRESET: str = "ultrafast"
    PLAYBACK_WEBRTC_OVERLAY_MAX_AGE_MS: int = 800

    # CP Plus playback clean restream stage. This is intentionally separate from
    # the live NVDEC restream: CP Plus H265 playback can start mid-GOP and is
    # fragile when OpenCV/AI reads it directly. FFmpeg buffers/decodes it first,
    # republishes a stable local H264 stream to MediaMTX playback_clean_*, then
    # pipeline_tracing reads that clean stream.
    PLAYBACK_CLEAN_RESTREAM_ENABLED: bool = True
    PLAYBACK_CLEAN_RESTREAM_INPUT_CODEC: str = "hevc"
    PLAYBACK_CLEAN_RESTREAM_DECODER: str = "hevc"  # software HEVC decoder by default
    PLAYBACK_CLEAN_RESTREAM_ENCODER: str = "libx264"
    PLAYBACK_CLEAN_RESTREAM_WIDTH: int = 1280
    PLAYBACK_CLEAN_RESTREAM_HEIGHT: int = 720
    PLAYBACK_CLEAN_RESTREAM_FPS: float = 20.0
    PLAYBACK_CLEAN_RESTREAM_BITRATE: str = "6000k"
    PLAYBACK_CLEAN_RESTREAM_BUFSIZE: str = "12000k"
    PLAYBACK_CLEAN_RESTREAM_GOP: int = 40
    PLAYBACK_CLEAN_RESTREAM_ALL_I: bool = False
    PLAYBACK_CLEAN_RESTREAM_PRESET: str = "ultrafast"
    PLAYBACK_CLEAN_RESTREAM_RTSP_TRANSPORT: str = "tcp"
    PLAYBACK_CLEAN_RESTREAM_LOG_DIR: str = "logs/playback_clean_restream"
    PLAYBACK_CLEAN_RESTREAM_PATH_PREFIX: str = "playback_clean_"
    PLAYBACK_CLEAN_RESTREAM_WARMUP_SECONDS: float = 6.0
    PLAYBACK_CLEAN_RESTREAM_FFMPEG_FLAGS: str = "-fflags +genpts -flags2 +showall -err_detect ignore_err -max_delay 1000000 -analyzeduration 3000000 -probesize 3000000"
    PLAYBACK_CLEAN_RESTREAM_FORCE_FPS_FILTER: bool = False
    PLAYBACK_CLEAN_RESTREAM_READY_PROBE: bool = True
    PLAYBACK_PUBLISHER_READY_TIMEOUT_SECONDS: float = 2.0

    # Optional playback overlay tuning used by patched pipeline_tracing draw paths.
    PLAYBACK_OVERLAY_FONT_SCALE: float = 0.45
    PLAYBACK_OVERLAY_THICKNESS: int = 1
    PLAYBACK_OVERLAY_LABEL_MAX_CHARS: int = 24
    PLAYBACK_SHOW_TRACK_ID: bool = True
    PLAYBACK_SHOW_UNKNOWN_LABELS: bool = False

    # Camera channel map, e.g. "11:101,12:301,10:201,9:401"
    CHANNEL_MAP: str = ""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def rtsp_username(self) -> str:
        return self.RTSP_USERNAME or self.RTSP_USER

    @property
    def rtsp_password(self) -> str:
        return self.RTSP_PASSWORD or self.RTSP_PASS

    @property
    def channel_map_dict(self) -> dict[int, int]:
        """
        Parse:
            "1:101,2:301"

        Into:
            {1: 101, 2: 301}
        """
        result: dict[int, int] = {}

        if not self.CHANNEL_MAP.strip():
            return result

        for pair in self.CHANNEL_MAP.split(","):
            pair = pair.strip()
            if not pair:
                continue

            if ":" not in pair:
                continue

            cam, channel = pair.split(":", 1)

            try:
                result[int(cam.strip())] = int(channel.strip())
            except ValueError:
                continue

        return result


settings = Settings()



# from pydantic_settings import BaseSettings, SettingsConfigDict
# from pydantic import Field

# class Settings(BaseSettings):
#     DATABASE_URL: str
#     pipeline_args: str = Field(..., env="PIPELINE_ARGS")
#     RTSP_USER: str = ""
#     RTSP_PASS: str = ""
#     RTSP_PORT: str = ""
#     RTSP_PATH: str = ""
#     RTSP_SCHEME: str = "rtsp"
#     RTSP_STREAM: str = ""
#     RTSP_URL_TEMPLATE: str = ""
#     JWT_SECRET: str
#     REDIS_URL: str
#     SECURE_COOKIES: bool
#     JWT_COOKIE_KEY: str
#     USER_INFO_COOKIE_KEY: str
#     REFRESH_COOKIE_KEY: str
#     TICKET_TTL: int
#     REFRESH_TTL: int
#     NVR_IP: str
#     NVR_USER: str
#     NVR_PASS: str
#     NVR_STREAM_KEY: str
#     MEDIAMTX_RTSP: str
#     MEDIAMTX_RTSP: str          # used by ffmpeg to push stream  e.g. rtsp://localhost:8554
#     MEDIAMTX_INTERNAL: str      # used by proxy to pull HLS     e.g. http://localhost:8888
#     CHANNEL_MAP: str
#     FFMPEG_BIN: str = "ffmpeg"       # ← add this

#     model_config = SettingsConfigDict(  # ← only ONE model_config (removed duplicate)
#         env_file=".env",
#         case_sensitive=True,
#         extra="forbid",
#     )

#     @property
#     def channel_map_dict(self) -> dict[int, int]:
#         """Parse '1:101,2:301' → {1: 101, 2: 301}"""
#         result = {}
#         for pair in self.CHANNEL_MAP.split(","):
#             cam, channel = pair.strip().split(":")
#             result[int(cam)] = int(channel)
#         return result


# settings = Settings()

# import os
# print("----------------------------------OS sees PIPELINE_ARGS =", os.environ.get("PIPELINE_ARGS"))



# ________________________________for hospital________________________________
# from pydantic_settings import BaseSettings, SettingsConfigDict
# from pydantic import Field
# import os


# class Settings(BaseSettings):
#     DATABASE_URL: str
#     pipeline_args: str = Field(..., env="PIPELINE_ARGS")

#     # Legacy RTSP fields
#     RTSP_USER: str = ""
#     RTSP_PASS: str = ""
#     RTSP_PORT: str = "554"
#     RTSP_PATH: str = ""
#     RTSP_SCHEME: str = "rtsp"
#     RTSP_STREAM: str = ""
#     RTSP_URL_TEMPLATE: str = ""

#     # New RTSP fields
#     RTSP_USERNAME: str = ""
#     RTSP_PASSWORD: str = ""
#     RTSP_CHANNEL: str = "1"
#     RTSP_SUBTYPE: str = "0"

#     JWT_SECRET: str
#     REDIS_URL: str
#     SECURE_COOKIES: bool
#     JWT_COOKIE_KEY: str
#     USER_INFO_COOKIE_KEY: str
#     REFRESH_COOKIE_KEY: str
#     TICKET_TTL: int
#     REFRESH_TTL: int

#     NVR_IP: str
#     NVR_USER: str
#     NVR_PASS: str
#     NVR_STREAM_KEY: str

#     MEDIAMTX_RTSP: str
#     HLS_BASE_URL: str
#     CHANNEL_MAP: str = ""

#     FFMPEG_BIN: str = "ffmpeg"

#     model_config = SettingsConfigDict(
#         env_file=".env",
#         case_sensitive=True,
#         extra="ignore",
#     )

#     @property
#     def channel_map_dict(self) -> dict[int, int]:
#         """
#         Parse CHANNEL_MAP like:
#         '1:101,2:301' -> {1: 101, 2: 301}

#         Returns empty dict if CHANNEL_MAP is blank.
#         """
#         result: dict[int, int] = {}
#         raw = (self.CHANNEL_MAP or "").strip()
#         if not raw:
#             return result

#         for pair in raw.split(","):
#             pair = pair.strip()
#             if not pair:
#                 continue
#             cam, channel = pair.split(":")
#             result[int(cam.strip())] = int(channel.strip())
#         return result

#     @property
#     def rtsp_username_effective(self) -> str:
#         return (self.RTSP_USERNAME or self.RTSP_USER or "").strip()

#     @property
#     def rtsp_password_effective(self) -> str:
#         return (self.RTSP_PASSWORD or self.RTSP_PASS or "").strip()


# settings = Settings()

# print("----------------------------------OS sees PIPELINE_ARGS =", os.environ.get("PIPELINE_ARGS"))

# # -----------------------------------NEW UPDATE_______________________FOR HOSPITAL
# from pydantic_settings import BaseSettings, SettingsConfigDict
# from pydantic import Field


# class Settings(BaseSettings):
#     # ===================== Core =====================
#     DATABASE_URL: str
#     pipeline_args: str = Field(..., env="PIPELINE_ARGS")

#     # ===================== RTSP (FIXED) =====================
#     RTSP_USERNAME: str = ""
#     RTSP_PASSWORD: str = ""
#     RTSP_PORT: int = 554
#     RTSP_CHANNEL: int = 1
#     RTSP_SUBTYPE: int = 0

#     RTSP_PATH: str = ""
#     RTSP_SCHEME: str = "rtsp"
#     RTSP_STREAM: str = ""
#     RTSP_URL_TEMPLATE: str = ""

#     # ===================== Auth =====================
#     JWT_SECRET: str
#     REDIS_URL: str
#     SECURE_COOKIES: bool

#     JWT_COOKIE_KEY: str
#     USER_INFO_COOKIE_KEY: str
#     REFRESH_COOKIE_KEY: str

#     TICKET_TTL: int
#     REFRESH_TTL: int

#     # ===================== NVR =====================
#     NVR_IP: str
#     NVR_USER: str
#     NVR_PASS: str
#     NVR_STREAM_KEY: str

#     # ===================== MediaMTX =====================
#     MEDIAMTX_RTSP: str        # e.g. rtsp://localhost:8554
#     MEDIAMTX_INTERNAL: str    # e.g. http://localhost:8888

#     # ===================== Camera =====================
#     CHANNEL_MAP: str

#     # ===================== FFmpeg =====================
#     FFMPEG_BIN: str = "ffmpeg"

#     # ===================== Settings Config =====================
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         case_sensitive=True,
#         extra="forbid",   # strict mode (safe now)
#     )

#     # ===================== Helpers =====================
#     @property
#     def channel_map_dict(self) -> dict[int, int]:
#         """Parse '1:101,2:301' → {1: 101, 2: 301}"""
#         result = {}
#         for pair in self.CHANNEL_MAP.split(","):
#             cam, channel = pair.strip().split(":")
#             result[int(cam)] = int(channel)
#         return result


# # ===================== Init =====================
# settings = Settings()


# # ===================== Debug =====================
# import os
# print("----------------------------------OS sees PIPELINE_ARGS =", os.environ.get("PIPELINE_ARGS"))
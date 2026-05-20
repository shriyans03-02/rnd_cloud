from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    DATABASE_URL: str

    # Your .env currently uses lowercase pipeline_args.
    pipeline_args: str = Field(default="")

    # RTSP config for this camera URL format:
    # rtsp://admin:Admin%40123@192.168.1.161:554/video/live?channel=1&subtype=0
    RTSP_USER: str = "admin"
    RTSP_PASS: str = "Admin%40123"

    # Optional aliases supported by embedding_service.py.
    # If these are set, they override RTSP_USER / RTSP_PASS in embedding_service.py.
    RTSP_USERNAME: str = ""
    RTSP_PASSWORD: str = ""

    RTSP_PORT: str = "554"
    RTSP_PATH: str = "/video/live"
    RTSP_SCHEME: str = "rtsp"
    RTSP_STREAM: str = ""

    RTSP_CHANNEL: str = "1"
    RTSP_SUBTYPE: str = "0"

    RTSP_URL_TEMPLATE: str = (
        "rtsp://{username}:{password}@{ip}:{port}/video/live"
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

    # NVR config
    NVR_IP: str = ""
    NVR_USER: str = ""
    NVR_PASS: str = ""
    NVR_STREAM_KEY: str = ""

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
    # Do not start one FFmpeg encoder per camera at backend startup.  Publishers
    # are started on demand when /v1/tracking/webrtc/{camera_id} is requested.
    # This is important for 12+ cameras on A100 because A100 has no NVENC;
    # libx264 encoding is CPU-bound.
    TRACKING_WEBRTC_AUTOSTART: bool = False
    TRACKING_WEBRTC_MAX_ACTIVE_PUBLISHERS: int = 0  # 0 = unlimited; set 4/6 if CPU is tight

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
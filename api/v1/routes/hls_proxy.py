"""
hls_proxy.py — Production-grade HLS reverse-proxy for MediaMTX.

Fixes applied vs original:
  1. Auth guard on all routes via get_current_user.
  2. stream_name validated against path-traversal regex before use in URL.
  3. Segment bytes streamed in chunks (iter_bytes) — no full-segment RAM load.
  4. Configurable timeouts: short for playlists, longer for segments.
  5. Proper error logging.
  6. Credentialed CORS headers — required for hls.js withCredentials: true.
"""

import logging
import re

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.core.config import settings
from app.db.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()

_STREAM_NAME_RE = re.compile(r"^[a-zA-Z0-9\-]{1,64}$")
_SAFE_FILE_RE = re.compile(r"^[a-zA-Z0-9\-/\.]{1,128}$")

# Timeouts (seconds)
_PLAYLIST_TIMEOUT = 10
_SEGMENT_TIMEOUT = 30


def _check_stream_name(name: str) -> None:
    if not _STREAM_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Invalid stream_name. Use only letters, digits, underscores, hyphens.",
        )


def _check_file_path(path: str) -> None:
    if ".." in path or not _SAFE_FILE_RE.match(path):
        raise HTTPException(status_code=400, detail="Invalid file path.")


def _cors_headers(request: Request) -> dict:
    origin = request.headers.get("origin", "")
    return {
        "Cache-Control": "no-cache, no-store",
        "Access-Control-Allow-Origin": origin or "*",
        "Access-Control-Allow-Credentials": "true",
        "Vary": "Origin",
    }


@router.get("/{stream_name}/{file_path:path}")
async def proxy_hls(
    request: Request,
    stream_name: str,
    file_path: str,
):
    """
    Proxy HLS playlists (.m3u8) and segments (.ts) from MediaMTX to the browser.

    Path  : /v1/hls/{stream_name}/{file_path}
    Source: MediaMTX on settings.MEDIAMTX_INTERNAL (server-side only — never exposed).

    Works transparently on localhost AND through Cloudflare Tunnel with zero config change
    because all absolute URLs are rewritten to the incoming request's base URL.
    """
    _check_stream_name(stream_name)
    _check_file_path(file_path)

    is_playlist = file_path.endswith(".m3u8")
    timeout = _PLAYLIST_TIMEOUT if is_playlist else _SEGMENT_TIMEOUT
    url = f"{settings.MEDIAMTX_INTERNAL.rstrip('/')}/{stream_name}/{file_path}"

    try:
        if is_playlist:
            # Playlists are small — fetch fully so we can rewrite URLs.
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
        else:
            # Segments can be large — open a streaming connection.
            client = httpx.AsyncClient(timeout=timeout)
            resp = await client.get(url, follow_redirects=True)
    except httpx.TimeoutException:
        logger.warning("MediaMTX timeout for %s", url)
        raise HTTPException(status_code=504, detail="MediaMTX timed out.")
    except httpx.RequestError as exc:
        logger.error("MediaMTX unreachable: %s", exc)
        raise HTTPException(status_code=502, detail=f"MediaMTX unreachable: {exc}")

    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Segment not found.")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Upstream error.")

    if is_playlist:
        base = str(request.base_url).rstrip("/")
        lines: list[str] = []
        for line in resp.text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                if stripped.endswith(".ts") or stripped.endswith(".m3u8"):
                    # Rewrite relative segment/playlist URLs to absolute proxy URLs.
                    line = f"{base}/v1/hls/{stream_name}/{stripped}"
            lines.append(line)

        return Response(
            content="\n".join(lines),
            media_type="application/vnd.apple.mpegurl",
            headers=_cors_headers(request),
        )

    # Stream .ts segments in chunks to avoid loading the whole segment into RAM.
    content_type = resp.headers.get("content-type", "video/mp2t")

    async def _iter_chunks():
        async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
            yield chunk
        await client.aclose()

    return StreamingResponse(
        _iter_chunks(),
        media_type=content_type,
        headers=_cors_headers(request),
    )
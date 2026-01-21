"""HTTP server for remote control of claudechic.

Enables external processes (like Claude in another terminal) to:
- Take screenshots (SVG or PNG)
- Send messages to the active agent
- Wait for agent idle
- Get screen content as text
- Exit the app (for restart)

Start with --remote-port flag or CLAUDECHIC_REMOTE_PORT env var.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from claudechic.app import ChatApp

log = logging.getLogger(__name__)

_app: ChatApp | None = None
_server: web.AppRunner | None = None


async def handle_screenshot(request: web.Request) -> web.Response:
    """Save screenshot. Query params: ?path=/tmp/shot.svg&format=svg|png

    For PNG, uses macOS qlmanage for conversion (falls back to SVG if unavailable).
    """
    if _app is None:
        return web.json_response({"error": "App not initialized"}, status=500)

    fmt = request.query.get("format", "svg")
    default_path = f"/tmp/claudechic-screenshot.{fmt}"
    path = request.query.get("path", default_path)

    try:
        # Always save SVG first
        svg_path = path if fmt == "svg" else path.replace(".png", ".svg")
        result_path = _app.save_screenshot(
            filename=Path(svg_path).name, path=str(Path(svg_path).parent)
        )

        if fmt == "png":
            # Convert SVG to PNG using macOS qlmanage
            import subprocess

            png_path = path if path.endswith(".png") else f"{path}.png"
            proc = await asyncio.create_subprocess_exec(
                "qlmanage",
                "-t",
                "-s",
                "1200",
                "-o",
                str(Path(png_path).parent),
                result_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await proc.wait()
            # qlmanage adds .png to the filename
            actual_png = f"{result_path}.png"
            if Path(actual_png).exists():
                # Rename to requested path
                Path(actual_png).rename(png_path)
                result_path = png_path

        return web.json_response({"path": result_path, "format": fmt})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_send(request: web.Request) -> web.Response:
    """Send a message to the active agent. Body: {"text": "message"}"""
    if _app is None:
        return web.json_response({"error": "App not initialized"}, status=500)

    try:
        data = await request.json()
        text = data.get("text", "")
    except Exception:
        # Plain text body
        text = await request.text()

    if not text:
        return web.json_response({"error": "No text provided"}, status=400)

    # Use the app's send method
    agent = _app._agent
    if agent is None:
        return web.json_response({"error": "No active agent"}, status=400)

    try:
        # Send to active agent (fire-and-forget, callbacks handle response)
        _app._send_to_active_agent(text)
        return web.json_response({"status": "sent", "text": text[:100]})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_screen_text(request: web.Request) -> web.Response:  # noqa: ARG001
    """Get current screen content as plain text.

    Returns the full screen rendered as text, preserving layout.
    """
    if _app is None:
        return web.json_response({"error": "App not initialized"}, status=500)

    try:
        # Export SVG and extract text content
        svg = _app.export_screenshot()
        svg_no_style = re.sub(r"<style>.*?</style>", "", svg, flags=re.DOTALL)
        texts = re.findall(r">([^<]+)<", svg_no_style)
        lines = [html.unescape(t.replace("&#160;", " ")) for t in texts if t.strip()]
        return web.json_response({"text": "\n".join(lines)})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_wait_idle(request: web.Request) -> web.Response:
    """Wait until active agent is idle. Query param: ?timeout=30"""
    if _app is None:
        return web.json_response({"error": "App not initialized"}, status=500)

    timeout = float(request.query.get("timeout", "30"))
    agent = _app._agent
    if agent is None:
        return web.json_response({"error": "No active agent"}, status=400)

    from claudechic.enums import AgentStatus

    try:
        start = asyncio.get_event_loop().time()
        while agent.status != AgentStatus.IDLE:
            if asyncio.get_event_loop().time() - start > timeout:
                return web.json_response(
                    {"error": "Timeout waiting for idle"}, status=408
                )
            await asyncio.sleep(0.1)
        return web.json_response({"status": "idle"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_status(request: web.Request) -> web.Response:  # noqa: ARG001
    """Get app/agent status."""
    if _app is None:
        return web.json_response({"error": "App not initialized"}, status=500)

    agent = _app._agent
    agents = []
    if _app.agent_mgr:
        for a in _app.agent_mgr:
            agents.append(
                {
                    "name": a.name,
                    "id": a.id,
                    "status": str(a.status),
                    "cwd": str(a.cwd),
                    "active": a.id == _app.agent_mgr.active_id,
                }
            )

    return web.json_response(
        {
            "agents": agents,
            "active_agent": agent.name if agent else None,
        }
    )


async def handle_exit(request: web.Request) -> web.Response:  # noqa: ARG001
    """Exit the app cleanly. Use this before restarting."""
    if _app is None:
        return web.json_response({"error": "App not initialized"}, status=500)

    # Schedule exit after response is sent
    async def do_exit():
        await asyncio.sleep(0.1)  # Let response complete
        if _app:
            _app.exit()

    asyncio.create_task(do_exit())
    return web.json_response({"status": "exiting"})


async def start_server(app: ChatApp, port: int) -> None:
    """Start the remote control HTTP server."""
    global _app, _server
    _app = app

    webapp = web.Application()
    webapp.router.add_get("/screenshot", handle_screenshot)
    webapp.router.add_post("/send", handle_send)
    webapp.router.add_get("/screen_text", handle_screen_text)
    webapp.router.add_get("/wait_idle", handle_wait_idle)
    webapp.router.add_get("/status", handle_status)
    webapp.router.add_post("/exit", handle_exit)

    runner = web.AppRunner(webapp)
    await runner.setup()
    _server = runner

    site = web.TCPSite(runner, "localhost", port)
    await site.start()
    log.info(f"Remote control server started on http://localhost:{port}")

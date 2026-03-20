import asyncio
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


async def convert_dxf_to_dwg(dxf_path: str) -> str | None:
    """
    Convert a DXF file to DWG using dxf2dwg (libredwg-utils).
    Returns the path to the resulting .dwg file, or None if conversion failed.
    Falls back gracefully — caller should send DXF if this returns None.
    """
    dwg_path = os.path.splitext(dxf_path)[0] + ".dwg"
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_dxf2dwg, dxf_path, dwg_path)
        return result
    except Exception as exc:
        logger.error("DXF→DWG conversion failed: %s", exc)
        return None


def _run_dxf2dwg(dxf_path: str, dwg_path: str) -> str | None:
    """Blocking call to dxf2dwg CLI."""
    try:
        proc = subprocess.run(
            ["dxf2dwg", dxf_path, "-o", dwg_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0 and os.path.exists(dwg_path):
            logger.info("DWG saved: %s", dwg_path)
            return dwg_path
        logger.warning("dxf2dwg exit %d: %s", proc.returncode, proc.stderr.strip())
        return None
    except FileNotFoundError:
        logger.warning("dxf2dwg not found — libredwg-utils not installed")
        return None
    except subprocess.TimeoutExpired:
        logger.error("dxf2dwg timed out for %s", dxf_path)
        return None
    except Exception as exc:
        logger.error("dxf2dwg error: %s", exc)
        return None

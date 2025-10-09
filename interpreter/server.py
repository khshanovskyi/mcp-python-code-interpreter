import asyncio
import logging
import secrets
import shutil
import time
from dataclasses import dataclass
from typing import Dict

from fastmcp import FastMCP

from config import Config
from models import ExecutionResult, SessionInfo
from notebook import NotebookKernel

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Session timeout: 30 minutes of inactivity
SESSION_TIMEOUT = 30 * 60  # seconds


@dataclass
class SessionHolder:
    """Track session with last activity timestamp."""
    kernel: NotebookKernel
    last_activity: float


mcp = FastMCP(
    name='Python Code Interpreter',
    instructions="""
    A stateful Python code execution environment with Jupyter kernel support.
    
    Features:
    - Execute Python code with persistent state across requests
    - Session management for maintaining context between executions
    - Secure session IDs (16-character random strings)
    - Automatic session cleanup after 30 minutes of inactivity
    - Support for matplotlib/seaborn/plotly visualizations
    - File generation and access via MCP resources (images, plots, data files)
    - Standard libraries available: pandas, numpy, matplotlib, seaborn, plotly, sympy
    
    Usage:
    1. First execution: Call execute_code without session_id (or with session_id="")
    2. The response will include a session_id - save this for subsequent requests
    3. Reuse the session_id to maintain state across multiple code executions
    4. Sessions expire after 30 minutes of inactivity - you'll get an error and need to recreate
    5. Files are returned as references (URIs), not inline content
    6. Access files via the kernel:// resource protocol
    
    Session Management:
    - Sessions use secure random IDs (16-character URL-safe strings)
    - Sessions automatically expire after 30 minutes of inactivity
    - When a session expires, all associated files are deleted
    - If you try to use an expired session, you'll receive a SessionExpiredError
    - Simply create a new session and re-execute your setup code
    
    The tool returns structured output including:
    - Standard output/error streams
    - Execution results
    - Error messages with tracebacks
    - File references with URIs (e.g., kernel://abc123xyz456/plot.png)
    
    To access generated files, use the resource URI returned in the files list.
    """
)

sessions: Dict[str, SessionHolder] = {}


def _generate_session_id() -> str:
    """
    Generate a secure, random session ID.

    Returns:
        16-character URL-safe random string
    """
    return secrets.token_urlsafe(12)  # 12 bytes = 16 chars in base64


def _cleanup_session(session_id: str) -> None:
    """
    Completely remove a session and all its files.

    Args:
        session_id: Session ID to cleanup
    """
    try:
        if session_id in sessions:
            session_info = sessions[session_id]

            session_info.kernel.shutdown()

            session_folder = session_info.kernel.session_folder
            if session_folder.exists():
                shutil.rmtree(session_folder)
                logger.info(f"Deleted session folder: {session_folder}")

            del sessions[session_id]
            logger.info(f"Session {session_id} completely removed")

    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {e}")


def _cleanup_expired_sessions() -> int:
    """
    Remove all expired sessions (inactive for > SESSION_TIMEOUT).
    Also cleans up any orphaned empty session folders.

    Returns:
        Number of sessions cleaned up
    """
    current_time = time.time()
    expired_sessions = []

    for session_id, session_info in sessions.items():
        if current_time - session_info.last_activity > SESSION_TIMEOUT:
            expired_sessions.append(session_id)

    for session_id in expired_sessions:
        logger.info(f"Cleaning up expired session {session_id}")
        _cleanup_session(session_id)

    # Also check for orphaned empty folders (from failed initializations)
    try:
        if Config.NOTEBOOKS_FOLDER.exists():
            for item in Config.NOTEBOOKS_FOLDER.iterdir():
                if item.is_dir() and item.name not in sessions:
                    # Check if it's an empty or nearly-empty folder
                    files = list(item.iterdir())
                    if len(files) == 0:
                        item.rmdir()
                        logger.info(f"Removed orphaned empty folder: {item.name}")
                    elif len(files) == 1 and files[0].name == 'jupyter_session_code.txt':
                        if files[0].stat().st_size == 0:
                            files[0].unlink()
                            item.rmdir()
                            logger.info(f"Removed orphaned folder with empty history: {item.name}")
    except Exception as e:
        logger.warning(f"Error checking for orphaned folders: {e}")

    if expired_sessions:
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    return len(expired_sessions)


def _session_exists(session_id: str) -> bool:
    """
    Check if a session folder exists on disk.

    Args:
        session_id: The session identifier

    Returns:
        True if session folder exists, False otherwise
    """
    session_folder = Config.NOTEBOOKS_FOLDER / session_id
    return session_folder.exists() and session_folder.is_dir()


def _cleanup_empty_session_folder(session_id: str) -> None:
    """
    Remove empty or nearly-empty session folder (error recovery).

    Args:
        session_id: The session identifier
    """
    try:
        session_folder = Config.NOTEBOOKS_FOLDER / session_id
        if not session_folder.exists():
            return

        files = list(session_folder.iterdir())

        if len(files) == 0:
            session_folder.rmdir()
            logger.info(f"Removed empty session folder: {session_id}")
        elif len(files) == 1 and files[0].name == 'jupyter_session_code.txt':
            session_file = files[0]
            if session_file.stat().st_size == 0:
                session_file.unlink()
                session_folder.rmdir()
                logger.info(f"Removed empty session folder with empty history: {session_id}")

    except Exception as e:
        logger.warning(f"Failed to cleanup empty session folder {session_id}: {e}")


def _get_or_create_session_sync(session_id: str) -> tuple[NotebookKernel, bool, str]:
    """
    Retrieve an existing session or create a new one (sync).

    Args:
        session_id: The session identifier (empty string for new session)

    Returns:
        Tuple of (NotebookKernel, is_new_session, actual_session_id)

    Raises:
        ValueError: If session_id is non-empty but doesn't exist (expired)
        Exception: If kernel initialization fails
    """
    is_new_session = False
    created_session_id = None

    _cleanup_expired_sessions()

    try:
        if not session_id or session_id == "0":
            session_id = _generate_session_id()
            created_session_id = session_id

            try:
                kernel = NotebookKernel(session_id)
            except Exception as e:
                logger.error(f"Failed to initialize kernel for session {session_id}: {e}")
                _cleanup_empty_session_folder(session_id)
                raise

            sessions[session_id] = SessionHolder(
                kernel=kernel,
                last_activity=time.time()
            )
            is_new_session = True
            logger.info(f"Created new session: {session_id}")
        else:
            if session_id not in sessions:
                if not _session_exists(session_id):
                    raise ValueError(
                        f"Session {session_id} not found or has expired. "
                        f"Sessions expire after 30 minutes of inactivity. "
                        f"Please create a new session (use session_id=\"\" or session_id=\"0\") and re-execute your setup code."
                    )

                logger.info(f"Loading existing session: {session_id}")
                kernel = NotebookKernel(session_id)
                kernel.load_from_file()
                sessions[session_id] = SessionHolder(
                    kernel=kernel,
                    last_activity=time.time()
                )
            else:
                sessions[session_id].last_activity = time.time()

        return sessions[session_id].kernel, is_new_session, session_id

    except Exception as e:
        if created_session_id and created_session_id in sessions:
            del sessions[created_session_id]
        raise


def _execute_code_sync(code: str, session_id: str) -> dict:
    """
    Synchronous code execution logic.

    Args:
        code: Python code to execute
        session_id: Session identifier

    Returns:
        Execution result dictionary
    """
    actual_session_id = None
    is_new = False

    try:
        notebook, is_new, actual_session_id = _get_or_create_session_sync(session_id)
    except ValueError as e:
        # Session expired or not found
        error_result = ExecutionResult(success=False)
        error_result.set_error("SessionExpiredError", str(e), [])
        return error_result.model_dump()
    except Exception as e:
        # Kernel initialization or other error
        error_result = ExecutionResult(success=False)
        error_result.set_error("InitializationError", str(e), [])
        return error_result.model_dump()

    result = notebook.execute_code(code)

    # Only set session_info if execution was successful AND it's a new session
    if is_new and result.success:
        result.session_info = SessionInfo(
            session_id=actual_session_id,
            instructions="Use this `session_id` in subsequent requests to maintain state. Session will expire after 30 minutes of inactivity."
        )
    elif is_new and not result.success:
        # New session but execution failed - clean up
        logger.warning(f"New session {actual_session_id} created but execution failed")
        _cleanup_session(actual_session_id)
        _cleanup_empty_session_folder(actual_session_id)

    if result.success:
        notebook.save_to_file()

    return result.model_dump()


@mcp.tool()
async def execute_code(code: str, session_id: str = "") -> dict:
    """Execute Python code in a persistent Jupyter kernel environment.

    Maintains stateful execution where variables, imports, and state persist across multiple calls in the same session.

    Args:
        code: Python code to execute (multi-line supported)
        session_id: Session identifier (empty string or "0" for first call, then reuse returned ID)

    Returns:
        dict with:
            - success (bool): Execution status
            - output (list): stdout/stderr text
            - result (str|None): Last expression value
            - error (str|None): Error message if failed
            - traceback (list): Full traceback if error
            - files (list): File references with URIs
            - session_info (dict|None): Session info for new sessions (includes session_id)
    """
    try:
        result = await asyncio.to_thread(_execute_code_sync, code, session_id)
        return result

    except Exception as e:
        logger.error(f"Unexpected error in execute_code: {e}", exc_info=True)
        error_result = ExecutionResult(success=False)
        error_result.set_error("ServerError", str(e), [])
        return error_result.model_dump()


@mcp.tool()
async def list_session_files(session_id: str) -> dict:
    """
    List all files generated in a session.

    Args:
        session_id: The session identifier

    Returns:
        dict with:
            - session_id (str): The session ID
            - files (list): List of file references with URIs
            - error (str): Error message if session not found
    """
    try:
        # Cleanup expired sessions first
        _cleanup_expired_sessions()

        if session_id not in sessions:
            if not _session_exists(session_id):
                return {
                    "session_id": session_id,
                    "files": [],
                    "error": f"Session {session_id} not found or has expired. Sessions expire after 30 minutes of inactivity."
                }

            kernel = NotebookKernel(session_id)
            kernel.load_from_file()
            sessions[session_id] = SessionHolder(
                kernel=kernel,
                last_activity=time.time()
            )
        else:
            # Update last activity
            sessions[session_id].last_activity = time.time()

        kernel = sessions[session_id].kernel

        files = []
        for filename, file_path in kernel.generated_files.items():
            if file_path.exists():
                mime_type = kernel._get_mime_type(file_path)
                size = file_path.stat().st_size
                uri = f"kernel://{session_id}/{filename}"

                files.append({
                    "uri": uri,
                    "mime_type": mime_type,
                    "name": filename,
                    "size": size
                })

        return {
            "session_id": session_id,
            "files": files
        }

    except Exception as e:
        logger.error(f"Error listing files for session {session_id}: {e}")
        return {
            "session_id": session_id,
            "files": [],
            "error": str(e)
        }


@mcp.tool()
async def clear_session(session_id: str) -> dict:
    """
    Manually clear a session and shutdown its kernel.
    Removes all session files and working directory.

    Args:
        session_id: The session identifier to clear

    Returns:
        dict with success status and message
    """
    try:
        if session_id in sessions or _session_exists(session_id):
            _cleanup_session(session_id)
            return {
                "success": True,
                "message": f"Session {session_id} cleared and all files deleted"
            }
        else:
            return {
                "success": False,
                "message": f"Session {session_id} not found"
            }
    except Exception as e:
        logger.error(f"Error clearing session {session_id}: {e}")
        return {"success": False, "error": str(e)}


@mcp.resource("kernel://{session_id}/{filename}")
async def get_session_file(session_id: str, filename: str):
    """
    Retrieve file content from a session by URI.

    The URI format is: kernel://session_id/filename

    Returns structured response with MIME type and base64-encoded content.
    """
    try:
        _cleanup_expired_sessions()

        logger.info(f"Fetching file: {filename} from session {session_id}")

        if session_id not in sessions:
            if not _session_exists(session_id):
                raise FileNotFoundError(
                    f"Session {session_id} not found or has expired. "
                    f"Sessions expire after 30 minutes of inactivity."
                )

            kernel = NotebookKernel(session_id)
            kernel.load_from_file()
            sessions[session_id] = SessionHolder(
                kernel=kernel,
                last_activity=time.time()
            )
        else:
            sessions[session_id].last_activity = time.time()

        kernel = sessions[session_id].kernel

        file_path = kernel.get_file(filename)
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        with open(file_path, 'rb') as f:
            file_bytes = f.read()

        mime_type = kernel._get_mime_type(file_path)

        logger.info(f"Successfully retrieved {filename} ({mime_type}, {len(file_bytes)} bytes)")

        if mime_type.startswith('text/') or mime_type in ['application/json', 'application/xml']:
            try:
                content = file_bytes.decode('utf-8')
                return content
            except UnicodeDecodeError:
                pass

        return file_bytes

    except Exception as e:
        logger.error(f"Error retrieving file {filename} from session {session_id}: {e}")
        raise


async def periodic_cleanup():
    """Background task to periodically cleanup expired sessions."""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            logger.debug("Running periodic session cleanup")
            cleaned = _cleanup_expired_sessions()
            if cleaned > 0:
                logger.info(f"Periodic cleanup: removed {cleaned} expired sessions")
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")


if __name__ == '__main__':
    import threading


    def run_cleanup_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(periodic_cleanup())


    cleanup_thread = threading.Thread(target=run_cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info("Started background session cleanup task (runs every 5 minutes)")

    # Run MCP server
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        log_level="DEBUG",
    )
from jupyter_client import KernelManager
from jupyter_client.blocking.client import BlockingKernelClient
from typing import Optional, List, Set, Dict
import logging
import time
import json
import base64
import mimetypes
from pathlib import Path

from models import ExecutionResult
from config import Config

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class NotebookKernel:
    """Manages a Jupyter kernel session for code execution with persistent state."""

    def __init__(self, session_id: str):
        """
        Initialize a new notebook kernel session.

        Args:
            session_id: Unique identifier for this session (secure random string)
        """
        self.session_id = session_id

        self.session_folder = Config.NOTEBOOKS_FOLDER / session_id

        try:
            self.session_folder.mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create session folder {session_id}: {e}")
            raise

        self.file_path = self.session_folder / 'jupyter_session_code.txt'
        self.history: List[str] = []

        self.working_dir = self.session_folder

        self.generated_files: Dict[str, Path] = {}  # filename -> full path

        self.kernel: Optional[KernelManager] = None
        self.client: Optional[BlockingKernelClient] = None

        try:
            self._initialize_kernel()
        except Exception as e:
            logger.error(f"Kernel initialization failed for session {session_id}: {e}")
            self.shutdown()
            raise

    def _initialize_kernel(self) -> None:
        """Initialize the Jupyter kernel and client."""
        try:
            self.kernel = KernelManager()
            self.kernel.start_kernel()

            self.client = self.kernel.blocking_client()
            self.client.start_channels()
            self.client.wait_for_ready(timeout=Config.KERNEL_TIMEOUT)

            setup_code = f"""
import os
os.chdir(r'{self.working_dir}')

# Matplotlib inline backend
%matplotlib inline

# Monkey-patch Plotly to auto-save on show()
try:
    import plotly.graph_objects as go
    _original_show = go.Figure.show
    
    def _auto_save_show(self, *args, **kwargs):
        import time
        # Auto-save to file
        filename = f"plotly_{{int(time.time() * 1000)}}.png"
        try:
            self.write_image(filename)
            print(f"Plotly figure saved as {{filename}}")
        except Exception as e:
            print(f"Warning: Could not save Plotly figure: {{e}}")
        # Call original show (won't display in standalone kernel, but maintains compatibility)
        return _original_show(self, *args, **kwargs)
    
    go.Figure.show = _auto_save_show
except ImportError:
    pass
"""
            self.client.execute(setup_code, silent=True)
            self.client.get_shell_msg(timeout=Config.KERNEL_TIMEOUT)

            try:
                while True:
                    self.client.get_iopub_msg(timeout=0.1)
            except:
                pass

            logger.info(f"Kernel initialized for session {self.session_id}")
            logger.debug(f"Working directory: {self.working_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize kernel: {e}")
            raise

    def _get_current_files(self) -> Set[Path]:
        """Get set of all files currently in working directory (excluding session history)."""
        if not self.working_dir.exists():
            return set()
        files = set()
        for item in self.working_dir.rglob('*'):
            if item.is_file() and item != self.file_path:  # Exclude the session history file
                files.add(item)
        return files

    def _detect_new_files(self, files_before: Set[Path]) -> List[Path]:
        """Detect files created during execution."""
        files_after = self._get_current_files()
        new_files = files_after - files_before
        return sorted(new_files)

    def _get_mime_type(self, file_path: Path) -> str:
        """Determine MIME type from file extension."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            return mime_type

        extension = file_path.suffix.lower()
        mime_map = {
            '.html': 'text/html',
            '.htm': 'text/html',
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.xml': 'application/xml',
        }
        return mime_map.get(extension, 'application/octet-stream')

    def _save_inline_image(self, image_bytes: bytes, extension: str) -> Path:
        """
        Save an inline image (from display_data) to working directory.

        Args:
            image_bytes: Binary image data
            extension: File extension (e.g., 'png', 'jpeg')

        Returns:
            Path to saved file
        """
        # Generate unique filename
        timestamp = int(time.time() * 1000)
        filename = f"inline_{timestamp}.{extension}"
        file_path = self.working_dir / filename

        with open(file_path, 'wb') as f:
            f.write(image_bytes)

        return file_path

    def _is_plotly_json(self, text: str) -> bool:
        """Check if text is a Plotly figure JSON."""
        if not text:
            return False
        text = text.strip().strip("'\"")
        return text.startswith('{"data":[') and '"layout":{' in text

    def _convert_plotly_to_image(self, json_str: str) -> Optional[bytes]:
        """Convert Plotly JSON to PNG bytes."""
        try:
            import plotly.io as pio

            json_str = json_str.strip().strip("'\"")
            fig_dict = json.loads(json_str)

            png_bytes = pio.to_image(fig_dict, format='png')
            return png_bytes

        except ImportError:
            logger.warning("plotly or kaleido not installed - cannot convert to image")
            return None
        except Exception as e:
            logger.warning(f"Failed to convert Plotly to image: {e}")
            return None

    def execute_code(self, code: str) -> ExecutionResult:
        """
        Execute Python code in the kernel (blocking operation).

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult containing outputs, errors, and file references
        """
        result = ExecutionResult(success=True)

        if not self.client:
            result.set_error("KernelError", "Kernel client not initialized", [])
            return result

        files_before = self._get_current_files()
        logger.debug(f"Files before execution: {len(files_before)}")

        try:
            deadline = time.time() + Config.KERNEL_TIMEOUT + 5

            self.client.execute(code)

            shell_msg = self.client.get_shell_msg(timeout=Config.KERNEL_TIMEOUT)
            logger.debug(f"Shell message: {shell_msg.get('content', {})}")

            while True:
                try:
                    timeout = max(deadline - time.time(), 0.1)
                    if timeout <= 0:
                        logger.warning("Execution timeout reached")
                        break

                    msg = self.client.get_iopub_msg(timeout=timeout)
                    msg_type = msg['msg_type']
                    content = msg['content']

                    logger.debug(f"Message type: {msg_type}")

                    if msg_type == 'status' and content.get('execution_state') == 'idle':
                        logger.debug("Received idle status, breaking loop")
                        break

                    elif msg_type == 'stream':
                        text = content.get('text', '')
                        if text:
                            result.add_output(text)
                            logger.debug(f"Stream output: {text[:100]}")

                    elif msg_type == 'execute_result':
                        data = content.get('data', {})
                        text_result = data.get('text/plain', '')
                        logger.debug(f"Execute result: {text_result[:200] if text_result else 'None'}")

                        if self._is_plotly_json(text_result):
                            image_bytes = self._convert_plotly_to_image(text_result)
                            if image_bytes:
                                file_path = self._save_inline_image(image_bytes, 'png')
                                self._track_file(file_path)
                            else:
                                result.result = text_result
                        else:
                            result.result = self._sanitize_result(text_result)

                    elif msg_type == 'display_data':
                        data = content.get('data', {})

                        # Handle Plotly figures (from fig.show())
                        if 'application/vnd.plotly.v1+json' in data:
                            try:
                                plotly_json = data['application/vnd.plotly.v1+json']
                                # Convert dict to JSON string if needed
                                if isinstance(plotly_json, dict):
                                    json_str = json.dumps(plotly_json)
                                else:
                                    json_str = plotly_json

                                image_bytes = self._convert_plotly_to_image(json_str)
                                if image_bytes:
                                    file_path = self._save_inline_image(image_bytes, 'png')
                                    self._track_file(file_path)
                                    logger.info(f"Captured Plotly figure from display_data")
                            except Exception as e:
                                logger.warning(f"Failed to convert Plotly display_data to image: {e}")

                        # Handle PNG images (matplotlib, etc.)
                        if 'image/png' in data:
                            png_bytes = base64.b64decode(data['image/png'])
                            file_path = self._save_inline_image(png_bytes, 'png')
                            self._track_file(file_path)

                        # Handle JPEG images
                        if 'image/jpeg' in data:
                            jpeg_bytes = base64.b64decode(data['image/jpeg'])
                            file_path = self._save_inline_image(jpeg_bytes, 'jpeg')
                            self._track_file(file_path)

                        # Handle SVG images
                        if 'image/svg+xml' in data:
                            svg_data = data['image/svg+xml']
                            if isinstance(svg_data, str):
                                svg_bytes = svg_data.encode('utf-8')
                            else:
                                svg_bytes = bytes(svg_data)
                            file_path = self._save_inline_image(svg_bytes, 'svg')
                            self._track_file(file_path)

                        if 'text/plain' in data:
                            result.add_output(data['text/plain'])

                    elif msg_type == 'error':
                        error_name = content.get('ename', 'Error')
                        error_value = content.get('evalue', '')
                        traceback = content.get('traceback', [])
                        result.set_error(error_name, error_value, traceback)
                        logger.error(f"Execution error: {error_name}: {error_value}")

                except Exception as e:
                    if "Timeout" in str(e):
                        logger.debug("Message timeout - assuming completion")
                        break
                    else:
                        logger.error(f"Error processing message: {e}")
                        break

            time.sleep(0.2)

            new_files = self._detect_new_files(files_before)
            logger.debug(f"New files detected: {len(new_files)}")

            for file_path in new_files:
                self._track_file(file_path)

            for filename, file_path in self.generated_files.items():
                try:
                    mime_type = self._get_mime_type(file_path)
                    size = file_path.stat().st_size
                    uri = f"kernel://{self.session_id}/{filename}"

                    result.add_output(f"📊 File created: {filename} ({mime_type}, {size} bytes)")
                    result.add_file_reference(uri, mime_type, filename, size)

                    logger.info(f"File reference added: {filename} ({mime_type}, {size} bytes)")
                except Exception as e:
                    logger.warning(f"Failed to create reference for {file_path}: {e}")

            if result.success:
                self.history.append(code)

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            result.set_error("ExecutionError", str(e), [])

        return result

    def _track_file(self, file_path: Path) -> None:
        """Track a generated file by its name."""
        filename = file_path.name
        if filename not in self.generated_files and filename != 'jupyter_session_code.txt':
            self.generated_files[filename] = file_path
            logger.debug(f"Tracking file: {filename}")

    def get_file(self, filename: str) -> Optional[Path]:
        """
        Get the path to a generated file.

        Args:
            filename: Name of the file

        Returns:
            Path to file if it exists, None otherwise
        """
        if filename == 'jupyter_session_code.txt':
            return None

        if filename in self.generated_files:
            path = self.generated_files[filename]
            if path.exists():
                return path

        file_path = self.working_dir / filename
        if file_path.exists() and file_path.is_file() and file_path != self.file_path:
            return file_path

        return None

    def save_to_file(self) -> None:
        """Persist the execution history to a file."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.history))
            logger.info(f"Session {self.session_id} saved to {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            raise

    def load_from_file(self) -> bool:
        """
        Load and execute code from a previously saved session.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.file_path.exists():
            logger.warning(f"No saved session found at {self.file_path}")
            return False

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            if code.strip():
                result = self.execute_code(code)
                if not result.success:
                    logger.error(f"Failed to reload session: {result.error}")
                    return False

            logger.info(f"Session {self.session_id} loaded from {self.file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the kernel gracefully."""
        try:
            if self.client:
                self.client.stop_channels()
            if self.kernel:
                self.kernel.shutdown_kernel(now=True)
            logger.info(f"Kernel shutdown for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def _sanitize_result(self, text_result: str) -> str:
        """
        Sanitize execution result to hide full file paths.

        If the result is a file path within the working directory,
        return just the filename. Otherwise, return as-is.

        Args:
            text_result: The result text from execution

        Returns:
            Sanitized result text
        """
        if not text_result:
            return text_result

        cleaned = text_result.strip().strip("'\"")

        try:
            result_path = Path(cleaned)

            if result_path.is_absolute() and result_path.exists():
                if self.working_dir in result_path.parents or result_path.parent == self.working_dir:
                    return result_path.name
        except (ValueError, OSError):
            pass

        return text_result

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures kernel is shutdown."""
        self.shutdown()
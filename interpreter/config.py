import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent


class Config:
    """Central configuration for the MCP server."""

    # Suppress Jupyter platformdirs warning
    os.environ['JUPYTER_PLATFORM_DIRS'] = '1'

    NOTEBOOKS_FOLDER = ROOT_DIR / 'notebooks'

    KERNEL_TIMEOUT = 10  # seconds

    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    @classmethod
    def initialize(cls) -> None:
        """Initialize necessary directories and configurations."""
        # Remove existing notebooks folder and all its contents
        if cls.NOTEBOOKS_FOLDER.exists():
            shutil.rmtree(cls.NOTEBOOKS_FOLDER)
            logger.info(f"Cleared existing notebooks folder: {cls.NOTEBOOKS_FOLDER}")

        # Create fresh empty folder
        cls.NOTEBOOKS_FOLDER.mkdir(exist_ok=True)
        logger.info(f"Initialized notebooks folder: {cls.NOTEBOOKS_FOLDER}")


Config.initialize()

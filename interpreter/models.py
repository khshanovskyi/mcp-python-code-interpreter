from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class SessionInfo(BaseModel):
    session_id: str = Field(description="Session ID")
    instructions: Optional[str] = Field(description="Additional instructions of how to use session_id")


class FileReference(BaseModel):
    """Reference to a file generated during execution (no binary content)."""

    uri: str = Field(description="Resource URI to access the file (e.g., 'kernel://session_123/plot.png')")
    mime_type: str = Field(description="MIME type of the file (e.g., 'image/png', 'image/jpeg')")
    name: str = Field(description="Filename")
    size: int = Field(description="File size in bytes")


class ExecutionResult(BaseModel):
    """Standardized response structure for code execution."""

    success: bool = Field(description="Whether the code execution completed successfully")
    output: list[str] = Field(
        default_factory=list,
        description="Standard output and print statements from execution"
    )
    result: Optional[str] = Field(
        default=None,
        description="The return value of the last expression, if any"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed (includes SessionExpiredError for expired sessions)"
    )
    traceback: list[str] = Field(
        default_factory=list,
        description="Full error traceback for debugging"
    )
    files: list[FileReference] = Field(
        default_factory=list,
        description="File references (not content) generated during execution"
    )
    session_info: Optional[SessionInfo] = Field(
        default=None,
        description="Session information for new sessions, including expiration notice"
    )

    def add_output(self, text: str) -> None:
        """Add text to output list."""
        self.output.append(text)

    def add_file_reference(
            self,
            uri: str,
            mime_type: str,
            name: str,
            size: int
    ) -> None:
        """
        Add a file reference (metadata only, no binary content).

        Args:
            uri: Resource URI to access the file
            mime_type: MIME type (e.g., 'image/png', 'image/jpeg')
            name: Filename
            size: File size in bytes
        """
        self.files.append(
            FileReference(uri=uri, mime_type=mime_type, name=name, size=size)
        )

    def set_error(self, error_name: str, error_value: str, traceback: list[str]) -> None:
        """Set error information."""
        self.success = False
        self.error = f"{error_name}: {error_value}"
        self.traceback = traceback

"""
Path normalization utilities for MODFLOW model files.

Handles conversion between Windows and Linux path separators in:
1. ZIP file entries
2. MODFLOW input files (OPEN/CLOSE references, external file paths)

MODFLOW executables on Linux require forward slashes, while models created
on Windows often contain backslashes. This module normalizes all paths to
forward slashes for cross-platform compatibility.
"""

import logging
import os
import re
import zipfile
from pathlib import Path
from typing import BinaryIO, List, Set, Tuple

from app.config import get_settings

logger = logging.getLogger(__name__)


# File extensions that may contain path references
PATH_REFERENCE_EXTENSIONS = {
    # Classic MODFLOW (MF2005/NWT/USG)
    '.nam', '.dis', '.disu', '.bas', '.bas6',
    '.rch', '.evt', '.wel', '.drn', '.riv', '.ghb', '.chd',
    '.sfr', '.lak', '.uzf', '.mnw', '.mnw2',
    '.lpf', '.upw', '.bcf', '.bcf6', '.hfb', '.hfb6',
    '.oc', '.pcg', '.nwt', '.sms', '.ims',
    # MF6 packages (may use OPEN/CLOSE for external arrays)
    '.tdis', '.gwf', '.ims6', '.tdis6', '.gwf6',
    '.disv', '.disv6', '.dis6',
    '.npf', '.npf6',       # Node Property Flow (hydraulic conductivity)
    '.sto', '.sto6',       # Storage (ss, sy)
    '.ic', '.ic6',         # Initial Conditions (strt)
    '.rcha', '.rcha6',     # Recharge (array)
    '.evta', '.evta6',     # Evapotranspiration (array)
    '.chd6', '.wel6', '.riv6', '.drn6', '.ghb6',  # MF6 stress packages
    '.sfr6', '.lak6', '.uzf6', '.maw6',           # MF6 advanced packages
    '.oc6',                # Output Control
    '.buy', '.buy6',       # Buoyancy
    '.csub', '.csub6',     # CSUB
    '.mvr', '.mvr6',       # Mover
}

# Patterns that indicate path references in MODFLOW files
PATH_PATTERNS = [
    # OPEN/CLOSE with path (quoted or unquoted)
    re.compile(r"(OPEN/CLOSE\s+)('[^']*'|\"[^\"]*\"|[^\s]+)", re.IGNORECASE),
    # External file references (DATA, DATA(BINARY))
    re.compile(r'(DATA(?:\(BINARY\))?\s+)(\d+\s+)([^\s]+)', re.IGNORECASE),
    # MODFLOW 6 file references (quoted or unquoted)
    re.compile(r"(FILEIN\s+)('[^']*'|\"[^\"]*\"|[^\s]+)", re.IGNORECASE),
    re.compile(r"(FILEOUT\s+)('[^']*'|\"[^\"]*\"|[^\s]+)", re.IGNORECASE),
    # Include files
    re.compile(r'(INCLUDE\s+)([^\s]+)', re.IGNORECASE),
]


def normalize_path(path: str) -> str:
    """
    Normalize a file path to use forward slashes.

    Parameters
    ----------
    path : str
        Path that may contain backslashes

    Returns
    -------
    str
        Path with all backslashes converted to forward slashes
    """
    return path.replace('\\', '/')


def normalize_zip_entry_name(name: str) -> str:
    """
    Normalize a ZIP entry name to use forward slashes.

    Parameters
    ----------
    name : str
        ZIP entry name that may contain backslashes

    Returns
    -------
    str
        Normalized entry name with forward slashes
    """
    return normalize_path(name)


def extract_zip_with_normalized_paths(
    zip_data: bytes,
    extract_dir: Path,
) -> Tuple[int, List[str]]:
    """
    Extract a ZIP file with normalized paths.

    Handles ZIP files created on Windows that have backslash paths.
    Extracts all files with forward-slash paths.

    Parameters
    ----------
    zip_data : bytes
        Raw bytes of the ZIP file
    extract_dir : Path
        Directory to extract files to

    Returns
    -------
    tuple of (int, list of str)
        Count of extracted files and list of normalized paths
    """
    import io

    settings = get_settings()
    max_uncompressed_bytes = settings.max_upload_size_mb * 2 * 1024 * 1024  # 2x compressed limit
    max_path_depth = 20

    extracted_paths = []
    count = 0

    with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
        # Check total uncompressed size to guard against zip bombs
        total_uncompressed = sum(info.file_size for info in zf.infolist())
        if total_uncompressed > max_uncompressed_bytes:
            raise ValueError(
                f"ZIP uncompressed size ({total_uncompressed / 1024 / 1024:.0f} MB) "
                f"exceeds limit ({max_uncompressed_bytes / 1024 / 1024:.0f} MB)"
            )

        for info in zf.infolist():
            # Normalize the path
            normalized_name = normalize_zip_entry_name(info.filename)

            # Skip directory entries
            if normalized_name.endswith('/'):
                continue

            # Skip symbolic links (external_attr bit 29 = symlink on Unix)
            if info.external_attr >> 28 == 0xA:
                logger.warning(f"Skipping symbolic link in ZIP: {normalized_name}")
                continue

            # Security check: prevent path traversal
            if normalized_name.startswith('/') or '..' in normalized_name:
                logger.warning(f"Skipping path traversal attempt in ZIP: {normalized_name}")
                continue

            # Limit path depth to prevent deeply nested extraction
            if normalized_name.count('/') > max_path_depth:
                logger.warning(f"Skipping deeply nested path in ZIP: {normalized_name}")
                continue

            # Create target path
            target_path = extract_dir / normalized_name

            # Verify the resolved path stays within extract_dir
            try:
                target_path.resolve().relative_to(extract_dir.resolve())
            except ValueError:
                logger.warning(f"Skipping path escaping extract dir: {normalized_name}")
                continue

            # Ensure parent directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract file content
            with zf.open(info) as src:
                content = src.read()

            # Write to normalized path
            target_path.write_bytes(content)
            extracted_paths.append(normalized_name)
            count += 1

    return count, extracted_paths


def normalize_modflow_file_paths(file_path: Path) -> Tuple[bool, int]:
    """
    Normalize path references inside a MODFLOW input file.

    Scans the file for path references (OPEN/CLOSE, external files, etc.)
    and converts any backslashes to forward slashes.

    Parameters
    ----------
    file_path : Path
        Path to the MODFLOW input file

    Returns
    -------
    tuple of (bool, int)
        (True if file was modified, count of paths normalized)
    """
    # Check if this file type might contain path references
    if file_path.suffix.lower() not in PATH_REFERENCE_EXTENSIONS:
        return False, 0

    try:
        # Read file content
        content = file_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        try:
            # Try latin-1 as fallback
            content = file_path.read_text(encoding='latin-1')
        except Exception:
            return False, 0

    modified = False
    replacements = 0
    lines = content.split('\n')
    new_lines = []

    for line in lines:
        original_line = line

        # Check for backslash in the line (quick check)
        if '\\' in line:
            # Check each pattern
            for pattern in PATH_PATTERNS:
                match = pattern.search(line)
                if match:
                    # Get the path part and normalize it
                    groups = match.groups()
                    for i, group in enumerate(groups):
                        if group and '\\' in group:
                            normalized = normalize_path(group)
                            line = line.replace(group, normalized)
                            if line != original_line:
                                replacements += 1
                                modified = True

            # Also check for bare paths with backslashes (like arrays\DIS\file.ref)
            # Common pattern in MODFLOW files
            if 'arrays\\' in line.lower() or '\\' in line:
                # Simple replacement for common patterns
                new_line = line
                # Find potential path segments
                words = line.split()
                for word in words:
                    if '\\' in word and not word.startswith('#'):
                        normalized_word = normalize_path(word)
                        new_line = new_line.replace(word, normalized_word)

                if new_line != line:
                    line = new_line
                    if line != original_line:
                        replacements += 1
                        modified = True

        new_lines.append(line)

    if modified:
        # Write back with Unix line endings
        new_content = '\n'.join(new_lines)
        # Ensure no BOM is written
        file_path.write_text(new_content, encoding='utf-8', newline='\n')

    return modified, replacements


def normalize_all_model_files(model_dir: Path) -> Tuple[int, int]:
    """
    Normalize path references in all MODFLOW files in a directory.

    Parameters
    ----------
    model_dir : Path
        Directory containing MODFLOW model files

    Returns
    -------
    tuple of (int, int)
        (count of files modified, total count of paths normalized)
    """
    files_modified = 0
    total_replacements = 0

    for file_path in model_dir.rglob('*'):
        if file_path.is_file():
            modified, replacements = normalize_modflow_file_paths(file_path)
            if modified:
                files_modified += 1
                total_replacements += replacements

    return files_modified, total_replacements

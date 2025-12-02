from pathlib import Path
import gzip
import shutil


class I3Compressor:
    """
    Compress .i3 files to .i3.gz.

    Usage:
    - If you give a directory path: all .i3 files in that folder are compressed.
    - If you give a single .i3 file path: only that file is compressed.
    """

    def __init__(self, delete_original: bool = False):
        """
        :param delete_original: If True, remove the original .i3 file
                                after creating the .i3.gz file.
        """
        self.delete_original = delete_original

    def compress(self, path: str):
        """
        Compress a directory of .i3 files or a single .i3 file.

        :param path: Directory path or single .i3 file path.
        """
        p = Path(path)

        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")

        if p.is_dir():
            # Case 1: path is a folder
            self._compress_directory(p)
        elif p.is_file():
            # Case 2: path is a single file
            self._compress_single_file(p)
        else:
            raise ValueError(f"Given path is neither a file nor a directory: {p}")

    def _compress_directory(self, folder: Path):
        """
        Find all .i3 files in the given folder and compress them.
        (Not recursive; only files directly in this folder.)
        """
        count = 0
        for i3_file in folder.glob("*.i3"):
            self._compress_file(i3_file)
            count += 1

        print(f"Compressed {count} .i3 file(s) in folder: {folder}")

    def _compress_single_file(self, file_path: Path):
        """
        Compress a single .i3 file.
        """
        if file_path.suffix != ".i3":
            raise ValueError(f"Single-file mode expects a .i3 file, got: {file_path}")

        self._compress_file(file_path)

    def _compress_file(self, i3_path: Path):
        """
        Do the actual .i3 -> .i3.gz compression for one file.
        """
        gz_path = i3_path.with_suffix(i3_path.suffix + ".gz")

        if gz_path.exists():
            print(f"Skipping (output already exists): {gz_path}")
            return

        print(f"Compressing: {i3_path} -> {gz_path}")

        # Stream copy: safe for large files
        with i3_path.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        if self.delete_original:
            i3_path.unlink()
            print(f"Deleted original file: {i3_path}")

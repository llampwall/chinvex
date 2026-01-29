from pathlib import Path
from chinvex.util import normalize_path_for_dedup

def test_normalize_path_converts_to_absolute():
    result = normalize_path_for_dedup("./src")
    assert Path(result).is_absolute()

def test_normalize_path_uses_forward_slashes():
    result = normalize_path_for_dedup(r"C:\Code\chinvex")
    assert "\\" not in result

def test_normalize_path_deduplicates_same_path():
    path1 = normalize_path_for_dedup("./src")
    path2 = normalize_path_for_dedup(Path.cwd() / "src")
    assert path1 == path2

def test_normalize_path_case_insensitive_on_windows():
    import platform
    if platform.system() != "Windows":
        return  # Skip on non-Windows
    path1 = normalize_path_for_dedup(r"C:\Code\Chinvex")
    path2 = normalize_path_for_dedup(r"c:\code\chinvex")
    assert path1 == path2

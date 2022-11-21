from urllib.request import urlopen, Request
from pathlib import Path
import tarfile
from tqdm import tqdm

from .config import DATASET_URL


def download_from_url(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urlopen(Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def extract_archive(file_path: Path, remove_tar: bool = True):
    with tarfile.open(str(file_path)) as f:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, str(file_path.parent))
    if remove_tar:
        file_path.unlink()


def download_and_extract_archive(root: str, filename: str) -> None:
    file_path_str = f"{root}/{filename}"
    file_path = Path(file_path_str)

    if file_path.exists():
        return
    elif file_path.suffix == "":
        targz_path_str = f"{file_path_str}.tar.gz"
        targz_path = Path(targz_path_str)
        url = f"{DATASET_URL}/files/{targz_path.name}?download=1"
        if not targz_path.exists():
            download_from_url(url, targz_path_str)
        extract_archive(targz_path)
    else:
        url = f"{DATASET_URL}/files/{file_path.name}?download=1"
        download_from_url(url, file_path_str)

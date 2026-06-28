import logging
import os
import shutil
import tempfile
import urllib.request


def download_file(url: str, dst: str) -> bool:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst)) as tmp:
            tmp_path = tmp.name
            with urllib.request.urlopen(url) as r:
                shutil.copyfileobj(r, tmp)
        os.replace(tmp_path, dst)
        return True
    except Exception as e:
        if tmp is not None:
            try:
                os.remove(tmp.name)
            except Exception:
                pass
        logging.error(f"Error downloading {url} -> {dst}: {e}")
        return False

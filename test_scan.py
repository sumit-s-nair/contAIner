import os
import sys
import subprocess
from pathlib import Path
import random
import urllib.request
import json
import time

sys.path.insert(0, os.path.abspath('src'))
from repo_scan.import_scan import scan_imports, IMPORT_TO_PACKAGE

def clone_repo(url, dest):
    if not os.path.exists(dest):
        print(f"Cloning {url} into {dest}...")
        subprocess.run(["git", "clone", url, dest], check=True)
    else:
        print(f"Repo {dest} already exists.")

def download_file(url, dest_dir, filename):
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
    else:
        print(f"File {dest_path} already exists.")

def main():
    test_dir = Path("test_repos")
    test_dir.mkdir(exist_ok=True)
    
    # 1. Webcam-Face-Detect (Uses cv2, sys)
    clone_repo("https://github.com/shantnu/Webcam-Face-Detect.git", str(test_dir / "Webcam-Face-Detect"))
    
    # 2. python-scripts by realpython (Multiple scripts, various imports)
    clone_repo("https://github.com/realpython/python-scripts.git", str(test_dir / "python-scripts"))
    
    # 3. min-char-rnn gist (Uses numpy)
    clone_repo("https://gist.github.com/d4dee566867f8291f086.git", str(test_dir / "min-char-rnn"))
    
    repos = [
        test_dir / "Webcam-Face-Detect",
        test_dir / "python-scripts",
        test_dir / "min-char-rnn"
    ]
    
    print("\n" + "="*50)
    print("SCANNING REPOS")
    print("="*50)
    
    for repo in repos:
        print(f"\nScanning {repo.name}...")
        deps = scan_imports(repo)
        if deps:
            for dep in deps:
                print(f"  - {dep.guessed_package_name} (from import '{dep.import_name}', conf: {dep.confidence})")
                for src in dep.sources[:2]:
                    print(f"      Source: {src.file}:{src.line}")
                if len(dep.sources) > 2:
                    print(f"      ... and {len(dep.sources)-2} more sources")
        else:
            print("  No external dependencies found.")
            
    print("\n" + "="*50)
    print("SPOT-CHECKING 15 RANDOM MAPPINGS AGAINST PYPI")
    print("="*50)
    
    random.seed(42) # For reproducibility during test
    keys = list(IMPORT_TO_PACKAGE.keys())
    sample_keys = random.sample(keys, min(15, len(keys)))
    
    for key in sample_keys:
        pkg_name = IMPORT_TO_PACKAGE[key]
        print(f"Checking {key} -> {pkg_name}...", end=" ")
        
        url = f"https://pypi.org/pypi/{pkg_name}/json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print("OK")
                else:
                    print(f"FAILED (Status: {response.status})")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print("NOT FOUND on PyPI")
            else:
                print(f"HTTP Error: {e.code}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(0.2) # be nice to PyPI

if __name__ == "__main__":
    main()

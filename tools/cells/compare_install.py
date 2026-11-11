import importlib
import importlib.metadata
import subprocess
import sys

EXTRA_PACKAGES = ['interpret-core', 'tabpfn']        # EBM、TabPFN
missing = []
for package in EXTRA_PACKAGES:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        missing.append(package)
if missing:
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], text=True,
                            capture_output=True, check=False)
    print('pip install: ', missing, 'exit code', result.returncode)
    if result.returncode != 0:
        print(result.stderr[-2000:])
    importlib.invalidate_caches()

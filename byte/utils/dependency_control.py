import subprocess
import sys

from byte.utils.error import PipInstallError
from byte.utils.log import byte_log


def prompt_install(package: str, warn: bool = False) -> None:  # pragma: no cover
    """
    Function used to prompt user to install a package.
    """
    package_name = str(package or "").strip().strip("'\"")
    if not package_name:
        raise PipInstallError(package)
    cmd = [sys.executable, "-m", "pip", "install", "-q", package_name]
    try:
        if warn and input(f"Install {package_name}? Y/n: ") != "Y":
            raise ModuleNotFoundError(f"No module named {package_name}")
        print(f"start to install package: {package_name}")
        subprocess.check_call(cmd)
        print(f"successfully installed package: {package_name}")
        byte_log.info("%s installed successfully!", package_name)
    except subprocess.CalledProcessError as e:
        raise PipInstallError(package) from e

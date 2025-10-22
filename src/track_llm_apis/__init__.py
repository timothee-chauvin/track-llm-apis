from pathlib import Path

from beartype.claw import beartype_this_package
from dotenv import load_dotenv

beartype_this_package()

# TODO remove the load_dotenv calls in other files once there's a unified way to invoke code
load_dotenv()


def get_assets_dir() -> Path:
    return Path(__file__).with_name("assets")

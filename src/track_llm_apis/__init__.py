from pathlib import Path

from beartype.claw import beartype_this_package

beartype_this_package()


def get_assets_dir() -> Path:
    return Path(__file__).with_name("assets")

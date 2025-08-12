set -Eeuo pipefail

# mm* are PITA, they do not list build dependencies; sam2 builds faster with ninja
uv sync --link-mode hardlink \
  --no-install-package mmcv-full \
  --no-install-package mmdet \
  --no-install-package mmengine \
  --no-install-package sam-2

# install them at last
MMCV_WITH_OPS=1 uv sync --link-mode hardlink --no-build-isolation

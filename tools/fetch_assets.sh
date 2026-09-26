#!/usr/bin/env bash
# Downloads third-party models used by optional scene bodies into assets/ (not committed).
# Athena, Museo Archeologico Nazionale, Florence — scan from the Three D Scans archive
# (https://threedscans.com/museo-archeologico-nazionale/athena/), which publishes its scans
# for free use; check the archive's terms for your own use.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets
if [ ! -f assets/athena.obj ]; then
  curl -fL --retry 3 -o assets/Athena.obj.zip https://threedscans.com/wp-content/uploads/2016/02/Athena.obj.zip
  unzip -o -q assets/Athena.obj.zip Athena.obj -d assets
  mv assets/Athena.obj assets/athena.obj
  rm -f assets/Athena.obj.zip
fi
echo "assets ready: $(ls assets)"

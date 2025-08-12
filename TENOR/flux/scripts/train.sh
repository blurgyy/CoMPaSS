#!/usr/bin/env bash

set -Eeuo pipefail

unset http_proxy
unset https_proxy

function ntfy() {
  msg=${1:-(empty message)}
  curl -fSsd "$msg" https://ntfy.blurgy.xyz/exp &
}

for train_config in ./train_configs/01-tau_v_0.3.yaml ./train_configs/02-* ./train_configs/03-*; do
  ntfy "Starting training with config '$train_config'"
  accelerate launch compass_train_lora.py --config "$train_config"
  ntfy "Finished training with config '$train_config'"
done

/usr/bin/shutdown

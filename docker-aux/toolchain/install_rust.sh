#!/bin/bash
set -eou pipefail

curl https://sh.rustup.rs -sSf | sh -s -- -y
rustc --version && pip3 install setuptools-rust
#!/bin/bash -x

PORT=$(shuf -n 1 -i 49152-65535)

echo "$PORT"

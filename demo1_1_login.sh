#!/bin/bash

set -ex

aws ecr get-login-password --region eu-central-1 \
    | docker login --username AWS --password-stdin 763104351884.dkr.ecr.eu-central-1.amazonaws.com/
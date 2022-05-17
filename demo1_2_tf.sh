#!/bin/bash 

docker run -it --rm -v $(pwd):/opt -w /opt \
    763104351884.dkr.ecr.eu-central-1.amazonaws.com/tensorflow-training:2.6.2-cpu-py38-ubuntu20.04 bash





# docker run -it --rm -v $(pwd):/opt -w /opt  \
#     763104351884.dkr.ecr.eu-central-1.amazonaws.com/tensorflow-training:2.8.0-cpu-py39-ubuntu20.04-e3 bash

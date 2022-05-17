# Israel 2022 AWS summit code snippets

```commandline
aws ecr get-login-password --region eu-central-1 \
    | docker login --username AWS --password-stdin 763104351884.dkr.ecr.eu-central-1.amazonaws.com/
```

Run in a container with Tensorflow 2.6.2 on ubuntu 20.04 with Python 3.8

```commandline
docker run -it --rm -v $(pwd):/opt -w /opt  \
    763104351884.dkr.ecr.eu-central-1.amazonaws.com/tensorflow-training:2.6.2-cpu-py38-ubuntu20.04 bash

python ./code/train.py
```

Run in a container with Tensorflow 2.8.0 on ubuntu 20.04 with Python 3.9

```commandline
docker run -it --rm -v $(pwd):/opt -w /opt  \
    763104351884.dkr.ecr.eu-central-1.amazonaws.com/tensorflow-training:2.8.0-cpu-py39-ubuntu20.04-e3 bash

# Install missing package in the container 
pip install sklearn

python ./code/train.py
```

Let's move to Amazon SageMaker

Run the same code but now using SageMaker SDK

```commandline
python ./local.py
```

The same as a hosted SageMaker training job

```commandline
python ./hosted.py
```

Now, let's use spot instances

```commandline
python ./spot.py
```


from sagemaker.tensorflow import TensorFlow

project_name = 'tensorflow-project-A1'
hyperparameters = {'epochs': 700, 'batch_size': 128, 'learning_rate': 0.01}

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

tensorflow_estimator = TensorFlow(
    source_dir='code',
    entry_point='train.py',
    instance_type='local',
    instance_count=1,
    hyperparameters=hyperparameters,
    role=DUMMY_IAM_ROLE,
    base_job_name='tf2',
    framework_version='2.8.0',
    py_version='py39')

inputs = {'train': 'file://./data/train', 'test': 'file://./data/test'}

tensorflow_estimator.fit(inputs)



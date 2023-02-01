

import sagemaker
from sagemaker.tensorflow import TensorFlow

project_name = 'tensorflow-project-A1'
hyperparameters = {'epochs': 120, 'batch_size': 64, 'learning_rate': 0.01}

sess = sagemaker.session.Session()
bucket = sess.default_bucket()
train_instance_type = 'ml.g4dn.xlarge'

tensorflow_estimator = TensorFlow(
    source_dir='code',
    entry_point='train.py',
    instance_type=train_instance_type,
    instance_count=1,
    hyperparameters=hyperparameters,
    role=sagemaker.get_execution_role(),
    base_job_name=project_name,
    framework_version='2.8.0',
    py_version='py39'
    # container_log_level='debug'
    )

inputs = {'train': f's3://{bucket}/{project_name}/train', 'test': f's3://{bucket}/{project_name}/test'}

tensorflow_estimator.fit(inputs,logs=False)


import sagemaker
from sagemaker.tensorflow import TensorFlow

project_name = 'tensorflow-project-A1'
hyperparameters = {'epochs': 70, 'batch_size': 128, 'learning_rate': 0.01}

sess = sagemaker.session.Session()
bucket = sess.default_bucket()
train_instance_type = 'ml.c5.xlarge'

tensorflow_estimator = TensorFlow(
    source_dir='code',
    entry_point='train.py',
    instance_type=train_instance_type,
    instance_count=1,
    hyperparameters=hyperparameters,
    role=sagemaker.get_execution_role(),
    base_job_name=f"{project_name}-spot",
    framework_version='2.8.0',
    py_version='py39',
    # Managed Spot training
    use_spot_instances=True,
    max_run=7200,
    max_wait=7200,
    checkpoint_s3_uri=f"s3://{bucket}/{project_name}/checkpoint"
)

inputs = {'train': f's3://{bucket}/{project_name}/train', 'test': f's3://{bucket}/{project_name}/test'}

tensorflow_estimator.fit(inputs)

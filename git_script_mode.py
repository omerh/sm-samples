import sagemaker
from sagemaker.tensorflow import TensorFlow

project_name = 'tensorflow-project-A1'
hyperparameters = {'epochs': 100, 'batch_size': 128, 'learning_rate': 0.01}
# github_config = {
#     'repo': 'https://github.com/omerh/tf-sample-train.git',
#     'branch': 'main'
# }

sess = sagemaker.session.Session()
bucket = sess.default_bucket()

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

tensorflow_estimator = TensorFlow(
    source_dir='./code',
    entry_point='train.py',
    # git_config=github_config,
    instance_type='ml.c5.xlarge',
    instance_count=1,
    hyperparameters=hyperparameters,
    role=sagemaker.get_execution_role(),
    base_job_name='tf2',
    framework_version='2.8.0',
    py_version='py39')

# inputs = {'train': 'file://./data/train', 'test': 'file://./data/test'}
inputs = {'train': f's3://{bucket}/{project_name}/train', 'test': f's3://{bucket}/{project_name}/test'}

tensorflow_estimator.fit(inputs)



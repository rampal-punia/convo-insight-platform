# llms/sagemaker_integration.py

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.pytorch import PyTorch
from sagemaker.sklearn import SKLearn
from sagemaker.model_monitor import DataCaptureConfig
from botocore.exceptions import ClientError


class SageMakerIntegration:
    def __init__(self, role_arn, bucket_name):
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.sagemaker_session = sagemaker.Session()
        self.s3_client = boto3.client('s3')

    def train_model(self, model_type, script_path, hyperparameters, train_data_path, output_path):
        if model_type == 'huggingface':
            estimator = HuggingFace(
                entry_point=script_path,
                instance_type='ml.p3.2xlarge',
                instance_count=1,
                role=self.role_arn,
                transformers_version='4.6',
                pytorch_version='1.7',
                py_version='py36',
                hyperparameters=hyperparameters
            )
        elif model_type == 'pytorch':
            estimator = PyTorch(
                entry_point=script_path,
                instance_type='ml.p3.2xlarge',
                instance_count=1,
                role=self.role_arn,
                framework_version='1.8',
                py_version='py36',
                hyperparameters=hyperparameters
            )
        elif model_type == 'sklearn':
            estimator = SKLearn(
                entry_point=script_path,
                instance_type='ml.m5.xlarge',
                instance_count=1,
                role=self.role_arn,
                framework_version='0.23-1',
                hyperparameters=hyperparameters
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        estimator.fit({'train': train_data_path})

        model = estimator.create_model()
        model.deploy(initial_instance_count=1, instance_type='ml.t2.medium')

        return model

    def deploy_model(self, model, endpoint_name):
        try:
            data_capture_config = DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=100,
                destination_s3_uri=f's3://{self.bucket_name}/data-capture'
            )

            predictor = model.deploy(
                initial_instance_count=1,
                instance_type='ml.t2.medium',
                endpoint_name=endpoint_name,
                data_capture_config=data_capture_config
            )
            return predictor
        except ClientError as e:
            print(f"Error deploying model: {e}")
            return None

    def monitor_model(self, endpoint_name):
        try:
            client = boto3.client('cloudwatch')
            response = client.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='Invocations',
                Dimensions=[
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': 'AllTraffic'}
                ],
                StartTime='2024-08-18T00:00:00Z',  # Adjust this based on your needs
                EndTime='2024-08-19T00:00:00Z',    # Adjust this based on your needs
                Period=3600,
                Statistics=['Sum']
            )
            return response
        except ClientError as e:
            print(f"Error monitoring model: {e}")
            return None

    def upload_to_s3(self, file_path, s3_key):
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            return f's3://{self.bucket_name}/{s3_key}'
        except ClientError as e:
            print(f"Error uploading file to S3: {e}")
            return None


# Usage example
if __name__ == "__main__":
    sagemaker_integration = SageMakerIntegration(
        'your-role-arn', 'your-bucket-name')

    # Upload training script and data to S3
    script_s3_uri = sagemaker_integration.upload_to_s3(
        'path/to/your/training_script.py', 'scripts/training_script.py')
    data_s3_uri = sagemaker_integration.upload_to_s3(
        'path/to/your/training_data.csv', 'data/training_data.csv')

    # Train and deploy model
    model = sagemaker_integration.train_model(
        model_type='huggingface',
        script_path=script_s3_uri,
        hyperparameters={'epochs': 3, 'learning_rate': 1e-5},
        train_data_path=data_s3_uri,
        output_path=f's3://{sagemaker_integration.bucket_name}/output'
    )

    predictor = sagemaker_integration.deploy_model(model, 'your-endpoint-name')

    # Monitor the deployed model
    monitoring_data = sagemaker_integration.monitor_model('your-endpoint-name')
    print(monitoring_data)

# apps/llms/tasks.py

from celery import shared_task
from .sagemaker_integration import SageMakerIntegration
from django.conf import settings
from django.core.management import call_command


sagemaker_integration = SageMakerIntegration(
    settings.AWS_ROLE_ARN, settings.AWS_S3_BUCKET)


@shared_task
def run_fine_tuning():
    call_command('fine_tune_llm')


@shared_task
def train_and_deploy_model(model_type, script_path, hyperparameters, train_data_path, output_path, endpoint_name):
    # Upload training script and data to S3
    script_s3_uri = sagemaker_integration.upload_to_s3(
        script_path, f'scripts/{script_path.split("/")[-1]}')
    data_s3_uri = sagemaker_integration.upload_to_s3(
        train_data_path, f'data/{train_data_path.split("/")[-1]}')

    # Train model
    model = sagemaker_integration.train_model(
        model_type=model_type,
        script_path=script_s3_uri,
        hyperparameters=hyperparameters,
        train_data_path=data_s3_uri,
        output_path=output_path
    )

    # Deploy model
    predictor = sagemaker_integration.deploy_model(model, endpoint_name)

    if predictor:
        print(f"Model deployed successfully to endpoint: {endpoint_name}")
    else:
        print("Failed to deploy model")


@shared_task
def monitor_model_performance(endpoint_name):
    monitoring_data = sagemaker_integration.monitor_model(endpoint_name)
    if monitoring_data:
        print(
            f"Monitoring data for endpoint {endpoint_name}: {monitoring_data}")
    else:
        print(
            f"Failed to retrieve monitoring data for endpoint {endpoint_name}")

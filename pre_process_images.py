import numpy as np
import os
from skimage import io
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import boto3
from azureml.core import Experiment
from azureml.core import Workspace, Run, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


def run_pre_processing(parameters):
    # trigger_azure_instance()
    # trigger_ec2_instance_and_workers()
    image_data_holder = read_images(parameters)
    pooled_image_data, pooled_image_labels = clean_pool_data(image_data_holder)
    data = split_data(parameters, pooled_image_data, pooled_image_labels)
    data['labels'] = list(image_data_holder.keys())
    save_original_data(data, parameters)
    return data


def trigger_ec2_instance_and_workers():
    region = 'us-east-1'
    instances = ['X-XXXXXXXX']
    ec2 = boto3.client('ec2', region_name=region)
    ec2.start_instances(InstanceIds=instances)
    print(f'started your instances: {instances}')


def trigger_azure_instance():
    ws = Workspace.from_config()
    cluster_name = "gpu-cluster"
    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing compute target')
    except ComputeTargetException:
        print('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                               max_nodes=4)
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    curated_env_name = 'AzureML-TensorFlow-2.2-GPU'
    tf_env = Environment.get(workspace=ws, name=curated_env_name)
    args = ['--data-directory', '/Users/ankushgupta/Documents/amazon_case_study/data/dog_images']
    src = ScriptRunConfig(source_directory='/Users/ankushgupta/Documents/pythonProject',
                          script='main.py',
                          compute_target=compute_target,
                          environment=tf_env)
    run = Experiment(workspace=ws, name='Tutorial-TF-Mnist').submit(src)
    run.wait_for_completion(show_output=True)


def read_images(parameters):
    # read the data in
    image_data_holder = {}
    for directory in os.listdir(parameters['data_directory']):
        sub_directory = os.path.join(parameters['data_directory'], directory)
        if sub_directory.split('/')[-1] != '.DS_Store' and sub_directory.split('/')[-1].split('.')[-1] != 'csv' \
                and sub_directory.split('/')[-1] != 'saved_model_weights':
            image_data_holder[sub_directory.split('/')[-1]] = {}
            path_to_jpgs = [os.path.join(sub_directory, _) for _ in os.listdir(sub_directory) if _.endswith(r".jpg")]
            for path in path_to_jpgs:
                image = io.imread(path)
                # try BGR2GRAY and RGB2GRAY
                # grey_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                image_data_holder[sub_directory.split('/')[-1]][path.split('/')[-1].split('.')[-2]] = image
        else:
            pass
    return image_data_holder


def read_submitted_image(image_path):
    image = io.imread(image_path)
    image = image[np.newaxis, :, :, :]
    image = image.astype(np.float32)
    if image.shape != (1, 28, 28, 3):
        image = cv2.resize(image, (1, 28, 28, 3), interpolation=cv2.INTER_LINEAR)
    return image


def clean_pool_data(image_data_holder):
    # intialize some information and structures
    labels = list(image_data_holder.keys())
    pooled_image_data = []
    pooled_image_labels = []
    # utilize the shape of the first image for the first folder as reference (for ease)
    ref_image_dim = np.shape(image_data_holder[list(labels)[0]][list(image_data_holder[list(labels)[0]])[0]])
    for global_key in labels:
        for local_key in list(image_data_holder[global_key].keys()):
            # normalize the image between 0 and 1
            image_data_holder[global_key][local_key] = image_data_holder[global_key][local_key] / np.max(image_data_holder[global_key][local_key])
            # making sure that all of the images are the same size
            dims_check = image_data_holder[global_key][local_key].shape
            if dims_check != ref_image_dim:
                final_image = cv2.resize(image_data_holder[global_key][local_key], ref_image_dim[:2], interpolation=cv2.INTER_LINEAR)
            else:
                final_image = image_data_holder[global_key][local_key]
            # pool data together for downstream tasks
            pooled_image_data.append(final_image)
            pooled_image_labels.append(np.where(np.asarray(labels) == np.asarray(global_key))[0][0])
            # clear the data from the original dictionary for memory purposes
            image_data_holder[global_key][local_key] = []
    return pooled_image_data, pooled_image_labels


def split_data(parameters, pooled_image_data, pooled_image_labels):
    # split the data into training/validation/test and reshape to input into model
    x_train, x_temp, y_train, y_temp = train_test_split(np.asarray(pooled_image_data), np.asarray(pooled_image_labels), test_size=1 - parameters['training_split'], random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=1 / ((1 - parameters['training_split']) / parameters['test_split']), random_state=42)
    data = dict(
        training_data = np.array(x_train.reshape(-1, x_train.shape[1], x_train.shape[2], x_train.shape[-1])),
        validation_data = np.array(x_val.reshape(-1, x_val.shape[1], x_val.shape[2], x_val.shape[-1])),
        test_data = np.array(x_test.reshape(-1, x_test.shape[1], x_test.shape[2], x_test.shape[-1])),
        full_data = np.array(np.asarray(pooled_image_data).reshape(-1, np.asarray(pooled_image_data).shape[1], np.asarray(pooled_image_data).shape[2], np.asarray(pooled_image_data).shape[-1]))
    )
    data['y_train'] = y_train
    data['y_test'] = y_test
    return data


def save_original_data(data, parameters):
    for channel in range(3):
        channel_specific_data = data['full_data'][:, :, :, channel]
        channel_specific_data_2 = channel_specific_data.reshape(channel_specific_data.shape[0], channel_specific_data.shape[1]*channel_specific_data.shape[2])
        pd.DataFrame(channel_specific_data_2).to_csv(os.path.join(parameters['data_directory'], 'channel_' + str(channel) + '.csv'), index=False)

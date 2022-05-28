# handle imports
import os
import argparse
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from pre_processing import run_pre_processing, read_submitted_image
from initialize_and_run_models import run_models_main, Convolutional_Autoencoder
from plot_metrics import run_plot_metrics
from visualizing_decoded_and_latent import get_decoded_and_latent
from find_similar_images import get_similar_images, calculate_KNN_and_cosine
import boto3


# read in parameters
def get_hyperparams():
    args = get_args().parse_args()
    parameters = vars(args)
    return parameters


def get_args():
    parser = argparse.ArgumentParser(description="Parameters For Neural Nets")
    # general
    parser.add_argument('--data_directory', type=str, default="/Users/ankushgupta/Documents/amazon_case_study/data/materials_images", help='directories for data')
    parser.add_argument('--run_hyperparameter_tuning', type=bool, default=False, help='whether or not to run hyperparameter running for convolutional autoencoder')
    parser.add_argument('--run_classifier_flag', type=bool, default=True, help='whether or not to run downstream classifier')
    # for splits
    parser.add_argument('--training_split', type=float, default=0.6, help='split for training split')
    parser.add_argument('--validation_split', type=float, default=0.2, help='split for validation split')
    parser.add_argument('--test_split', type=float, default=0.2, help='split for test split')
    # for classifier
    parser.add_argument('--classifier_lr', type=float, default=0.0005, help='learning rate for classifier')
    parser.add_argument('--classifier_epochs', type=int, default=800, help='epochs for classifier')
    # fixed hyperparams
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for model')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dim for model')
    parser.add_argument('--epochs', type=int, default=20, help='epochs for model')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate for model')
    parser.add_argument('--num_encoder_conv_layers', type=int, default=2, help='# of convolutional / dropout blocks')
    parser.add_argument('--num_decoder_conv_layers', type=int, default=2, help='# of convolutional / dropout blocks (deocder)')
    parser.add_argument('--dropout_val', type=float, default=0.0, help='dropout val')
    parser.add_argument('--augment', type=bool, default=False, help='whether to augment the data or not')
    # for convolutional autoencoder hyperparameter optimization
    parser.add_argument('--tuning_batch_size', type=list, default=[10], help='batch size for model')
    parser.add_argument('--tuning_latent_dim', type=list, default=[20], help='latent dim for model')
    parser.add_argument('--tuning_epochs', type=list, default=[10, 50, 100], help='epochs for hyperparameter optimization')
    parser.add_argument('--tuning_learning_rate', type=list, default=[1e-1, 0.5e-1, 1e-2, 0.5e-2, 1e-3, 0.5e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5, 1e-6, 0.5e-6, 1e-7, 0.5e-7],
                        help='learning rate for hyperparameter optimization')
    return parser


_ = os.system('clear')
# get parameters
parameters = get_hyperparams()

if os.path.exists(os.path.join(parameters['data_directory'], 'encoded_image_data.csv')):
    response = input(
        'Existing Encoded Image Data Found In ' + parameters['data_directory'] + '\n'
        'Would You Like To Overwrite It? (Y/N)?   '
    )
    if response == 'Y':
        case = 1
    else:
        case = 2
else:
    case = 1
if case == 1:
    print('\n\nRunning Neural Nets For Image Encoding!')
    # pre processing of images
    data = run_pre_processing(parameters)
    # running models
    encoded, models, metrics = run_models_main(parameters, data)
    # plot metrics and downstream classification results
    run_plot_metrics(models, metrics, encoded, data)
    # visualizing the latent space and decoded image
    get_decoded_and_latent(models, data, parameters, encoded)
    # saving full encoded set to .csv and outputting into same directory --> loads in and takes user input
    pd.DataFrame(encoded['encoded_full_set']).to_csv(os.path.join(parameters['data_directory'], 'encoded_image_data.csv'))
    _ = os.system('clear')
    print('\nFinished Running Neural Nets!')
    encoded_full_set = encoded['encoded_full_set']
elif case == 2:
    encoded_full_set = pd.read_csv(os.path.join(parameters['data_directory'], 'encoded_image_data.csv'), index_col=False)
    encoded_full_set = encoded_full_set[encoded_full_set.columns[1:]].to_numpy()
    channel_holder = []
    for channel in range(3):
        channel_data = pd.read_csv(os.path.join(parameters['data_directory'], 'channel_' + str(channel) + '.csv')).to_numpy()
        channel_data2 = channel_data.reshape(channel_data.shape[0], int(np.sqrt(channel_data.shape[1])), int(np.sqrt(channel_data.shape[1])))
        channel_holder.append(channel_data2)
    full_original_data = np.stack(channel_holder, axis=3)

# ask for image index as input ---> outputs the most similar images
use_existing_image_or_no = input('\n\nWould You Like to Submit An Image (Y/N)?:  ')
if use_existing_image_or_no == 'N':
    case2 = 3
elif use_existing_image_or_no == 'Y':
    case2 = 4

if case2 == 3:
    image_index = input('\n\nPlease Enter Index for Image:  ')
elif case2 == 4:
    image_path = input('\n\nPlease Enter Path Directly To Image:  ')
    image = read_submitted_image(image_path)
    if case == 1:
        full_original_data = data['full_data']
    original_with_input = np.empty((full_original_data.shape[0]+1, full_original_data.shape[1], full_original_data.shape[2],
                                    full_original_data.shape[3]))
    original_with_input[:] = np.NaN
    original_with_input[:full_original_data.shape[0], :, :, :] = full_original_data
    original_with_input[full_original_data.shape[0], :, :, :] = image
    convolutional_autoencoder = Convolutional_Autoencoder(parameters)
    convolutional_autoencoder.build(input_shape=original_with_input.shape)
    convolutional_autoencoder.load_weights(os.path.join(parameters['data_directory'], 'saved_model_weights', 'epoch_20.hdf5'))
    convolutional_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                                      loss=tf.keras.losses.MeanSquaredError())
    encoded_original_with_input = convolutional_autoencoder.encoder.predict(original_with_input)
num_similar_images = input('\nPlease Enter # Of Similar Images To Find:  ')
num_similar_images = int(num_similar_images)

print('\n\nProceeding To Find Similar Images...!')

if case2 == 4:
    calculate_KNN_and_cosine(encoded_original_with_input, original_with_input, num_similar_images, image_index=0, input_image_flag=True)
elif case2 == 3 and case == 2:
    get_similar_images(full_original_data, encoded_full_set, int(image_index), int(num_similar_images))
elif case2 == 3 and case == 1:
    get_similar_images(data['full_data'], encoded_full_set, int(image_index), int(num_similar_images))

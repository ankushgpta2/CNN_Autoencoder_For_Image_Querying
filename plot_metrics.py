import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def run_plot_metrics(models, metrics, encoded, data):
    plot_autoencoder_classifier_metrics(metrics['reconstruction_metrics'], metrics['classification_metrics'])
    plot_confusion_matrics(models['downstream_classifier'], encoded, data)


def plot_autoencoder_classifier_metrics(reconstruction_metrics, classification_metrics):
    # plot the training and validation loss metrics
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 12))

    # plot the image reconstruction loss
    ax1.plot(range(len(reconstruction_metrics.history['loss'])), reconstruction_metrics.history['loss'], label='Training Loss', linewidth=5, color='dodgerblue')
    ax1.plot(range(len(reconstruction_metrics.history['val_loss'])), reconstruction_metrics.history['val_loss'], label='Validation Loss', linewidth=5, color='red')
    ax1.legend(fontsize=14), ax1.set_xlabel('Epochs', fontsize=14), ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=14)
    ax1.set_title('Image Reconstruction Loss', fontsize=16)

    # plot the classification metrics
    ax2.plot(range(len(classification_metrics.history['loss'])), classification_metrics.history['loss'], label='Training Loss', linewidth=5, color='dodgerblue')
    ax3.plot(range(len(classification_metrics.history['accuracy'])), classification_metrics.history['accuracy'], label='Training Accuracy', linewidth=5, color='green')
    ax2.legend(fontsize=14), ax2.set_xlabel('Epochs', fontsize=14), ax2.set_ylabel('Categorical Cross Entropy', fontsize=14)
    ax3.legend(fontsize=14), ax3.set_xlabel('Epochs', fontsize=14), ax3.set_ylabel('Accuracy', fontsize=14)
    ax2.set_title('Classification Loss On Encoded Images', fontsize=16)
    ax3.set_title('Classification Accuracy on Encoded Images', fontsize=16)
    plt.suptitle('Metrics From Autoencoder and Classifier', fontsize=22, fontweight='bold')
    plt.show()


def plot_confusion_matrics(downstream_classifier, encoded, data):
    # plot confusion matrix of the classification results
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    train_predictions = downstream_classifier.predict(encoded['encoded_training_set'])
    train_predicted_labels = [np.where(train_predictions[x] == np.max(train_predictions[x]))[0][0] for x in range(np.shape(train_predictions)[0])]
    train_norm_matrix = confusion_matrix(data['y_train'], train_predicted_labels, normalize='true')
    sns.heatmap(train_norm_matrix, ax=axes[0], annot=True, yticklabels=data['labels'], xticklabels=data['labels'])
    axes[0].set_title('Encoded Training Data', fontsize=16)

    test_predictions = downstream_classifier.predict(encoded['encoded_test_set'])
    test_predicted_labels = [np.where(test_predictions[x] == np.max(test_predictions[x]))[0][0] for x in range(np.shape(test_predictions)[0])]
    test_norm_matrix = confusion_matrix(data['y_test'], test_predicted_labels, normalize='true')
    sns.heatmap(test_norm_matrix, ax=axes[1], annot=True, yticklabels=data['labels'], xticklabels=data['labels'])
    axes[1].set_title('Encoded Test Data', fontsize=16)

    plt.suptitle('Confusion Matrices From Encoded Training/Test Data', fontweight='bold', fontsize=20)
    plt.show()

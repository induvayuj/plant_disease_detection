"""
Model Evaluation and Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, roc_auc_score, accuracy_score
)
import logging
import os

from config import CLASS_NAMES, LOGS_PATH

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set and return metrics
    """
    logger.info("Evaluating model on test set...")
    
    # Get predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_labels, y_pred)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    return {
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs,
        'y_true': y_test_labels,
        'accuracy': accuracy
    }


def plot_training_history(history, save_path=None):
    """
    Plot training history (accuracy and loss)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Training Accuracy', marker='o')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history['loss'], label='Training Loss', marker='o')
    axes[1].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    return cm


def plot_top_predictions(model, X_test, y_test, num_samples=9, save_path=None):
    """
    Plot images with top predictions
    """
    y_pred_probs = model.predict(X_test[:num_samples])
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test[:num_samples], axis=1) if y_test.ndim > 1 else y_test[:num_samples]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, ax in enumerate(axes):
        ax.imshow(X_test[idx].astype('uint8'))
        true_label = CLASS_NAMES[y_test_labels[idx]]
        pred_label = CLASS_NAMES[y_pred[idx]]
        confidence = np.max(y_pred_probs[idx])
        
        color = 'green' if y_pred[idx] == y_test_labels[idx] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                    color=color, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Predictions plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(y_true, y_pred, save_path=None):
    """
    Print and save classification report
    """
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    
    logger.info("\nClassification Report:")
    logger.info(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        logger.info(f"Classification report saved to {save_path}")
    
    return report


def plot_top_k_accuracy(model, X_test, y_test, k_values=[1, 3, 5], save_path=None):
    """
    Plot top-k accuracy for different k values
    """
    y_pred_probs = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    
    top_k_accuracies = []
    
    for k in k_values:
        top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
        correct = sum(y_test_labels[i] in top_k_preds[i] for i in range(len(y_test_labels)))
        accuracy = correct / len(y_test_labels)
        top_k_accuracies.append(accuracy)
        logger.info(f"Top-{k} Accuracy: {accuracy:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.bar([f'Top-{k}' for k in k_values], top_k_accuracies, color='skyblue', edgecolor='navy')
    plt.ylabel('Accuracy')
    plt.title('Top-K Accuracy')
    plt.ylim([0, 1])
    
    for i, v in enumerate(top_k_accuracies):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Top-K accuracy plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_evaluation_report(model, X_test, y_test, experiment_name='evaluation'):
    """
    Generate complete evaluation report with all metrics and plots
    """
    logger.info("Generating complete evaluation report...")
    
    # Create evaluation directory
    eval_dir = os.path.join(LOGS_PATH, experiment_name)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # Generate plots
    plot_training_history({}, os.path.join(eval_dir, 'training_history.png'))
    plot_confusion_matrix(y_true, y_pred, os.path.join(eval_dir, 'confusion_matrix.png'))
    plot_top_k_accuracy(model, X_test, y_test, save_path=os.path.join(eval_dir, 'top_k_accuracy.png'))
    plot_top_predictions(model, X_test, y_test, save_path=os.path.join(eval_dir, 'sample_predictions.png'))
    
    # Save classification report
    print_classification_report(y_true, y_pred, os.path.join(eval_dir, 'classification_report.txt'))
    
    logger.info(f"Evaluation report saved to {eval_dir}")

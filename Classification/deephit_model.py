import numpy as np # For numerical operations
import tensorflow.keras.backend as K # For backend tensor operations in TensorFlow
from tensorflow.keras.models import Sequential # To create a sequential neural network model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization  # Added BatchNormalization for training stability
from tensorflow.keras.optimizers import Adam # To use the Adam optimizer during training
from tensorflow.keras.utils import to_categorical # To convert labels into one-hot encoded format
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # To control training with callbacks

def ranking_loss(y_true, y_pred):
    """
    Computes a ranking loss which encourages the average predicted probability
    for the positive class to exceed that of the negative class by at least a fixed margin.
    
    Parameters:
    - y_true: Ground truth one-hot encoded labels.
    - y_pred: Predicted probabilities for each class.
    
    Returns:
    - A tensor representing the ranking loss value.
    """
    # Extract the predicted probability for class 1 (second column) and clip it to avoid numerical issues.
    p = K.clip(y_pred[:, 1], 1e-7, 1-1e-7)
    # Create a mask for positive samples: 1 if the true class is positive, 0 otherwise.
    pos_mask = K.cast(K.equal(K.argmax(y_true, axis=-1), 1), K.floatx())
    # Create a mask for negative samples: complement of the positive mask.
    neg_mask = 1 - pos_mask
    # Compute the mean predicted probability for positive samples.
    pos_mean = K.sum(p * pos_mask) / (K.sum(pos_mask) + K.epsilon())
    # Compute the mean predicted probability for negative samples.
    neg_mean = K.sum(p * neg_mask) / (K.sum(neg_mask) + K.epsilon())
    # Define a fixed margin value that the difference between positive and negative means should exceed.
    margin = 0.1
    # Return the ranking loss: if (pos_mean - neg_mean) is less than margin, return the difference, otherwise zero.
    return K.maximum(0.0, margin - (pos_mean - neg_mean))

def combined_loss(y_true, y_pred):
    """
    Combined loss function that adds the standard categorical crossentropy loss
    and a weighted ranking loss.
    
    Parameters:
    - y_true: Ground truth one-hot encoded labels.
    - y_pred: Predicted probabilities for each class.
    
    Returns:
    - A tensor representing the total loss.
    """
    # Compute categorical crossentropy loss.
    ce = K.categorical_crossentropy(y_true, y_pred)
    # Compute ranking loss.
    rl = ranking_loss(y_true, y_pred)
    # Define weight for the ranking loss component.
    alpha = 0.2
    # Return the sum of categorical crossentropy loss and the weighted ranking loss.
    return ce + alpha * rl

class DeephitModel:
    def __init__(self, input_dim, hidden_units=64):
        """
        Initializes the DeepHit model with a specified input dimension and hidden layer size.
        
        Parameters:
        - input_dim: Integer, number of input features.
        - hidden_units: Integer, number of neurons in each hidden layer.
        """
        # Build a sequential neural network model.
        self.model = Sequential([
            # Input layer.
            Input(shape=(input_dim,)),
            # First hidden dense layer with ReLU activation.
            Dense(hidden_units, activation='relu'),
            # BatchNormalization layer to stabilize and accelerate training.
            BatchNormalization(),
            # Dropout layer to prevent overfitting.
            Dropout(0.2),
            # Second hidden dense layer with ReLU activation.
            Dense(hidden_units, activation='relu'),
            # Another BatchNormalization layer.
            BatchNormalization(),
            # Second Dropout layer.
            Dropout(0.2),
            # Output layer with softmax activation for binary classification.
            Dense(2, activation='softmax')
        ])
        # Compile the model with Adam optimizer, combined loss function, and accuracy metric.
        # Gradient clipping (clipnorm=1.0) is applied to avoid exploding gradients.
        self.model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
                           loss=combined_loss,
                           metrics=['accuracy'])
    
    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, class_weight=None):
        """
        Trains the DeepHit model using early stopping and learning rate reduction callbacks.
        
        Parameters:
        - X: Numpy array of training features.
        - y: Numpy array of one-hot encoded training labels.
        - epochs: Integer, maximum number of training epochs.
        - batch_size: Integer, training batch size.
        - validation_data: Tuple (val_X, val_y) of validation features and one-hot encoded labels.
        - class_weight: Dictionary, mapping class indices to a weight to handle class imbalance.
        """
        # Choose which metric to monitor for early stopping: validation loss if validation data is provided.
        monitor_metric = 'val_loss' if validation_data is not None else 'loss'
        callbacks = [
            EarlyStopping(monitor=monitor_metric, patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=2, min_lr=1e-6)
        ]
        # Fit the model with the specified parameters.
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0,
                       validation_data=validation_data, callbacks=callbacks,
                       class_weight=class_weight)
    
    def predict(self, X):
        """
        Predicts class labels for the input features.
        
        Parameters:
        - X: Numpy array of input features.
        
        Returns:
        - A numpy array of predicted class labels (0 or 1).
        """
        # Predict probabilities and select the class with the highest probability.
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input.
        
        Parameters:
        - X: Numpy array of input features.
        
        Returns:
        - A numpy array of predicted probabilities for each class.
        """
        return self.model.predict(X)

def train_deephit(train_data, class_weight=None):
    """
    Trains a DeepHit model using the provided training data.
    
    Parameters:
    - train_data: Dictionary with keys "X", "y", and optionally "val_X", "val_y".
                 "X" is the training features and "y" are the corresponding labels (assumed to be 0/1).
    - class_weight: Optional dictionary for class weights to handle imbalanced data.
    
    Returns:
    - A trained instance of DeephitModel.
    """
    # Convert training features and labels to numpy arrays.
    X = np.array(train_data["X"])
    y = np.array(train_data["y"])
    # Ensure the feature matrix has two dimensions.
    if (X.ndim == 1):
        X = X.reshape(-1, 1)
    # Validate that there is at least one feature.
    if (X.shape[1] == 0):
        raise ValueError("Invalid input: X has no features")
    # Determine the number of input features.
    input_dim = X.shape[1]
    # Convert labels to one-hot encoded format for binary classification.
    y_onehot = to_categorical(y, num_classes=2)
    
    # Initialize the DeepHit model.
    model = DeephitModel(input_dim=input_dim, hidden_units=64)
    
    # If validation data is provided, prepare and use it during training.
    if ("val_X" in train_data and "val_y" in train_data):
        val_X = np.array(train_data["val_X"])
        val_y = np.array(train_data["val_y"])
        if (val_X.ndim == 1):
            val_X = val_X.reshape(-1, 1)
        val_y_onehot = to_categorical(val_y, num_classes=2)
        model.fit(X, y_onehot, epochs=train_data["epochs"], batch_size=train_data["batch_size"], 
                  validation_data=(val_X, val_y_onehot), class_weight=class_weight)
    else:
        # Train without validation data if not provided.
        model.fit(X, y_onehot, epochs=train_data["epochs"], batch_size=train_data["batch_size"], class_weight=class_weight)
    
    return model

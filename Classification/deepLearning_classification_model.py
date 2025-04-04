import numpy as np # Import Numpy For Numerical Operations
import tensorflow.keras.backend as K # Import Keras Backend For Tensor Operations
from tensorflow.keras.models import Sequential # Import Sequential Model Constructor
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, BatchNormalization # Added BatchNormalization for stability
from tensorflow.keras.optimizers import Adam # Import Adam Optimizer For Model Training
from tensorflow.keras.utils import to_categorical # Import Utility To Convert Labels To One-Hot Encoding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Import Callbacks For Training Control

def transformation_layer(x):
    """
    Applies a logarithmic transformation to simulate the transformation of the risk function.
    
    This function applies the operation log(1 + x) on the input tensor.
    To avoid issues with taking the logarithm of zero or negative values, the input is first clipped.
    
    Parameters:
    - x: Input tensor.
    
    Returns:
    - A tensor representing the log-transformed input.
    """
    # Clip the input to ensure 1.0 + x is greater than zero.
    clipped_input = K.clip(1.0 + x, K.epsilon(), None)
    # Return the natural logarithm of the clipped input.
    return K.log(clipped_input)

class DeepLearningClassifier:
    def __init__(self, input_dim, hidden_units=64):
        """
        Initializes a deep neural network model for binary classification using the Transformation approach.
        This model includes a custom transformation layer to simulate risk transformation.
        
        Parameters:
        - input_dim: Integer, the number of input features.
        - hidden_units: Integer, the number of neurons in each hidden layer.
        """
        # Build a sequential model with:
        # - An input layer that accepts 'input_dim' features.
        # - Two Dense layers with ReLU activation, each followed by BatchNormalization and Dropout.
        # - A Lambda layer that applies the custom transformation_layer.
        # - An output Dense layer with softmax activation for binary classification.
        self.model = Sequential([
            Input(shape=(input_dim,)),                    # Input layer with specified dimension.
            Dense(hidden_units, activation='relu'),       # First dense layer with ReLU activation.
            BatchNormalization(),                         # BatchNormalization for improved training stability.
            Dropout(0.2),                                 # Dropout layer to reduce overfitting.
            Dense(hidden_units, activation='relu'),       # Second dense layer with ReLU activation.
            BatchNormalization(),                         # Additional BatchNormalization.
            Lambda(transformation_layer),                 # Custom transformation layer to simulate risk transformation.
            Dense(2, activation='softmax')                # Output layer for binary classification.
        ])
        # Compile the model with the Adam optimizer, using categorical crossentropy loss,
        # and including accuracy as a performance metric. Gradient clipping is applied to avoid exploding gradients.
        self.model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, class_weight=None):
        """
        Trains the Transformation model using early stopping and learning rate reduction callbacks.
        
        Parameters:
        - X: Numpy array of training features.
        - y: Numpy array of one-hot encoded training labels.
        - epochs: Integer, maximum number of training epochs.
        - batch_size: Integer, size of each training batch.
        - validation_data: Tuple (val_X, val_y) for validation during training.
        - class_weight: Dictionary mapping class indices to weights, to handle class imbalance.
        """
        # Determine the metric to monitor: 'val_loss' if validation data is provided; otherwise 'loss'.
        monitor_metric = 'val_loss' if validation_data is not None else 'loss'
        callbacks = [
            EarlyStopping(monitor=monitor_metric, patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=2, min_lr=1e-6)
        ]
        # Fit the model with the provided training data and callbacks.
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0,
                       validation_data=validation_data, callbacks=callbacks,
                       class_weight=class_weight)
    
    def predict(self, X):
        """
        Predicts class labels for the given input features.
        
        Parameters:
        - X: Numpy array of input features.
        
        Returns:
        - A numpy array of predicted class labels (0 or 1) for each sample.
        """
        # Generate predictions (probabilities) and choose the class with the highest probability.
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input features.
        
        Parameters:
        - X: Numpy array of input features.
        
        Returns:
        - A numpy array of predicted probabilities for each class.
        """
        return self.model.predict(X)

def train_deepLearning(train_data, class_weight=None):
    """
    Trains a Transformation model using the provided training data.
    
    Parameters:
    - train_data: A dictionary with the following keys:
         "X": Numpy array of training features.
         "y": Numpy array of training labels (assumed to be 0 or 1).
         Optionally, "val_X" and "val_y" for validation data.
    - class_weight: Optional dictionary for class weights to handle imbalanced data.
    
    Returns:
    - A trained instance of DeepLearningClassifier.
    """
    # Convert training features and labels to numpy arrays.
    X = np.array(train_data["X"])
    y = np.array(train_data["y"])
    # Ensure that X has at least two dimensions.
    if (X.ndim == 1):
        X = X.reshape(-1, 1)
    # Validate that there is at least one feature.
    if (X.shape[1] == 0):
        raise ValueError("Invalid input: X has no features")
    # Determine the number of input features.
    input_dim = X.shape[1]
    # Convert labels to one-hot encoded format for binary classification.
    y_onehot = to_categorical(y, num_classes=2)
    
    # Initialize the DeepLearningClassifier with the specified input dimension and hidden units.
    model = DeepLearningClassifier(input_dim=input_dim, hidden_units=64)
    
    # If validation data is provided in the dictionary, prepare and use it during training.
    if (("val_X" in train_data) and ("val_y" in train_data)):
        val_X = np.array(train_data["val_X"])
        val_y = np.array(train_data["val_y"])
        if (val_X.ndim == 1):
            val_X = val_X.reshape(-1, 1)
        val_y_onehot = to_categorical(val_y, num_classes=2)
        # Train the model using both training and validation data.
        model.fit(X, y_onehot, epochs=train_data["epochs"], batch_size=train_data["batch_size"], 
                  validation_data=(val_X, val_y_onehot), class_weight=class_weight)
    else:
        # Train the model using only the training data if no validation data is provided.
        model.fit(X, y_onehot, epochs=train_data["epochs"], batch_size=train_data["batch_size"], class_weight=class_weight)
    
    # Return the trained DeepLearningClassifier instance.
    return model

import tensorflow as tf

# Calculates the difference between two face embeddings in a Siamese Neural Network.
# Inherited from tf.keras.layers.Layer to build a custom Keras layer
class L1Dist(tf.keras.layers.Layer):

    # Constructor method
    def __init__(self, **kwargs):

        # Super refers parent class and init calls parent class constructor
        super().__init__(**kwargs)



    # "call" defines the actual computation logic that executes when data flows 
    # through this layer during model inference or training
    # Takes 'inputs' as a single list: [input_embedding, validation_embedding]
    def call(self, inputs):
        # Unpack the list into two separate variables
        input_embedding, validation_embedding = inputs
        
        # Step 2: Unwrap inner lists if Keras 3 wrapped them in [[Tensor1], [Tensor2]]
        if isinstance(input_embedding, (list, tuple)):
            input_embedding = input_embedding[0]
        if isinstance(validation_embedding, (list, tuple)):
            validation_embedding = validation_embedding[0]

        # Returns the resulting 4,096-dimensional distance vector |E_1 - E_2| to be 
        # passed into the final classification layer
        return tf.math.abs(input_embedding-validation_embedding)
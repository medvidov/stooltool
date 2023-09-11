import sys
import numpy as np
import tensorflow as tf
from utils import decode_and_resize_image, get_encoded_image

class StoolToolCNN:
    """
    StoolToolCNN object for use when doing inference server side using trained classifier.
    """
    def __init__(self, model_path: str):
        """
        Constructor for StoolToolCNN object.

        Input:
            model_path: String path to saved model want to load for inference.
        """
        # Load the model
        self._model = tf.keras.models.load_model(model_path)

    def inference(self, image_path: str):
        """
        Perform inference using the loaded classifier.

        Input:
            image_path: String path to image want to perform inference on.

        Output:
            Prediction scores for each label on the predicted image.
        """
        # Load image for inference
        crop_size = self._model.input.shape[1:3]
        image = decode_and_resize_image((get_encoded_image(image_path), tf.convert_to_tensor(crop_size)))

        # Perform inference
        image = np.expand_dims(image, axis=0)
        prediction = self._model.predict(image)

        # Return the probabilities for each class
        return tf.nn.softmax(prediction)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python inference.py [MODEL PATH] [IMAGE PATH]')
        sys.exit()
    
    # Load model and perform inference
    model = StoolToolCNN(sys.argv[1])
    predictions = model.inference(sys.argv[2])

    # print predictions
    print(predictions)
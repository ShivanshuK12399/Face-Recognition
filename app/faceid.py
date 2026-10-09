# Import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other stuff 
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies 
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Defines custom application class inheriting from Kivy's App base class.
class CamApp(App):

    # method construct and return the application UI layout.
    def build(self):

        # Initializes Kivy GUI components for displaying images, buttons, and labels.

        # self img will display the live video stream.
        # Occupies 100% of the window width and 80% of the window height.4
        self.img1 = Image(size_hint=(1,0.8))

        # Creates the "Verify" Button widget.
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,0.1))

        # Creates a Status Text Label with initial text 'Verification Uninitiated'.
        self.verification_label = Label(text='Verification Uninitiated', size_hint=(1,0.1))


        # add items to layout 
        # Creates a vertical layout container that stacks items from top to bottom.
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)


        # load tensorflow model
        self.model = tf.keras.models.load_model('model.h5', custom_objects={'L1Dist':L1Dist})


        # Initializing the Webcam
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,250)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,250)


        # A Kivy timer that repeatedly calls self.update at 30fps
        Clock.schedule_interval(self.update, 1.0/30.0)     

        return layout


    def update(self, *args):
        success, frame = self.capture.read()

        # Flips the image vertically along the X-axis
        # Why? OpenCV places (0,0) at the top-left, but Kivy places (0,0) at the bottom-left. Flipping prevents the webcam image from displaying upside-down!
        buf = cv2.flip(frame,0).tostring() # Converts the image array into a raw byte string so Kivy can read it.

        # Creates a new texture object with the same dimensions and color format as the frame.
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')

        # Copies the raw byte string data from the buffer into the texture.
        img_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte') 

        # Assigns the updated texture to the Image widget.
        self.img1.texture = img_texture 


    def preprocess(self, file_path): 
        # Read file from disk as raw bytes
        byte_img=tf.io.read_file(file_path)

        # Decode JPEG into uint8 tensor (H, W, 3)
        # converts the .jpg photo file into a 3D grid of color numbers (Red, Green, Blue)
        img=tf.io.decode_jpeg(byte_img)

        # Resize to standard target dimensions (105x105)
        img =tf.image.resize(img,(105,105))

        # Rescale pixel values from [0, 255] to [0.0, 1.0]
        img=img/255.0
        
        return img


    # Verification function 
    def verify(self, *args):

        detection_threshold = 0.5
        verification_threshold = 0.5

        # Capturing the Live Image on Verification Click
        save_path = os.path.join('application_data','input_image','input_image.jpg')
        success, frame = self.capture.read(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,250)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,250)
        cv2.imwrite(save_path, frame)
        

        results=[]
        verification_folder=os.listdir(os.path.join('application_data','verification_images'))

        for image in verification_folder:

            input_image=self.preprocess(os.path.join('application_data','input_image','input_image.jpg'))
            validation_image=self.preprocess(os.path.join('application_data','verification_images', image))

            # np.expand_dims inserts an extra dimension of size 1 at the specified axis
            # Add batch dimension to create 4D tensors: (1, 100, 100, 3)
            input_batch = np.expand_dims(input_image, axis=0)
            val_batch = np.expand_dims(validation_image, axis=0)

            # make predictions 
            result = self.model.predict([input_batch, val_batch], verbose=0)
            results.append(result)
    
        # np.sum() counts the total number of True matches {eg. 3 out of 5 photos}
        detection=np.sum(np.array(results) > detection_threshold)

        verification=detection/len(verification_folder) # Calculating Verification Percentage
        verified=verification>verification_threshold # Matching to the Threshold

        # Updating the Status Label
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # Log out details
        Logger.info(results)
        Logger.info(np.sum(np.array(results) > 0.5))

        return results, verified



# If you run faceid.py directly python automatically sets __name__ = '__main__'
# If another Python file imports faceid.py python sets __name__ = 'faceid' (not '__main__').
if __name__=='__main__':
    CamApp().run() # Runs the Kivy application.
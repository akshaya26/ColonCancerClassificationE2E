""""
This will remove specular reflections from image
"""
print("before torchvision")
import torchvision
print("after torchvision")
import os
import cv2
import numpy as np
print(7)
import torchvision.transforms as transforms
print(8)
from PIL import Image

def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((512,512)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.2627,0.2455))
    ])

def show_img(img,name):
    window_name=name
    cv2.imshow(window_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inpaint(image_bytes):
    # Open the image file
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR format
    if image is None:
        raise ValueError("Failed to decode image bytes")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to the HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # show_img(image_hsv,"HSV_image")
    # Split the HSV image into individual channels
    h, s, v = cv2.split(image_hsv)
    # show_img(h,"H")
    # show_img(s,"S")
    # show_img(v,"V")
    # Define the threshold for saturation
    saturation_threshold = 15 #intial 50

    # Create a mask for low saturated areas
    low_saturation_mask = s<saturation_threshold
    low_saturation_mask = low_saturation_mask.astype(np.uint8) * 255
    # print(low_saturation_mask.shape)
    # show_img(low_saturation_mask,"low_saturation_mask")
    # print(cv2.cvtColor(low_saturation_mask, None))

    # contours, _ = cv2.findContours(low_saturation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    low_saturation_mask2 = cv2.dilate(low_saturation_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)),iterations=2)
    # show_img(low_saturation_mask2,"dilate")
    # low_saturation_mask2 = cv2.erode(low_saturation_mask2,(5,5),iterations = 2)
    # show_img(low_saturation_mask2,"erode")

    # Perform inpainting to fill low saturation areas
    inpaint_image = cv2.inpaint(image, low_saturation_mask2,4, cv2.INPAINT_NS)
    # show_img(inpaint_image,"inpaint_image")

    # Convert the inpainted image array back to PIL image
    # inpaint_image_pil = Image.fromarray(cv2.cvtColor(inpaint_image, cv2.COLOR_BGR2RGB))

    # # Display the resulting image
    # show_image(inpaint_image_pil)
    return inpaint_image


def preprocess_image(image_bytes):
    image = inpaint(image_bytes) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    image_pil = Image.fromarray(image) #This is to be done because transform accepts PIL image
    transform = get_transform()
    return transform(image_pil).unsqueeze(0)




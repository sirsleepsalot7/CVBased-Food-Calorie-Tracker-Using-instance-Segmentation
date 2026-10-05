
from roboflow import Roboflow
rf = Roboflow(api_key="0y6UP6S5rcAQe0zZxQhM")
project = rf.workspace("wisdyoussef").project("fruits-xlypp")
version = project.version(1)
dataset = version.download("coco-segmentation")
                

# Downloads and unzips the dataset into a subfolder named 'food_dataset'
dataset = version.download("yolov8", location="food_dataset")
print("Dataset successfully downloaded to /food_dataset")
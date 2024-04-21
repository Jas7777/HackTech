# Import the necessary libraries
from PIL import Image
from matplotlib import pyplot as plt
from numpy import asarray
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
# load the image and convert into
# numpy array
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
cv_img = []
y = []
for img in glob.glob(r'C:\Users\Jas\desktop\test9\non-autistic\*.jpg'):
   y.append(0)
   img1 = Image.open(img)
   # Added code
   img1 = img1.resize((550, 600))
   reshaped_image=np.transpose(img1)

   # reshape image being weights are diffrent
   x1 = np.array(img1)
   x = x1.flatten()
   cv_img.append(x)


for img in glob.glob(r'C:\Users\Jas\desktop\test9\autistic\*.jpg'):
   y.append(1)
   img1 = Image.open(img)
   img1 = img1.resize((550, 600))
   reshaped_image=np.transpose(img1)


   x1 = np.array(img1)
   x = x1.flatten()
   cv_img.append(x)

X = cv_img




X_train, X_test, y_train, y_test = train_test_split(X, y, )
reg = LinearRegression( fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
reg1 = LinearRegression( fit_intercept=True, copy_X=True, n_jobs=None, positive=False)

# asarray() class is used to convert
# PIL images into NumPy arrays


predictions1=reg1.predict(X_test)
predictions=reg.predict(X_test)
for k in range(10):
   print(predictions1[k], y_test[k])
counter=0
for k in range(len(X_test)):
   if(predictions[k]==y_test[k]):
       counter=counter+1
print(counter/len(X_test))
w1=0.5
w2=0.5
for i in range(len(X_test)):
   pr=predictions[i]*w1+predictions1[i]*w2
   error=abs(predictions[i]-y_test[i])
   error1 = abs(predictions1[i] - y_test[i])
   w1=w1/2**error
   w2=w2/2**error1
   Sum=w1+w2
   w1=w1/Sum
   w2=w2/Sum

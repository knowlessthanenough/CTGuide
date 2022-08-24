## CT Guide neeedle insertion using imu and TCN

### Overview

 Overall purpose: To establish a system that can assist doctors to perform thoracentesis, and solve the current problem of relying on doctors' experience and intuition to perform surgery

 The project is divided into two parts. The first part is puncture guidance, and the second part is lung volume prediction.

 The purpose of the first part: Let the doctor know the current needle angle, the target angle, the required depth of puncture, and provide a graphical interface for the doctor to operate intuitively. This part is done by basic trigonometric function.
 
  The purpose of the second part: let the doctor know when to tell the patient to hold the breath and perform the puncture only when the lung volume is close to the picture. This part is done by TCN network


### First part

To achieve this goal, there is in total three things needed to know. 1. needle projection angle and 2.target angle in axial and Sagittal view, 3.target length of the needle.

1.By fixing a wireless IMU module in the needle allow us to know the 3D angle of the needle(euler angle). After transform, it become two projection angle in axial and Sagittal view. 

2.After receive the CT screen image (this is because in my case i can't do any change to the CT screen computer so image need to send to another computer). The target angle require operator(e.g. doctor) to click two point (this is a 3D point) on the axial view image (two point usually located in different image. Target length of the needle is the euclidean distance of two point. Target angle is the angle between base line and that line formed by two point. (i dont know how to explain clearer, i am sorry)


### Second part

To achieve this goal i need to know the lung volume of the patient in real time and do prediction(cause there is a reaction time for both doctor and patient). however if using tradtional method it is inconvenient and expensive.(cause the equipment is disposable). So i use a respiration belt to replace. To transform force data to volume and do prediction, i use a TCN network to do so. 

After wear the belt and clicking the save lung volume button(before sceen )when the sensor detect the force remain the same for a period of thime. that volume data transform by TCN with be save.(consider as the lung volume when taking CT) The belt will constandly send data and tansfom in to volume. when the volume after 2s is close(i am not sure how many %) to CT screen volume blue light will on and doctor can tell the patient to hold breath. if green light on doctor can do insertion.





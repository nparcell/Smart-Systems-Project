# Smart-Systems-Project

Goal of this project is to demonstrate the utility of controlling a chemical system using a neural net. 

The system used will be a simple tank draining system with a single outlet volumetric flow, and 3 input streams QA, QB, and QC. After the system has been created, PID process control will be implemented to control the tank height/level. There will be random disturbances in QB, and QC, and QA will be controlled to maintain proper level height in tank. h (height) and V (volume) will change in response to these disturbances.

By monitoring QA in response to changes in QB, QC, h, and V in the system, data can be sent to Workspace in a MATLAB Simulink simulation. In using this data, a neural net can be trained and tested in both Tensorflow and MATLAB neural net packages. The corresponding QA values will be the label for the data. 

After using a neural net to make test predictions, it was found that it is more favorable to have a training dataset that is more of a "ramp" style set point change with higher random variability. Then, when the neural net "sees" something new, such as a step set point change with some random variability, the neural net performs with higher predictability. An example of how it performs is shown in the image below:

![alt text](https://github.com/nparcell/Smart-Systems-Project/blob/master/Traindata%20-%20ramp2%20and%20testdata%20-%20train2.png)

The neural net predictions almost exactly match the process control simulation data. 

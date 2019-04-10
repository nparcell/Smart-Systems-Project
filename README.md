# Smart-Systems-Project

Goal of this project is to demonstrate the utility of controlling a chemical system using a neural net. 

The system used will be a simple tank draining system with a single outlet volumetric flow, and 3 input streams QA, QB, and QC. After the system has been created, PID process control will be implemented to control the tank height/level. There will be random disturbances in QB, and QC, and QA will be controlled to maintain proper level height in tank. h (height) and V (volume) will change in response to these disturbances.

By monitoring QA in response to changes in QB, QC, h, and V in the system, data can be sent to Workspace in a MATLAB Simulink simulation. In using this data, a neural net can be trained and tested in both Tensorflow and MATLAB neural net packages. The corresponding QA values will be the label for the data. 

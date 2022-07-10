# ASl-alphabet-translation
translate ASL alphabet from camera feed

## Capabilities
The ASL -alphabet-traslation is able to recognice users hand following the ASL alphabet and translate it into letters and print into word in the terminal.

## Usage
this is a tool to help people with disabilities communicate with people that do not understand American Sign Language(like me!). 

## How it works
The program uses a TensorFlow model trained through Teachable machine. 
*important* the model used can be significantly improved by running more epochs, collecting a bigger data set. There current model is trained with 100 images for each class
[a,b,bg(ie. background),c,d,delete,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z] under a white background/lighting. The data ran throught 500 epochs. If you want to improve the model, follow the Create Your Own Model section.
<p align="center">
  <img src="/model.savedmodel/community-teachable-machine-2.png" width="300">
</p>
## Set Up a jetson nano(a jetson dev kit is recommended to run the program)
1. With the jetson nano,  Flash microSD Card with the latest version of the JetPack software package as per instruction on this [NVIDIA Web site](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write);

2. Then deploy Jetson inference libraries, as explained on this [GitHub repo's page](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md);

3. And, finally, install TensorFlow framework on your Jetson Nano device as per instruction on this [NVIDIA Web site](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html).

## Run the program
1. use git clone https://github.com/EthanYX05/ASl-alphabet-translation.git to import the file onto the nano home directory.
2. cd into asl_image
3. run the comman: python3 asl_recog.py
4. Wait(this process could take up to minutes depending of the memory) until you see ["init", any letter] in the terminal, which then the program is running. 

## Create your own model
1. open [Teachable Machine](https://teachablemachine.withgoogle.com/) Web site and click "Get Started" button
2. select image project
3. create classes following the order provided in the model.savedmodel/labels.txt file provided. (note that the order is different).
4. click train model. 
5. export the model as Tensorflow .pb file
6. scp the model.savedmodel folder you created to the jetson nano
7. run the asl_recog.py script.

## Dependency
Tensorflow
python3 


## Reference
Template of the python code is based off of https://github.com/LazaUK/NvidiaJetsonNano-TeachableMachine credit to LazaUK, with changes to how output display(no display needed), how model is loaded, and organized output.
Same as LazaUK I used Feng Wang's function for RGBA-to-RGB conversion, and customised Microsoft's demo script to setup connectivity with Azure IoT Hub.

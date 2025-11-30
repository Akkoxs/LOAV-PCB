Project 3 for AER850 - YOLOv11 for PCB component placement and verification

LOAV-PCB = Look Once And Verify - Printed Circuit Board 

We will be identifying the following PCB SMD Components:
0 'Button' 
1 'Capacitor'
2 'Connector'
3 'Diode'  
4 'Electrolytic Capacitor'
5 'IC'
6 'Inductor'
7 'Led'
8 'Pads'
9 'Pins'
10'Resistor'
11'Switch'
12'Transistor'

DATA DIRECTORY

Project 3 Data/
    Project 3 Data/
            data/
                evaluation/
 		test/
		    images/
		    labels/
                train/
		    images/
		    labels/
		    labels.cache
                valid/
		    images/
		    labels/
		    labels.cache
		data.yaml
            prediction imgs/
	    .DS_Store
	    motherboard_image.JPEG

MODELS TRAINED
v0 - epochs = 5, batch = 8, imgsz = 900
v1 - epochs = 200, batch = 8, imgsz = 900
v2 - epochs = 200, batch = 2, imgsz = 
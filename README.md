Setup

    Create the virtual environment (must be done only once)

python3 -m venv venv

source venv/bin/activate (linux) venv\Scripts\activate (windows)

pip3 install -r requirements.txt 

    If the virtual environment has been made, skip "python3 -m venv venv" source venv/bin/activate


test using tensor flow
- trainTensorflow.py : trainiere model von dataset

    usage python3 trainTensorFlow.py -dataset_url chimie-dataset.tgz
    the tgz file MUST have a DATASET subdirectory
- testModel.py : testet ein model mit einem Bild

    usage python3 testModel.py -image Becherglass-250ml-neu-25.jpg --tensorflow -model chimie-transparent.keras

- CaptureImg.py : nimmt ein snapshot von der kamara
- evaluate_model.py : evaluiert ein Model : schaut ob die test bilder in einer xml datei die entsprechenden Ergebnisse zeigen

    usage python3 evaluate_model.py  -model chimie-conny.keras  --canny

dataset management
- contours.py : use opencv canny to produce a dataset only with surroundings
    usage : python3 contours.py -dir "/home/christian/git/ksh-ki/DATASET-bg neu" -output /home/christian/tmp/dataset
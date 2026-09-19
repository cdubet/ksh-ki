test using tensor flow
- trainTensorflow.py : trainiere model von dataset
- testModel.py : testet ein model mit einem Bild
- CaptureImg.py : nimmt ein snapshot von der kamara
- evaluate_model.py : evaluiert ein Model : schaut ob die test bilder in einer xml datei die entsprechenden Ergebnisse zeigen

dataset management
- contours.py : use opencv canny to produce a dataset only with surroundings
    usage : -dir "/home/christian/git/ksh-ki/DATASET-bg neu" -output /home/christian/tmp/dataset
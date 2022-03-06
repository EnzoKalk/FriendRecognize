# FriendRecognize

## Preparing datasets

All images must be splitted in "data/trainingSet"
so that for each friend there are a couple of folders:

```
|-- data
|   |-- trainingSet
|       |-- Vincenzo
|       |-- Angelo
|       |-- Dima
|       |-- Giovanna
|       |-- Noemi
|       |-- noVincenzo
|       |-- noAngelo
|       |-- noDima
|       |-- noGiovanna
|       |-- noNoemi
```

P.S. 
The class '1_ExtractAndAlignFaces.py' it is usefull to
extract faces from "data/detectedFaces/source/"
in "data/detectedFaces/extracted_and_aligned_faces/".
Anyway, each face extracted must be moved into
the specific folder in "data/trainingSet"

## Training phase

Before training, make sure that the 
following value are set in 'config.yml':

- friends -> with/without
- libs -> face_cascade, predictor
- data -> training
- models

To start the training:

```
python 2_Main.py
```

After training, the resulting models are saved in:

```
|-- models
|   |-- Vincenzo
|   |   |-- predictor.pkl
|   |-- Angelo
|   |   |-- predictor.pkl
|   |-- Dima
|   |   |-- predictor.pkl
|   |-- Giovanna
|   |   |-- predictor.pkl
|   |-- Noemi
|   |   |-- predictor.pkl
```

## Recognize phase

Before test, make sure that the 
following value are set in 'config.yml':

- friends -> with/without
- libs -> face_cascade, predictor
- models

To start the recognition:

```
python 3_CameraRecognizer.py
```

## License+
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International License. To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
PO Box 1866, Mountain View, CA 94042, USA.

# FL-FD: Federated learning-based fall detection with multimodal data fusion
This is a Pytorch implementation of FL-FD in the following paper:

Qi, Pian, Diletta Chiaro, and Francesco Piccialli, FL-FD: Federated learning-based fall detection with multimodal data fusion, Information Fusion (2023). [Paper](https://www.sciencedirect.com/science/article/pii/S1566253523002063/)

# Requirements
python>=3.6

pytorch>=0.4

# Note
Download FALL-UP dataset.

A)
download sensor data from Google drive [here](https://drive.google.com/file/d/bc1qk55vk7wjgzg3pmxlh59rv5dlgewd9jem5nrt4w/view) 
then place the file into the folder 'dataset'.

B)
download camera data follow the tutorial [here](https://github.com/jpnm561/HAR-UP/tree/master/DataBaseDownload) 
then place the file into the folder 'dataset'.

camera data like follow:

ParentFolder\
             \Subject#\
                      \Activity#\
                               \Trial#\
                                      \downloadedFile(1)
                                      ...
                                      \donwnloadedFile(i)

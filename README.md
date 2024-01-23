# FL-FD: Federated learning-based fall detection with multimodal data fusion
This is a Pytorch implementation of FL-FD in the following paper:

Pian Qi, Diletta Chiaro, and Francesco Piccialli, FL-FD: Federated learning-based fall detection with multimodal data fusion, Information Fusion (2023). [Paper](https://www.sciencedirect.com/science/article/pii/S1566253523002063/)

# Abstract
Multimodal data fusion is a critical element of fall detection systems, as it provides more comprehensive information than single-modal data. Yet, data heterogeneity between sources has posed a challenge for the effective fusion of such data. This paper proposes a novel multimodal data fusion method under a federated learning (FL) framework that addresses the privacy concerns of users while exploiting the complementarity of such data. Specifically, we fuse time-series data from wearable sensors and visual data from cameras at the input level, where the data is first transformed into images using the Gramian Angular Field (GAF) method. Moreover, each user is treated as a private client in the FL system whereby the fall detection model is trained without requiring the sharing of user data. The proposed method is evaluated using the UP-Fall dataset, where we perform different fall detection tasks: binary classification for fall and non-fall detection yields a remarkable accuracy of 99.927%, while multi-classification for different fall activity recognition attains an accurate result of 89.769%.


# Requirements
python>=3.6

pytorch>=0.4

# Note
Follow the tutorial [here](https://github.com/jpnm561/HAR-UP/tree/master/DataBaseDownload/) to download the FALL-UP dataset and place the files into the "dataset" folder.

## Acknowledgments
This work was supported by the following projects: 

 - PNRR project FAIR -  Future AI Research (PE00000013), Spoke 3, under the NRRP MUR program funded by the NextGenerationEU.
 - PNRR Centro Nazionale HPC, Big Data e Quantum Computing, (CN\_00000013)(CUP: E63C22000980007), under the NRRP MUR program funded by the NextGenerationEU.

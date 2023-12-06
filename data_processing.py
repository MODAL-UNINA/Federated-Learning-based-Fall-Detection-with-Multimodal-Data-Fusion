import os
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField
from PIL import Image
import imageio
import pickle


def re_value(arr):
    rzero = np.min(arr)
    arr = arr + np.abs(rzero)
    r255 = np.amax(arr)
    if r255 != 0:
        fcon = 255/r255
        arr = arr*fcon
        arr = arr.unsqueeze(0)
    return arr


# path
sensor_path = 'FL-FD/dataset'
camera_path = 'FL-FD/dataset/camera'


### processing sensor data
sensor_file = pd.read_csv(os.path.join(sensor_path, 'CompleteDataSet.csv'), skiprows=2, header=None)

# keep columns in csv file, drop the rest. since we only need accelerometer and angular velocity data.
keep_columns = [1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,22,23,24,25,26,27,29,30,31,32,33,34,43,44,45]

time = pd.to_datetime(sensor_file.iloc[:,0], format='%Y-%m-%dT%H:%M:%S.%f')
values = np.array(sensor_file.iloc[:,keep_columns])

df = pd.DataFrame(values, index=time, 
                  columns=['AnkleAccelerometer_x', 'AnkleAccelerometer_y', 'AnkleAccelerometer_z', 
                        'AnkleAngularVelocity_x', 'AnkleAngularVelocity_y', 'AnkleAngularVelocity_z', 
                        'RightPocketAccelerometer_x', 'RightPocketAccelerometer_y', 'RightPocketAccelerometer_z',
                        'RightPocketAngularVelocity_x', 'RightPocketAngularVelocity_y', 'RightPocketAngularVelocity_z', 
                        'BeltAccelerometer_x', 'BeltAccelerometer_y', 'BeltAccelerometer_z', 
                        'BeltAngularVelocity_x', 'BeltAngularVelocity_y', 'BeltAngularVelocity_z', 
                        'NeckAccelerometer_x', 'NeckAccelerometer_y', 'NeckAccelerometer_z', 
                        'NeckAngularVelocity_x', 'NeckAngularVelocity_y', 'NeckAngularVelocity_z', 
                        'WristAccelerometer_x', 'WristAccelerometer_y', 'WristAccelerometer_z', 
                        'WristAngularVelocity_x', 'WristAngularVelocity_y', 'WristAngularVelocity_z', 
                        'Subject', 
                        'Activity', 
                        'Trial'])


# drop subject 5, 9, since they have no camera data
df = df.drop(df[df.Subject == 5].index)
df = df.drop(df[df.Subject == 9].index)

# for each subject, group by 'Activity', for each activity, group by 'Trial'
S_A_T = df.groupby(['Subject', 'Activity', 'Trial'])

# create a dictionary to save GAF of each sensor
GAF_data = {}

for key, value in S_A_T:

    value = value.iloc[0:140, :]

    index_each = value.index
    gaf = GramianAngularField(method='difference')  # GAF method

    # Ankle
    AnkleAccelerometer = np.sqrt(value['AnkleAccelerometer_x']**2 + value['AnkleAccelerometer_y']**2 + value['AnkleAccelerometer_z']**2)
    gaf_AnkleAccelerometer = gaf.fit_transform(np.array(AnkleAccelerometer).reshape(1, -1))
                                               
    AnkleAngularVelocity = np.sqrt(value['AnkleAngularVelocity_x']**2 + value['AnkleAngularVelocity_y']**2 + value['AnkleAngularVelocity_z']**2)
    gaf_AnkleAngularVelocity = gaf.fit_transform(np.array(AnkleAngularVelocity).reshape(1, -1))

    gaf_Ankle = np.dstack((gaf_AnkleAccelerometer[0], gaf_AnkleAngularVelocity[0]))     # stack two GAFs

    # RightPocket
    RightPocketAccelerometer = np.sqrt(value['RightPocketAccelerometer_x']**2 + value['RightPocketAccelerometer_y']**2 + value['RightPocketAccelerometer_z']**2)
    gaf_RightPocketAccelerometer = gaf.fit_transform(np.array(RightPocketAccelerometer).reshape(1, -1))
    
    RightPocketAngularVelocity = np.sqrt(value['RightPocketAngularVelocity_x']**2 + value['RightPocketAngularVelocity_y']**2 + value['RightPocketAngularVelocity_z']**2)
    gaf_RightPocketAngularVelocity = gaf.fit_transform(np.array(RightPocketAngularVelocity).reshape(1, -1))

    gaf_RightPocket = np.dstack((gaf_RightPocketAccelerometer[0], gaf_RightPocketAngularVelocity[0]))   # stack two GAFs

    # Belt
    BeltAccelerometer = np.sqrt(value['BeltAccelerometer_x']**2 + value['BeltAccelerometer_y']**2 + value['BeltAccelerometer_z']**2)
    gaf_BeltAccelerometer = gaf.fit_transform(np.array(BeltAccelerometer).reshape(1, -1))

    BeltAngularVelocity = np.sqrt(value['BeltAngularVelocity_x']**2 + value['BeltAngularVelocity_y']**2 + value['BeltAngularVelocity_z']**2)
    gaf_BeltAngularVelocity = gaf.fit_transform(np.array(BeltAngularVelocity).reshape(1, -1))

    gaf_Belt = np.dstack((gaf_BeltAccelerometer[0], gaf_BeltAngularVelocity[0]))  # stack two GAFs

    # Neck
    NeckAccelerometer = np.sqrt(value['NeckAccelerometer_x']**2 + value['NeckAccelerometer_y']**2 + value['NeckAccelerometer_z']**2)
    gaf_NeckAccelerometer = gaf.fit_transform(np.array(NeckAccelerometer).reshape(1, -1))
    
    NeckAngularVelocity = np.sqrt(value['NeckAngularVelocity_x']**2 + value['NeckAngularVelocity_y']**2 + value['NeckAngularVelocity_z']**2)
    gaf_NeckAngularVelocity = gaf.fit_transform(np.array(NeckAngularVelocity).reshape(1, -1))

    gaf_Neck = np.dstack((gaf_NeckAccelerometer[0], gaf_NeckAngularVelocity[0]))  # stack two GAFs

    # Wrist
    WristAccelerometer = np.sqrt(value['WristAccelerometer_x']**2 + value['WristAccelerometer_y']**2 + value['WristAccelerometer_z']**2)
    gaf_WristAccelerometer = gaf.fit_transform(np.array(WristAccelerometer).reshape(1, -1))
    
    WristAngularVelocity = np.sqrt(value['WristAngularVelocity_x']**2 + value['WristAngularVelocity_y']**2 + value['WristAngularVelocity_z']**2)
    gaf_WristAngularVelocity = gaf.fit_transform(np.array(WristAngularVelocity).reshape(1, -1))

    gaf_Wrist = np.dstack((gaf_WristAccelerometer[0], gaf_WristAngularVelocity[0]))  # stack two GAFs

    # save GAF of each sensor
    each_gaf = {'GAF_Ankle': gaf_Ankle,
                'GAF_RightPocket': gaf_RightPocket,
                'GAF_Belt': gaf_Belt,
                'GAF_Neck': gaf_Neck,
                'GAF_Wrist': gaf_Wrist}
    
    GAF_data[key] = each_gaf    # sunch as key = (1, 1, 1) means subject 1, activity 1, trial 1


### processing camera data
camera_dir = os.path.join(camera_path)
camera_dir_list = os.listdir(camera_dir)
camera_dir_list.sort()
camera_filenames = [camera_dir + "/" + filename for filename in camera_dir_list]

# create a dictionary to save camera data
Camera_data = {}

for i in range(len(camera_filenames)):
    key = (camera_filenames[i].split("/")[6])[:-7]
    all_gray_img = []
    delt_list = []
    
    photos_dir_list = os.listdir(camera_filenames[i])
    photos_dir_list.sort()
    photos_filenames = [camera_filenames[i] + "/" + photo for photo in photos_dir_list]

    # resize all images to 140*140, convert to gray image
    for j in range(140):
        temp_rgb_arr = imageio.imread(photos_filenames[j])  # rgb array
        
        temp_rgb = Image.fromarray(temp_rgb_arr).convert('RGB')   # rgb image
        
        tem_gray = temp_rgb.resize((140,140)).convert('L')    # gray image
        
        all_gray_img.append(tem_gray)

    # calculate the difference between two adjacent gray images
    for k in range(len(all_gray_img) - 1):
        delt = np.abs(np.array(all_gray_img[k + 1]).astype(np.int) - np.array(all_gray_img[k]).astype(np.int))
        delt_list.append(delt)
    
    delt_gray_arr = sum(delt_list)   # average gray image
    arr = re_value(delt_gray_arr)
    
    Camera_data[key] = arr


# combine GAF and Camera data, using the same key
combined_keys = list(GAF_data.keys())
combined_keys.sort()

GAF_Camera_data = {}

for i in range(len(combined_keys)):

    key = combined_keys[i]

    Ankle = np.concatenate(GAF_data[key]['GAF_Ankle'].transpose(2,0,1), Camera_data[key], axis=0)
    RightPocket = np.concatenate(GAF_data[key]['GAF_RightPocket'].transpose(2,0,1), Camera_data[key], axis=0)
    Belt = np.concatenate(GAF_data[key]['GAF_Belt'].transpose(2,0,1), Camera_data[key], axis=0)
    Neck = np.concatenate(GAF_data[key]['GAF_Neck'].transpose(2,0,1), Camera_data[key], axis=0)
    Wrist = np.concatenate(GAF_data[key]['GAF_Wrist'].transpose(2,0,1), Camera_data[key], axis=0)

    l = [Ankle, RightPocket, Belt, Neck, Wrist]

    GAF_Camera_data[key] = l


# split data into train and test set
Train_data = {}
Test_data = {}

for key, value in GAF_Camera_data.items():
    # create label, each activity has a label
    if key[1] == '1':
        label = 0
    elif key[1] == '2':
        label = 1
    elif key[1] == '3':
        label = 2
    elif key[1] == '4':
        label = 3
    elif key[1] == '5':
        label = 4
    elif key[1] == '6':
        label = 5
    elif key[1] == '7':
        label = 6
    elif key[1] == '8':
        label = 7
    elif key[1] == '9':
        label = 8
    elif key[1] == '10':
        label = 9
    elif key[1] == '11':
        label = 10
    else:
        continue

    value.append(label)

    # trial 1 and 2 for train, trial 3 for test
    if key[-1] == '1' or key[-1] == '2':

        Train_data[key] = value

    elif key[-1] == '3':
        Test_data[key] = value


# save dataset as pickle file
with open('Train.pkl', 'wb') as f:
    pickle.dump(Train_data, f)

with open('Test.pkl', 'wb') as f:
    pickle.dump(Test_data, f)

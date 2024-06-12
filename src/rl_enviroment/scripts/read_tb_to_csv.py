from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm
from select_names import select_name

import os
def tb_to_csv(tb_path,i):
    # load log data
    # names = select_name(tb_path)
    # if names['use_pid']:
    #     out_path1 = 'use_pid_'+ names['leader_mode'] +'_' +'speed_'+names['speed']+'.csv'
    # else:
    #     if names['no_lstm']:
    #         out_path1 = 'no_lstm_'+ names['leader_mode'] +'_' + 'speed_' + names['speed']+'.csv'
    #     else:
    #         out_path1 = 'lstm_' + names['leader_mode'] +'_' + 'speed_'+ names['speed']+'.csv'
    #     out_path = os.path.join('many_csv_file3',out_path1)

    event_data = event_accumulator.EventAccumulator(tb_path)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    # print(keys)
    df = pd.DataFrame(columns=keys)  # my first column is training loss per iteration, so I abandon it
    for key in tqdm(keys):
        
        df[key] = pd.DataFrame(event_data.Scalars(key)).value
    
    csv_floder = 'random_target'
    if not os.path.exists(csv_floder):
                        os.makedirs(csv_floder)
    df.to_csv(os.path.join(csv_floder,'{}.csv'.format(i)))

    print("u_shape")


# tb_to_csv('test_p_finetune_room_chan/use_pid_False_noise_False_tn_0.0_pn_0.01_speed_0.3reward_249.82984237675012_zhen_r_-1.7015762324986032/events.out.tfevents.1716464822.lmf-bjut')

def list_files_in_directory(directory):
    files_list = []
    excluded_folder = 'leader_follower_trajectory'
    for root, dirs, files in os.walk(directory):
        if excluded_folder in root.split(os.sep):
            continue
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(file_path)
    return files_list

directory_path = 'test_random_target_finetune_room_chan' 
files = list_files_in_directory(directory_path)
for k,file in enumerate(files):
    tb_to_csv(file,k)
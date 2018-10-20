# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------------------
#-----data: 2018.05.19
#-----function: get the Valence-Arousal values of each frame in each subject using the map of emotion to VA
#-----input: emotion file
#-----output: VA file with timestamp of each subject
#-----------------------------------------------------------------------------------------------------

import numpy as np
import xlrd
import xlwt
import os

path_code = ''
emotion_path = path_code + '../data/emotion/'
VA_path = path_code + '../features/VA/'
time_csv = path_code + '../data/emotion/'
save_path = path_code + '../features/VA_withtimestamp/'

#the map of emotion to Valence-Arousal(VA)
emotion = np.array(
    [ #   V,   A
        [3.5, 3.0], # 'sadness'
        [7.0, 3.0], # 'neutral'
        [4.0, 6.5], # 'disgust'
        [3.5, 7.0], # 'anger'
        [6.0, 6.5], # 'surprise'
        [2.0, 6.0], # 'fear'
        [7.5, 7.5], # 'happiness'
    ])
#the map of emtoion to VA according to the probability of emotion
def map_to_VA(emotion_prob):
    valence = np.sum(emotion[:,0] * emotion_prob) / 100
    arousal = np.sum(emotion[:,1] * emotion_prob) / 100
    return np.array([valence, arousal])

def map_to_VA_all(emotion_prob_array):
    VA_list = list(map(map_to_VA, emotion_prob_array))
    return np.array(VA_list)

#read the emtoion probability file
def read_emo_prob_array(path):
    workbook = xlrd.open_workbook(path)
    sheets = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheets[0])
    emo_prob_array = []
    for i in range(1, worksheet.nrows):
        row = worksheet.row(i)
        row_data = []
        for j in range(2, worksheet.ncols-1):
            row_data.append(worksheet.cell_value(i, j))
        emo_prob_array.append(row_data)
    return np.array(emo_prob_array)

#save the VA values to Excel file
def write_VA_to_xls(VA_list, path):
    wb = xlwt.Workbook()
    sheet = wb.add_sheet('VA')
    sheet.write(0, 0, 'valence')
    sheet.write(0, 1, 'arousal')
    for i in range(VA_list.shape[0]):
        for j in range(VA_list.shape[1]):
            sheet.write(i+1, j, VA_list[i,j])
    wb.save(path)

def get_file_list(path):
    file_list = os.listdir(path)
    return file_list


#add the timestamp to the VA file
def batch_process(file_path_VA, file_path_time):
    VA_path_list = []
    for file_path_VA, sub_dirs_VA, files_VA in os.walk(file_path_VA):
        files_VA = sorted(files_VA)
        print (file_path_VA, sub_dirs_VA, files_VA)

    for file_name_VA in files_VA:
        VA_path_list.append(file_path_VA + file_name_VA)

    time_path_list = []
    for file_path_time, sub_dirs_time, files_time in os.walk(file_path_time):
        files_time = sorted(files_time)
        print (file_path_time, sub_dirs_time, files_time)

    for file_name_time in files_time:
        time_path_list.append(file_path_time + file_name_time)


    for i in range(len(VA_path_list)):
        VA_path_list_temp = xlrd.open_workbook(VA_path_list[i])
        VA = VA_path_list_temp.sheet_by_index(0)
        time_path_list_temp = xlrd.open_workbook(time_path_list[i])
        time = time_path_list_temp.sheet_by_index(0)

        wb = xlwt.Workbook()
        sheet = wb.add_sheet('VA')
        sheet.write(0, 0, 'time_stamp')
        sheet.write(0, 1, 'valence')
        sheet.write(0, 2, 'arousal')

        row_number = VA.nrows
        for j in range(1, row_number):
            sheet.write(j, 0, time.cell(j, 1).value)
            sheet.write(j, 1, VA.cell(j, 0).value)
            sheet.write(j, 2, VA.cell(j, 1).value)
        excel_path = save_path + str((time.cell(1, 0).value)) + '.xls'
        wb.save(excel_path)

#main function
if __name__ == '__main__':
    read_dir_path = emotion_path
    write_dir_path = VA_path
    file_list = get_file_list(read_dir_path)
    for i in range(len(file_list)):
        read_file_path = read_dir_path+file_list[i]
        file_num = file_list[i].split('.')[0]
        write_file_path = write_dir_path + file_num + '_VA.xls'

        emo_prob_array = read_emo_prob_array(read_file_path)
        VA_list = map_to_VA_all(emo_prob_array)
        write_VA_to_xls(VA_list, write_file_path)
    batch_process(VA_path, time_csv)


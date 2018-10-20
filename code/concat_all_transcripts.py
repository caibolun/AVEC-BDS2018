#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
concatenate all transcripts into a DataFrame file
"""
import pandas as pd
import os
import re

FILE_PATH = '../data/translatation_check'

topic_map = {1:'why_come', 2:'man_pic',
             3:'worst_memory', 4:'count1',
             5:'count2', 6:'family_pic',
             7:'best_memory'}

files_path = os.listdir(FILE_PATH + os.sep + 'train')
files_path.extend(os.listdir(FILE_PATH + os.sep + 'test'))
files_path.extend(os.listdir(FILE_PATH + os.sep + 'dev'))
files_path = [a for a in files_path if not (a.startswith('~') or a.startswith('.'))]


feature_dict_list = []

for file_path in files_path:
        
    test_file = pd.read_excel(FILE_PATH + os.sep + 
                              file_path.split('_')[0] +  os.sep + file_path)
 
    
    feature_dict = {}
    feature_dict = {'Instance_name':file_path.split('.')[0]}
    print(feature_dict['Instance_name'] + ' is being processed')
    
    topic_column = test_file['topic']
    
    # aggregate all transcripts for target person
    all_transripts = [re.sub('\xa0','',a) for a in test_file['transcript'].values]
    all_transripts = '.'.join(all_transripts)
    
    # count the number of words of all transcripts for target person
    num_words = len(re.split('[ .]',all_transripts))
    
    feature_dict['all_transripts'] = all_transripts
    feature_dict['all_transripts_num_words'] = num_words
    
    # save all transripts as txt for later usage
    f = open('transcript_txt'+ os.sep +'all_transcript_txt' + 
             os.sep + file_path.split('.')[0] + '.txt', 'w')
    f.write(all_transripts)
    f.close()
    
    for i in range(1,8):
        is_cur_topic = topic_column == i
        if sum(is_cur_topic) > 0:
            temp_pd = test_file[is_cur_topic]
            
            start_time_column = temp_pd['start_time'].values
            end_time_column = temp_pd['end_time'].values
            
            start_time_str =  start_time_column[0]
            min_sec = start_time_str.split(':')
            start_time = int(min_sec[0]) * 60 + int(min_sec[-1])
            
            end_time_str =  end_time_column[-1]
            min_sec = end_time_str.split(':')
            end_time = int(min_sec[0]) * 60 + int(min_sec[-1])
            
            transcript_column = temp_pd['transcript'].values
            transcript_column = [re.sub('\xa0','',a) for a in transcript_column]
            transcript = '.'.join(transcript_column)
            
            feature_dict[topic_map[i] + '_exist'] = 1
            feature_dict[topic_map[i] + '_starttime(s)'] = start_time
            feature_dict[topic_map[i] + '_endtime(s)'] = end_time
            feature_dict[topic_map[i] + '_transcript'] = transcript
            
            # count the number of words of target topic
            num_words = len(re.split('[ .]',transcript))
            feature_dict[topic_map[i] + '_num_words'] = num_words
            
        else:
            feature_dict[topic_map[i] + '_exist'] = 0
            feature_dict[topic_map[i] + '_starttime(s)'] = ''
            feature_dict[topic_map[i] + '_endtime(s)'] = ''
            feature_dict[topic_map[i] + '_transcript'] = ''
            feature_dict[topic_map[i] + '_num_words'] = 0
            
     # save topic transripts as txt for later usage
        f = open('transcript_txt'+ os.sep + topic_map[i] + 
                 '_txt' + os.sep + file_path.split('.')[0] + '.txt', 'w')
        f.write(feature_dict[topic_map[i] + '_transcript'])
        f.close()
                       
    feature_dict_list.append(feature_dict)
    print(feature_dict['Instance_name'] + ' is processed successfully')
    
table = pd.DataFrame.from_dict(feature_dict_list)
table = table[['Instance_name', 'all_transripts', 'all_transripts_num_words', 'why_come_exist','why_come_starttime(s)',
              'why_come_endtime(s)','why_come_transcript', 'why_come_num_words',
              'man_pic_exist', 'man_pic_starttime(s)', 'man_pic_endtime(s)', 
              'man_pic_transcript','man_pic_num_words','worst_memory_exist', 'worst_memory_starttime(s)', 
              'worst_memory_endtime(s)','worst_memory_transcript','worst_memory_num_words','count1_exist',
              'count1_starttime(s)','count1_endtime(s)','count1_transcript', 'count2_exist',
              'count2_starttime(s)','count2_endtime(s)','count2_transcript',
              'family_pic_exist', 'family_pic_starttime(s)', 'family_pic_endtime(s)',
              'family_pic_transcript','family_pic_num_words', 'best_memory_exist', 'best_memory_starttime(s)',
              'best_memory_endtime(s)', 'best_memory_transcript','best_memory_num_words'
              ]]

table = table.sort_values(by=['Instance_name'])
table.to_csv('../data/7topics_data.csv',index=False)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

extract a binary feature of whether the patients admitted he had mental disorder.
1: The patient admitted he had mental disorder
0: The patient refused to admit he had mental disorder

"""

import pandas as pd
import re


topics_data = pd.read_csv('../data/7topics_data.csv')

why_come_transrcipts = topics_data['why_come_transcript']

admit_sick_regex = 'psychology problem|heal|treatment|attack|treated|therapy|check|examined|program|health|bipolar|recover|disorder|manic|relax|uncomfortable|test|survey|questionnaire|medicine|medication|sick|illness|diease|sickness'
refuse_admit_regex = 'don’t know|do not know|not[^.]*my[^.]*will|do not remember|not have a disease|had no problem|took.*by force'

admit_sick_list = []
for one_transcript in why_come_transrcipts:
    if type(one_transcript) != str: 
        admit_sick_list.append(0)
        continue
    
    if len(re.findall(refuse_admit_regex, one_transcript)) != 0:
        admit_sick_list.append(0)
        continue
        
    if len(re.findall(admit_sick_regex, one_transcript)) != 0:
        admit_sick_list.append(1)
    else:
        admit_sick_list.append(0)
        
        

admit_sick_feature_pd = pd.DataFrame({'Intance_name':topics_data['Instance_name'],
                                   'why_come_transcript':topics_data['why_come_transcript'],
                                   'admit_sick_feature_auto':admit_sick_list})
#admit_sick_feature_pd = admit_sick_feature_pd[['Intance_name','why_come_transcript',
#                                              'admit_sick_feature_auto']]

admit_sick_feature_pd = 1 - admit_sick_feature_pd[['admit_sick_feature_auto']]

admit_sick_feature_pd.to_csv('../features/semantic_features.csv',index=None, header=None)

#coding=utf-8
#-----------------------------------------------------------------------------------------------------
#-----data: 2018.05.14
#-----function: get the emotion of each frame in each subject using face++ toolkit
#-----input: all video
#-----output: emtoion file of each subject
#-----------------------------------------------------------------------------------------------------
import numpy as np
import cv2 as cv
import sys,os
import time
import xlwt
import json
from PIL import Image
import requests
from json import JSONDecoder
import argparse
#import urllib2


parser = argparse.ArgumentParser(description='Etract the emotion feature by Face++')
parser.add_argument('-k', '--key', type=str, default='', help="Key for Face++")
parser.add_argument('-s', '--secret', type=str, default='', help="secret for Face++")
args = parser.parse_args()
#API parameters
key = args.key
secret = args.secret

#import face++ API
def use_API(img):
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    # boundary = '-------------------%s' %hex(int(time.time() * 1000))
    # data = []
    # data.append('--%s' % boundary)
    # data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    # data.append(key)
    # data.append('--%s' % boundary)
    # data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    # data.append(secret)
    # data.append('--%s' % boundary)
    # fr = open(img, 'rb')
    # data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    # data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    # data.append(fr.read())
    # fr.close()
    # data.append('1')
    # data.append('--%s' % boundary)
    # data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    # data.append("gender,age,emotion,ethnicity")
    # data.append('--%s--\r\n' % boundary)

   #  http_body = '\r\n'.join(data)
   #  request = urllib2.Request(http_url)  #http request
   #  request.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
   #  request.add_data(http_body)
    data = {"api_key":key, "api_secret": secret , "return_attributes":"gender,age,emotion,ethnicity"}
    files = {"image_file": open(img,"rb")}
    response = requests.post(http_url, data = data, files= files)
    req_con = response.content.decode('utf-8')
    req_dict = JSONDecoder().decode(req_con)

    # try:
    #     resp = urllib2.urlopen(request, timeout=5)
    #     global qrcont
    #     qrcont = json.load(resp)
    #     print qrcont
    # except urllib2.HTTPError as e:
    #     print e.read()

    return req_dict


#get time stamp from video
def get_time_stamp(path):
    global time_last, time_difference
    time_present = time.clock()
    time_temp = time_present - time_last
    time_difference =  time_difference + time_temp
    time_last = time_present
    #return day_stamp, time_stamp, hour_stamp
    return time_difference

#get the size of video frame
def image_size(img):
    Img = Image.open(img)
    w,h = Img.size
    return w,h

#using the face++ API, get the emotion of each frame and save as Excel file
def Write_EXCEL(img, worksheet, row, files_name, time_stamp_fromvideo):
    parsed = use_API(img)
    if not parsed['faces']:
        print('This picture do not have any face!')
    else:
        print('attribute!')
        for list_item in parsed['faces']:
            filename, extension = os.path.splitext(files_name)  #prpare the excel file
            worksheet.write(row, 0, filename)
            worksheet.write(row, 1, label=time_stamp_fromvideo)

            emotion = []
            for key1, value1 in list_item.items():
                if key1 == 'attributes':
                    for key2, value2 in value1.items():
                        if key2 == 'emotion':
                            for key3, value3 in value2.items():
                                if key3 == 'sadness':
                                    worksheet.write(row, 2, value3)
                                    emotion.append(value3)
                                elif key3 == 'neutral':
                                    worksheet.write(row, 3, value3)
                                    emotion.append(value3)
                                elif key3 == 'disgust':
                                    worksheet.write(row, 4, value3)
                                    emotion.append(value3)
                                elif key3 == 'anger':
                                    worksheet.write(row, 5, value3)
                                    emotion.append(value3)
                                elif key3 == 'surprise':
                                    worksheet.write(row, 6, value3)
                                    emotion.append(value3)
                                elif key3 == 'fear':
                                    worksheet.write(row, 7, value3)
                                    emotion.append(value3)
                                else:   #happiness
                                    worksheet.write(row, 8, value3)
                                    emotion.append(value3)
                        else:
                            pass
                # elif key1 == 'face_token':
                #     worksheet.write(row, 10, value1)
                else:
                    pass
            worksheet.write(row, 9, emotion.index(max(emotion)))  #0-neutral,1-sadness,2-disgust，3-anger，4-surprise，5-fear，6-happiness

            row +=1
            print('success, the pic' + str(files_name) + 'was detected!')
    return row, worksheet




timeF = 10
path_video = '../data/video/'    #file_path
#path_video = 'F:\\avec\\BDS\\test\\video'
#path_picture = r'temp/picture'
path_save = '../data/emotion/'
#path_save = 'F:\\avec\\BDS\\test\\feature\\'
path_temp = '../temp/temp_picture.jpg'
#create excel



#go through all video including the dev/test/train video, and get the emotion file for each video for the convenience of getting emotion features
for root, dirs, files_video in os.walk(path_video):
    print(files_video)
    for files_name in files_video:
        print(files_name)
        count = 1
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet(files_name, cell_overwrite_ok=True)
        title = ['video_ID', 'time_stamp',  'sadness', 'neutral', 'digust', 'anger', 'surprise', 'fear', 'happiness', 'emotion']
        for col in range(len(title)):
            worksheet.write(0, col, title[col])
        row = 1

        video_address = path_video + files_name
        video = cv.VideoCapture( video_address )
        if (video.isOpened() == False):
            print("error opening video stream or file!")
        while (video.isOpened()):
            ret, frame = video.read()
            time_stamp_fromvideo = video.get( cv.CAP_PROP_POS_MSEC )
            if(count % timeF == 0):
                cv.imwrite(path_temp, frame)
                count = count + 1
            else:
                count = count + 1
                continue

            if ret == True:
                try:
                    print('now, the program is going to deal with'  + ' pic ' + str(files_name))
                    w, h = image_size(path_temp)
                    if w<48 or h<48 or w>4096 or h>4096:
                        print('invalid image size')
                    else:
                        row, worksheet =  Write_EXCEL(path_temp, worksheet,row, files_name,time_stamp_fromvideo)

                except:
                    print('error！')
                    time.sleep(3)
                    print('the program is going to work')
                    print('now, the program is going to deal with '  + ' pic ' + str(files_name))
                    row, worksheet = Write_EXCEL(path_temp, worksheet, row, files_name,time_stamp_fromvideo)
                # cv.imshow('Frame', frame)
            else:
                break
        worksheet.write(1, 1, 0)
        address_save = path_save + files_name + '.xls'
        workbook.save(address_save)
        print('the current video' +  files_name  + ' is done')


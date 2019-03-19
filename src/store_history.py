import m3u8
import requests
import time
#import cv2
#import numpy as np
import logging

appkey = '00633c496fbe46b49751f96de8057f70'
secret = '69cc22b33bfa54498734680ffbb6e023'

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def request(appkey=None, secret=None,target=None,accessToken=None,channel=None):
    channels = [2,4,5]
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    if secret:
        r = requests.post('https://open.ys7.com/api/lapp/token/get', headers=headers, data = {'appKey': appkey, 'appSecret': secret})
    else:
        r = requests.post('https://open.ys7.com/api/lapp/live/address/limited', headers=headers, data = {'accessToken': accessToken, 'deviceSerial': 'C63000956', 'channelNo': channel})
    
    #print(r.text)
    if r.status_code == requests.codes.ok:
        result = r.json()
        data = result['data']
       
        if secret:
            token = data['accessToken']
            expireT = data['expireTime']
            return {'token':token, 'expiretime':expireT}
            
        else:
            res = data['hdAddress']
            return res
    else:
        if secret:
            logger.error("getting connetion failed for accessToken inquiry" + str(r.status_code) + ',' + r.text)
        elif r.status_code == 10002:
            result = request(appkey=appkey,secret=secret)
        elif r.status_code == 20002:
            logger.error('device is offline')
        else:
            logger.error("getting connetion failed for live address" + str(r.status_code) + ',' + r.text)
    
def getM3u8(url):
    m3u8_obj = m3u8.load(url)  # this could also be an absolute filename

    ts_url_list = []

    base_uri = m3u8_obj.base_uri

    ts_list = m3u8_obj.files

    return ts_list

def save(ts_url_list):
    #print(ts_url_list)
    error_get = []
    path = '/Users/pro/Documents/GIX/courses/thesis/axle_detection/wieght-limit-recgonize/data/ts/'
    for ts in ts_url_list:
        try:
            movie_name = str(time.time())+'.ts'
            movie = requests.get(ts, headers = {'Connection':'close'}, timeout=60)
            with open(movie_name, 'wb') as movie_content:
                movie_content.writelines(movie)
            
        except:
            error_get.append(_url)
            continue
        
    if error_get:
        logger.warning('cannot get videos:'+ error_get)
        save(error_get)

if __name__ == "__main__":
    accessToken = 'at.7au8lk9n4q5ccic55yk1a7ieanui8fn7-5j1xvqx2kb-1st2uu3-5jgeg8jcc'
    channel = 2
    while True:
        r = request(accessToken=accessToken, channel=channel)
        #print(type(r))
        if type(r) == str:
            ts_list = getM3u8(r)
            save(ts_list)
        else:
            accessToken = r['token']
            #r = request(accessToken=accessToken, channel=channel)
        time.sleep(10)
    
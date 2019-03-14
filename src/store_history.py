import m3u8
import requests
import time
#import cv2
#import numpy as np
import logging
import os
import sys, getopt

appkey = '00633c496fbe46b49751f96de8057f70'
secret = '69cc22b33bfa54498734680ffbb6e023'

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def request(appkey=None, secret=None,target=None,accessToken=None,deviceSerial=None,channel=None):
    #channels = [2,4,5]
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    if secret:
        r = requests.post('https://open.ys7.com/api/lapp/token/get', headers=headers, data = {'appKey': appkey, 'appSecret': secret})
    else:
        r = requests.post('https://open.ys7.com/api/lapp/live/address/limited', headers=headers, data = {'accessToken': accessToken, 'deviceSerial': deviceSerial, 'channelNo': channel})
    
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

    #ts_url_list = []

    #base_uri = m3u8_obj.base_uri

    ts_list = m3u8_obj.files

    return ts_list

def save(ts_url_list, path):
    #print(ts_url_list)
    error_get = []
    #'/Users/GIX/Downloads/wieght-limit-recgonize/data/'
    os.chdir(path)
    for ts in ts_url_list:
        try:
            movie_name = str(time.time())+'.ts'
            movie = requests.get(ts, headers = {'Connection':'close'}, timeout=60)
            with open(movie_name, 'wb') as movie_content:
                movie_content.writelines(movie)
            
        except:
            error_get.append(ts)
            continue
        
    if error_get:
        logger.warning('cannot get videos:'+ str(error_get))
        save(error_get,path)

def main(argv):
    try:
        opts, agrs = getopt.getopt(argv,"-h-d:-c:-p:")
        #print(opts)
    except getopt.GetoptError:
        print('Usage:store_history.py -d <deviceSerial> -c <channelNum> -p <path>' )
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: store_history.py -d <deviceSerial> -c <channelNum> -p <path>')
            sys.exit()
        elif opt == "-d":
            deviceSerial = arg
        elif opt == '-c':
            channel = int(arg)
        elif opt == '-p':
            path = arg
    return deviceSerial, channel, path

if __name__ == "__main__":
    accessToken = 'at.7au8lk9n4q5ccic55yk1a7ieanui8fn7-5j1xvqx2kb-1st2uu3-5jgeg8jcc'
    deviceSerial, channel, path = main(sys.argv[1:])
    while True:
        r = request(accessToken=accessToken,deviceSerial=deviceSerial,channel=channel)
        #print(type(r))
        if type(r) == str:
            ts_list = getM3u8(r)
            save(ts_list,path)
        else:
            accessToken = r['token']
            logger.info('newest accessToken:' + accessToken )
            #r = request(accessToken=accessToken, channel=channel)
        time.sleep(1)
    
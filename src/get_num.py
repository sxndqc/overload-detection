import m3u8
import requests
import subprocess
import os
import time
import cv2

def getM3u8(url):
    m3u8_obj = m3u8.load(url)  # this could also be an absolute filename

    ts_url_list = []

    base_uri = m3u8_obj.base_uri

    ts_list = m3u8_obj.files

    for _ts in ts_list:

        ts_url = base_uri + _ts

        ts_url_list.append(ts_url)

    # print ts_url

    # response = requests.head(ts_url)

    # if response.status_code == 200:
    #     print "URL 没问题"

    return ts_url_list

def download_movie(movie_url, _path):
    os.chdir(_path)
    print('>>>[+] downloading...')
    print('-' * 60)
    error_get = []

    for _url in movie_url:
        movie_name = str(_url.split("/")[-1]).split("?")[0]
        
        try:
            movie = requests.get(_url, headers = {'Connection':'close'}, timeout=60)
            with open(movie_name, 'wb') as movie_content:
                movie_content.writelines(movie)
            print('>>>[+] File ' + movie_name + ' done')
        # Picture part **********************************************************************************************************
            print("Image Conversion Test: \n")
            cap = cv2.VideoCapture(_url)
            ret, image= cap.read()
            timestamp = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
            cv2.imwrite(timestamp+".jpg",image)
            processed = process(image)
            cv2.imwrite(timestamp + ".png", processed)
            print('>>>[+] File ' + movie_name + ' done')
        # ***********************************************************************************************************************
        
        except:
            error_get.append(_url)
            continue

    # 如果没有不成功的请求就结束
    if error_get:
        # print u'共有%d个请求失败' % len(file_list)
        print('-' * 60)
        download_movie(error_get, _path)
    else:
        print('>>>[+] Download successfully!!!')


def process(image):
    image_cut = image[560:600, 680:740]
    image_rotater = cv2.getRotationMatrix2D((30,20),30,1)
    image_rotated = cv2.warpAffine(image_cut,image_rotater,(120,80))
    image_fine_cut = image_rotated[20:60,40:100]
    image_warped =  cv2.cvtColor(image_fine_cut, cv2.COLOR_BGR2GRAY)
    #image_bw = cv2.threshold(image_warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return image_warped

if __name__ == "__main__":
    #TODO: to request accessToken and fresh it
    url = "http://hls.open.ys7.com/openlive/77bd93fe20054c8893f9f0b309f83e31.hd.m3u8"
    path = "../pics/ts"
    #ts_url_list = []
    #while True:
    path = './'
    ts_url_list = getM3u8(url)
            
        #time.sleep(5)
    download_movie(ts_url_list,path)
    
        
    #cmd_str = hebing('./',timestamp+".mp4")
    #runConvertMp4(cmd_str)
    #time.sleep(5)
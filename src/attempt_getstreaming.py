import m3u8
import requests
import subprocess
import os
import time
"""
get m3u8ts files
"""
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
            # 'Connection':'close' 防止请求端口占用
            # timeout=30    防止请求时间超长连接
            movie = requests.get(_url, headers = {'Connection':'close'}, timeout=60)
            with open(movie_name, 'wb') as movie_content:
                movie_content.writelines(movie)
            print('>>>[+] File ' + movie_name + ' done')
        # 捕获异常，记录失败请求
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

"""
combine ts files
ls * | perl -nale 'chomp;push @a, $_;END{printf "ffmpeg -i \"concat:%s\" -acodec copy -vcodec copy -absf aac_adtstoasc out.mp4\n", join("|",@a)}' 
"""
def hebing(path,outfile):
    cmd_str = None
    filelist = []
    
    for file in os.listdir("./"):
        print(file)
        if len(file.split(".")) == 2:
            if file.split(".")[1] == 'ts':
                filelist.append('./' + file)
    s = '|'.join(filelist)
    print(s)
    cmd_str = 'ffmpeg -i \"concat:' + s + '\" ' + '-acodec copy -vcodec copy -absf aac_adtstoasc ' + path + outfile
    print(cmd_str)
    return cmd_str

def runConvertMp4(cmd_str):
    #str_env = "/Users/huqingen/Desktop/Finger/tool/ffmpeg/"
    subprocess.call(cmd_str, shell=True)
    for file in os.listdir("./"):
        print(file)
        if len(file.split(".")) == 2:
            if file.split(".")[1] == 'ts':
                os.remove(file)
if __name__ == "__main__":
    #TODO: to request accessToken and fresh it
    url = "http://hls.open.ys7.com/openlive/77bd93fe20054c8893f9f0b309f83e31.hd.m3u8"
    path = "../pics/ts"
    #ts_url_list = []
    while True:
        path = './'
        ts_url_list = getM3u8(url)
            
            #time.sleep(5)
        download_movie(ts_url_list,path)
        timestamp = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
        
        cmd_str = hebing('./',timestamp+".mp4")
        runConvertMp4(cmd_str)
        time.sleep(5)
    
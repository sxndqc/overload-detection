import cv2
import subprocess as sp
import numpy
VIDEO_URL = "http://hls.open.ys7.com/openlive/77bd93fe20054c8893f9f0b309f83e31.hd.m3u8"

#cv2.namedWindow("GoPro")

FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
#FFMPEG_BIN = "ffmpeg.exe" # on Windows

pipe = sp.Popen([ FFMPEG_BIN, "-i", VIDEO_URL,
           "-loglevel", "quiet", # no text output
           "-an",   # disable audio
           "-f", "image2pipe",
           "-pix_fmt", "bgr24",
           "-vcodec", "rawvideo", "-"],
           stdin = sp.PIPE, stdout = sp.PIPE)
print(pipe)
while True:
    raw_image = pipe.stdout.read(720*1280*3) # read 1 frame
    #print(raw_image)
    image =  numpy.fromstring(raw_image, dtype='uint8').reshape((720,1280,3))
    print(image)
    cv2.imshow("test",image)
    
    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()
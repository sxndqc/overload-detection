#相关性 点积归一化（
import cv2
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

def get_perspective_mat():
  
    src_points = np.array([[687., 551.], [681., 571.], [749., 580.], [744., 603.]], dtype = "float32")
    dst_points = np.array([[681., 540.], [681., 560.], [736.,540.], [736., 560.]], dtype = "float32")
 
    M = cv2.getPerspectiveTransform(src_points, dst_points)
 
    return M

def cut(image):
    global M
    rows,cols = image.shape[:2]
    res = cv2.warpPerspective(image, M, (rows, cols), cv2.INTER_LINEAR)

    im = cv2.resize(res[540:560, 696:736],(600,200))
    
    return im

def main(imgS):
    
    img = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)

    #print(im.shape)
    #im2 = custom_blur_demo(im)
    #cv2.waitKey(0)
    #new_temp = []
    templates = pkl.load(open("templates.pkl","rb"))
    #print(len(templates))
    #for template in templates:
    #    im = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #    new_temp.append(im)
    #pkl.dump(new_temp, open("template_storage.pkl","rb"))
    #w, h = template.shape[::-1]
    
    #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    meth = 'cv2.TM_CCOEFF_NORMED'
    img2 = img.copy()
    method = eval(meth)
    #print(method)
    #cv2.imshow('',img2)
    #cv2.waitKey(0)
    horizon = np.zeros((10, img.shape[1]))
    
    #print(templates[0])
    colors = ['aqua','black','coral', 'cyan', 'darkgreen','deeppink',
                    'gold','indigo','lavender','mediumaquamarine']
    for i,template in enumerate(templates):
        #print(i)
        img = img2.copy()
        # Apply template Matching
        #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #print(template)
        #cv2.imshow('',template)
        #cv2.waitKey(0)
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        w, h = template.shape[::-1]

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    #if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #    top_left = min_loc
    #else:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        horizon[i][:res.shape[1]] = np.max(res, 0)
        x = np.arange(0,img.shape[1],1)
    """    
        plt.figure(1)
        plt.scatter(x, horizon[i], color=colors[i], label = str(i),alpha=0.6)
        
    xt = np.arange(0.,600.,50.)
    yt = np.arange(-1.,1.,0.05) 
    plt.xticks(xt)   
    plt.yticks(yt)
    plt.grid()
    plt.show()   
    """
    turning_tops = []
    threshold = 0.70
    for num, series in enumerate(horizon):
        for loc in range(1,len(series)-1):
            if (series[loc-1]<series[loc]) and (series[loc+1]<series[loc]) and (series[loc]>threshold):
               turning_tops.append((loc, series[loc], num)) 
    
    turning_tops.sort(key=lambda x:x[0])
    
    print(turning_tops)
    
    locs = [55, 140, 225, 310, 395, 1000]
    maxs = [0,0,0,0,0,1000000]
    nums = [-1,-1,-1,-1,-1, -1]
    k = 0
    tonari = 10
    #print(turning_tops)[]
    for i in turning_tops:
        if (abs(i[0]-locs[k])<tonari) and (i[1]>maxs[k]):
            maxs[k] = i[1]
            nums[k] = i[2]
            continue
        if (abs(i[0]-locs[k+1])<tonari) and (i[1]>maxs[k+1]):
            k += 1
            maxs[k] = i[1]
            nums[k] = i[2]
            continue
    
    if nums[0]>=0:
        print(nums[0],nums[1],nums[2],nums[3],nums[4])
    else:
        print(nums[1],nums[2],nums[3],nums[4])
      

if __name__ == "__main__":
    
    video = cv2.VideoCapture("../../src/ch04_20190312153219.mp4")
    count = 0
    M = get_perspective_mat()
    while True:
        count += 1
        _,frame = video.read()
        if count%600==0:
            img = cut(frame)
            main(img)
            """
    M = get_perspective_mat()
    img = cv2.imread("../2019-03-18-16:16:29.jpg")
    #cv2.imshow('',img)
    #cv2.waitKey(0)
    img = cut(img)
    main(img)
    """

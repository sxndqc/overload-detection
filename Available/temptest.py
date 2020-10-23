#相关性 点积归一化（
import cv2
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from clustering import *

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
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.uint8)
    #img = cv2.dilate(img, kernel, iterations=1)
    #img = cv2.filter2D(img, 1, kernel)
    #cv2.imshow('',img)
    #cv2.waitKey(0)
    """
    #解决9\3识别问题
    """
    #print(im.shape)
    #im2 = custom_blur_demo(im)
    #cv2.waitKey(0)
    #new_temp = []
    templates = pkl.load(open("templates.pkl","rb"))
    
    #templates[1] = templates[1][:,45:]
    #cv2.imshow('',templates[1])
    #cv2.waitKey(0)
    #print(len(templates))
    #for template in templates:
    #    im = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #    new_temp.append(im)
    #pkl.dump(new_temp, open("template_storage.pkl","rb"))
    #w, h = template.shape[::-1]
    
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    
    meth = methods[1]
    img2 = img.copy()
    method = eval(meth)
    #print(method)
    #cv2.imshow('',img2)
    #cv2.waitKey(0)
    horizon = np.zeros((11, img.shape[1]))
    maximum = np.zeros((10, img.shape[1]), dtype=np.uint8)
    
    #print(templates[0])
    colors = ['blue','black','coral', 'cyan', 'darkgreen','deeppink',
                    'gold','indigo','lavender','yellow', 'red']
    plt.ion()
    plt.clf()
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
        maximum[i][template.shape[1]//2:res.shape[1]+template.shape[1]//2] = np.argmax(res, 0)
        horizon[i][template.shape[1]//2:res.shape[1]+template.shape[1]//2] = np.max(res, 0)
        x = np.arange(0,img.shape[1],1)
    #"""    
        plt.figure(1)
        plt.scatter(x, horizon[i], color=colors[i], label = str(i),alpha=0.6)
    
    method = eval(methods[2])
    maxi_index = np.argmax(horizon[:9], axis=0)
    template = templates[8]
    #cv2.imshow('',template)
    #cv2.waitKey(0)
    weight = 1e8
    res = cv2.matchTemplate(img, template, method)
    #print(np.mean(maximum, axis=0))
    for i in range(res.shape[1]):
        #print(maxi_index[i+template.shape[1]//2])
        #if maxi_index[i+template.shape[1]//2]==10:
        #    print(horizon[:, i+template.shape[1]//2])
        #print(maximum[maxi_index[i+template.shape[1]//2]][i+template.shape[1]//2])
        #改成半截之后只需要看右半截的，自然就平衡了
        horizon[10][i+template.shape[1]//2] = res[min(maximum[maxi_index[i+template.shape[1]//2]][i+template.shape[1]//2], res.shape[0]-1)][i]/weight# - res[min(maximum[maxi_index[i+template.shape[1]//2]][i+template.shape[1]//2], res.shape[0]-1)][i]/weight
        
    #horizon[10][:res.shape[1]] = res[np.int32(np.round(np.mean(maximum, axis=0)))[:res.shape[1]],np.arange(res.shape[1])]/weight
    x = np.arange(0,img.shape[1],1)
    plt.scatter(x, horizon[10], color=colors[10], label = str(10), alpha=0.6)
        
    xt = np.arange(0.,600.,50.)
    #yt = np.arange(-1.,1.,0.05) 
    plt.xticks(xt)   
    #plt.yticks(yt)
    plt.grid()
    plt.draw()   
    plt.pause(0.01)
    #"""
    turning_tops = []
    threshold = 0.60   #必须要很宽才行
    #backend ={5:[], 6:[], 7:[], 8:[], 9:[]}
    for num, series in enumerate(horizon[:9]):
        for loc in range(1,len(series)-1):
            if (series[loc-1]<series[loc]) and (series[loc+1]<series[loc]) and (series[loc]>threshold):
               turning_tops.append((loc, series[loc], num))
               #backend[np.floor(series[loc]*10)].append((loc, series[loc], num))
               #位置、相似度、数字 
    
    turning_tops.sort(key=lambda x:x[0])
    
    #print(turning_tops)
    
    #摒弃这种方法
    """
    
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
    
    #以下部分是确定位置的方法
    #用四个位置去套用就行
    #套那个高的点
    """
    """
    for series in enumerate(horizon[10]): #做一次卷积
        for loc in range(1,len(series)-1):
            if (series[loc-1]<series[loc]) and (series[loc+1]<series[loc]) and (series[loc]>threshold):
               turning_tops.append((loc, series[loc], num))
               backend[np.floor(series[loc]*10)].append((loc, series[loc], num))
    """
    tonari = 12
    
    #合并一定范围内的最大值，以免歧义；最大值周围20以内的都pop
    #不好用聚类，因为有些中间数不好归类，而且把非数字位的归到一起了也不好
    i = 0
    while i<len(turning_tops):
        #i
        j = i-1
        while (j>=0) and (turning_tops[i][0]-turning_tops[j][0]<tonari) and (turning_tops[i][1]>turning_tops[j][1]):
            turning_tops.pop(j)
            i -= 1
            j = i-1
            
        j = i+1
        while (j<len(turning_tops)) and (turning_tops[j][0]-turning_tops[i][0]<tonari) and (turning_tops[i][1]>turning_tops[j][1]):
            turning_tops.pop(j)
        i += 1
    #这样就干净了
    
    four_pos = np.array([0, 85, 170, 255, 1000])
    nums = [-1,-1,-1,-1,-1, -1]
    final_pos = np.array([False])
    light_threshold = 0.8
    for j in range(600-255-tonari):#找到最后一个四个并列的，作为四个卡的位置
        k = 0
        tnums = [-1,-1,-1,-1,-1]
        maxs = np.array([0.,0.,0.,0., 10000.])
        locs = four_pos + j
        for i in turning_tops:
            
            if (abs(i[0]-locs[k])<tonari) and (i[1]>maxs[k]):# and (horizon[10][i[0]]>0.9):
                maxs[k] = i[1]
                tnums[k+1] = i[2]
                continue
            if (abs(i[0]-locs[k+1])<tonari) and (i[1]>maxs[k+1]):# and (horizon[10][i[0]]>0.9):
                k += 1
                if k>3:
                    break
                maxs[k] = i[1]
                tnums[k+1] = i[2]
                continue
        #问题出在越界了；一段边界如果最后的那个最大值，仍然在最大范围的话，那就会取它，即使它不是领域中的最大值
        #应该合并tonari范围内的最大值
        if (maxs.all()!=0) and (horizon[10][locs[3]]>light_threshold):
            final_pos = locs
            nums = tnums
    
    one_threshold = 0.5                
    if not final_pos.all():
        print("Error")
    elif final_pos[0]>95:
        final_pos -= tonari #因为最后确定的final_pos一定是最接近右边缘的
        loc = final_pos[0]-85
        maxi = 0.
        print("kkk")
        print(loc)
        for i in turning_tops:
            if (abs(i[0]-loc)<tonari) and (i[1]>maxi) and (horizon[10][i[0]]>one_threshold):
                maxi = i[1]
                nums[0] = i[2]
            if i[0] - loc>tonari:
                break
                
    if nums[0]>=0:
        print(nums[0],nums[1],nums[2],nums[3],nums[4])
    else:
        print(nums[1],nums[2],nums[3],nums[4])
        
    #再画图
    turning_tops = np.array(turning_tops)
    plt.scatter(turning_tops[:, 0], turning_tops[:, 1], color="black" ,alpha=0.6)
    #print([final_pos[0]-85] + final_pos[:4])
    #plt.scatter([final_pos[0]-85] + list(final_pos[:4]), [1,1,1,1,1], color="blue" )
    #xt = np.arange(0.,600.,50.)
    #yt = np.arange(-1.,1.,0.05) 
    #plt.xticks(xt)   
    #plt.yticks(yt)
    plt.grid()
    plt.draw()
    cv2.imshow('a', imgS)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    
    video = cv2.VideoCapture("vlc-record-2019-05-06-09h15m52s-ch04_20190312153219.mp4-.mp4")
    count = 0
    M = get_perspective_mat()
    
    while True:
        count += 1
        _,frame = video.read()
        try:
            frame.shape
        except:
            print("Finished")
            break
        if count%6==0:            
            #img = cut(frame)
            #cv2.namedWindow("capture",0)
            #cv2.resizeWindow("capture", 200, 200)
            #cv2.imshow("capture", frame[500:700, 600:800])
            #cv2.waitKey(6)
            #img = ROIExtraction(frame)
            NotFound = True
            while NotFound:
                try:
                    img = ROIExtraction(frame)
                    NotFound = False
                except:
                    NotFound = True
            #img = cv2.imread("../pics/2019-03-18-16:07:57.png")
            main(img)
            
            """
    M = get_perspective_mat()
    img = cv2.imread("../2019-03-18-16:16:29.jpg")
    #cv2.imshow('',img)
    #cv2.waitKey(0)
    img = cut(img)
    main(img)
    """
    
    
    """
    方法：相似度从上往下搜寻最高点，直到找到四个等差数列。
    
    遇到的问题：
    ①clustering中的filter有时候根本不显示，然后就报错找不到approx
    但是filter这个不显示并不报错，后面的程序还在继续执行
    ②首位1的光线太暗，无法判定是不是数字
    ③末位的0可能被识别成其他数字，而这个数字的亮度不够导致无法找到最后一位数
    ④9容易被识别成3
        """
 
    



#................SOM...................#
#                 VS                 
#................SOM+..................#


#...............inrto..................#

'''
        image processing project
         Segmentation
          image with SOM neural network and with new method of SOM+
           denoised by wavelet translation
            and test the quality by
             ssim and topolygy, psnr quantization error
              all of the progect has beed made without special library
               Ahmadali Jamali
                02.21.2022
                 Teheran

                                    '''

#Start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#..............libraries...............#

import numpy as np
import math 
from skimage import io, img_as_float
import cv2
import random
import matplotlib.image as mpimg
import time
#.............Functions................#

#normalization:
def norm(matrix):
    
    a = len(matrix)
    b = len(matrix[0])
    l = []
    for i in range(a):
        for j in range(b):
            l.append(matrix[i][j])
            
    mi = min(l)
    ma = max(l)
    di = ma - mi
    
    for i in range(a):
        for j in range(b):
            matrix[i][j] = (matrix[i][j] - mi)/di
            
    return np.array(matrix)

#normalized between [a,b], as preparing image for wavelet translation:
def normalization(array,a,b):

    m = len(array)
    n = len(array[0])
    sample = []
    for i in range(m):
        for j in range(n):
            sample.append(array[i][j])
    maximum = max(sample)
    minimum = min(sample)

    old_range = maximum - minimum
    new_range = abs(a - b)

    for i in range(m):
        for j in range(n):
            array[i][j] = a+(array[i][j] - minimum)*new_range/old_range

    return array

#wavelet transform:
def wt(image):

    m = len(image)
    n = len(image[0])
    
    H = normalization(image,200,255)
    L = normalization(image,0,200)

    HH = normalization(H,240,255)
    HL = normalization(H,200,240)

    LH = normalization(L,177,200)

    for i in range(m):
        for j in range(n):

            image[i][j] = (HH[i][j] + HL[i][j] + LH[i][j])/3

    return np.array(image)        

#////////////////////////////////First Algorithm of Classic SOM/////////////////////////////////////    
#self_organized map neural network:
def som(image):

    m = len(image)
    n = len(image[0])
    product = m*n
    print('\n')
    #image size, row and column
    print('//////////som//////////\n')
    print('//////////////////////\n')
    print(' Image size',m,n,'\n')
    print('----------------------')

    topography = 0
    #random self organized map
    mm = 8
    nn = 8 #size for neck 6*6, brain1 3*3,Head 5*5,knee 7*7 a= 0.58.
    print(' The map size:',mm,'\n')
    print('----------------------')    
    map_neuralnetwork = []
    itteration  = 1
    for _ in range(mm):     #weight is earnd from last part
        column = []
        for _ in range(nn):
            if itteration %2 == 0:
                column.append(random.uniform(10,25)) #random range
                itteration = itteration + 1
            else :
                column.append(random.uniform(0,10.5000))
                itteration = itteration +1
        map_neuralnetwork.append(column) #map_neuralnetwork size bydefault m,n
    
    #updating the map:
    new_image = []
    a = 0.88 #learning coefficient 0.8
    print(' The learning coefficient',a,'\n')
    print('----------------------')
    alfa = []
    t = 0
    for i in range(mm):
        alfac = []
        for j in range(nn):
            alfac.append(a)
        alfa.append(alfac)
    for i in range(m):
        new_value = []
        for j in range(n):
            #print(sd,m*n)
                #finding winner
                winner = []
                for k in range(mm):
                    for l in range(nn):
                        result = round((image[i][j] - map_neuralnetwork[k][l])**2,4)
                        winner.append(result)        
                winmin = min(winner)
                counterw = 0
                for v in range(mm):
                    if winmin == winner[v]:
                       counterw = counterw + 1
                    if counterw == 2:
                       winmin2 = winner[v]
                       break
                
                else:
                    secondw = []
                    for f in range(mm):
                        if winmin - winner[v] !=0:
                           secondw.append(abs(winmin - winner[v]))
                    winmin2 = min(secondw)                 
                                       
                #find the adress of min and update the weight and alpha:
                lenght = len(winner)
                one = True
                two = True
                indexn,indexm = 0,0
                for z in range(lenght):
                    if winmin == winner[z] and one == True:
                       indexm = math.ceil(z/nn)-1
                       indexn = z - indexm*nn -1
                       #updayted weight:
                       map_neuralnetwork[indexm][indexn] = map_neuralnetwork[indexm][indexn] + alfa[indexm][indexn]*math.sqrt(winmin)#round
                       alfa[indexm][indexn] = a*math.e**(-1*t/product) #updayted alpha
                       t = t + 1
                       #print(map_neuralnetwork[indexm][indexn])
                       new_value.append(z)
                       one = False
                       #second winner: 
                    if winmin2 == winner[z] and two == True:
                       indexm2 = math.ceil(z/nn)-1
                       indexn2 = z - indexm*nn -1
                       if indexn2 != indexn or indexm2 != indexm:
                          two = False
                          
                    if one == False and two == False:
                        
                       #topography:
                       if abs(indexm - indexm2)  == 1 and indexn == indexn2:
                           neighborhood = 0
                           topography = topography + neighborhood
                           break
                       elif abs(indexm - indexm2)  == 1 and abs(indexn - indexn2)  == 1:
                           neighborhood = 0
                           topography = topography + neighborhood
                           break
                       elif abs(indexn - indexn2)  == 1 and indexm == indexm2:
                           neighborhood = 0
                           topography = topography + neighborhood
                           break
                       else:
                           neighborhood = 1
                           topography = topography + neighborhood       
                           break
        #make a new array           
        new_image.append(new_value)
          
    #translation pixel    
    for s in range(m):
        for q in range(n):
            #print(new_image[s][q])
            indexmm = math.ceil(new_image[s][q]/nn)-1
            indexnn = new_image[s][q] - indexmm*nn -1
            new_image[s][q] = map_neuralnetwork[indexmm][indexnn]
    
    norm(new_image)
    np.array(map_neuralnetwork)
    cluster = []
    for i in range(mm):
        for j in range(nn):
            cluster.append(int(map_neuralnetwork[i][j]))
    uni = np.unique(cluster)
    mean = sum(uni)/len(uni)
    variance = 0
    for i in range(len(uni)):
        variance = ((uni[i] - mean)**2)/len(uni) + variance
    
    print('the variance of clustering:',variance)     
    print('the number of clustering',len(uni))
    print(uni)
    #topography error:
    top = round(topography/product,4)
    print(' The topography result: ',top,'\n',)
    print('----------------------')
    return np.array(new_image)

#/////////////////////////////////Fitness Function Errors////////////////////////////////
#quantization erroe:
def quantizationerror(org,image):

    m = len(image)
    n = len(image[0])
    l = m*n

    s = 0
    for i in range(m):
        for j in range(n):
            s = s + abs(org[i][j] - image[i][j])

    mse = round(s/l,2)
    
    return mse    

#Structural similarity:
def ssim(org,seg):

    m = len(org)
    n = len(org[0])
    mn = m*n
    #sum of orginal and segmentatied image
    so,ss = 0,0
    dato,dats = [],[]
    for i in range(m):
        for j in range(n):
            dato.append(org[i][j])
            dats.append(seg[i][j])
            so = so +org[i][j]
            ss = ss + seg[i][j]
    #mean of images:        
    mean_orginal_image = so/ mn
    mean_segmentaion = ss/mn
    
    so,ss = 0,0
    cov = 0
    for i in range(len(dato)):
        so = (dato[i]- mean_orginal_image)**2 + so
        ss = (dats[i]-  mean_segmentaion)**2 + ss
        cov =cov + (dato[i] - mean_orginal_image)*(dats[i] - mean_segmentaion )/(len(dato))
    
    varianceo = so/mn
    variances = ss/mn
    np.array(dato)
    np.array(dats)
    
    #dynamic range:
    lo = len(np.unique(dato)) - 2
    ls = len(np.unique(dats)) - 2
    k1 = 0.01
    k2 = 0.03
    c1 = (k1*lo)**2
    c2 = (k2*ls)**2
    print(' The variance of orginal image ',round(varianceo,3),'\n')
    print('----------------------')
    ssimimage = (2*mean_orginal_image*mean_segmentaion+c1)*(2*cov+c2)/((mean_orginal_image**2+mean_segmentaion**2+c1)*(c2+varianceo**2+variances**2))
    
    return round(ssimimage,4)

    
#...............input..................#

#main function:
def main():
    start_time = time.time()
    #load image / gray and float 
    img = img_as_float(io.imread('p.jpg',as_gray = True))

    #kernel
    kernel = np.ones((5,5),np.float32)/25
    gaussian_kernel = np.array([[1/16, 1/4, 1/16],
                                [1/8, 1/4,  1/8],
                                [1/16, 1/8, 1/16]])
    
    #convolution
    #conv_using_cv2 = cv2.filter2D(img, -1, gaussian_kernel, borderType = cv2.BORDER_CONSTANT)

    #output/show
    
    #orginal image:
    #cv2.imshow('org',img)

    #blured image:
    #cv2.imshow('blured image',conv_using_cv2)

    #test with iterative evaluation:
    test = 1
    er00 = []
    er11 = []
    er22 = []
    for i in range(test):
        
        print('test Number: ',i+1,'/',test)
        #Self_organized map neural network:
        
        segmentation = som(img)

        #show
        #news2 = new_som_part2(news1,map_neuralnetwork)
        
        #cv2.imwrite('segmentation1.jpeg',segmentation)
        print("--- %s seconds ---" % (time.time() - start_time))
        mpimg.imsave('savedimage.jpeg',segmentation) 
        #evaluation the SOM

        #quantization error
        er0 = quantizationerror(img,segmentation)
        print(" The quantization error result:",er0,'\n')
        print('----------------------')
        er00.append(er0)
        #ssim
        er2 = ssim(img,segmentation)
        print(' The ssim error result:',er2,'\n')
        print('----------------------')
        er11.append(er2)
    

    quantization = round(sum(er00)/test,3)
    ssimerror = round(sum(er11)/test,3)
    
    print('----------------------\n')
    print('quantization :',quantization,'\n',)
    print('ssim :',ssimerror,'\n',)

    #output/showing
    #cv2.waitkey()
    #cv2.destruAllwindow()


   
#...............output.................#

#output/main function:    
if __name__ == "__main__":
    
    main()
                
input()

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<END#

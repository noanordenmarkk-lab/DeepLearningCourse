import numpy as np

from scipy.signal import convolve2d
from imageio import imread

import matplotlib.pyplot as plt

imgN  = 20
image = np.random.randn(imgN,imgN)

kernelN = 7
Y,X = np.meshgrid(np.linspace(-3,3,kernelN),np.linspace(-3,3,kernelN))
kernel = np.exp( -(X**2+Y**2)/7 )

#now for the convolution
convoutput = np.zeros((imgN,imgN))
halfKr = kernelN//2

for rowi in range(halfKr,imgN-halfKr):
    for coli in range(halfKr,imgN-halfKr):

        #cut out a piece of the image
        pieceOfImg = image[rowi-halfKr:rowi+halfKr+1,:]      # get the rows
        pieceOfImg = pieceOfImg[:,coli-halfKr:coli+halfKr+1] # extract the columns
        
        #dot product: element-wise multiply and sum (and flup the kernel for "real convolution")
        dotprod = np.sum( pieceOfImg*kernel[::-1,::-1] )

        #store the results for this pixel
        convoutput[rowi,coli] = dotprod

convoutput2 = convolve2d(image,kernel,mode='valid')

fig,ax = plt.subplots(2,2, figsize=(8,8))

ax[0,0].imshow(image)
ax[0,0].set_title('image')

ax[0,1].imshow(kernel)
ax[0,1].set_title('kernel')

ax[1,0].imshow(convoutput)
ax[1,0].set_title('convoutput')

ax[1,1].imshow(convoutput2)
ax[1,1].set_title('convuotput2')

plt.legend()
plt.show()
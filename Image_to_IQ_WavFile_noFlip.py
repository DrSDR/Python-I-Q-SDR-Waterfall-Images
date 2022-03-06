
# program reads in a jpg and saves as wavefile
# program converts jpg into ifft signal to see image in sdr waterfall
# equation  (wpixels * hpixels)/fs    =    tx time
# so make 800x800 image in paint,
# with 50khz , tx time is (800*800) / 50e3 = 12.8seconds
# then use gnuradio to tx wave file to sdr tx hardware
# program was first in matlab, but now python so folks can test for free






import numpy as np  # N-dimensional arrays
# from scipy.signal import firwin, lfilter  # filtering

from PIL import Image  # read images 

# from skimage.io import imread  # read an image file
from scipy.io import wavfile  # writing wav files

import matplotlib.pyplot as plt  # showing an image plot

from tkinter import filedialog  # open/save a file through GUI

# import sys  # aborting the script

filename = filedialog.askopenfilename()  # get a jpg file, window pops up, go get jpg
data = Image.open(filename)   # open the jpg file
data = np.array(data)      # turn jpg into [w x h x 3]  matrix of numbers


# convert the jpg to grayscale using 1950s NTCS math
# now we lazy and just code it.
data = 0.2989*data[:,:,0] + 0.5870*data[:,:,1] + 0.1140*data[:,:,2]   # grayscale math old school
# data is now a [w x h]  matrix,     so 3d to 2d



# show the image why not
plt.figure(1)
plt.imshow(data, cmap='gray') # imshow colormap = grayscale
plt.title('the picture ')
plt.show()  # show the image

# # flip image up side down
# data = np.flip(data,axis=0)
# plt.figure(2)
# plt.imshow(data,cmap='gray')
# plt.title('image now flipped')
# plt.show()

#now the crazy part, need to do fftshift to place dc terms at extreme left and right
# crazy math parts, drinking strong beer is a good idea, beyond this part 
data = np.fft.fftshift(data,axes=1)  #fftshift along width of image
plt.figure(3)
plt.imshow(data,cmap='gray')
plt.title('image now fftshift to dc regions')
plt.show()

#finally the ifft part
# this part converts the image into time domain signals
# this is the i/q data we send to transmitter for jamming.
data = np.fft.ifft(data,axis=1)  #ifft along width of image matrix
data = np.fft.ifftshift(data,axes=1)  #ifftshift along width of image





#flatten matrix into 1xN vector
data = data.flatten();    # 1xN vector

# normalize to -1 to 1 range
data /= np.max(np.abs(data))

#now scale to int16 range  -32768 to 32767
scale = 32767
data = np.multiply(data,scale)



# get data into [samples,2] for wav save, left = real , right = imag  
data = np.array([np.real(data), np.imag(data)]).T  # need transpose since wavefile.write expects shape (Nsamples, Nchannels)
data = data.astype(np.int16)  # int16


# see the signal before wav file save
plt.figure(4)
plt.plot(data[:,0],label='real data')
plt.plot(data[:,1],label='imag data')
plt.legend()
plt.title('I/Q time signal int16 ranges')
plt.show()


filetypes = [('WAV File', '*.wav')]
pathname = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=filetypes)

fs = 50e3  # sampling freq of wave file


wavfile.write(pathname, int(round(fs)), data )  # sampling rate needs to be an integer (samples per sec)











# Python-I-Q-SDR-Waterfall-Images
in here are python scripts to read in image and convert to wav file.
one can use gnuradio to tx wav file
script creates a i/q wav file that has spectrum of jpg image
open ms paint, make image 800 x 800
set background as black, and text to white
draw or write what you want
save jpg file
then run python script  (install spyder or anaconda on your pc)  all free
run script
it pops open window to find jpg
then saves wav file.
in gnuradio use wavread (set to 2ch)
then use float to complex block
then send to sdr hardware sink block
set fs of sink block to wav file sample freq
script as is fs = 50khz
so tx time is  (800*800) / 50e3 = 12.8 seconds
connect sdr hardware to rx with waterfall 
you should see image in sdr waterfall
if image is flipped, run the no flip python script instead
no need to have matlab anymore
thanks to python numpy, scipy and pil,  all math done in python.
see thegmr140 youtube videos for examples of fun
enjoy

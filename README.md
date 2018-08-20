#  catch the eye

this simple framework uses a deep learning network to find the eye in the image of a face (for instance the image that will be taken by a WebCam) and then to predict the direction of the gaze

## install

````
pip3 install --user easydict
git clone https://github.com/laurentperrinet/CatchTheEye
cd CatchTheEye/
python3 Regard.py
````

## results

````
$ python3 Regard.py
--------------------------------------------------
Default parameters
--------------------------------------------------
Accuracy=96.2% +/- 0.0%  in 28483.2 seconds
--------------------------------------------------
 parameter scan
--------------------------------------------------
--------------------------------------------------
 base= 2
--------------------------------------------------
--------------------------------------------------
 parameter scan : data
--------------------------------------------------
scanning over size = [130, 154, 183, 218, 260, 309, 367, 437, 520]
For parameter size = 130 ,  Accuracy=97.5% +/- 0.0%  in 7176.8 seconds
For parameter size = 154 ,  Accuracy=95.7% +/- 0.4%  in 10124.3 seconds
For parameter size = 183 ,  Accuracy=96.2% +/- 0.3%  in 15962.0 seconds
For parameter size = 218 ,  Accuracy=96.2% +/- 0.0%  in 20079.4 seconds
For parameter size = 260 , Accuracy=100.0% +/- 0.0%  in 28498.9 seconds
For parameter size = 309 ,  Accuracy=96.9% +/- 0.0%  in 44837.7 seconds
For parameter size = 367 ,  Accuracy=94.0% +/- 0.3%  in 63373.4 seconds
For parameter size = 437 ,  Accuracy=95.5% +/- 0.2%  in 88630.5 seconds
For parameter size = 520 , Accuracy=92.3% +/- 0.3%  in 109674.5 seconds
scanning over fullsize = [175, 208, 247, 294, 350, 416, 494, 588, 700]
For parameter fullsize = 175 ,  Accuracy=95.5% +/- 0.3%  in 29079.5 seconds
For parameter fullsize = 208 ,  Accuracy=97.5% +/- 0.0%  in 29111.6 seconds
For parameter fullsize = 247 ,  Accuracy=95.0% +/- 0.0%  in 29093.0 seconds
For parameter fullsize = 294 ,  Accuracy=97.5% +/- 0.2%  in 29117.6 seconds
For parameter fullsize = 350 ,  Accuracy=96.8% +/- 0.2%  in 29121.2 seconds
For parameter fullsize = 416 ,  Accuracy=96.0% +/- 0.6%  in 29082.4 seconds
For parameter fullsize = 494 ,  Accuracy=95.3% +/- 0.3%  in 29123.9 seconds
For parameter fullsize = 588 ,  Accuracy=95.3% +/- 0.4%  in 29100.3 seconds
For parameter fullsize = 700 ,  Accuracy=91.6% +/- 0.4%  in 29232.6 seconds
scanning over crop = [160, 190, 226, 269, 320, 380, 452, 538, 640]
For parameter crop = 160 ,  Accuracy=94.5% +/- 0.6%  in 29102.9 seconds
For parameter crop = 190 ,  Accuracy=98.7% +/- 0.2%  in 29116.4 seconds
For parameter crop = 226 ,  Accuracy=92.8% +/- 0.4%  in 29052.8 seconds
For parameter crop = 269 ,  Accuracy=94.2% +/- 0.3%  in 28752.1 seconds
For parameter crop = 320 ,  Accuracy=95.0% +/- 0.0%  in 29120.9 seconds
For parameter crop = 380 ,  Accuracy=96.1% +/- 0.3%  in 29084.2 seconds
For parameter crop = 452 ,  Accuracy=98.1% +/- 0.2%  in 29120.6 seconds
For parameter crop = 538 ,  Accuracy=98.7% +/- 0.0%  in 29064.7 seconds
For parameter crop = 640 ,  Accuracy=96.5% +/- 0.6%  in 29095.6 seconds
scanning over mean = [ 0.18        0.21405728  0.25455844  0.30272271  0.36        0.42811456
  0.50911688  0.60544542  0.72      ]
For parameter mean = 0.18 ,  Accuracy=95.4% +/- 0.3%  in 29098.9 seconds
For parameter mean = 0.2140572807 ,  Accuracy=97.5% +/- 0.0%  in 29099.5 seconds
For parameter mean = 0.254558441227 ,  Accuracy=98.7% +/- 0.2%  in 29113.6 seconds
For parameter mean = 0.302722709491 ,  Accuracy=93.7% +/- 0.0%  in 29121.3 seconds
For parameter mean = 0.36 ,  Accuracy=97.2% +/- 0.3%  in 29111.9 seconds
For parameter mean = 0.428114561401 ,  Accuracy=98.1% +/- 0.2%  in 29080.5 seconds
For parameter mean = 0.509116882454 ,  Accuracy=93.5% +/- 2.0%  in 29075.1 seconds
For parameter mean = 0.605445418983 ,  Accuracy=95.0% +/- 0.2%  in 29067.8 seconds
For parameter mean = 0.72 ,  Accuracy=91.4% +/- 15.0%  in 29056.6 seconds
scanning over std = [ 0.15        0.17838107  0.21213203  0.25226892  0.3         0.35676213
  0.42426407  0.50453785  0.6       ]
For parameter std = 0.15 ,  Accuracy=95.0% +/- 0.0%  in 29099.4 seconds
For parameter std = 0.17838106725 ,  Accuracy=96.2% +/- 0.2%  in 29050.0 seconds
For parameter std = 0.212132034356 ,  Accuracy=95.0% +/- 0.2%  in 29068.3 seconds
For parameter std = 0.252268924576 ,  Accuracy=95.3% +/- 0.4%  in 29077.8 seconds
For parameter std = 0.3 ,  Accuracy=97.5% +/- 0.2%  in 29055.2 seconds
For parameter std = 0.356762134501 ,  Accuracy=94.6% +/- 0.8%  in 29077.9 seconds
For parameter std = 0.424264068712 ,

````

## LICENCE

All code is released under the GPL license, see the LICENSE file.

The database files of faces are distributed for the learning of the network, but should not be re-used outside this repository.

The precise license is https://creativecommons.org/licenses/by-nc-nd/4.0/

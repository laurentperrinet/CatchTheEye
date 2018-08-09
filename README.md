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
For parameter size = 520 ,

````

## LICENCE

All code is released under the GPL license, see the LICENSE file.

The database files of faces are distributed for the learning of the network, but should not be re-used outside this repository.

The precise license is https://creativecommons.org/licenses/by-nc-nd/4.0/

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

In summary, by scanning different parameters, we prove that :
 - using the GPU improves speed by a factor ~10,
 - SGD is more performant than ADAM in most cases for this problem,
 - there seem to be 2 optimal strategies, diabolo or inverted diabolo

````
$ python3 Regard.py
--------------------------------------------------
Default parameters
--------------------------------------------------
scanning over no_cuda = [True, False]
For parameter no_cuda = True ,  Accuracy=95.8% +/- 0.5%  in 13856.7 seconds
For parameter no_cuda = False ,  Accuracy=95.0% +/- 0.0%  in 1413.7 seconds
--------------------------------------------------
 parameter scan
--------------------------------------------------

--------------------------------------------------
Using ADAM
--------------------------------------------------
scanning over lr = [ 0.01        0.01189207  0.01414214  0.01681793  0.02        0.02378414
  0.02828427  0.03363586  0.04      ]
For parameter lr = 0.01 ,  Accuracy=33.6% +/- 0.9%  in 1392.1 seconds
For parameter lr = 0.01189207115 ,  Accuracy=36.5% +/- 0.0%  in 1394.0 seconds
For parameter lr = 0.0141421356237 ,  Accuracy=32.7% +/- 0.0%  in 1392.1 seconds
For parameter lr = 0.0168179283051 ,  Accuracy=22.0% +/- 0.0%  in 1393.3 seconds
For parameter lr = 0.02 ,  Accuracy=23.3% +/- 0.0%  in 1393.7 seconds
For parameter lr = 0.0237841423001 ,  Accuracy=32.1% +/- 0.0%  in 1393.8 seconds
For parameter lr = 0.0282842712475 ,  Accuracy=39.0% +/- 0.0%  in 1393.6 seconds
For parameter lr = 0.0336358566101 ,  Accuracy=31.4% +/- 0.0%  in 1394.0 seconds
For parameter lr = 0.04 ,  Accuracy=33.3% +/- 0.0%  in 1394.1 seconds
scanning over momentum = [ 0.24        0.28540971  0.33941125  0.40363028  0.48        0.57081942
  0.67882251  0.80726056  0.96      ]
For parameter momentum = 0.24 ,  Accuracy=29.6% +/- 0.0%  in 1393.6 seconds
For parameter momentum = 0.285409707601 ,  Accuracy=32.7% +/- 0.0%  in 1393.2 seconds
For parameter momentum = 0.33941125497 ,  Accuracy=33.3% +/- 0.0%  in 1393.3 seconds
For parameter momentum = 0.403630279322 ,  Accuracy=32.1% +/- 0.0%  in 1393.2 seconds
For parameter momentum = 0.48 ,  Accuracy=36.5% +/- 0.0%  in 1393.6 seconds
For parameter momentum = 0.570819415201 ,  Accuracy=33.3% +/- 0.0%  in 1394.1 seconds
For parameter momentum = 0.678822509939 ,  Accuracy=30.8% +/- 0.0%  in 1393.3 seconds
For parameter momentum = 0.807260558644 ,  Accuracy=35.8% +/- 0.0%  in 1394.3 seconds
For parameter momentum = 0.96 ,  Accuracy=37.7% +/- 0.0%  in 1393.1 seconds
scanning over batch_size = [8, 9, 11, 13, 16, 19, 22, 26, 32]
For parameter batch_size = 8 ,  Accuracy=35.8% +/- 0.0%  in 1391.1 seconds
For parameter batch_size = 9 ,  Accuracy=32.1% +/- 0.0%  in 1395.0 seconds
For parameter batch_size = 11 ,  Accuracy=35.2% +/- 0.0%  in 1393.7 seconds
For parameter batch_size = 13 ,  Accuracy=32.7% +/- 0.0%  in 1392.7 seconds
For parameter batch_size = 16 ,  Accuracy=25.2% +/- 0.0%  in 1395.8 seconds
For parameter batch_size = 19 ,  Accuracy=30.2% +/- 0.0%  in 1393.7 seconds
For parameter batch_size = 22 ,  Accuracy=32.1% +/- 0.0%  in 1396.2 seconds
For parameter batch_size = 26 ,  Accuracy=37.1% +/- 0.0%  in 1392.6 seconds
For parameter batch_size = 32 ,  Accuracy=22.6% +/- 0.0%  in 1397.2 seconds
scanning over epochs = [20, 23, 28, 33, 40, 47, 56, 67, 80]
For parameter epochs = 20 ,  Accuracy=33.3% +/- 0.0%  in 702.0 seconds
For parameter epochs = 23 ,  Accuracy=33.3% +/- 0.0%  in 805.4 seconds
For parameter epochs = 28 ,  Accuracy=34.0% +/- 0.0%  in 978.5 seconds
For parameter epochs = 33 ,  Accuracy=32.1% +/- 0.0%  in 1152.8 seconds
For parameter epochs = 40 ,  Accuracy=34.0% +/- 0.0%  in 1393.2 seconds
For parameter epochs = 47 ,  Accuracy=35.2% +/- 0.0%  in 1635.5 seconds
For parameter epochs = 56 ,  Accuracy=35.8% +/- 0.0%  in 1948.0 seconds
For parameter epochs = 67 ,  Accuracy=35.2% +/- 0.0%  in 2328.6 seconds
For parameter epochs = 80 ,  Accuracy=38.4% +/- 0.0%  in 2777.7 seconds
--------------------------------------------------
 parameter scan : network
--------------------------------------------------
scanning over conv1_kernel_size = [3, 4, 4, 5, 7, 8, 9, 11, 14]
For parameter conv1_kernel_size = 3 ,  Accuracy=93.1% +/- 0.2%  in 1392.5 seconds
For parameter conv1_kernel_size = 4 ,  Accuracy=96.8% +/- 0.2%  in 1392.2 seconds
For parameter conv1_kernel_size = 4 ,  Accuracy=95.0% +/- 0.2%  in 1392.5 seconds
For parameter conv1_kernel_size = 5 ,  Accuracy=95.8% +/- 0.3%  in 1393.8 seconds
For parameter conv1_kernel_size = 7 ,  Accuracy=100.0% +/- 0.0%  in 1392.7 seconds
For parameter conv1_kernel_size = 8 ,  Accuracy=96.7% +/- 0.3%  in 1392.2 seconds
For parameter conv1_kernel_size = 9 ,  Accuracy=94.8% +/- 0.3%  in 1391.7 seconds
For parameter conv1_kernel_size = 11 ,  Accuracy=98.1% +/- 0.0%  in 1392.9 seconds
For parameter conv1_kernel_size = 14 ,  Accuracy=95.0% +/- 0.0%  in 1393.2 seconds
scanning over conv1_dim = [2, 2, 2, 3, 4, 4, 5, 6, 8]
For parameter conv1_dim = 2 ,  Accuracy=97.4% +/- 0.3%  in 1390.8 seconds
For parameter conv1_dim = 2 ,  Accuracy=96.3% +/- 0.2%  in 1391.4 seconds
For parameter conv1_dim = 2 ,  Accuracy=93.3% +/- 0.3%  in 1393.2 seconds
For parameter conv1_dim = 3 ,  Accuracy=93.6% +/- 0.3%  in 1390.3 seconds
For parameter conv1_dim = 4 ,  Accuracy=97.0% +/- 0.3%  in 1391.3 seconds
For parameter conv1_dim = 4 ,  Accuracy=96.9% +/- 0.0%  in 1392.2 seconds
For parameter conv1_dim = 5 ,  Accuracy=97.4% +/- 0.2%  in 1390.3 seconds
For parameter conv1_dim = 6 ,  Accuracy=95.8% +/- 0.3%  in 1391.1 seconds
For parameter conv1_dim = 8 ,  Accuracy=94.3% +/- 0.0%  in 1391.4 seconds
scanning over conv2_kernel_size = [2, 2, 3, 4, 5, 5, 7, 8, 10]
For parameter conv2_kernel_size = 2 ,  Accuracy=94.6% +/- 0.4%  in 1392.3 seconds
For parameter conv2_kernel_size = 2 ,  Accuracy=94.7% +/- 1.4%  in 1392.8 seconds
For parameter conv2_kernel_size = 3 ,  Accuracy=93.5% +/- 1.0%  in 1393.9 seconds
For parameter conv2_kernel_size = 4 ,  Accuracy=95.1% +/- 0.3%  in 1391.4 seconds
For parameter conv2_kernel_size = 5 ,  Accuracy=97.5% +/- 0.0%  in 1397.1 seconds
For parameter conv2_kernel_size = 5 ,  Accuracy=98.1% +/- 0.0%  in 1394.9 seconds
For parameter conv2_kernel_size = 7 ,  Accuracy=98.9% +/- 0.5%  in 1391.6 seconds
For parameter conv2_kernel_size = 8 ,  Accuracy=97.2% +/- 0.6%  in 1392.9 seconds
For parameter conv2_kernel_size = 10 ,  Accuracy=96.8% +/- 0.2%  in 1393.3 seconds
scanning over conv2_dim = [6, 7, 9, 10, 13, 15, 18, 21, 26]
For parameter conv2_dim = 6 ,  Accuracy=94.3% +/- 0.3%  in 1391.3 seconds
For parameter conv2_dim = 7 ,  Accuracy=96.7% +/- 0.3%  in 1392.0 seconds
For parameter conv2_dim = 9 ,  Accuracy=98.1% +/- 0.0%  in 1392.7 seconds
For parameter conv2_dim = 10 ,  Accuracy=95.8% +/- 0.3%  in 1393.3 seconds
For parameter conv2_dim = 13 ,  Accuracy=98.5% +/- 0.3%  in 1392.3 seconds
For parameter conv2_dim = 15 ,  Accuracy=97.4% +/- 0.3%  in 1402.0 seconds
For parameter conv2_dim = 18 ,  Accuracy=97.0% +/- 0.4%  in 1392.3 seconds
For parameter conv2_dim = 21 ,  Accuracy=95.6% +/- 0.0%  in 1393.1 seconds
For parameter conv2_dim = 26 ,  Accuracy=97.5% +/- 0.5%  in 1396.5 seconds
scanning over stride1 = [2, 2, 2, 3, 4, 4, 5, 6, 8]
For parameter stride1 = 2 ,  Accuracy=94.0% +/- 0.3%  in 1395.0 seconds
For parameter stride1 = 2 ,  Accuracy=98.2% +/- 0.2%  in 1393.9 seconds
For parameter stride1 = 2 ,  Accuracy=95.0% +/- 0.0%  in 1394.7 seconds
For parameter stride1 = 3 ,  Accuracy=96.4% +/- 0.3%  in 1392.8 seconds
For parameter stride1 = 4 ,  Accuracy=97.4% +/- 0.3%  in 1393.1 seconds
For parameter stride1 = 4 ,  Accuracy=97.4% +/- 0.2%  in 1392.3 seconds
For parameter stride1 = 5 ,  Accuracy=100.0% +/- 0.0%  in 1395.4 seconds
For parameter stride1 = 6 ,  Accuracy=93.3% +/- 0.4%  in 1396.8 seconds
For parameter stride1 = 8 ,  Accuracy=97.5% +/- 0.0%  in 1398.9 seconds
scanning over stride2 = [2, 2, 2, 3, 4, 4, 5, 6, 8]
For parameter stride2 = 2 ,  Accuracy=97.5% +/- 0.0%  in 1394.1 seconds
For parameter stride2 = 2 ,  Accuracy=95.6% +/- 0.0%  in 1396.8 seconds
For parameter stride2 = 2 ,  Accuracy=97.5% +/- 0.0%  in 1396.6 seconds
For parameter stride2 = 3 ,  Accuracy=97.5% +/- 0.0%  in 1398.3 seconds
For parameter stride2 = 4 ,  Accuracy=98.1% +/- 0.0%  in 1400.0 seconds
For parameter stride2 = 4 ,  Accuracy=98.4% +/- 0.3%  in 1400.7 seconds
For parameter stride2 = 5 ,  Accuracy=95.3% +/- 0.3%  in 1401.1 seconds
For parameter stride2 = 6 ,  Accuracy=93.7% +/- 0.0%  in 1399.5 seconds
For parameter stride2 = 8 ,  Accuracy=95.0% +/- 0.2%  in 1400.2 seconds
scanning over dimension = [12, 14, 17, 21, 25, 29, 35, 42, 50]
For parameter dimension = 12 ,  Accuracy=98.7% +/- 0.2%  in 1395.9 seconds
For parameter dimension = 14 ,  Accuracy=96.9% +/- 0.0%  in 1399.3 seconds
For parameter dimension = 17 ,  Accuracy=97.5% +/- 0.0%  in 1401.3 seconds
For parameter dimension = 21 ,  Accuracy=97.5% +/- 0.0%  in 1400.3 seconds
For parameter dimension = 25 ,  Accuracy=98.2% +/- 0.2%  in 1400.7 seconds
For parameter dimension = 29 ,  Accuracy=98.4% +/- 0.5%  in 1400.7 seconds
For parameter dimension = 35 ,  Accuracy=98.1% +/- 0.0%  in 1400.0 seconds
For parameter dimension = 42 ,  Accuracy=98.1% +/- 0.0%  in 1396.8 seconds
For parameter dimension = 50 ,  Accuracy=99.4% +/- 0.0%  in 1395.0 seconds
--------------------------------------------------
 parameter scan : data
--------------------------------------------------
scanning over size = [90, 107, 127, 151, 180, 214, 254, 302, 360]
For parameter size = 90 ,  Accuracy=96.5% +/- 0.5%  in 1207.4 seconds
For parameter size = 107 ,  Accuracy=98.1% +/- 0.0%  in 1234.2 seconds
For parameter size = 127 ,  Accuracy=97.5% +/- 0.0%  in 1273.8 seconds
For parameter size = 151 ,  Accuracy=97.4% +/- 0.2%  in 1327.7 seconds
For parameter size = 180 ,  Accuracy=98.4% +/- 0.4%  in 1399.5 seconds
For parameter size = 214 ,  Accuracy=97.5% +/- 0.0%  in 1494.6 seconds
For parameter size = 254 ,  Accuracy=92.8% +/- 0.4%  in 1625.1 seconds
For parameter size = 302 ,  Accuracy=98.2% +/- 0.2%  in 1804.5 seconds
For parameter size = 360 ,  Accuracy=96.1% +/- 0.4%  in 2057.0 seconds
scanning over fullsize = [175, 208, 247, 294, 350, 416, 494, 588, 700]
For parameter fullsize = 175 ,  Accuracy=99.0% +/- 0.5%  in 1201.6 seconds
For parameter fullsize = 208 ,  Accuracy=95.8% +/- 0.3%  in 1225.3 seconds
For parameter fullsize = 247 ,  Accuracy=96.7% +/- 0.4%  in 1259.5 seconds
For parameter fullsize = 294 ,  Accuracy=97.3% +/- 0.4%  in 1322.9 seconds
For parameter fullsize = 350 ,  Accuracy=96.2% +/- 0.2%  in 1400.2 seconds
For parameter fullsize = 416 ,  Accuracy=97.8% +/- 0.3%  in 1506.9 seconds
For parameter fullsize = 494 ,  Accuracy=96.9% +/- 0.0%  in 1651.2 seconds
For parameter fullsize = 588 ,  Accuracy=94.9% +/- 0.2%  in 1849.7 seconds
For parameter fullsize = 700 ,  Accuracy=96.4% +/- 0.5%  in 2129.1 seconds
scanning over crop = [175, 208, 247, 294, 350, 416, 494, 588, 700]
For parameter crop = 175 ,  Accuracy=96.5% +/- 1.1%  in 1276.0 seconds
For parameter crop = 208 ,  Accuracy=94.0% +/- 0.8%  in 1292.7 seconds
For parameter crop = 247 ,  Accuracy=98.2% +/- 0.2%  in 1318.1 seconds
For parameter crop = 294 ,  Accuracy=96.8% +/- 0.2%  in 1351.7 seconds
For parameter crop = 350 ,  Accuracy=98.1% +/- 0.0%  in 1402.0 seconds
For parameter crop = 416 ,  Accuracy=98.7% +/- 0.0%  in 1476.9 seconds
For parameter crop = 494 ,  Accuracy=97.4% +/- 0.2%  in 1555.9 seconds
For parameter crop = 588 ,  Accuracy=99.9% +/- 0.3%  in 1664.4 seconds
For parameter crop = 700 ,  Accuracy=97.4% +/- 0.2%  in 1825.9 seconds
scanning over mean = [ 0.18        0.21405728  0.25455844  0.30272271  0.36        0.42811456
  0.50911688  0.60544542  0.72      ]
For parameter mean = 0.18 ,  Accuracy=95.6% +/- 0.0%  in 1400.2 seconds
For parameter mean = 0.2140572807 ,  Accuracy=95.6% +/- 0.0%  in 1400.4 seconds
For parameter mean = 0.254558441227 ,  Accuracy=94.7% +/- 0.4%  in 1401.2 seconds
For parameter mean = 0.302722709491 ,  Accuracy=98.1% +/- 0.0%  in 1401.7 seconds
For parameter mean = 0.36 ,  Accuracy=95.6% +/- 0.0%  in 1399.1 seconds
For parameter mean = 0.428114561401 ,  Accuracy=96.9% +/- 0.0%  in 1400.9 seconds
For parameter mean = 0.509116882454 ,  Accuracy=98.6% +/- 0.4%  in 1400.6 seconds
For parameter mean = 0.605445418983 ,  Accuracy=97.5% +/- 0.0%  in 1400.7 seconds
For parameter mean = 0.72 ,  Accuracy=94.5% +/- 0.3%  in 1400.6 seconds
scanning over std = [ 0.15        0.17838107  0.21213203  0.25226892  0.3         0.35676213
  0.42426407  0.50453785  0.6       ]
For parameter std = 0.15 ,  Accuracy=96.9% +/- 0.0%  in 1400.1 seconds
For parameter std = 0.17838106725 ,  Accuracy=98.0% +/- 0.3%  in 1399.5 seconds
For parameter std = 0.212132034356 ,  Accuracy=98.1% +/- 0.0%  in 1402.2 seconds
For parameter std = 0.252268924576 ,  Accuracy=95.8% +/- 0.4%  in 1399.6 seconds
For parameter std = 0.3 ,  Accuracy=97.5% +/- 0.0%  in 1401.8 seconds
For parameter std = 0.356762134501 ,  Accuracy=98.6% +/- 0.4%  in 1404.4 seconds
For parameter std = 0.424264068712 ,  Accuracy=96.2% +/- 0.0%  in 1401.3 seconds
For parameter std = 0.504537849152 ,  Accuracy=93.3% +/- 0.4%  in 1401.5 seconds
For parameter std = 0.6 ,  Accuracy=98.1% +/- 0.2%  in 1397.7 seconds

````

## LICENCE

All code is released under the GPL license, see the LICENSE file.

The database files of faces are distributed for the learning of the network, but should not be re-used outside this repository.

The precise license is https://creativecommons.org/licenses/by-nc-nd/4.0/

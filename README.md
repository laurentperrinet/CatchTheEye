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
$ ipython3 Regard.py
scanning over no_cuda = [True, False]
For parameter no_cuda = True ,  
For parameter no_cuda = False ,  Accuracy=95.0% +/- 0.0%  in 2163.6 seconds
--------------------------------------------------
 parameter scan
--------------------------------------------------
--------------------------------------------------
 base= 10
--------------------------------------------------
 parameter scan : learning
--------------------------------------------------
Using SGD
--------------------------------------------------
scanning over lr = [ 0.002       0.00355656  0.00632456  0.01124683  0.02        0.03556559
  0.06324555  0.11246827  0.2       ]
For parameter lr = 0.002 ,  Accuracy=94.6% +/- 5.0%  in 2162.8 seconds
For parameter lr = 0.004 ,  Accuracy=94.9% +/- 1.4%  in 2162.5 seconds
For parameter lr = 0.006 ,  Accuracy=98.7% +/- 0.8%  in 2165.9 seconds
For parameter lr = 0.011 ,  Accuracy=94.9% +/- 0.1%  in 2168.4 seconds
For parameter lr = 0.020 ,  Accuracy=95.0% +/- 0.2%  in 2166.5 seconds
For parameter lr = 0.036 ,  Accuracy=97.5% +/- 0.0%  in 2170.0 seconds
For parameter lr = 0.063 ,  Accuracy=95.9% +/- 0.3%  in 2167.5 seconds
For parameter lr = 0.112 ,  Accuracy=11.3% +/- 0.0%  in 2166.5 seconds
For parameter lr = 0.200 ,  Accuracy=6.3% +/- 0.0%  in 2169.5 seconds
scanning over momentum = [ 0.01        0.01778279  0.03162278  0.05623413  0.1         0.17782794
  0.31622777  0.56234133  1.        ]
For parameter momentum = 0.010 ,  Accuracy=96.5% +/- 0.3%  in 2163.1 seconds
For parameter momentum = 0.018 ,  Accuracy=96.6% +/- 0.3%  in 2169.7 seconds
For parameter momentum = 0.032 ,  Accuracy=96.1% +/- 0.3%  in 2168.1 seconds
For parameter momentum = 0.056 ,  Accuracy=96.2% +/- 0.3%  in 2167.7 seconds
For parameter momentum = 0.100 ,  Accuracy=92.5% +/- 0.1%  in 2165.7 seconds
For parameter momentum = 0.178 ,  Accuracy=98.4% +/- 0.6%  in 2165.9 seconds
For parameter momentum = 0.316 ,  Accuracy=96.2% +/- 0.3%  in 2166.2 seconds
For parameter momentum = 0.562 ,  Accuracy=95.7% +/- 0.4%  in 2166.3 seconds
For parameter momentum = 1.000 ,  Accuracy=30.0% +/- 3.5%  in 2168.1 seconds
scanning over batch_size = [1, 2, 5, 8, 16, 28, 50, 89, 160]
For parameter batch_size = 1 ,  Accuracy=31.4% +/- 0.0%  in 2566.2 seconds
For parameter batch_size = 2 ,  Accuracy=95.1% +/- 0.2%  in 2285.3 seconds
For parameter batch_size = 5 ,  Accuracy=98.2% +/- 0.2%  in 2196.0 seconds
For parameter batch_size = 8 ,  Accuracy=98.7% +/- 0.0%  in 2181.3 seconds
For parameter batch_size = 16 ,  Accuracy=97.4% +/- 0.2%  in 2168.9 seconds
For parameter batch_size = 28 ,  Accuracy=96.4% +/- 0.3%  in 2157.1 seconds
For parameter batch_size = 50 ,  Accuracy=95.9% +/- 0.8%  in 2161.1 seconds
For parameter batch_size = 89 ,  Accuracy=94.3% +/- 12.0%  in 2272.1 seconds
For parameter batch_size = 160 ,  Accuracy=92.4% +/- 8.6%  in 2313.5 seconds
scanning over epochs = [4, 7, 12, 22, 40, 71, 126, 224, 400]
For parameter epochs = 4 ,  Accuracy=92.1% +/- 12.8%  in 231.2 seconds
For parameter epochs = 7 ,  Accuracy=96.2% +/- 4.8%  in 393.2 seconds
For parameter epochs = 12 ,  Accuracy=97.4% +/- 0.7%  in 661.4 seconds
For parameter epochs = 22 ,  Accuracy=96.8% +/- 0.4%  in 1203.4 seconds
For parameter epochs = 40 ,  Accuracy=98.7% +/- 0.0%  in 2166.7 seconds
For parameter epochs = 71 ,  Accuracy=97.4% +/- 0.2%  in 3834.5 seconds
For parameter epochs = 126 ,  Accuracy=98.1% +/- 0.0%  in 6800.7 seconds
For parameter epochs = 224 ,  Accuracy=96.9% +/- 0.0%  in 12058.8 seconds
For parameter epochs = 400 ,  Accuracy=98.0% +/- 0.2%  in 21926.0 seconds
--------------------------------------------------
Using ADAM
--------------------------------------------------
scanning over lr = [ 0.01        0.01189207  0.01414214  0.01681793  0.02        0.02378414
  0.02828427  0.03363586  0.04      ]
For parameter lr = 0.010 ,  Accuracy=33.3% +/- 0.0%  in 2264.3 seconds
For parameter lr = 0.012 ,  Accuracy=32.7% +/- 0.0%  in 2269.8 seconds
For parameter lr = 0.014 ,  Accuracy=23.3% +/- 0.0%  in 2270.0 seconds
For parameter lr = 0.017 ,  Accuracy=39.0% +/- 0.0%  in 2268.2 seconds
For parameter lr = 0.020 ,  Accuracy=95.0% +/- 0.2%  in 2166.5 seconds
For parameter lr = 0.024 ,  For parameter lr = 0.028 ,  Accuracy=33.5% +/- 0.4%  in 2283.1 seconds
For parameter lr = 0.034 ,  For parameter lr = 0.040 ,  Accuracy=25.9% +/- 2.3%  in 2283.7 seconds
scanning over momentum = [ 0.05        0.05946036  0.07071068  0.08408964  0.1         0.11892071
  0.14142136  0.16817928  0.2       ]
For parameter momentum = 0.050 ,  For parameter momentum = 0.059 ,  Accuracy=23.3% +/- 0.0%  in 2281.1 seconds
For parameter momentum = 0.071 ,  For parameter momentum = 0.084 ,  Accuracy=39.0% +/- 0.0%  in 2285.3 seconds
For parameter momentum = 0.100 ,  Accuracy=92.5% +/- 0.1%  in 2165.7 seconds
For parameter momentum = 0.119 ,  For parameter momentum = 0.141 ,  Accuracy=33.3% +/- 0.0%  in 2285.6 seconds
For parameter momentum = 0.168 ,  For parameter momentum = 0.200 ,  Accuracy=32.7% +/- 0.0%  in 2233.0 seconds
scanning over batch_size = [8, 9, 11, 13, 16, 19, 22, 26, 32]
For parameter batch_size = 8 ,  Accuracy=98.7% +/- 0.0%  in 2181.3 seconds
For parameter batch_size = 9 ,  For parameter batch_size = 11 ,  Accuracy=32.1% +/- 0.0%  in 2287.8 seconds
For parameter batch_size = 13 ,  For parameter batch_size = 16 ,  Accuracy=97.4% +/- 0.2%  in 2168.9 seconds
For parameter batch_size = 19 ,  Accuracy=33.3% +/- 0.0%  in 2258.1 seconds
For parameter batch_size = 22 ,  For parameter batch_size = 26 ,  Accuracy=35.8% +/- 0.0%  in 2305.6 seconds
For parameter batch_size = 32 ,  scanning over epochs = [20, 23, 28, 33, 40, 47, 56, 67, 80]
For parameter epochs = 20 ,  Accuracy=35.8% +/- 0.0%  in 1160.3 seconds
For parameter epochs = 23 ,  Accuracy=35.2% +/- 0.0%  in 1249.3 seconds
For parameter epochs = 28 ,  Accuracy=25.2% +/- 0.0%  in 1516.9 seconds
For parameter epochs = 33 ,  Accuracy=32.1% +/- 0.0%  in 1785.4 seconds
For parameter epochs = 40 ,  Accuracy=98.7% +/- 0.0%  in 2166.7 seconds
For parameter epochs = 47 ,  Accuracy=22.6% +/- 0.0%  in 2535.1 seconds
For parameter epochs = 56 ,  Accuracy=33.3% +/- 0.0%  in 3020.0 seconds
For parameter epochs = 67 ,  Accuracy=33.3% +/- 0.0%  in 3610.2 seconds
For parameter epochs = 80 ,  Accuracy=35.2% +/- 0.0%  in 4279.1 seconds
--------------------------------------------------
 parameter scan : network
--------------------------------------------------
scanning over conv1_kernel_size = [3, 4, 4, 5, 7, 8, 9, 11, 14]
For parameter conv1_kernel_size = 3 ,  Accuracy=96.6% +/- 0.4%  in 2157.3 seconds
For parameter conv1_kernel_size = 4 ,  Accuracy=96.5% +/- 0.3%  in 2159.2 seconds
For parameter conv1_kernel_size = 4 ,  Accuracy=96.5% +/- 0.3%  in 2159.2 seconds
For parameter conv1_kernel_size = 5 ,  Accuracy=100.0% +/- 0.1%  in 2162.2 seconds
For parameter conv1_kernel_size = 7 ,  Accuracy=94.9% +/- 0.1%  in 2146.4 seconds
For parameter conv1_kernel_size = 8 ,  Accuracy=94.9% +/- 0.2%  in 2160.4 seconds
For parameter conv1_kernel_size = 9 ,  Accuracy=96.9% +/- 0.1%  in 2156.5 seconds
For parameter conv1_kernel_size = 11 ,  Accuracy=98.1% +/- 0.1%  in 2161.1 seconds
For parameter conv1_kernel_size = 14 ,  Accuracy=96.8% +/- 0.1%  in 2155.6 seconds
scanning over conv1_dim = [4, 4, 5, 6, 8, 9, 11, 13, 16]
For parameter conv1_dim = 4 ,  Accuracy=98.1% +/- 0.0%  in 2160.6 seconds
For parameter conv1_dim = 4 ,  Accuracy=98.1% +/- 0.0%  in 2160.6 seconds
For parameter conv1_dim = 5 ,  Accuracy=96.4% +/- 0.3%  in 2146.3 seconds
For parameter conv1_dim = 6 ,  Accuracy=95.9% +/- 0.7%  in 2158.4 seconds
For parameter conv1_dim = 8 ,  Accuracy=95.6% +/- 0.0%  in 2161.9 seconds
For parameter conv1_dim = 9 ,  Accuracy=97.6% +/- 0.3%  in 2160.3 seconds
For parameter conv1_dim = 11 ,  Accuracy=97.4% +/- 0.3%  in 2160.2 seconds
For parameter conv1_dim = 13 ,  Accuracy=98.1% +/- 0.0%  in 2158.1 seconds
For parameter conv1_dim = 16 ,  Accuracy=96.2% +/- 0.0%  in 2158.9 seconds
scanning over conv2_kernel_size = [3, 4, 4, 5, 7, 8, 9, 11, 14]
For parameter conv2_kernel_size = 3 ,  Accuracy=93.1% +/- 0.5%  in 2157.5 seconds
For parameter conv2_kernel_size = 4 ,  Accuracy=95.0% +/- 0.2%  in 2157.8 seconds
For parameter conv2_kernel_size = 4 ,  Accuracy=95.0% +/- 0.2%  in 2157.8 seconds
For parameter conv2_kernel_size = 5 ,  Accuracy=95.0% +/- 0.6%  in 2160.2 seconds
For parameter conv2_kernel_size = 7 ,  Accuracy=98.1% +/- 0.0%  in 2158.9 seconds
For parameter conv2_kernel_size = 8 ,  Accuracy=98.7% +/- 0.0%  in 2156.7 seconds
For parameter conv2_kernel_size = 9 ,  Accuracy=99.4% +/- 0.0%  in 2156.0 seconds
For parameter conv2_kernel_size = 11 ,  Accuracy=97.5% +/- 0.0%  in 2157.0 seconds
For parameter conv2_kernel_size = 14 ,  Accuracy=97.3% +/- 0.4%  in 2164.5 seconds
scanning over conv2_dim = [6, 7, 9, 10, 13, 15, 18, 21, 26]
For parameter conv2_dim = 6 ,  Accuracy=97.5% +/- 0.0%  in 2156.5 seconds
For parameter conv2_dim = 7 ,  Accuracy=97.5% +/- 0.1%  in 2157.0 seconds
For parameter conv2_dim = 9 ,  Accuracy=94.8% +/- 0.3%  in 2155.6 seconds
For parameter conv2_dim = 10 ,  Accuracy=98.6% +/- 0.4%  in 2154.5 seconds
For parameter conv2_dim = 13 ,  Accuracy=98.1% +/- 0.0%  in 2160.0 seconds
For parameter conv2_dim = 15 ,  Accuracy=98.1% +/- 0.1%  in 2156.4 seconds
For parameter conv2_dim = 18 ,  Accuracy=96.8% +/- 0.1%  in 2159.2 seconds
For parameter conv2_dim = 21 ,  Accuracy=98.1% +/- 0.1%  in 2156.4 seconds
For parameter conv2_dim = 26 ,  Accuracy=98.1% +/- 0.1%  in 2157.4 seconds
scanning over stride1 = [1, 1, 1, 1, 2, 2, 2, 3, 4]
For parameter stride1 = 1 ,  Accuracy=96.2% +/- 0.1%  in 2440.4 seconds
For parameter stride1 = 1 ,  Accuracy=96.2% +/- 0.1%  in 2440.4 seconds
For parameter stride1 = 1 ,  Accuracy=96.2% +/- 0.1%  in 2440.4 seconds
For parameter stride1 = 1 ,  Accuracy=96.2% +/- 0.1%  in 2440.4 seconds
For parameter stride1 = 2 ,  Accuracy=96.9% +/- 0.0%  in 2154.5 seconds
For parameter stride1 = 2 ,  Accuracy=96.9% +/- 0.0%  in 2154.5 seconds
For parameter stride1 = 2 ,  Accuracy=96.9% +/- 0.0%  in 2154.5 seconds
For parameter stride1 = 3 ,  Accuracy=97.4% +/- 0.4%  in 2158.8 seconds
For parameter stride1 = 4 ,  Accuracy=98.7% +/- 0.0%  in 2153.5 seconds
scanning over stride2 = [2, 2, 2, 3, 4, 4, 5, 6, 8]
For parameter stride2 = 2 ,  Accuracy=95.5% +/- 0.2%  in 2158.6 seconds
For parameter stride2 = 2 ,  Accuracy=95.5% +/- 0.2%  in 2158.6 seconds
For parameter stride2 = 2 ,  Accuracy=95.5% +/- 0.2%  in 2158.6 seconds
For parameter stride2 = 3 ,  Accuracy=95.8% +/- 0.3%  in 2157.9 seconds
For parameter stride2 = 4 ,  Accuracy=96.4% +/- 0.3%  in 2158.6 seconds
For parameter stride2 = 4 ,  Accuracy=96.4% +/- 0.3%  in 2158.6 seconds
For parameter stride2 = 5 ,  Accuracy=95.0% +/- 0.0%  in 2160.6 seconds
For parameter stride2 = 6 ,  Accuracy=96.4% +/- 0.6%  in 2159.8 seconds
For parameter stride2 = 8 ,  Accuracy=96.0% +/- 0.6%  in 2159.5 seconds
scanning over dimension = [12, 14, 17, 21, 25, 29, 35, 42, 50]
For parameter dimension = 12 ,  Accuracy=96.9% +/- 0.0%  in 2162.0 seconds
For parameter dimension = 14 ,  Accuracy=98.6% +/- 0.2%  in 2159.3 seconds
For parameter dimension = 17 ,  Accuracy=98.1% +/- 0.0%  in 2160.5 seconds
For parameter dimension = 21 ,  Accuracy=96.2% +/- 0.0%  in 2159.7 seconds
For parameter dimension = 25 ,  Accuracy=96.0% +/- 0.5%  in 2158.1 seconds
For parameter dimension = 29 ,  Accuracy=97.9% +/- 0.3%  in 2157.9 seconds
For parameter dimension = 35 ,  Accuracy=97.5% +/- 0.0%  in 2158.7 seconds
For parameter dimension = 42 ,  Accuracy=96.9% +/- 0.0%  in 2158.0 seconds
For parameter dimension = 50 ,  Accuracy=98.1% +/- 0.1%  in 2157.5 seconds
--------------------------------------------------
 parameter scan : data
--------------------------------------------------
scanning over size = [90, 107, 127, 151, 180, 214, 254, 302, 360]
For parameter size = 90 ,  Accuracy=95.5% +/- 0.4%  in 1801.2 seconds
For parameter size = 107 ,  Accuracy=92.6% +/- 0.3%  in 1853.7 seconds
For parameter size = 127 ,  Accuracy=97.3% +/- 0.3%  in 1914.8 seconds
For parameter size = 151 ,  Accuracy=98.4% +/- 0.3%  in 2019.0 seconds
For parameter size = 180 ,  Accuracy=97.5% +/- 0.3%  in 2156.8 seconds
For parameter size = 214 ,  Accuracy=96.8% +/- 0.2%  in 2350.5 seconds
For parameter size = 254 ,  Accuracy=95.5% +/- 0.2%  in 2611.2 seconds
For parameter size = 302 ,  Accuracy=96.9% +/- 0.0%  in 3022.7 seconds
For parameter size = 360 ,  Accuracy=97.9% +/- 0.3%  in 4472.9 seconds
scanning over fullsize = [87, 104, 123, 147, 175, 208, 247, 294, 350]
For parameter fullsize = 87 ,  Accuracy=98.7% +/- 0.0%  in 2050.0 seconds
For parameter fullsize = 104 ,  Accuracy=98.2% +/- 0.2%  in 2053.9 seconds
For parameter fullsize = 123 ,  Accuracy=97.5% +/- 0.1%  in 2088.5 seconds
For parameter fullsize = 147 ,  Accuracy=98.5% +/- 0.6%  in 2119.4 seconds
For parameter fullsize = 175 ,  Accuracy=96.0% +/- 0.4%  in 2158.8 seconds
For parameter fullsize = 208 ,  Accuracy=96.3% +/- 0.2%  in 2207.1 seconds
For parameter fullsize = 247 ,  Accuracy=96.7% +/- 0.7%  in 2269.9 seconds
For parameter fullsize = 294 ,  Accuracy=95.1% +/- 1.9%  in 2396.9 seconds
For parameter fullsize = 350 ,  Accuracy=92.8% +/- 1.1%  in 2556.4 seconds
scanning over crop = [87, 104, 123, 147, 175, 208, 247, 294, 350]
For parameter crop = 87 ,  Accuracy=98.1% +/- 0.0%  in 2093.3 seconds
For parameter crop = 104 ,  Accuracy=94.9% +/- 0.4%  in 2103.8 seconds
For parameter crop = 123 ,  Accuracy=95.2% +/- 0.5%  in 2118.2 seconds
For parameter crop = 147 ,  Accuracy=98.3% +/- 0.4%  in 2137.6 seconds
For parameter crop = 175 ,  Accuracy=96.5% +/- 0.3%  in 2158.5 seconds
For parameter crop = 208 ,  Accuracy=96.9% +/- 0.0%  in 2189.8 seconds
For parameter crop = 247 ,  Accuracy=96.8% +/- 0.4%  in 2242.2 seconds
For parameter crop = 294 ,  Accuracy=98.0% +/- 0.5%  in 2309.3 seconds
For parameter crop = 350 ,  Accuracy=99.8% +/- 0.4%  in 2390.0 seconds
scanning over mean = [ 0.18        0.21405728  0.25455844  0.30272271  0.36        0.42811456
  0.50911688  0.60544542  0.72      ]
For parameter mean = 0.180 ,  Accuracy=96.9% +/- 0.0%  in 2147.1 seconds
For parameter mean = 0.214 ,  Accuracy=98.1% +/- 0.1%  in 2148.3 seconds
For parameter mean = 0.255 ,  Accuracy=95.6% +/- 0.0%  in 2161.8 seconds
For parameter mean = 0.303 ,  Accuracy=96.1% +/- 0.3%  in 2158.5 seconds
For parameter mean = 0.360 ,  Accuracy=96.6% +/- 0.3%  in 2161.1 seconds
For parameter mean = 0.428 ,  Accuracy=96.9% +/- 0.0%  in 2158.6 seconds
For parameter mean = 0.509 ,  Accuracy=97.3% +/- 0.4%  in 2154.3 seconds
For parameter mean = 0.605 ,  ^[
````

## LICENCE

All code is released under the GPL license, see the LICENSE file.

The database files of faces are distributed for the learning of the network, but should not be re-used outside this repository.

The precise license is https://creativecommons.org/licenses/by-nc-nd/4.0/

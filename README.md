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
Scanning over no_cuda = [True]
For parameter no_cuda = True ,  Accuracy=96.4% +/- 0.8%  in 837.5 seconds
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
scanning over lr = [0.0035     0.00622398 0.01106797 0.01968195 0.035      0.06223978
 0.11067972 0.19681946 0.35      ]
For parameter lr = 0.004 ,  Accuracy=33.6% +/- 5.8%  in 860.1 seconds
For parameter lr = 0.006 ,  Accuracy=66.4% +/- 4.9%  in 849.4 seconds
For parameter lr = 0.011 ,  Accuracy=91.7% +/- 1.8%  in 849.7 seconds
For parameter lr = 0.020 ,  Accuracy=95.3% +/- 1.2%  in 861.4 seconds
For parameter lr = 0.035 ,  Accuracy=96.6% +/- 0.8%  in 866.1 seconds
For parameter lr = 0.062 ,  Accuracy=74.0% +/- 31.7%  in 1514.4 seconds
For parameter lr = 0.111 ,  Accuracy=36.9% +/- 23.0%  in 1513.7 seconds
For parameter lr = 0.197 ,  Accuracy=36.9% +/- 18.8%  in 878.8 seconds
For parameter lr = 0.350 ,  Accuracy=26.1% +/- 2.5%  in 880.9 seconds
scanning over momentum = [0.005      0.0088914  0.01581139 0.02811707 0.05       0.08891397
 0.15811388 0.28117066 0.5       ]
For parameter momentum = 0.005 ,  Accuracy=96.5% +/- 1.9%  in 884.6 seconds
For parameter momentum = 0.009 ,  Accuracy=96.4% +/- 1.1%  in 884.1 seconds
For parameter momentum = 0.016 ,  Accuracy=94.7% +/- 6.6%  in 1428.5 seconds
For parameter momentum = 0.028 ,  Accuracy=96.5% +/- 1.4%  in 1430.4 seconds
For parameter momentum = 0.050 ,  Accuracy=96.6% +/- 0.8%  in 886.1 seconds
For parameter momentum = 0.089 ,  Accuracy=97.1% +/- 1.0%  in 886.8 seconds
For parameter momentum = 0.158 ,  Accuracy=96.8% +/- 0.8%  in 885.5 seconds
For parameter momentum = 0.281 ,  Accuracy=97.0% +/- 1.5%  in 1428.6 seconds
For parameter momentum = 0.500 ,  Accuracy=88.9% +/- 21.2%  in 1423.5 seconds
scanning over batch_size = [0, 1, 2, 4, 8, 14, 25, 44, 80]
For parameter batch_size = 0 ,   currently locked with  _tmp_scanning_batch_size__0.npy_lock
For parameter batch_size = 1 ,  Accuracy=28.1% +/- 7.1%  in 940.8 seconds
For parameter batch_size = 2 ,  Accuracy=46.4% +/- 29.8%  in 778.7 seconds
For parameter batch_size = 4 ,  Accuracy=96.8% +/- 1.7%  in 1485.7 seconds
For parameter batch_size = 8 ,  Accuracy=96.5% +/- 0.8%  in 1417.7 seconds
For parameter batch_size = 14 ,  Accuracy=95.3% +/- 1.5%  in 723.2 seconds
For parameter batch_size = 25 ,  Accuracy=88.0% +/- 4.7%  in 716.6 seconds
For parameter batch_size = 44 ,  Accuracy=61.4% +/- 7.4%  in 1838.6 seconds
For parameter batch_size = 80 ,  Accuracy=38.4% +/- 8.0%  in 2026.1 seconds
scanning over epochs = [2, 3, 6, 11, 20, 35, 63, 112, 200]
For parameter epochs = 2 ,  Accuracy=28.0% +/- 3.9%  in 81.0 seconds
For parameter epochs = 3 ,  Accuracy=33.5% +/- 7.0%  in 117.0 seconds
For parameter epochs = 6 ,  Accuracy=74.0% +/- 7.5%  in 224.5 seconds
For parameter epochs = 11 ,  Accuracy=93.4% +/- 2.4%  in 402.2 seconds
For parameter epochs = 20 ,  Accuracy=96.6% +/- 0.8%  in 722.8 seconds
For parameter epochs = 35 ,  Accuracy=96.6% +/- 0.8%  in 2378.0 seconds
For parameter epochs = 63 ,  Accuracy=96.9% +/- 0.9%  in 2252.0 seconds
For parameter epochs = 112 ,  Accuracy=96.7% +/- 1.0%  in 7698.7 seconds
For parameter epochs = 200 ,  Accuracy=96.9% +/- 1.0%  in 13544.5 seconds
--------------------------------------------------
Using ADAM
--------------------------------------------------
scanning over lr = [0.0175     0.02081112 0.02474874 0.02943137 0.035      0.04162225
 0.04949747 0.05886275 0.07      ]
For parameter lr = 0.018 ,  Accuracy=25.8% +/- 0.0%  in 1487.6 seconds
For parameter lr = 0.021 ,  Accuracy=25.8% +/- 0.0%  in 1473.3 seconds
For parameter lr = 0.025 ,  Accuracy=25.8% +/- 0.0%  in 816.7 seconds
For parameter lr = 0.029 ,  Accuracy=25.8% +/- 0.0%  in 795.5 seconds
For parameter lr = 0.035 ,  Accuracy=25.8% +/- 0.0%  in 1470.9 seconds
For parameter lr = 0.042 ,  Accuracy=25.8% +/- 0.0%  in 796.8 seconds
For parameter lr = 0.049 ,  Accuracy=25.8% +/- 0.0%  in 797.2 seconds
For parameter lr = 0.059 ,  Accuracy=25.8% +/- 0.0%  in 1429.2 seconds
For parameter lr = 0.070 ,  Accuracy=25.7% +/- 0.4%  in 801.7 seconds
scanning over momentum = [0.025      0.02973018 0.03535534 0.04204482 0.05       0.05946036
 0.07071068 0.08408964 0.1       ]
For parameter momentum = 0.025 ,  Accuracy=25.8% +/- 0.0%  in 798.6 seconds
For parameter momentum = 0.030 ,  Accuracy=25.8% +/- 0.0%  in 1464.2 seconds
For parameter momentum = 0.035 ,  Accuracy=25.8% +/- 0.0%  in 796.5 seconds
For parameter momentum = 0.042 ,  Accuracy=25.8% +/- 0.0%  in 797.2 seconds
For parameter momentum = 0.050 ,  Accuracy=25.8% +/- 0.0%  in 1169.9 seconds
For parameter momentum = 0.059 ,  Accuracy=25.8% +/- 0.0%  in 795.9 seconds
For parameter momentum = 0.071 ,  Accuracy=25.8% +/- 0.0%  in 944.4 seconds
For parameter momentum = 0.084 ,  Accuracy=25.8% +/- 0.0%  in 798.1 seconds
For parameter momentum = 0.100 ,  Accuracy=25.8% +/- 0.0%  in 1273.2 seconds
scanning over batch_size = [4, 4, 5, 6, 8, 9, 11, 13, 16]
For parameter batch_size = 4 ,  Accuracy=25.8% +/- 0.0%  in 1007.8 seconds
For parameter batch_size = 4 ,  Accuracy=25.8% +/- 0.0%  in 1007.8 seconds
For parameter batch_size = 5 ,  Accuracy=25.8% +/- 0.0%  in 1610.4 seconds
For parameter batch_size = 6 ,  Accuracy=25.8% +/- 0.0%  in 858.5 seconds
For parameter batch_size = 8 ,  Accuracy=25.8% +/- 0.0%  in 1395.6 seconds
For parameter batch_size = 9 ,  Accuracy=25.8% +/- 0.0%  in 1413.1 seconds
For parameter batch_size = 11 ,  Accuracy=25.8% +/- 0.0%  in 803.6 seconds
For parameter batch_size = 13 ,  Accuracy=25.8% +/- 0.0%  in 1606.7 seconds
For parameter batch_size = 16 ,  Accuracy=25.8% +/- 0.0%  in 737.5 seconds
scanning over epochs = [10, 11, 14, 16, 20, 23, 28, 33, 40]
For parameter epochs = 10 ,  Accuracy=25.8% +/- 0.0%  in 719.4 seconds
For parameter epochs = 11 ,  Accuracy=25.8% +/- 0.0%  in 412.9 seconds
For parameter epochs = 14 ,  Accuracy=25.8% +/- 0.0%  in 558.6 seconds
For parameter epochs = 16 ,  Accuracy=25.8% +/- 0.0%  in 1151.6 seconds
For parameter epochs = 20 ,  Accuracy=25.8% +/- 0.0%  in 1412.9 seconds
For parameter epochs = 23 ,  Accuracy=25.8% +/- 0.0%  in 904.6 seconds
For parameter epochs = 28 ,  Accuracy=25.8% +/- 0.0%  in 2010.2 seconds
For parameter epochs = 33 ,  Accuracy=25.8% +/- 0.0%  in 1488.9 seconds
For parameter epochs = 40 ,  Accuracy=25.8% +/- 0.0%  in 2898.4 seconds
--------------------------------------------------
 parameter scan : network
--------------------------------------------------
scanning over conv1_kernel_size = [3, 4, 4, 5, 7, 8, 9, 11, 14]
For parameter conv1_kernel_size = 3 ,  Accuracy=81.1% +/- 7.2%  in 878.0 seconds
For parameter conv1_kernel_size = 4 ,  Accuracy=95.7% +/- 1.0%  in 833.7 seconds
For parameter conv1_kernel_size = 4 ,  Accuracy=95.7% +/- 1.0%  in 833.7 seconds
For parameter conv1_kernel_size = 5 ,  Accuracy=97.7% +/- 1.0%  in 1241.9 seconds
For parameter conv1_kernel_size = 7 ,  Accuracy=96.6% +/- 0.8%  in 872.4 seconds
For parameter conv1_kernel_size = 8 ,  Accuracy=95.4% +/- 2.6%  in 912.4 seconds
For parameter conv1_kernel_size = 9 ,  Accuracy=95.6% +/- 2.1%  in 896.6 seconds
For parameter conv1_kernel_size = 11 ,  Accuracy=97.2% +/- 1.2%  in 1526.4 seconds
For parameter conv1_kernel_size = 14 ,  Accuracy=92.9% +/- 2.5%  in 1813.0 seconds
scanning over conv1_dim = [3, 4, 4, 5, 7, 8, 9, 11, 14]
For parameter conv1_dim = 3 ,  Accuracy=95.0% +/- 2.6%  in 840.8 seconds
For parameter conv1_dim = 4 ,  Accuracy=94.6% +/- 3.6%  in 821.1 seconds
For parameter conv1_dim = 4 ,  Accuracy=94.6% +/- 3.6%  in 821.1 seconds
For parameter conv1_dim = 5 ,  Accuracy=94.9% +/- 1.6%  in 845.4 seconds
For parameter conv1_dim = 7 ,  Accuracy=96.5% +/- 0.8%  in 1355.7 seconds
For parameter conv1_dim = 8 ,  Accuracy=96.9% +/- 1.2%  in 891.6 seconds
For parameter conv1_dim = 9 ,  Accuracy=96.1% +/- 0.9%  in 1213.3 seconds
For parameter conv1_dim = 11 ,  Accuracy=97.0% +/- 1.0%  in 934.7 seconds
For parameter conv1_dim = 14 ,  Accuracy=96.9% +/- 1.1%  in 950.4 seconds
scanning over conv2_kernel_size = [3, 4, 4, 5, 7, 8, 9, 11, 14]
For parameter conv2_kernel_size = 3 ,  Accuracy=93.9% +/- 2.9%  in 856.6 seconds
For parameter conv2_kernel_size = 4 ,  Accuracy=93.3% +/- 2.1%  in 865.3 seconds
For parameter conv2_kernel_size = 4 ,  Accuracy=93.3% +/- 2.1%  in 865.3 seconds
For parameter conv2_kernel_size = 5 ,  Accuracy=91.1% +/- 2.2%  in 853.8 seconds
For parameter conv2_kernel_size = 7 ,  Accuracy=96.5% +/- 0.8%  in 871.5 seconds
For parameter conv2_kernel_size = 8 ,  Accuracy=97.3% +/- 0.7%  in 888.8 seconds
For parameter conv2_kernel_size = 9 ,  Accuracy=95.3% +/- 5.0%  in 923.2 seconds
For parameter conv2_kernel_size = 11 ,  Accuracy=96.2% +/- 1.6%  in 887.6 seconds
For parameter conv2_kernel_size = 14 ,  Accuracy=83.4% +/- 26.8%  in 936.7 seconds
scanning over conv2_dim = [6, 7, 9, 10, 13, 15, 18, 21, 26]
For parameter conv2_dim = 6 ,  Accuracy=96.1% +/- 1.6%  in 900.4 seconds
For parameter conv2_dim = 7 ,  Accuracy=95.6% +/- 2.6%  in 878.4 seconds
For parameter conv2_dim = 9 ,  Accuracy=94.6% +/- 1.3%  in 869.9 seconds
For parameter conv2_dim = 10 ,  Accuracy=92.9% +/- 13.1%  in 895.1 seconds
For parameter conv2_dim = 13 ,  Accuracy=96.5% +/- 0.8%  in 874.7 seconds
For parameter conv2_dim = 15 ,  Accuracy=96.8% +/- 1.1%  in 915.9 seconds
For parameter conv2_dim = 18 ,  Accuracy=96.9% +/- 0.9%  in 902.9 seconds
For parameter conv2_dim = 21 ,  Accuracy=96.8% +/- 1.8%  in 878.6 seconds
For parameter conv2_dim = 26 ,  Accuracy=96.2% +/- 2.2%  in 954.1 seconds
scanning over stride1 = [1, 1, 1, 1, 2, 2, 2, 3, 4]
For parameter stride1 = 1 ,  Accuracy=94.1% +/- 1.9%  in 1363.2 seconds
For parameter stride1 = 1 ,  Accuracy=94.1% +/- 1.9%  in 1363.2 seconds
For parameter stride1 = 1 ,  Accuracy=94.1% +/- 1.9%  in 1363.2 seconds
For parameter stride1 = 1 ,  Accuracy=94.1% +/- 1.9%  in 1363.2 seconds
For parameter stride1 = 2 ,  Accuracy=96.5% +/- 0.8%  in 904.8 seconds
For parameter stride1 = 2 ,  Accuracy=96.5% +/- 0.8%  in 904.8 seconds
For parameter stride1 = 2 ,  Accuracy=96.5% +/- 0.8%  in 904.8 seconds
For parameter stride1 = 3 ,  Accuracy=96.3% +/- 1.3%  in 980.9 seconds
For parameter stride1 = 4 ,  Accuracy=94.6% +/- 2.1%  in 857.8 seconds
scanning over stride2 = [2, 2, 2, 3, 4, 4, 5, 6, 8]
For parameter stride2 = 2 ,  Accuracy=94.8% +/- 1.3%  in 919.5 seconds
For parameter stride2 = 2 ,  Accuracy=94.8% +/- 1.3%  in 919.5 seconds
For parameter stride2 = 2 ,  Accuracy=94.8% +/- 1.3%  in 919.5 seconds
For parameter stride2 = 3 ,  Accuracy=96.1% +/- 1.2%  in 921.1 seconds
For parameter stride2 = 4 ,  Accuracy=96.5% +/- 0.8%  in 874.2 seconds
For parameter stride2 = 4 ,  Accuracy=96.5% +/- 0.8%  in 874.2 seconds
For parameter stride2 = 5 ,  Accuracy=96.1% +/- 3.6%  in 918.4 seconds
For parameter stride2 = 6 ,  Accuracy=97.9% +/- 1.4%  in 912.1 seconds
For parameter stride2 = 8 ,  Accuracy=96.8% +/- 1.5%  in 876.6 seconds
scanning over dimension = [15, 17, 21, 25, 30, 35, 42, 50, 60]
For parameter dimension = 15 ,  Accuracy=95.0% +/- 2.7%  in 895.1 seconds
For parameter dimension = 17 ,  Accuracy=96.4% +/- 1.4%  in 888.3 seconds
For parameter dimension = 21 ,  Accuracy=94.8% +/- 1.8%  in 876.2 seconds
For parameter dimension = 25 ,  Accuracy=97.3% +/- 1.3%  in 864.1 seconds
For parameter dimension = 30 ,  Accuracy=96.6% +/- 0.8%  in 861.3 seconds
For parameter dimension = 35 ,  Accuracy=96.4% +/- 1.4%  in 875.1 seconds
For parameter dimension = 42 ,  Accuracy=91.1% +/- 15.4%  in 878.9 seconds
For parameter dimension = 50 ,  Accuracy=96.1% +/- 1.0%  in 870.0 seconds
For parameter dimension = 60 ,  Accuracy=95.3% +/- 2.5%  in 916.0 seconds
--------------------------------------------------
 parameter scan : data
--------------------------------------------------
scanning over size = [32, 38, 45, 53, 64, 76, 90, 107, 128]
For parameter size = 32 ,  Accuracy=94.4% +/- 2.0%  in 722.6 seconds
For parameter size = 38 ,  Accuracy=95.3% +/- 1.7%  in 714.0 seconds
For parameter size = 45 ,  Accuracy=94.1% +/- 5.4%  in 854.2 seconds
For parameter size = 53 ,  Accuracy=96.1% +/- 3.2%  in 837.4 seconds
For parameter size = 64 ,  Accuracy=96.6% +/- 0.8%  in 860.5 seconds
For parameter size = 76 ,  Accuracy=95.4% +/- 1.1%  in 994.2 seconds
For parameter size = 90 ,  Accuracy=89.3% +/- 12.5%  in 1370.8 seconds
For parameter size = 107 ,  Accuracy=95.4% +/- 1.5%  in 1868.7 seconds
For parameter size = 128 ,  Accuracy=94.6% +/- 2.0%  in 2412.1 seconds
scanning over fullsize = [32, 38, 45, 53, 64, 76, 90, 107, 128]
For parameter fullsize = 32 ,  Accuracy=85.2% +/- 5.1%  in 950.7 seconds
For parameter fullsize = 38 ,  Accuracy=89.3% +/- 6.7%  in 877.9 seconds
For parameter fullsize = 45 ,  Accuracy=93.0% +/- 5.2%  in 873.5 seconds
For parameter fullsize = 53 ,  Accuracy=94.9% +/- 1.7%  in 869.5 seconds
For parameter fullsize = 64 ,  Accuracy=96.6% +/- 0.8%  in 876.3 seconds
For parameter fullsize = 76 ,  Accuracy=94.8% +/- 5.6%  in 888.0 seconds
For parameter fullsize = 90 ,  Accuracy=95.0% +/- 1.9%  in 883.0 seconds
For parameter fullsize = 107 ,  Accuracy=89.4% +/- 3.1%  in 892.1 seconds
For parameter fullsize = 128 ,  Accuracy=77.6% +/- 7.2%  in 909.6 seconds
scanning over crop = [32, 38, 45, 53, 64, 76, 90, 107, 128]
For parameter crop = 32 ,  Accuracy=76.3% +/- 5.0%  in 888.2 seconds
For parameter crop = 38 ,  Accuracy=87.3% +/- 5.4%  in 866.8 seconds
For parameter crop = 45 ,  Accuracy=91.0% +/- 5.5%  in 890.1 seconds
For parameter crop = 53 ,  Accuracy=95.7% +/- 1.5%  in 875.1 seconds
For parameter crop = 64 ,  Accuracy=96.6% +/- 0.8%  in 848.3 seconds
For parameter crop = 76 ,  Accuracy=90.2% +/- 15.2%  in 895.8 seconds
For parameter crop = 90 ,  Accuracy=92.0% +/- 7.0%  in 873.8 seconds
For parameter crop = 107 ,  Accuracy=84.5% +/- 16.4%  in 880.8 seconds
For parameter crop = 128 ,  Accuracy=78.2% +/- 14.1%  in 917.6 seconds
scanning over mean = [0.18       0.21405728 0.25455844 0.30272271 0.36       0.42811456
 0.50911688 0.60544542 0.72      ]
For parameter mean = 0.180 ,  Accuracy=79.6% +/- 27.8%  in 849.7 seconds
For parameter mean = 0.214 ,  Accuracy=86.2% +/- 21.6%  in 850.0 seconds
For parameter mean = 0.255 ,  Accuracy=92.0% +/- 15.3%  in 874.9 seconds
For parameter mean = 0.303 ,  Accuracy=92.9% +/- 15.4%  in 850.6 seconds
For parameter mean = 0.360 ,  Accuracy=96.6% +/- 0.8%  in 849.4 seconds
For parameter mean = 0.428 ,  Accuracy=96.1% +/- 1.5%  in 875.3 seconds
For parameter mean = 0.509 ,  Accuracy=96.1% +/- 1.6%  in 848.8 seconds
For parameter mean = 0.605 ,  Accuracy=96.8% +/- 1.3%  in 850.1 seconds
For parameter mean = 0.720 ,  Accuracy=95.4% +/- 5.5%  in 875.4 seconds
scanning over std = [0.15       0.17838107 0.21213203 0.25226892 0.3        0.35676213
 0.42426407 0.50453785 0.6       ]
For parameter std = 0.150 ,  Accuracy=97.1% +/- 0.9%  in 849.1 seconds
For parameter std = 0.178 ,  Accuracy=96.6% +/- 2.2%  in 852.1 seconds
For parameter std = 0.212 ,  Accuracy=96.8% +/- 1.3%  in 881.6 seconds
For parameter std = 0.252 ,  Accuracy=97.3% +/- 1.1%  in 851.0 seconds
For parameter std = 0.300 ,  Accuracy=96.6% +/- 0.8%  in 850.7 seconds
For parameter std = 0.357 ,  Accuracy=96.7% +/- 1.1%  in 875.1 seconds
For parameter std = 0.424 ,  Accuracy=95.6% +/- 1.0%  in 850.0 seconds
For parameter std = 0.505 ,  Accuracy=95.2% +/- 1.1%  in 850.0 seconds
For parameter std = 0.600 ,  Accuracy=95.5% +/- 1.4%  in 875.4 seconds
--------------------------------------------------
 base= 2
--------------------------------------------------
 parameter scan : learning
--------------------------------------------------
Using SGD
--------------------------------------------------
scanning over lr = [0.0175     0.02081112 0.02474874 0.02943137 0.035      0.04162225
 0.04949747 0.05886275 0.07      ]
For parameter lr = 0.018 ,  Accuracy=94.5% +/- 1.8%  in 850.4 seconds
For parameter lr = 0.021 ,  Accuracy=94.8% +/- 2.7%  in 849.2 seconds
For parameter lr = 0.025 ,  Accuracy=95.4% +/- 1.4%  in 875.4 seconds
For parameter lr = 0.029 ,  Accuracy=96.0% +/- 1.1%  in 849.1 seconds
For parameter lr = 0.035 ,  Accuracy=96.6% +/- 0.8%  in 866.1 seconds
For parameter lr = 0.042 ,  Accuracy=96.6% +/- 1.0%  in 850.5 seconds
For parameter lr = 0.049 ,  Accuracy=89.3% +/- 21.2%  in 881.7 seconds
For parameter lr = 0.059 ,  Accuracy=82.5% +/- 28.4%  in 853.7 seconds
For parameter lr = 0.070 ,  Accuracy=65.1% +/- 33.4%  in 853.4 seconds
scanning over momentum = [0.025      0.02973018 0.03535534 0.04204482 0.05       0.05946036
 0.07071068 0.08408964 0.1       ]
For parameter momentum = 0.025 ,  Accuracy=96.1% +/- 1.3%  in 877.0 seconds
For parameter momentum = 0.030 ,  Accuracy=96.4% +/- 1.1%  in 848.7 seconds
For parameter momentum = 0.035 ,  Accuracy=96.5% +/- 1.4%  in 853.4 seconds
For parameter momentum = 0.042 ,  Accuracy=96.4% +/- 1.1%  in 875.3 seconds
For parameter momentum = 0.050 ,  Accuracy=96.6% +/- 0.8%  in 886.1 seconds
For parameter momentum = 0.059 ,  Accuracy=96.4% +/- 1.0%  in 852.3 seconds
For parameter momentum = 0.071 ,  Accuracy=96.3% +/- 1.7%  in 849.6 seconds
For parameter momentum = 0.084 ,  Accuracy=96.7% +/- 1.1%  in 875.5 seconds
For parameter momentum = 0.100 ,  Accuracy=96.8% +/- 1.5%  in 850.8 seconds
scanning over batch_size = [4, 4, 5, 6, 8, 9, 11, 13, 16]
For parameter batch_size = 4 ,  Accuracy=96.8% +/- 1.7%  in 1485.7 seconds
For parameter batch_size = 4 ,  Accuracy=96.8% +/- 1.7%  in 1485.7 seconds
For parameter batch_size = 5 ,  Accuracy=96.3% +/- 4.3%  in 857.8 seconds
For parameter batch_size = 6 ,  Accuracy=96.7% +/- 1.4%  in 907.3 seconds
For parameter batch_size = 8 ,  Accuracy=96.5% +/- 0.8%  in 1417.7 seconds
For parameter batch_size = 9 ,  Accuracy=96.4% +/- 1.2%  in 846.9 seconds
For parameter batch_size = 11 ,  Accuracy=93.8% +/- 9.4%  in 847.7 seconds
For parameter batch_size = 13 ,  Accuracy=95.1% +/- 1.5%  in 926.9 seconds
For parameter batch_size = 16 ,  Accuracy=94.3% +/- 2.4%  in 880.0 seconds
scanning over epochs = [10, 11, 14, 16, 20, 23, 28, 33, 40]
For parameter epochs = 10 ,  Accuracy=92.0% +/- 3.4%  in 426.2 seconds
For parameter epochs = 11 ,  Accuracy=93.4% +/- 2.4%  in 402.2 seconds
For parameter epochs = 14 ,  Accuracy=96.1% +/- 2.0%  in 611.3 seconds
For parameter epochs = 16 ,  Accuracy=96.4% +/- 1.3%  in 680.6 seconds
For parameter epochs = 20 ,  Accuracy=96.6% +/- 0.8%  in 722.8 seconds
For parameter epochs = 23 ,  Accuracy=96.6% +/- 0.8%  in 976.7 seconds
For parameter epochs = 28 ,  Accuracy=96.8% +/- 1.0%  in 1183.7 seconds
For parameter epochs = 33 ,  Accuracy=96.9% +/- 1.0%  in 1399.6 seconds
For parameter epochs = 40 ,  Accuracy=97.0% +/- 0.7%  in 1589.2 seconds
--------------------------------------------------
Using ADAM
--------------------------------------------------
scanning over lr = [0.0175     0.02081112 0.02474874 0.02943137 0.035      0.04162225
 0.04949747 0.05886275 0.07      ]
For parameter lr = 0.018 ,  Accuracy=25.8% +/- 0.0%  in 1487.6 seconds
For parameter lr = 0.021 ,  Accuracy=25.8% +/- 0.0%  in 1473.3 seconds
For parameter lr = 0.025 ,  Accuracy=25.8% +/- 0.0%  in 816.7 seconds
For parameter lr = 0.029 ,  Accuracy=25.8% +/- 0.0%  in 795.5 seconds
For parameter lr = 0.035 ,  Accuracy=25.8% +/- 0.0%  in 1470.9 seconds
For parameter lr = 0.042 ,  Accuracy=25.8% +/- 0.0%  in 796.8 seconds
For parameter lr = 0.049 ,  Accuracy=25.8% +/- 0.0%  in 797.2 seconds
For parameter lr = 0.059 ,  Accuracy=25.8% +/- 0.0%  in 1429.2 seconds
For parameter lr = 0.070 ,  Accuracy=25.7% +/- 0.4%  in 801.7 seconds
scanning over momentum = [0.025      0.02973018 0.03535534 0.04204482 0.05       0.05946036
 0.07071068 0.08408964 0.1       ]
For parameter momentum = 0.025 ,  Accuracy=25.8% +/- 0.0%  in 798.6 seconds
For parameter momentum = 0.030 ,  Accuracy=25.8% +/- 0.0%  in 1464.2 seconds
For parameter momentum = 0.035 ,  Accuracy=25.8% +/- 0.0%  in 796.5 seconds
For parameter momentum = 0.042 ,  Accuracy=25.8% +/- 0.0%  in 797.2 seconds
For parameter momentum = 0.050 ,  Accuracy=25.8% +/- 0.0%  in 1169.9 seconds
For parameter momentum = 0.059 ,  Accuracy=25.8% +/- 0.0%  in 795.9 seconds
For parameter momentum = 0.071 ,  Accuracy=25.8% +/- 0.0%  in 944.4 seconds
For parameter momentum = 0.084 ,  Accuracy=25.8% +/- 0.0%  in 798.1 seconds
For parameter momentum = 0.100 ,  Accuracy=25.8% +/- 0.0%  in 1273.2 seconds
scanning over batch_size = [4, 4, 5, 6, 8, 9, 11, 13, 16]
For parameter batch_size = 4 ,  Accuracy=25.8% +/- 0.0%  in 1007.8 seconds
For parameter batch_size = 4 ,  Accuracy=25.8% +/- 0.0%  in 1007.8 seconds
For parameter batch_size = 5 ,  Accuracy=25.8% +/- 0.0%  in 1610.4 seconds
For parameter batch_size = 6 ,  Accuracy=25.8% +/- 0.0%  in 858.5 seconds
For parameter batch_size = 8 ,  Accuracy=25.8% +/- 0.0%  in 1395.6 seconds
For parameter batch_size = 9 ,  Accuracy=25.8% +/- 0.0%  in 1413.1 seconds
For parameter batch_size = 11 ,  Accuracy=25.8% +/- 0.0%  in 803.6 seconds
For parameter batch_size = 13 ,  Accuracy=25.8% +/- 0.0%  in 1606.7 seconds
For parameter batch_size = 16 ,  Accuracy=25.8% +/- 0.0%  in 737.5 seconds
scanning over epochs = [10, 11, 14, 16, 20, 23, 28, 33, 40]
For parameter epochs = 10 ,  Accuracy=25.8% +/- 0.0%  in 719.4 seconds
For parameter epochs = 11 ,  Accuracy=25.8% +/- 0.0%  in 412.9 seconds
For parameter epochs = 14 ,  Accuracy=25.8% +/- 0.0%  in 558.6 seconds
For parameter epochs = 16 ,  Accuracy=25.8% +/- 0.0%  in 1151.6 seconds
For parameter epochs = 20 ,  Accuracy=25.8% +/- 0.0%  in 1412.9 seconds
For parameter epochs = 23 ,  Accuracy=25.8% +/- 0.0%  in 904.6 seconds
For parameter epochs = 28 ,  Accuracy=25.8% +/- 0.0%  in 2010.2 seconds
For parameter epochs = 33 ,  Accuracy=25.8% +/- 0.0%  in 1488.9 seconds
For parameter epochs = 40 ,  Accuracy=25.8% +/- 0.0%  in 2898.4 seconds
--------------------------------------------------
 parameter scan : network
--------------------------------------------------
scanning over conv1_kernel_size = [3, 4, 4, 5, 7, 8, 9, 11, 14]
For parameter conv1_kernel_size = 3 ,  Accuracy=81.1% +/- 7.2%  in 878.0 seconds
For parameter conv1_kernel_size = 4 ,  Accuracy=95.7% +/- 1.0%  in 833.7 seconds
For parameter conv1_kernel_size = 4 ,  Accuracy=95.7% +/- 1.0%  in 833.7 seconds
For parameter conv1_kernel_size = 5 ,  Accuracy=97.7% +/- 1.0%  in 1241.9 seconds
For parameter conv1_kernel_size = 7 ,  Accuracy=96.6% +/- 0.8%  in 872.4 seconds
For parameter conv1_kernel_size = 8 ,  Accuracy=95.4% +/- 2.6%  in 912.4 seconds
For parameter conv1_kernel_size = 9 ,  Accuracy=95.6% +/- 2.1%  in 896.6 seconds
For parameter conv1_kernel_size = 11 ,  Accuracy=97.2% +/- 1.2%  in 1526.4 seconds
For parameter conv1_kernel_size = 14 ,  Accuracy=92.9% +/- 2.5%  in 1813.0 seconds
scanning over conv1_dim = [3, 4, 4, 5, 7, 8, 9, 11, 14]
For parameter conv1_dim = 3 ,  Accuracy=95.0% +/- 2.6%  in 840.8 seconds
For parameter conv1_dim = 4 ,  Accuracy=94.6% +/- 3.6%  in 821.1 seconds
For parameter conv1_dim = 4 ,  Accuracy=94.6% +/- 3.6%  in 821.1 seconds
For parameter conv1_dim = 5 ,  Accuracy=94.9% +/- 1.6%  in 845.4 seconds
For parameter conv1_dim = 7 ,  Accuracy=96.5% +/- 0.8%  in 1355.7 seconds
For parameter conv1_dim = 8 ,  Accuracy=96.9% +/- 1.2%  in 891.6 seconds
For parameter conv1_dim = 9 ,  Accuracy=96.1% +/- 0.9%  in 1213.3 seconds
For parameter conv1_dim = 11 ,  Accuracy=97.0% +/- 1.0%  in 934.7 seconds
For parameter conv1_dim = 14 ,  Accuracy=96.9% +/- 1.1%  in 950.4 seconds
scanning over conv2_kernel_size = [3, 4, 4, 5, 7, 8, 9, 11, 14]
For parameter conv2_kernel_size = 3 ,  Accuracy=93.9% +/- 2.9%  in 856.6 seconds
For parameter conv2_kernel_size = 4 ,  Accuracy=93.3% +/- 2.1%  in 865.3 seconds
For parameter conv2_kernel_size = 4 ,  Accuracy=93.3% +/- 2.1%  in 865.3 seconds
For parameter conv2_kernel_size = 5 ,  Accuracy=91.1% +/- 2.2%  in 853.8 seconds
For parameter conv2_kernel_size = 7 ,  Accuracy=96.5% +/- 0.8%  in 871.5 seconds
For parameter conv2_kernel_size = 8 ,  Accuracy=97.3% +/- 0.7%  in 888.8 seconds
For parameter conv2_kernel_size = 9 ,  Accuracy=95.3% +/- 5.0%  in 923.2 seconds
For parameter conv2_kernel_size = 11 ,  Accuracy=96.2% +/- 1.6%  in 887.6 seconds
For parameter conv2_kernel_size = 14 ,  Accuracy=83.4% +/- 26.8%  in 936.7 seconds
scanning over conv2_dim = [6, 7, 9, 10, 13, 15, 18, 21, 26]
For parameter conv2_dim = 6 ,  Accuracy=96.1% +/- 1.6%  in 900.4 seconds
For parameter conv2_dim = 7 ,  Accuracy=95.6% +/- 2.6%  in 878.4 seconds
For parameter conv2_dim = 9 ,  Accuracy=94.6% +/- 1.3%  in 869.9 seconds
For parameter conv2_dim = 10 ,  Accuracy=92.9% +/- 13.1%  in 895.1 seconds
For parameter conv2_dim = 13 ,  Accuracy=96.5% +/- 0.8%  in 874.7 seconds
For parameter conv2_dim = 15 ,  Accuracy=96.8% +/- 1.1%  in 915.9 seconds
For parameter conv2_dim = 18 ,  Accuracy=96.9% +/- 0.9%  in 902.9 seconds
For parameter conv2_dim = 21 ,  Accuracy=96.8% +/- 1.8%  in 878.6 seconds
For parameter conv2_dim = 26 ,  Accuracy=96.2% +/- 2.2%  in 954.1 seconds
scanning over stride1 = [1, 1, 1, 1, 2, 2, 2, 3, 4]
For parameter stride1 = 1 ,  Accuracy=94.1% +/- 1.9%  in 1363.2 seconds
For parameter stride1 = 1 ,  Accuracy=94.1% +/- 1.9%  in 1363.2 seconds
For parameter stride1 = 1 ,  Accuracy=94.1% +/- 1.9%  in 1363.2 seconds
For parameter stride1 = 1 ,  Accuracy=94.1% +/- 1.9%  in 1363.2 seconds
For parameter stride1 = 2 ,  Accuracy=96.5% +/- 0.8%  in 904.8 seconds
For parameter stride1 = 2 ,  Accuracy=96.5% +/- 0.8%  in 904.8 seconds
For parameter stride1 = 2 ,  Accuracy=96.5% +/- 0.8%  in 904.8 seconds
For parameter stride1 = 3 ,  Accuracy=96.3% +/- 1.3%  in 980.9 seconds
For parameter stride1 = 4 ,  Accuracy=94.6% +/- 2.1%  in 857.8 seconds
scanning over stride2 = [2, 2, 2, 3, 4, 4, 5, 6, 8]
For parameter stride2 = 2 ,  Accuracy=94.8% +/- 1.3%  in 919.5 seconds
For parameter stride2 = 2 ,  Accuracy=94.8% +/- 1.3%  in 919.5 seconds
For parameter stride2 = 2 ,  Accuracy=94.8% +/- 1.3%  in 919.5 seconds
For parameter stride2 = 3 ,  Accuracy=96.1% +/- 1.2%  in 921.1 seconds
For parameter stride2 = 4 ,  Accuracy=96.5% +/- 0.8%  in 874.2 seconds
For parameter stride2 = 4 ,  Accuracy=96.5% +/- 0.8%  in 874.2 seconds
For parameter stride2 = 5 ,  Accuracy=96.1% +/- 3.6%  in 918.4 seconds
For parameter stride2 = 6 ,  Accuracy=97.9% +/- 1.4%  in 912.1 seconds
For parameter stride2 = 8 ,  Accuracy=96.8% +/- 1.5%  in 876.6 seconds
scanning over dimension = [15, 17, 21, 25, 30, 35, 42, 50, 60]
For parameter dimension = 15 ,  Accuracy=95.0% +/- 2.7%  in 895.1 seconds
For parameter dimension = 17 ,  Accuracy=96.4% +/- 1.4%  in 888.3 seconds
For parameter dimension = 21 ,  Accuracy=94.8% +/- 1.8%  in 876.2 seconds
For parameter dimension = 25 ,  Accuracy=97.3% +/- 1.3%  in 864.1 seconds
For parameter dimension = 30 ,  Accuracy=96.6% +/- 0.8%  in 861.3 seconds
For parameter dimension = 35 ,  Accuracy=96.4% +/- 1.4%  in 875.1 seconds
For parameter dimension = 42 ,  Accuracy=91.1% +/- 15.4%  in 878.9 seconds
For parameter dimension = 50 ,  Accuracy=96.1% +/- 1.0%  in 870.0 seconds
For parameter dimension = 60 ,  Accuracy=95.3% +/- 2.5%  in 916.0 seconds
--------------------------------------------------
 parameter scan : data
--------------------------------------------------
scanning over size = [32, 38, 45, 53, 64, 76, 90, 107, 128]
For parameter size = 32 ,  Accuracy=94.4% +/- 2.0%  in 722.6 seconds
For parameter size = 38 ,  Accuracy=95.3% +/- 1.7%  in 714.0 seconds
For parameter size = 45 ,  Accuracy=94.1% +/- 5.4%  in 854.2 seconds
For parameter size = 53 ,  Accuracy=96.1% +/- 3.2%  in 837.4 seconds
For parameter size = 64 ,  Accuracy=96.6% +/- 0.8%  in 860.5 seconds
For parameter size = 76 ,  Accuracy=95.4% +/- 1.1%  in 994.2 seconds
For parameter size = 90 ,  Accuracy=89.3% +/- 12.5%  in 1370.8 seconds
For parameter size = 107 ,  Accuracy=95.4% +/- 1.5%  in 1868.7 seconds
For parameter size = 128 ,  Accuracy=94.6% +/- 2.0%  in 2412.1 seconds
scanning over fullsize = [32, 38, 45, 53, 64, 76, 90, 107, 128]
For parameter fullsize = 32 ,  Accuracy=85.2% +/- 5.1%  in 950.7 seconds
For parameter fullsize = 38 ,  Accuracy=89.3% +/- 6.7%  in 877.9 seconds
For parameter fullsize = 45 ,  Accuracy=93.0% +/- 5.2%  in 873.5 seconds
For parameter fullsize = 53 ,  Accuracy=94.9% +/- 1.7%  in 869.5 seconds
For parameter fullsize = 64 ,  Accuracy=96.6% +/- 0.8%  in 876.3 seconds
For parameter fullsize = 76 ,  Accuracy=94.8% +/- 5.6%  in 888.0 seconds
For parameter fullsize = 90 ,  Accuracy=95.0% +/- 1.9%  in 883.0 seconds
For parameter fullsize = 107 ,  Accuracy=89.4% +/- 3.1%  in 892.1 seconds
For parameter fullsize = 128 ,  Accuracy=77.6% +/- 7.2%  in 909.6 seconds
scanning over crop = [32, 38, 45, 53, 64, 76, 90, 107, 128]
For parameter crop = 32 ,  Accuracy=76.3% +/- 5.0%  in 888.2 seconds
For parameter crop = 38 ,  Accuracy=87.3% +/- 5.4%  in 866.8 seconds
For parameter crop = 45 ,  Accuracy=91.0% +/- 5.5%  in 890.1 seconds
For parameter crop = 53 ,  Accuracy=95.7% +/- 1.5%  in 875.1 seconds
For parameter crop = 64 ,  Accuracy=96.6% +/- 0.8%  in 848.3 seconds
For parameter crop = 76 ,  Accuracy=90.2% +/- 15.2%  in 895.8 seconds
For parameter crop = 90 ,  Accuracy=92.0% +/- 7.0%  in 873.8 seconds
For parameter crop = 107 ,  Accuracy=84.5% +/- 16.4%  in 880.8 seconds
For parameter crop = 128 ,  Accuracy=78.2% +/- 14.1%  in 917.6 seconds
scanning over mean = [0.18       0.21405728 0.25455844 0.30272271 0.36       0.42811456
 0.50911688 0.60544542 0.72      ]
For parameter mean = 0.180 ,  Accuracy=79.6% +/- 27.8%  in 849.7 seconds
For parameter mean = 0.214 ,  Accuracy=86.2% +/- 21.6%  in 850.0 seconds
For parameter mean = 0.255 ,  Accuracy=92.0% +/- 15.3%  in 874.9 seconds
For parameter mean = 0.303 ,  Accuracy=92.9% +/- 15.4%  in 850.6 seconds
For parameter mean = 0.360 ,  Accuracy=96.6% +/- 0.8%  in 849.4 seconds
For parameter mean = 0.428 ,  Accuracy=96.1% +/- 1.5%  in 875.3 seconds
For parameter mean = 0.509 ,  Accuracy=96.1% +/- 1.6%  in 848.8 seconds
For parameter mean = 0.605 ,  Accuracy=96.8% +/- 1.3%  in 850.1 seconds
For parameter mean = 0.720 ,  Accuracy=95.4% +/- 5.5%  in 875.4 seconds
scanning over std = [0.15       0.17838107 0.21213203 0.25226892 0.3        0.35676213
 0.42426407 0.50453785 0.6       ]
For parameter std = 0.150 ,  Accuracy=97.1% +/- 0.9%  in 849.1 seconds
For parameter std = 0.178 ,  Accuracy=96.6% +/- 2.2%  in 852.1 seconds
For parameter std = 0.212 ,  Accuracy=96.8% +/- 1.3%  in 881.6 seconds
For parameter std = 0.252 ,  Accuracy=97.3% +/- 1.1%  in 851.0 seconds
For parameter std = 0.300 ,  Accuracy=96.6% +/- 0.8%  in 850.7 seconds
For parameter std = 0.357 ,  Accuracy=96.7% +/- 1.1%  in 875.1 seconds
For parameter std = 0.424 ,  Accuracy=95.6% +/- 1.0%  in 850.0 seconds
For parameter std = 0.505 ,  Accuracy=95.2% +/- 1.1%  in 850.0 seconds
For parameter std = 0.600 ,  Accuracy=95.5% +/- 1.4%  in 875.4 seconds
````

## LICENCE

All code is released under the GPL license, see the LICENSE file.

The database files of faces are distributed for the learning of the network, but should not be re-used outside this repository.

The precise license is https://creativecommons.org/licenses/by-nc-nd/4.0/

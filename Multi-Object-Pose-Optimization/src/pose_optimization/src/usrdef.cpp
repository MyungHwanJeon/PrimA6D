#include "usrdef.h"

void addNormal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals)
{  
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr searchTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    searchTree->setInputCloud(cloud);

    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimator;
    normalEstimator.setInputCloud (cloud);
    normalEstimator.setSearchMethod(searchTree);
    normalEstimator.setKSearch(50);
    normalEstimator.compute(*normals);

    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
}

char getch()
{
    fd_set set;
    struct timeval timeout;
    int rv;
    char buff = 0;
    int len = 1;
    int filedesc = 0;
    FD_ZERO(&set);
    FD_SET(filedesc, &set);

    timeout.tv_sec = 0;
    timeout.tv_usec = 1000;

    rv = select(filedesc + 1, &set, NULL, NULL, &timeout);

    struct termios old = {0};
    if (tcgetattr(filedesc, &old) < 0)
        ROS_ERROR("tcsetattr()");
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(filedesc, TCSANOW, &old) < 0)
        ROS_ERROR("tcsetattr ICANON");

    if(rv == -1)
    {}
        //ROS_ERROR("select");
    else if(rv == 0)
    {}
        //ROS_INFO("no_key_pressed");
    else
    {
        read(filedesc, &buff, len );
    }

    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(filedesc, TCSADRAIN, &old) < 0)
        ROS_ERROR ("tcsetattr ~ICANON");
    return (buff);
}

model_info get_model_param(std::string dataset, int idx)
{
    std::map<int, model_info> ycb_model_info;
    ycb_model_info[1] = model_info(172.063, -51.223, 51.223, -70.072, 102.289, 102.446, 140.144);
    ycb_model_info[2] = model_info(269.573, -35.865, -81.9885, -106.743, 71.73, 163.977, 213.486);
    ycb_model_info[3] = model_info(198.377, -24.772, -47.024, -88.0075, 49.544, 94.048, 176.015);
    ycb_model_info[4] = model_info(120.543, -33.927, -33.875, -51.0185, 67.854, 67.75, 102.037);
    ycb_model_info[5] = model_info(196.463, -48.575, -33.31, -95.704, 97.15, 66.62, 191.408);
    ycb_model_info[6] = model_info(89.797, -42.755, -42.807, -16.7555, 85.51, 85.614, 33.511);
    ycb_model_info[7] = model_info(142.543, -68.924, -64.3955, -19.414, 137.848, 128.791, 38.828);
    ycb_model_info[8] = model_info(114.053, -44.6775, -50.5545, -15.06, 89.355, 101.109, 30.12);
    ycb_model_info[9] = model_info(129.54, -51.0615, -30.161, -41.8185, 102.123, 60.322, 83.637);
    ycb_model_info[10] = model_info(197.796, -54.444, -89.206, -18.335, 108.888, 178.412, 36.67);
    ycb_model_info[11] = model_info(259.534, -74.4985, -72.3845, -121.32, 148.997, 144.769, 242.64);
    ycb_model_info[12] = model_info(259.566, -51.203, -33.856, -125.32, 102.406, 67.712, 250.64);
    ycb_model_info[13] = model_info(161.922, -80.722, -80.5565, -27.485, 161.444, 161.113, 54.97);
    ycb_model_info[14] = model_info(124.99, -58.483, -46.5375, -40.692, 116.966, 93.075, 81.384);
    ycb_model_info[15] = model_info(226.17, -92.1205, -93.717, -28.6585, 184.241, 187.434, 57.317);
    ycb_model_info[16] = model_info(237.299, -51.9755, -51.774, -102.945, 103.951, 103.548, 205.89);
    ycb_model_info[17] = model_info(203.973, -48.04, -100.772, -7.858, 96.08, 201.544, 15.716);
    ycb_model_info[18] = model_info(121.365, -10.5195, -60.4225, -9.4385, 21.039, 120.845, 18.877);
    ycb_model_info[19] = model_info(174.746, -59.978, -85.639, -19.575, 119.956, 171.278, 39.15);
    ycb_model_info[20] = model_info(217.094, -104.897, -82.18, -18.1665, 209.794, 164.36, 36.333);
    ycb_model_info[21] = model_info(102.903, -26.315, -38.921, -25.5655, 52.63, 77.842, 51.131);

    std::map<int, model_info> tless_model_info;
    tless_model_info[1] = model_info(63.515100, -17.495800, -17.495800, -30.600000, 34.991600, 34.991600, 61.200000);
    tless_model_info[2] = model_info(66.151200, -21.644800, -21.644800, -30.851100, 43.289600, 43.289600, 61.702200);
    tless_model_info[3] = model_info(65.349100, -23.883700, -23.884200, -30.835100, 47.762700, 47.761600, 61.670200);
    tless_model_info[4] = model_info(80.725700, -19.997800, -19.992900, -39.000000, 39.995600, 39.985800, 78.000000);
    tless_model_info[5] = model_info(108.690000, -47.500000, -26.750000, -29.500000, 95.000000, 53.500000, 59.000000);
    tless_model_info[6] = model_info(108.265000, -44.700000, -25.000000, -27.750000, 89.400000, 50.000000, 55.500000);
    tless_model_info[7] = model_info(178.615000, -75.000000, -44.700000, -30.750000, 150.000000, 89.400000, 61.500000);
    tless_model_info[8] = model_info(217.156000, -93.039300, -52.667200, -30.007300, 186.079000, 105.334000, 60.014600);
    tless_model_info[9] = model_info(144.546000, -60.625000, -39.250000, -31.349000, 121.250000, 78.500000, 62.698000);
    tless_model_info[10] = model_info(90.211200, -40.329900, -21.003600, -31.751300, 80.659800, 42.007200, 63.502600);
    tless_model_info[11] = model_info(76.597800, -33.145600, -24.121000, -27.650000, 66.291200, 48.242000, 55.300000);
    tless_model_info[12] = model_info(86.010900, -39.165600, -28.813500, -28.300000, 78.331200, 57.627000, 56.600000);
    tless_model_info[13] = model_info(58.125700, -19.997100, -19.994300, -23.000000, 39.994200, 39.988600, 46.000000);
    tless_model_info[14] = model_info(71.947100, -22.067300, -22.067300, -32.550000, 44.134600, 44.134600, 65.100000);
    tless_model_info[15] = model_info(68.569200, -22.273800, -22.277100, -27.500000, 44.547600, 44.554200, 55.000000);
    tless_model_info[16] = model_info(69.188300, -27.633800, -27.631700, -23.500000, 55.267600, 55.263400, 47.000000);
    tless_model_info[17] = model_info(112.839000, -53.887100, -53.881600, -30.050000, 107.774000, 107.763000, 60.100000);
    tless_model_info[18] = model_info(110.982000, -49.360500, -49.369400, -31.886900, 98.721000, 98.738800, 63.773800);
    tless_model_info[19] = model_info(89.068900, -32.750000, -38.250000, -23.500000, 65.500000, 76.500000, 47.000000);
    tless_model_info[20] = model_info(98.888700, -41.500000, -37.750000, -23.500000, 83.000000, 75.500000, 47.000000);
    tless_model_info[21] = model_info(92.252700, -38.450000, -39.489100, -21.500000, 76.900000, 78.978200, 43.000000);
    tless_model_info[22] = model_info(92.252700, -38.450000, -39.489100, -21.993500, 76.900000, 78.978200, 43.987000);
    tless_model_info[23] = model_info(142.587000, -68.974200, -36.750000, -26.032600, 137.948000, 73.500000, 52.065200);
    tless_model_info[24] = model_info(84.736000, -21.497700, -21.498500, -40.300000, 42.995400, 42.997000, 80.600000);
    tless_model_info[25] = model_info(108.801000, -48.000000, -30.750000, -30.400600, 96.000000, 61.500000, 60.801200);
    tless_model_info[26] = model_info(108.801000, -48.000000, -30.750000, -30.392700, 96.000000, 61.500000, 60.785400);
    tless_model_info[27] = model_info(152.495000, -54.250000, -54.250000, -28.000000, 108.500000, 108.500000, 56.000000);
    tless_model_info[28] = model_info(124.778000, -49.750000, -49.750000, -24.200000, 99.500000, 99.500000, 48.400000);
    tless_model_info[29] = model_info(134.227000, -56.500000, -39.000000, -28.400000, 113.000000, 78.000000, 56.800000);
    tless_model_info[30] = model_info(88.753800, -40.000000, -40.000000, -25.300000, 80.000000, 80.000000, 50.600000);

    if (dataset.compare("ycbv") == 0)
    {
        return ycb_model_info[idx];
    }
    else if (dataset.compare("tless") == 0)
    {
        return tless_model_info[idx];
    }
}
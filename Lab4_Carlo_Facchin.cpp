// Computer Vision 2021 (Carlo Facchin - 1234374) - LAB 4 / Project1

#include "Extended_panoramic_utils.h"

//libraries already included in header

using namespace cv;
using namespace std;

// [MAIN] ----------------------------------------------------------------------
//
// Main panorama test options
// 
// path = 'cortile', format = '.png', number of images = 24, half_FoV_angle = 27.
// path = 'dolomites', format = '.png', number of images = 23, half_FoV_angle = 27.
// path = 'terrazzo', format = '.png', number of images = 24, half_FoV_angle = 38.
// path = 'kitchen', format = '.bmp', number of images = 23 not used last 3, half_FoV_angle = 33.
// path = 'lab-manual-ok', format = '.png', number of images = 11 not used the first, half_FoV_angle = 33.
// path = 'lab', format = '.bmp', number of images = 12, half_FoV_angle = 33.


int main()
{
    //Selecting settings
    string img_path = "cortile/";

    string img_format = ".png";
   
    int image_number = 24;

    double half_FoV_angle = 27.;

    
    //Initialization of the output canvas 
    Mat s_output;
    Mat smooth_output;

    //Images loading and projection
    vector<Mat> src = PanoramicFunctions::img_load(img_path, img_format, image_number, false);
    vector<Mat> cylindrical_src;
    PanoramicFunctions::img_proj(src, cylindrical_src, half_FoV_angle, true);


    vector<vector<KeyPoint>> keypoints_vec;
    vector<Mat> descriptors_vec;

    //Feature extraction using ORB algorithm
    PanoramicFunctions::ORB_extract(cylindrical_src, descriptors_vec, keypoints_vec);


    vector<vector<DMatch>> match;

    //Matching the features between two consecutive images
    PanoramicFunctions::feature_match(descriptors_vec, match);


    float ratio = 1.5;
    vector<vector<DMatch>> BestMatches;

    //Selecting the optimal matches with the threshold value ratio
    PanoramicFunctions::match_select(match, BestMatches, ratio);


    vector<Mat> H_mats;

    // Finding the translation matrices between image with RANSAC algorithm
    PanoramicFunctions::RANSAC_homography(BestMatches, H_mats, keypoints_vec);



    //Merge of the images in the output canvas
    s_output = PanoramicFunctions::ToCanvas(cylindrical_src, H_mats, img_path, true);

    namedWindow("Simple_panorama", WINDOW_AUTOSIZE);

    if (img_path == "kitchen/")
    {
        resize(s_output, s_output, Size(s_output.cols / 3., s_output.rows / 3.));
        imshow("Simple_panorama", s_output);
        waitKey(0);
    }

    else 
    {
        resize(s_output, s_output, Size(s_output.cols / 2.5, s_output.rows / 2.5));
        imshow("Simple_panorama", s_output);
        waitKey(0);
    }
   


    int d1 = 30;
    
    //Merge of the images in the output canvas fading the edges with addWeighted function and closing the loop
    smooth_output = PanoramicFunctions::ToCanvas_smooth(cylindrical_src, H_mats, d1, img_path, true);

    namedWindow("Faded_panorama", WINDOW_AUTOSIZE);

    if (img_path == "kitchen/")
    {
        resize(smooth_output, smooth_output, Size(smooth_output.cols / 3., smooth_output.rows / 3.));
        imshow("Faded_panorama", smooth_output);
        waitKey(0);
    }

    else
    {
        resize(smooth_output, smooth_output, Size(smooth_output.cols / 2.5, smooth_output.rows / 2.5));
        imshow("Faded_panorama", smooth_output);
        waitKey(0);
    }
    

}
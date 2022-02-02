#pragma once

// Computer Vision 2021 (Carlo Facchin - 1234374) - LAB 4 / Project1

#ifndef EXT_LAB4__PANORAMIC__UTILS__H
#define EXT_LAB4__PANORAMIC__UTILS__H

#include <memory>
#include <iostream>


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

class PanoramicFunctions
{
public:
    static
        cv::Mat cylindricalProj(
            const cv::Mat& image,
            const double angle);


    // [RGB_cylindricalProj]------------------------------------------------------------------
    // 
    //Same projection function without BGR2GRAY conversion
    // image = input image Matrix
    // angle = half of the Field of View angle of the camera used to take the photos
    static cv::Mat RGB_cylindricalProj(const cv::Mat& image, const double angle);

    // [img_load]------------------------------------------------------------------
    // 
    // Images loading function, return a vector of the Mat of all the loaded images
    // path dir = directory of images folder
    // img format = the format of the images
    // n = the number of images
    // equalization = flag for enabling/disabling the equalization of the three channels
    static vector<Mat> img_load(string path_dir, string img_format, int n, bool equalization = false);


    // [img_proj]------------------------------------------------------------------
    // 
    //Compute the cylindricalProjection of the imput images (vector of Mat) selecting the RGB or the grey proj function
    // input = vector of the Mat of all the loaded images
    // output = vector of the Mat of all the projected images
    // half_FoV_angle = half of the Field of View of the camera used to take the photos
    // RGB_select = flag for selecting greyscale or color processing 
    static void img_proj(vector<Mat> input, vector<Mat>& output, double half_FoV_angle, bool RGB_select=false);


    // [ORB_extract]------------------------------------------------------------------
    // 
    //Extract the ORB features and compute descriptors of the input images
    // input = vector of the Mat of all the loaded images
    // descriptors_vec = vector of Mat that cointain the computed descriptors of the images
    // keypoints_vec = vector of vectors that cointain the keypoints of the images
    static void ORB_extract(vector<Mat> input, vector<Mat>& descriptors_vec, vector<vector<KeyPoint>>& keypoints_vec);


    // [feature_match]------------------------------------------------------------------
    // 
    //Compute the matching of the features of two consecutive images
    // descriptors_vec = vector of Mat that cointain the computed descriptors of the image
    // matches_vec = vector of vectors that cointain the computed matches of the image
    // verbose = flag for enabling/disabling the verbose in cout output
    static void feature_match(vector<Mat> descriptors_vec, vector<vector<DMatch>>& matches_vec, bool verbose=true);


    // [match_select]------------------------------------------------------------------
    // 
    //Select the optimal matches
    // matches_vec = vector of vectors that cointain the computed matches of the image
    // selected_matches_vec = vector of the selected matches 
    // ratio = filtering threshold value
    static void match_select(vector<vector<DMatch>> matches_vec, vector<vector<DMatch>>& selected_matches_vec, float ratio);


    // [RANSAC_homography]------------------------------------------------------------------
    // 
    //RANSAC translation computation
    // selected_matches_vec = vector of the selected matches
    // output = vector of the computed translation matrices between two images
    // keypoints_vec = vector of vectors that cointain the keypoints of the image
    static void RANSAC_homography(vector<vector<DMatch>> selected_matches_vec, vector<Mat>& output, vector<vector<KeyPoint>> keypoints_vec);


    // [ToCanvas]------------------------------------------------------------------
    // 
    //Merge the images copying them in a canvas. Return the output canvas.
    // input = vector of the Mat of all the loaded images
    // H_mats = vector of the computed translation matrices between two images
    // path = saving directory path folder
    // save_img = flag for enabling/disabling saving
    static Mat ToCanvas(vector<Mat> input, vector<Mat> H_mats, const string& path = "/", bool save_img = false);


    // [ToCanvas_smooth]------------------------------------------------------------------
    // 
    //Merge the images copying them in a canvas fading the edges with addWeighted function, closed loop. Return the output canvas.
    // input = vector of the Mat of all the loaded images
    // panoramic - destination canvas of merged panoramic image
    // H_mats = vector of the computed translation matrices between two images
    // d1 = overlap size value between the images where the fade is applied
    // path = saving directory path folder
    // save_img = flag for enabling / disabling saving
    static Mat ToCanvas_smooth(vector<Mat> input, vector<Mat> H_mats, int d1, const string& path = "/", bool save_img = false);


};

#endif // EXT_LAB4__PANORAMIC__UTILS__H
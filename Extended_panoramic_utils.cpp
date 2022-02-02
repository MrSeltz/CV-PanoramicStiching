// Computer Vision 2021 (Carlo Facchin - 1234374) - LAB 4 / Project1

#include "Extended_panoramic_utils.h"
#include <memory>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>


cv::Mat PanoramicFunctions::cylindricalProj(
    const cv::Mat& image,
    const double angle)
{
    cv::Mat tmp, result;
    cv::cvtColor(image, tmp, cv::COLOR_BGR2GRAY);
    result = tmp.clone();


    double alpha(angle / 180 * CV_PI);
    double d((image.cols / 2.0) / tan(alpha));
    double r(d / cos(alpha));
    double d_by_r(d / r);
    int half_height_image(image.rows / 2);
    int half_width_image(image.cols / 2);


    for (int x = -half_width_image + 1,
        x_end = half_width_image; x < x_end; ++x)
    {
        for (int y = -half_height_image + 1,
            y_end = half_height_image; y < y_end; ++y)
        {
            double x1(d * tan(x / r));
            double y1(y * d_by_r / cos(x / r));

            if (x1 < half_width_image &&
                x1 > -half_width_image + 1 &&
                y1 < half_height_image &&
                y1 > -half_height_image + 1)
            {
                result.at<uchar>(y + half_height_image, x + half_width_image)
                    = tmp.at<uchar>(round(y1 + half_height_image),
                        round(x1 + half_width_image));
            }
        }
    }

    return result;
}

//------------------------------------------------------------------
// Same projection function without BGR2GRAY conversion

cv::Mat PanoramicFunctions::RGB_cylindricalProj(const cv::Mat& image, const double angle)
{
    cv::Mat tmp, result;
    tmp = image;
    result = tmp.clone();


    double alpha(angle / 180 * CV_PI);
    double d((image.cols / 2.0) / tan(alpha));
    double r(d / cos(alpha));
    double d_by_r(d / r);
    int half_height_image(image.rows / 2);
    int half_width_image(image.cols / 2);

    for (int x = -half_width_image + 1, x_end = half_width_image; x < x_end; ++x)
    {
        for (int y = -half_height_image + 1, y_end = half_height_image; y < y_end; ++y)
        {
            double x1(d * tan(x / r));
            double y1(y * d_by_r / cos(x / r));

            if (x1 < half_width_image &&
                x1 > -half_width_image + 1 &&
                y1 < half_height_image &&
                y1 > -half_height_image + 1)
            {
                result.at<Vec3b>(y + half_height_image, x + half_width_image)
                    = tmp.at<Vec3b>(round(y1 + half_height_image), 
                        round(x1 + half_width_image));
            }
        }
    }
    return result;
}

//--------------------------------------------------------------------------------
// Images loading function from the directory path and the number of the image, optional equalization of the three channels

vector<Mat> PanoramicFunctions::img_load(string path_dir, string img_format, int n, bool equalization)
{
    // Variable initialization
    
    string format = img_format;
    
    vector<Mat> img_vec;
    string tmp;

    Mat lab_image;
    Mat equalized_img;
    Mat equalized_bgr;

    vector<Mat> channels;
    vector<Mat> lab_planes(3);

    // Images loading cycle
    for (int i = 0; i < n; i++)
    {

        if (i < 9)
            tmp= path_dir + "i0" +to_string(i+1) + format;

        else
            tmp= path_dir + "i" +to_string(i+1) + format;

        Mat img = imread(tmp, IMREAD_COLOR);

        //Image not loaded case, empty Mat
        if (img.empty())
            cerr << "Image NOT loaded: " << tmp << endl;
        else
        {
            //Mat not empty, succesfully loaded and pushed to the vector of <Mat>
            cout << "Image loaded: " << tmp << endl;

            if (equalization)
            {
            
                //Separate the 3 channels BGR of the image
                vector<Mat> bgr;
                split(img, bgr);

                //Initialize the equalized BGR channels
                Mat bgr_eq[3];

                //Equalize each channel
                equalizeHist(bgr[0], bgr_eq[0]);
                equalizeHist(bgr[1], bgr_eq[1]);
                equalizeHist(bgr[2], bgr_eq[2]);

                //Merge back equalized channels to a single BGR image
                vector<Mat> channels;
                Mat equalized_img;
                channels.push_back(bgr_eq[0]);
                channels.push_back(bgr_eq[1]);
                channels.push_back(bgr_eq[2]);
                merge(channels, equalized_img);

                //Push the equalized image in the output vector
                img_vec.push_back(equalized_img);
            }

            else
            {
                img_vec.push_back(img);
            }
            
        }
    }

    return img_vec;
}



//------------------------------------------------------------------------------------------------------------------
// Compute the cylindricalProjection of the imput images (vector of Mat) selecting the RGB or the grey proj function

void PanoramicFunctions::img_proj(vector<Mat> input, vector<Mat>& output, double half_FoV_angle, bool RGB_select)
{
    // Images projection cycle
    for (int i = 0; i < input.size(); i++)
    {

        // RGB or greyscale selection
        if (RGB_select)
            output.push_back(RGB_cylindricalProj(input[i], half_FoV_angle));
        
        else
            output.push_back(cylindricalProj(input[i], half_FoV_angle));

        
        cout << "Image projected: " << i << endl;
    }
    return;
}

// Extract the ORB features and compute descriptors of the input images
void PanoramicFunctions::ORB_extract(vector<Mat> input, vector<Mat>& descriptors_vec, vector<vector<KeyPoint>>& keypoints_vec)
{
    // Init and create ORB feature object
    Ptr<Feature2D> orb_feature = ORB::create();

    //Images keypoints and descriptor extraction cycle
    for (int i = 0; i < input.size(); i++)
    {
        // Init temp Mat to be pushed in the vector of keypoints and descriptors
        Mat tmp_Mat;
        vector<KeyPoint> tmp_kp;


        // Feature detecting and computation
        orb_feature->detectAndCompute(input[i],Mat(), tmp_kp, tmp_Mat);

        descriptors_vec.push_back(tmp_Mat);
        keypoints_vec.push_back(tmp_kp);
    }

    return;
}

// Compute the matching of the features of two consecutive images
void PanoramicFunctions::feature_match(vector<Mat> descriptors_vec, vector<vector<DMatch>>& matches_vec, bool verbose)
{
    vector<DMatch> img_matches;


    // int 4 is the selction of normType = NORM_HAMMING, that should be used with ORB. crossCheck = true.
    Ptr<BFMatcher> matcher = BFMatcher::create(4, true);

    for (int i = 0; i < descriptors_vec.size() - 1; i++)
    {
        matcher->match(descriptors_vec[i], descriptors_vec[i+1], img_matches, Mat());

        matches_vec.push_back(img_matches);

        if (verbose)
        {
            cout << "Match computed of images: " << i + 1 << " and " << i + 2 << endl;
        }
        
    }

    // Close loop matching connection (last image with first image)
    matcher->match(descriptors_vec.back(), descriptors_vec[0], img_matches, Mat());
    matches_vec.push_back(img_matches);

    if (verbose)
    {
        cout << "Closing loop match computed." << endl;
    }
    
    return;
}

// Select the optimal matches
void PanoramicFunctions::match_select(vector<vector<DMatch>> matches_vec, vector<vector<DMatch>>& selected_matches_vec, float ratio)
{
    for (int i = 0; i < matches_vec.size(); i++)
    {
        float ratio_threshold = ratio;

        vector<DMatch> Matches;
       
        float distance;
        float min_distance = -1.;
        int match_s = int(matches_vec[i].size());
        Mat tab(match_s, 1, CV_32F);

        // Find the minumun distance between matchpoints
        for (int j = 0; j < match_s; j++)
        {
            distance = matches_vec[i][j].distance;

            // min_distance update
            if (min_distance < 0 || distance < min_distance)
                min_distance = distance;

        }

        // Adjusting the ratio for having a filtering of at least 80 matches between images couples
        do
        {
            for (int j = 0; j < int(matches_vec[i].size()); j++)
            {
                if (matches_vec[i][j].distance < min_distance * ratio_threshold)
                    Matches.push_back(matches_vec[i][j]);
            }

           
            ratio_threshold = 2 + ratio_threshold;

        } while (Matches.size() < 80);

        selected_matches_vec.push_back(Matches);
    }

    return;
}

// RANSAC translation computation
void PanoramicFunctions::RANSAC_homography(vector<vector<DMatch>> selected_matches_vec, vector<Mat>& output, vector<vector<KeyPoint>> keypoints_vec)
{
    //For every image except the closing loop between the last and the first image
    for (int i = 0; i < selected_matches_vec.size()-1; i++)
    {
        // Init matching keypoints vectors and iterator
        vector<Point2f> source_points;
        vector<Point2f> dest_points;
        vector<DMatch>::iterator t;

        // Iterate matches of the image
        for (t = selected_matches_vec[i].begin(); t != selected_matches_vec[i].end()-1; ++t)
        {
            //Push the keypoints of the selected matches in the vectors
            source_points.push_back(keypoints_vec[i+1][t->trainIdx].pt);
            dest_points.push_back(keypoints_vec[i][t->queryIdx].pt);
        }


        // Compute the homography matrix H with RANSAC algorithm
        Mat H_mats = findHomography(source_points, dest_points, RANSAC);

        // Append the homography to output vector
        output.push_back(H_mats);
        waitKey(0);
    }

    //Closing loop homography
    vector<Point2f> source_points;
    vector<Point2f> dest_points;
    vector<DMatch>::iterator t;

    for (t = selected_matches_vec.back().begin(); t != selected_matches_vec.back().end() - 1; ++t)
    {
        source_points.push_back(keypoints_vec[0][t->trainIdx].pt);
        dest_points.push_back(keypoints_vec.back()[t->queryIdx].pt);
    }

    Mat H_mats = findHomography(source_points, dest_points, RANSAC);
    output.push_back(H_mats);

    return;
}

//Merge the images copying them in a canvas
Mat PanoramicFunctions::ToCanvas(vector<Mat> input, vector<Mat> H_mats, const string& path, bool save_img)
{
    // Initialize the translation offset vector vector pushing 0.
    vector<double> x_offset;
    x_offset.push_back(0.);


    for (int i = 0; i < H_mats.size(); i++)
    {
        
        double t = H_mats[i].at<double>(Point(2, 0));

        // For the first element is pushed the first offset then is pushed the sum of the current translation offset and the previous one
        if (i == 0)
            x_offset.push_back(t);
        
        else
        {
            x_offset.push_back(t + x_offset[i]);
        }
    }

    // Initialize the output canvas
    Mat output_canvas(Size((int)x_offset.back() + input[0].cols, input[0].rows), input[0].type());
    output_canvas = Scalar(0);

    // Merge the translated images in the canvas
    for (int i = 0; i < input.size(); i++)
    {
        input[i].copyTo(output_canvas(Rect(x_offset[i], 0, input[i].cols, input[i].rows)));
    }

    if (save_img)
    {
        imwrite(path + "panorama_v1.jpg", output_canvas);
    }
        
    return output_canvas;
}

// Merge the images copying them in a canvas fading the edges with addWeighted function, closed loop
Mat PanoramicFunctions::ToCanvas_smooth(vector<Mat> input, vector<Mat> H_mats, int d1, const string& path, bool save_img)
{
    // Initialize x-offset vector pushing 0.
    vector<int> x_offset;
    x_offset.push_back(0.);

    for (int i = 0; i < H_mats.size(); i++)
    {

        double t = H_mats[i].at<double>(Point(2, 0));

        // For the first element is pushed the first offset then is pushed the sum of the current translation offset and the previous one
        if (i == 0)
            x_offset.push_back(t);

        else
        {
            x_offset.push_back(t + x_offset[i]);
        }
    }


    // Initialize the output canvas
    Mat output_canvas(Size((int)x_offset.back() + input[0].cols - 1, input[0].rows), input[0].type());
    output_canvas = Scalar(0);

    // Initialize the overlap size between the images
    double d2 = 1. / d1;
    int last_offset = (int)H_mats.back().at<double>(Point(2, 0));


    for (int i = 0; i < input.size() - 1; i++)
    {
        //Copied in the canvas first section of the image, closedloop fading is computed later
        if (i == 0)
        {
            for (int j = 0; j < input[i].cols; j++)
            {
                input[i].col(j).copyTo(output_canvas.col(j));
            }
        }

        // Copied in the canvas the second section of the image
        for (int k = 0; k < (int)H_mats[i].at<double>(Point(2, 0)); k++)
        {
            input[i+1].col(input[i+1].cols - (int)H_mats[i].at<double>(Point(2, 0)) + k)
                .copyTo(output_canvas.col(input[i+1].cols + x_offset[i] + k));
        }

        // Compute the transition fading with the addWeighted function at the end of the image
        int k = 1;
        for (int j = input[i].cols + (int)floor(x_offset[i]) - d1; j < input[i].cols + (int)floor(x_offset[i]); j++)
        {

            Mat sct;

            addWeighted(output_canvas.col(j), 
                (1 - k * d2), 
                input[i+1].col(j - (int)floor(x_offset[i]) - (int)H_mats[i].at<double>(Point(2, 0))),
                (k * d2),
                0., sct);
            
            k++;
            sct.copyTo(output_canvas.col(j));
        }
    }

    // Fading between the last and the first image
    int offset_size = x_offset.size();
    int k = 1;

    for (int j = input.back().cols + x_offset[offset_size-2] - d1; j < input.back().cols + (int)(x_offset[offset_size-2] - 1); j++)
    {

        Mat sct;

        addWeighted(output_canvas.col(j),
            (1 - k * d2),
            output_canvas.col(j - x_offset[offset_size-2] - (int)H_mats.back().at<double>(Point(2, 0))),
            (k * d2),
            0., sct);
        
        
        k++;
        sct.copyTo(output_canvas.col(j));
    }

    int H_size = H_mats.size();

    for (int j = 0; j < (int)H_mats.back().at<double>(Point(2, 0)); j++)
    {
        int temp = (int)H_mats[H_size-1].at<double>(Point(2, 0));

        output_canvas.col(input[0].cols - temp + j).copyTo(output_canvas.col(input[0].cols + x_offset[H_size-1] + j - 1));
    }

    // Merging in the canvas the last section of the last image in the first one
    for (int k = 0; k < input[0].cols-1; k++)
    {
        output_canvas.col((int)x_offset.back() + k).copyTo(output_canvas.col(k));
    }


    // Save the panoramic image in jpg format
    if (save_img)
    {
        imwrite(path + "panorama_v2.jpg", output_canvas);
    }
        

    return output_canvas;
}




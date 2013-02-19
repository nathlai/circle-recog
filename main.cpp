#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cv.h"
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace cv;

int edge_threshy = 200;
int center_threshy = 80;
int blur_threshy = 4;
int const max_threshold = 500;
Mat orig_src, orig_gray, src, src_gray;

int contrast_threshy = 20;
int brightness_threshy = 0;

float image_height = 600.;

int min_circle_radius = 5;
int max_circle_radius = 100;


std::string logfile_output = "";

void print_radii(vector<Vec3f> circles);

vector<Vec3f> radii_vector;

Mat applySobel(Mat src_gray)
{
    /// Generate grad_x and grad_y
    Mat grad, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;
    
    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    
    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    
    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    return grad;
}

void drawHough(int, void*)
{
    
    Mat new_image = Mat::zeros( orig_src.size(), orig_src.type() );
    double c_thresh = contrast_threshy/20.;
    
    for( int y = 0; y < orig_src.rows; y++ )
    {
        for( int x = 0; x < orig_src.cols; x++ )
        {
            for( int c = 0; c < 3; c++ )
            {
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(c_thresh*( orig_src.at<Vec3b>(y,x)[c] ) + brightness_threshy );
            }
        }
    }
    imshow("after contrast and brightness", new_image);
    
    
    
    cvtColor( new_image, orig_gray, CV_BGR2GRAY );
    
    vector<Vec3f> circles;
    src = orig_src.clone();
    src_gray = orig_gray.clone();
    
    printf("EDGE: %d\nCENTER: %d\nBLUR: %d\n\n", edge_threshy, center_threshy, blur_threshy * 2 + 1);
    
    GaussianBlur( src_gray, src_gray, Size(blur_threshy * 2 + 1, blur_threshy * 2 + 1), 0, 0 );
    //src_gray = applySobel(src_gray);
    //GaussianBlur( src_gray, src_gray, Size(9, 9), 0, 0 );
    imshow("after blur", src_gray);
    

    
    
    
    
    
    
    
    
    /// Apply the Hough Transform to find the circles
    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, 0.1, edge_threshy, center_threshy, min_circle_radius, max_circle_radius );
    
    /// Draw the circles detected
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }
    radii_vector = circles;
    
    imshow( "Hough Circle Transform Demo", src );
    //imshow( "Hough Circle Transform Demo", src_gray );
    //waitKey(0);
    //imwrite("/Users/Nath/Desktop/circle2.jpg", src);
}

Mat
set_image_resolution(Mat value){
    Mat small_image, resized_image;
    
    float percent = image_height/value.rows;
    
    resize(value, resized_image, Size(),percent, percent, INTER_LINEAR);
    return resized_image;
}

void
print_log_file(
               string name, float blur, bool sobel, float circle_centers, float canny_threshold,
               float center_threshold, float circles, float rows, float columns, double time)
{
    std::string tempstr = logfile_output + "circle_log_file.txt";
    std::ofstream logfile;
    logfile.open(tempstr.c_str(), std::ios_base::app);
    if (logfile.is_open())
    {
        logfile << "----------------------------------------------------------------------\n";
        logfile << "Picture: "<< name << std::endl;
        logfile << "Sobel filter: "<< sobel << std::endl;
        logfile << "Blur amount: "<< blur << std::endl;
        logfile << "Minimum Circle distance: "<< circle_centers << std::endl;
        logfile << "Canny line detection threshold: "<< canny_threshold << std::endl;
        logfile << "Center threshold: "<< center_threshold << std::endl;
        logfile << "Rows: "<< rows << std::endl;
        logfile << "Columns: "<< columns << std::endl;
        logfile << "Number of circles: "<< circles << std::endl;
        logfile << "Milliseconds: "<< time/1000 << std::endl;
        logfile << std::endl;
        logfile.close();
    }
    else std::cout << "Unable to open file";
}


void
print_radii(vector<Vec3f> circles)
{
    std::string tempstr = logfile_output + "radii_file.txt";
    std::ofstream radiifile (tempstr.c_str());
    if (radiifile.is_open())
    {
        for (int i = 0; i < circles.size(); i++) {
            radiifile << "X: " << circles[i][0] << " Y: " << circles[i][1] << " Radius: " << circles[i][2] << std::endl;
        }
        radiifile.close();
    }
    else std::cout << "Unable to open file";
}

int main(int argc, char** argv)
{
    string window_name = "Hough Circle Transform Demo";
    clock_t start;
    double duration;

    /// Read the image
    orig_src = imread( argv[1], 1 );
    logfile_output = argv[2];
    
    if( !orig_src.data )
    { return -1; }
    
    
    orig_src = set_image_resolution(orig_src);

    namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
    createTrackbar( "Hough Edge:", window_name, &edge_threshy, max_threshold, drawHough );
    createTrackbar( "Hough Center:", window_name, &center_threshy, max_threshold, drawHough );
    createTrackbar( "Gaussian Blur:", window_name, &blur_threshy, 31, drawHough );
    createTrackbar( "Contrast:", window_name, &contrast_threshy, 60, drawHough );
    
    
    
    start = clock();
    
    drawHough(0, 0);
    duration = (clock() - start ) / (double) CLOCKS_PER_SEC;

    
    waitKey(0);
    
    print_log_file(argv[1], blur_threshy * 2 + 1, false, 100, 100, center_threshy, radii_vector.size(), orig_src.rows, orig_src.cols, duration);
    
    print_radii(radii_vector);
    return 0;
}

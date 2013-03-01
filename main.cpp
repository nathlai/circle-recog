#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cv.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <ctime>

using namespace cv;

int edge_threshy = 200;
int center_threshy = 80;
int blur_threshy = 4;
int const max_threshold = 500;
float cntr_distance;

// orig_src - original source picture
// orig_gray - original source picture, grayscale
// src, src_gray - clones of originals, used to do one pass of Hough an display
Mat orig_src, orig_gray, src, src_gray;

int contrast_threshy = 20;
int brightness_threshy = 0;

float image_height = 600.;
string file_name;

float original_row_amount;
float original_column_amount;

int min_circle_radius = 0;
int max_circle_radius = 0;

Scalar color = Scalar(0, 0, 0);

std::string logfile_output = "";

vector<Vec3f> radii_vector;

vector <vector<Vec3f> > circle_list;
int list_counter = 0;


/*
 This function helps to colorize the values of our circles to overlay on top of the image
 */
Scalar BlueGreenRed( double f )
{
    f = min(max(1. - f, 0.), 1.);
	f = f * (2./3.);	// convert 0.-1. into 0.-(2/3)
    
    double r = 1.;
    double g = 0.0;
    double b = 1.  -  6. * ( f - (5./6.) );
    
    if( f <= (5./6.) )
    {
        r = 6. * ( f - (4./6.) );
        g = 0.;
        b = 1.;
    }
    
    if( f <= (4./6.) )
    {
        r = 0.;
        g = 1.  -  6. * ( f - (3./6.) );
        b = 1.;
    }
    
    if( f <= (3./6.) )
    {
        r = 0.;
        g = 1.;
        b = 6. * ( f - (2./6.) );
    }
    
    if( f <= (2./6.) )
    {
        r = 1.  -  6. * ( f - (1./6.) );
        g = 1.;
        b = 0.;
    }
    
    if( f <= (1./6.) )
    {
        r = 1.;
        g = 6. * f;
    }
    
    //std::cout << Scalar(b, g, r) << std::endl;
    
    return Scalar(b * 255, g * 255, r * 255);
}

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
    vector<Vec3f> circles;
    //src = orig_src.clone();
 
    // Apply contrast
    /*
    for( int y = 0; y < orig_src.rows; y++ )
    {
        for( int x = 0; x < orig_src.cols; x++ )
        {
            for( int c = 0; c < 3; c++ )
            {
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(c_thresh*( orig_src.at<Vec3b>(y,x)[c] ) + brightness_threshy );
            }
        }
    }*/
    //imshow("after contrast and brightness", new_image);
    
    // Convert to grayscale
    cvtColor( orig_src, orig_gray, CV_BGR2GRAY );
    src_gray = orig_gray.clone();
    
    //Mat dst_this;
    //equalizeHist(src_gray, dst_this);
    
    //imshow("source", src_gray);
    //imshow("histog", dst_this);
    //src_gray = dst_this;
    //printf("EDGE: %d\nCENTER: %d\nBLUR: %d\n\n", edge_threshy, center_threshy, blur_threshy * 2 + 1);
    
    // Blur input picture
    GaussianBlur( src_gray, src_gray, Size(blur_threshy * 2 + 1, blur_threshy * 2 + 1), 0, 0 );
    //imshow("after blur", src_gray);
    
    /// Apply the Hough Transform to find the circles
    cntr_distance = src.rows/8;
    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, cntr_distance, edge_threshy, center_threshy, min_circle_radius, max_circle_radius );
    
    /// Draw the circles detected
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        //circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        circle( src, center, radius, color, 3, 8, 0 );
    }
    radii_vector = circles;
    
    circle_list.push_back( circles );
    
    //imshow( "Hough Circle Transform Demo", src );
    //imshow( "Hough Circle Transform Demo", src_gray );
    //waitKey(0);
}


/*
 This function takes in the current mat and then resizes it and returns the new resized mat
 */
Mat
set_image_resolution(Mat value){
    Mat small_image, resized_image;
    
    float percent = image_height/value.rows;
    
    resize(value, resized_image, Size(),percent, percent, INTER_LINEAR);
    return resized_image;
}

/*
 This function prints out the values for each pass.
 */
 void
print_log_file(
               string name, float blur, bool sobel, float circle_centers, float canny_threshold,
               float center_threshold, float circles, float rows, float columns, double time, int run_number, Scalar temp_col)
{
    std::string tempstr = logfile_output + "circle_log_file.txt";
    std::ofstream logfile;
    logfile.open(tempstr.c_str(), std::ios_base::app);
    if (logfile.is_open())
    {
        logfile << "----------------------------------------------------------------------\n";
        logfile << run_number << std::endl;
        //logfile << "Picture: "<< name << std::endl;
        //logfile << "Sobel filter: "<< sobel << std::endl;
        logfile << "Blur amount: "<< blur << std::endl;
        logfile << "Circle Center distance: "<< circle_centers << std::endl;
        logfile << "Canny edge threshold: "<< canny_threshold << std::endl;
        logfile << "Center threshold: "<< center_threshold << std::endl;
        logfile << "Color: " << temp_col<< std::endl;
        //logfile << "Rows: "<< rows << std::endl;
        //logfile << "Columns: "<< columns << std::endl;
        logfile << "Number of circles: "<< circles << std::endl;
        logfile << "Milliseconds: "<< time/1000 << std::endl;
        logfile << std::endl;
        logfile.close();
    }
    else std::cout << "Unable to open file";
}

/*
 This function runs once at the beginning to print out the default values for the image, rows, cols, name
 */
void print_log_header(string name, float rows, float columns, float old_rows, float old_columns)
{
    time_t now = time(0);
    tm *ltm = localtime(&now);
    std::string tempstr = logfile_output + "circle_log_file.txt";
    std::ofstream logfile;
    logfile.open(tempstr.c_str(), std::ios_base::app);
    if (logfile.is_open())
    {
        logfile << "----------------------------------------------------------------------\n";
        logfile << "----------------------------------------------------------------------\n";
        logfile << ltm->tm_year + 1900<<":"<< ltm->tm_mon + 1 <<":"<< ltm->tm_mday <<":"<< ltm->tm_hour <<":"<< ltm->tm_min <<":"<< ltm->tm_sec <<std::endl;
        logfile << "Picture: "<< name << std::endl;
        logfile << "Original_Rows: "<< old_rows << std::endl;
        logfile << "Original_Columns: "<< old_columns << std::endl;
        
        logfile << "Rows: "<< rows << std::endl;
        logfile << "Columns: "<< columns << std::endl;
        logfile << std::endl;
        logfile.close();
    }
}

/*
 This function just outputs the circles.
 */
void
print_radii_values(vector<Vec3f> circles)
{
    std::string tempstr = logfile_output + "circle_log_file.txt";
    std::ofstream logfile;
    logfile.open(tempstr.c_str(), std::ios_base::app);
    if (logfile.is_open()){
        logfile << "X,Y,R" << std::endl;
        for (int i = 0; i < circles.size(); i++) {
            logfile << circles[i][0] << "," << circles[i][1] << "," << circles[i][2] << std::endl;
        }
        logfile.close();
    }
    else std::cout << "Unable to open file";
}

void passes(int low, int high)
{
    int run_number = 0;
    double input_color = 0.;
    double inc = 1. / ((high - low) / 5.);
    for(int i = low; i <= high; i += 5)
    {
        clock_t start;
        double duration;
        color = BlueGreenRed(input_color);
        
        start = clock();
        drawHough(0, 0);
        duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
        
        edge_threshy = i;
        input_color += inc;
        print_log_file(file_name, blur_threshy * 2 + 1, false, cntr_distance, edge_threshy, center_threshy, radii_vector.size(), orig_src.rows, orig_src.cols, duration, run_number, color);
        run_number++;
        
        print_radii_values(radii_vector);
        
    }
    imshow( "Hough Circle Transform Demo", src );
}

/*
 Writes the circle list to the console
 */
void write_circle_list()
{
    for (int i = 0; i < circle_list.size(); i++) {
        vector<Vec3f> temp = circle_list[i];
        for (int j = 0; j < temp.size(); j++) {
            std::cout << temp[j] << std::endl;
        }
    }
}

int main(int argc, char** argv)
{
    string window_name = "Hough Circle Transform Demo";

    /// Read the image
    orig_src = imread( argv[1], 1 );
    logfile_output = argv[2];
    if( !orig_src.data )
    { return -1; }
    
    original_row_amount = orig_src.rows;
    original_column_amount = orig_src.cols;
    
    file_name = argv[1];
    
    // Resize image, for consistency
    orig_src = set_image_resolution(orig_src);
    src = orig_src.clone();
    
    print_log_header(argv[1], orig_src.rows, orig_src.cols, original_row_amount, original_column_amount);

    // Create window and trackbars
    namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
    createTrackbar( "Hough Edge:", window_name, &edge_threshy, max_threshold, drawHough );
    createTrackbar( "Hough Center:", window_name, &center_threshy, max_threshold, drawHough );
    createTrackbar( "Gaussian Blur:", window_name, &blur_threshy, 31, drawHough );
    createTrackbar( "Contrast:", window_name, &contrast_threshy, 60, drawHough );
    
    // Run circle detection, track timing also
    
    passes(50, 200);
    
    waitKey(0);
    imwrite(logfile_output + "circle_recog.jpg", src);
    
    write_circle_list();
    return 0;
}

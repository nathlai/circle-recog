#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace cv;

int edge_threshy = 200;
int center_threshy = 80;
int blur_threshy = 4;
int const max_threshold = 500;
Mat orig_src, orig_gray, src, src_gray;

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
    vector<Vec3f> circles;
    src = orig_src.clone();
    src_gray = orig_gray.clone();
    
    printf("EDGE: %d\nCENTER: %d\nBLUR: %d\n\n", edge_threshy, center_threshy, blur_threshy * 2 + 1);
    
    GaussianBlur( src_gray, src_gray, Size(blur_threshy * 2 + 1, blur_threshy * 2 + 1), 0, 0 );
    //src_gray = applySobel(src_gray);
    //GaussianBlur( src_gray, src_gray, Size(9, 9), 0, 0 );
    
    /// Apply the Hough Transform to find the circles
    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, edge_threshy, center_threshy, 0, 0 );
    
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
    
    imshow( "Hough Circle Transform Demo", src );
    //imshow( "Hough Circle Transform Demo", src_gray );
    //waitKey(0);
    //imwrite("/Users/Nath/Desktop/circle2.jpg", src);
}

void
print_log_file(
               string name, float blur, bool sobel, float circle_centers, float canny_threshold,
               float center_threshold, float circles, float rows, float columns, double time)
{
    std::ofstream logfile ("/Users/Nath/Desktop/circle_log_file.txt");
    if (logfile.is_open())
    {
        
        logfile << "Hough Circle Recognition\n\n";
        logfile << "Picture: "<< name << std::endl;
        logfile << "Sobel filter: "<< sobel << std::endl;
        logfile << "Blur amount: "<< blur << std::endl;
        logfile << "Minimum Circle distance: "<< circle_centers << std::endl;
        logfile << "Canny line detection threshold: "<< canny_threshold << std::endl;
        logfile << "Center threshold: "<< center_threshold << std::endl;
        logfile << "Rows: "<< rows << std::endl;
        logfile << "Columns: "<< columns << std::endl;
        logfile << "Number of circles: "<< circles << std::endl;
        logfile << "Milliseconds: "<< time << std::endl;
	logfile.close();
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
    
    if( !orig_src.data )
    { return -1; }

    namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
    createTrackbar( "Hough Edge:", window_name, &edge_threshy, max_threshold, drawHough );
    createTrackbar( "Hough Center:", window_name, &center_threshy, max_threshold, drawHough );
    createTrackbar( "Gaussian Blur:", window_name, &blur_threshy, 31, drawHough );
    
    start = clock();
    cvtColor( orig_src, orig_gray, CV_BGR2GRAY );
    drawHough(0, 0);
    duration = (clock() - start ) / (double) CLOCKS_PER_SEC;

    print_log_file(argv[1], blur_threshy * 2 + 1, false, 100, 100, center_threshy, 100, orig_src.rows, orig_src.cols, duration);
    waitKey(0);
    return 0;
}

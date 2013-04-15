/*
 Nathan Lai
 Matthew Okazaki
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cv.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <ctime>
#include <map>
#include <getopt.h>

using namespace cv;

int edge_threshy = 200;
int center_threshy = 80;
int blur_threshy = 4;
int const max_threshold = 500;
float image_height = 600.;
float cntr_distance = image_height/8;
bool debugmode = false;

int debug_passes_counter = 0;

bool debugmode_passes = false;
int total_circles = 0;
int total_aggregated_circles = 0;



int pixel_tolerance = 30;
int radius_tolerance = 30;
int circle_occurence = 5;
std::map<string, Vec4f> aggregated_map;
std::map<string, Vec4f>::iterator map_iterator;



// orig_src - original source picture
// orig_gray - original source picture, grayscale
// src, src_gray - clones of originals, used to do one pass of Hough an display
Mat orig_src, orig_gray, src, src_gray;

int contrast_threshy = 20;
int brightness_threshy = 0;


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

string convertInt(int number)
{
    std::stringstream ss;
    ss << number;
    return ss.str();
}

string hash_function(Vec3f circle_vector)
{
    int xhash = circle_vector[0] / pixel_tolerance;
    int yhash = circle_vector[1] / pixel_tolerance;
    int rhash = circle_vector[2] / radius_tolerance;
    
    return convertInt(xhash)+"_"+convertInt(yhash)+"_"+convertInt(rhash);
}


void hash_insert(Vec3f circle_vector)
{
    Vec4f temp;
    string hash = hash_function(circle_vector);
    map_iterator = aggregated_map.find(hash);
    if (map_iterator != aggregated_map.end()) {
        //it is already added
        temp = aggregated_map.at(hash);
        temp[0] = (temp[0] + circle_vector[0])/2.;
        temp[1] = (temp[1] + circle_vector[1])/2.;
        temp[2] = (temp[2] + circle_vector[2])/2.;
        temp[3]++;
        aggregated_map.at(hash) = temp;
        
    } else {
        //it has not been added
        temp[0] = circle_vector[0];
        temp[1] = circle_vector[1];
        temp[2] = circle_vector[2];
        temp[3] = 1;
        
        aggregated_map.insert(std::pair<string, Vec4f>(hash, temp));
    }
    
}



void drawHough(int, void*)
{
    Mat new_image = Mat::zeros( orig_src.size(), orig_src.type() );
    double c_thresh = contrast_threshy/20.;
    vector<Vec3f> circles;
    
    // Convert to grayscale
    cvtColor( orig_src, orig_gray, CV_BGR2GRAY );
    src_gray = orig_gray.clone();

    //printf("EDGE: %d\nCENTER: %d\nBLUR: %d\n\n", edge_threshy, center_threshy, blur_threshy * 2 + 1);
    
    // Blur input picture
    GaussianBlur( src_gray, src_gray, Size(blur_threshy * 2 + 1, blur_threshy * 2 + 1), 0, 0 );
    //imshow("after blur", src_gray);
    
    /// Apply the Hough Transform to find the circles
    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, cntr_distance, edge_threshy, center_threshy, min_circle_radius, max_circle_radius );

    Mat src_copy = src.clone();
    
    /// Draw the circles detected
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        //circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        if (debugmode) {
            circle( src_copy, center, radius, Scalar(0,255, 0), 2, 8, 0 );
        } else {
            total_circles++;
            hash_insert(circles[i]);
            
            //this displays the cumulative circles onto the screen
            circle( src, center, radius, color, 2, 8, 0 );
        }
        
    }
    radii_vector = circles;
    
    circle_list.push_back( circles );
    
    if (debugmode) {
        imshow( "Hough Circle Transform Demo", src_copy );
        std::cout << edge_threshy << " " << blur_threshy << " " << center_threshy << std::endl;
    }
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

void print_aggregate_logfile(string name, float rows, float columns, float old_rows, float old_columns)
{
}

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

void passes(int low, int high, int lowBlur, int highBlur, int lowCenter, int highCenter)
{
    int run_number = 0;
    double input_color = 0.;
    double inc = 1. / ((
                        ceil((high - low + 1.) / 5.) *
                        (highBlur - lowBlur + 1.) *
                        ceil((highCenter - lowCenter + 1.) / 5.) /*
                        2.  this is for the multi pass for radius distance*/
                        ));
    /*
     //multipass for radius distance
     for(int h = 0; h < 2; h++)
    {
        if (h==0) {
            min_circle_radius = 0;
            max_circle_radius = 0;
        } else {
            min_circle_radius = 0;
            max_circle_radius = 0;
        }
    */
    for(int i = low; i <= high; i += 5)
    {
        edge_threshy = i;
        for(int j = lowBlur; j <= highBlur; j++)
        {
            blur_threshy = j;
            for(int k = lowCenter; k <= highCenter; k += 5)
            {
                if ( edge_threshy < (0.75* blur_threshy * blur_threshy - 34.*blur_threshy + 450) ){
                    debug_passes_counter++;
                    center_threshy = k;
                    clock_t start;
                    double duration;
                    color = BlueGreenRed(input_color);
                    
                    start = clock();
                    drawHough(0, 0);
                    duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
                    
                    input_color += inc;
                    if(debugmode_passes)
                    {
                        print_log_file(file_name, blur_threshy * 2 + 1, false, cntr_distance, edge_threshy, center_threshy, radii_vector.size(), orig_src.rows, orig_src.cols, duration, run_number, color);
                        print_radii_values(radii_vector);
                    }
                    run_number++;

                }

            }
        }
    }
    //}
    
    total_aggregated_circles = aggregated_map.size();
    imshow( "Passes Without Aggregation", src );
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

/*
 This writes the aggregated list to the console
 */

void write_aggregate_list()
{
    for (map_iterator = aggregated_map.begin(); map_iterator != aggregated_map.end() ; map_iterator++) {
        std::cout<< map_iterator->second << '\n';
    }
}

/*
 This draws the aggregated list to an image and then dispays it.
 */
void draw_aggregate_list()
{
    Mat agg_src = orig_src.clone();
    for (map_iterator = aggregated_map.begin(); map_iterator != aggregated_map.end() ; map_iterator++) {
        
        Vec4f temp = map_iterator->second;
        if (temp[3] >= circle_occurence) {
            Point center(cvRound(temp[0]), cvRound(temp[1]));
            int radius = cvRound(temp[2]);
            circle( agg_src, center, radius, Scalar(0,255, 0), 2, 8, 0 );
            total_aggregated_circles++;
        }
    }
    imshow("aggregated list", agg_src);
    imwrite(logfile_output + "circle_aggregate.jpg", agg_src);
}

int main(int argc, char** argv)
{
    int blur_low = 3;
    int blur_high = 20;
    int cent_low = 80;
    int cent_high = 80;
    int edge_low = 100;
    int edge_high = 300;
    
    int c;
    char *token;
    
    while ((c = getopt (argc, argv, "dsi:o:b:e:c:r:p:")) != -1)
        switch (c)
    {
        case 's':
            debugmode_passes = true;
        case 'd':
            debugmode = true;
            break;
        case 'i':
            orig_src = imread( optarg, 1 );
            file_name = optarg;
            break;
        case 'o':
            logfile_output = optarg;
            break;
        case 'b':
            token = strtok(optarg, "-");
            blur_low = atoi(token);
            token = strtok(NULL, "-");
            blur_high = atoi(token);
            break;
        case 'e':
            token = strtok(optarg, "-");
            edge_low = atoi(token);
            token = strtok(NULL, "-");
            edge_high = atoi(token);
            break;
        case 'c':
            token = strtok(optarg, "-");
            cent_low = atoi(token);
            token = strtok(NULL, "-");
            cent_high = atoi(token);
            break;
        case 'r':
            token = strtok(optarg, "-");
            radius_tolerance = atoi(token);
            break;
        case 'p':
            token = strtok(optarg, "-");
            pixel_tolerance = atoi(token);
            break;
        case '?':
            if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                         "Unknown option character `\\x%x'.\n",
                         optopt);
            return 1;
        default:
            abort();
    }
    
    
    
    string window_name = "Hough Circle Transform Demo";

    // Check the image
    if( !orig_src.data )
        { return -1; }
    
    original_row_amount = orig_src.rows;
    original_column_amount = orig_src.cols;
    
    // Resize image, for consistency
    orig_src = set_image_resolution(orig_src);
    src = orig_src.clone();

    if(debugmode){
        printf("GETOPT ARGS\nBLUR:%d to %d\nEDGE:%d to %d\nCENT:%d to %d\n", blur_low, blur_high, edge_low, edge_high, cent_low, cent_high);
        // Create window and trackbars
        namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
        createTrackbar( "Hough Edge:", window_name, &edge_threshy, max_threshold, drawHough );
        createTrackbar( "Hough Center:", window_name, &center_threshy, max_threshold, drawHough );
        createTrackbar( "Gaussian Blur:", window_name, &blur_threshy, 31, drawHough );
        drawHough(0, 0);
        waitKey(0);
        std::cout << edge_threshy << " " << blur_threshy << " " << center_threshy << std::endl;
    }
    else {
        // Run circle detection, track timing also
        if(debugmode_passes)
            print_log_header(file_name, orig_src.rows, orig_src.cols, original_row_amount, original_column_amount);

        passes(edge_low, edge_high, blur_low, blur_high, cent_low, cent_high);
        draw_aggregate_list();
        
        waitKey(0);
        imwrite(logfile_output + "circle_recog.jpg", src);
    

        //write_circle_list();
        //std::cout << "--------------------------"<<std::endl;
        //write_aggregate_list();

        write_circle_list();
        std::cout << "--------------------------"<<std::endl;
        write_aggregate_list();
        if(!debugmode_passes)
            print_aggregate_logfile(file_name, orig_src.rows, orig_src.cols, original_row_amount, original_column_amount);
            // print new log file

    }
    
    std::cout << debug_passes_counter << std::endl;
    std::cout << pixel_tolerance << " " << radius_tolerance;
    return 0;
}

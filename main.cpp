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


//gui opener

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_File_Chooser.H>
#include <FL/FL_Int_Input.H>
#include <FL/Fl_Slider.H>
#include <FL/Fl_Text_Display.H>
#include <FL/Fl_Text_Editor.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Multiline_Output.H>

#include "sliders.h"

using namespace cv;

/*
 global variables
 */
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

int blur_low = 3;
int blur_high = 20;
int cent_low = 80;
int cent_high = 80;
int edge_low = 100;
int edge_high = 300;

//Controls the post processing aggregation algorithm, 
int pixel_tolerance = 30;
int radius_tolerance = 30;
int circle_occurence = 10;

std::map<string, Vec4f> aggregated_map;
std::map<string, Vec4f>::iterator map_iterator;

//Image Strings
string no_aggregation = "No Aggregation";
string one_aggregation = "One Aggregation";
string end_aggregation = "Final Aggregation";




// orig_src - original source picture
// orig_gray - original source picture, grayscale
// src, src_gray - clones of originals, used to do one pass of Hough an display
Mat orig_src, orig_gray, src, src_gray;


int contrast_threshy = 20;
int brightness_threshy = 0;

// global variable of the name of the file to open
string file_name="";

//original size of the image that was loaded
float original_row_amount;
float original_column_amount;

int min_circle_radius = 0;
int max_circle_radius = 0;

Scalar color = Scalar(0, 0, 0);

std::string logfile_output = "";

vector<Vec3f> radii_vector;

vector <vector<Vec3f> > circle_list;
int list_counter = 0;


//Gui global variables for sliders and messages
SliderInput *max_blur_slider;
SliderInput *min_blur_slider;

SliderInput *max_edge_slider;
SliderInput *min_edge_slider;

SliderInput *max_cent_slider;
SliderInput *min_cent_slider;


SliderInput *ptol_slider;
SliderInput *rtol_slider;
SliderInput *occurence_slider;

Fl_Text_Display *disp;
Fl_Text_Display *message_disp;
Fl_Text_Buffer *message_buff;
Fl_Text_Buffer *buff;
Fl_Text_Buffer *sbuff;

Fl_Text_Display::Style_Table_Entry stable[] = {
    // FONT COLOR      FONT FACE   FONT SIZE
    // --------------- ----------- --------------
    {  FL_RED,         FL_BOLD, 15 }, // A - Red
    {  FL_DARK_YELLOW, FL_COURIER, 12 }, // B - Yellow
    {  FL_DARK_GREEN,  FL_COURIER, 12 }, // C - Green
    {  FL_BLUE,        FL_COURIER, 12 }, // D - Blue
};


/*
 This function helps to colorize the values of our circles to overlay on top of the image.
 It uses a double that is between 0 and 1 to determine the color to pass back
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


/*
 This function would apply the sobel filter to the image, although this was not necessary because
 Open cv's Hough gradient already applies canny filter to find edges and uses the sobel filter in order to get the gradient.
 */
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

/*
 this function converts the int to a string value and then returns it.  This is necessary to get the string hash value
 */
string convertInt(int number)
{
    std::stringstream ss;
    ss << number;
    return ss.str();
}


/*
 This is the hashing function.  It takes in the circle vector and hashes it
 This was the original hashing function and only does the first group of buckets
 */
string hash_function(Vec3f circle_vector)
{
    int xhash = circle_vector[0] / pixel_tolerance;
    int yhash = circle_vector[1] / pixel_tolerance;
    int rhash = circle_vector[2] / radius_tolerance;
    
    return convertInt(xhash)+"_"+convertInt(yhash)+"_"+convertInt(rhash);
}

/*
 this takes the circle vector and adds it to the global hash table
 */
void hash_insert(Vec3f circle_vector)
{
    Vec4f temp;
    
    //generate our hash value
    string hash = hash_function(circle_vector);
    map_iterator = aggregated_map.find(hash);
    
    // we need to check to see if the hash exists already.  if it does we combine it with the already added value, if not we add it to our hash table
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

/*
 This is the modular hash function in order to allow us to shift the buckets in which circles are placed
 We use 0 - 1 for x y and r in order to shift the bucket location.
 0, 0, 0 is the default. 
 because of the formula (circle_vector[0] + x * (pixel_tolerance *.5), an x value of 2 will move the bucket over exactly 1 spot, resulting in the same hash as if it was 0
 1 is half a grid length while 2 is a full grid length
 This function determines how we shift the buckets, 0 doesn't shift, values of 2 shift it exactly one 
 */
string hash_function_modular(Vec3f circle_vector, float x, float y, float r)
{
    int xhash = (circle_vector[0] + x * (pixel_tolerance *.5))/ pixel_tolerance;
    int yhash = (circle_vector[1] + y * (pixel_tolerance *.5))/ pixel_tolerance;
    int rhash = (circle_vector[2] + r * (radius_tolerance * .5))/ radius_tolerance;
    
    return convertInt(xhash)+"_"+convertInt(yhash)+"_"+convertInt(rhash);
}


/*
 this is the modular insert for our hash maps.
 circle_vector is the vec3f that we want to store. 
 map is the map that we want to put it into.
 iterator is the iterator that we will use in order to find the value in the map
 hash_number is the int which determines which hash function to use
 */
std::map<string, Vec4f> hash_insert_modular(Vec3f circle_vector, std::map<string, Vec4f> map, std::map<string, Vec4f>::iterator iterator, int hash_number)
{
    Vec4f temp;
    string hash = hash_function(circle_vector);
    switch (hash_number) {
        case 1:
            hash = hash_function_modular(circle_vector, 0, 0, 0);
            break;
        case 2:
            hash = hash_function_modular(circle_vector, 1, 1, 0);
            break;
        case 3:
            hash = hash_function_modular(circle_vector, 1, 0, 0);
            break;
        case 4:
            hash = hash_function_modular(circle_vector, 0, 1, 0);
            break;
        case 5:
            hash = hash_function_modular(circle_vector, 0, 0, 1);
            break;
        case 6:
            hash = hash_function_modular(circle_vector, 1, 1, 1);
            break;
        case 7:
            hash = hash_function_modular(circle_vector, 1, 0, 1);
            break;
        case 8:
            hash = hash_function_modular(circle_vector, 0, 1, 1);
            break;
        default:
            break;
    }
    iterator = map.find(hash);
    if (iterator != map.end()) {
        //it is already added
        temp = map.at(hash);
        temp[0] = (temp[0] + circle_vector[0])/2.;
        temp[1] = (temp[1] + circle_vector[1])/2.;
        temp[2] = (temp[2] + circle_vector[2])/2.;
        temp[3]++;
        map.at(hash) = temp;
        
    } else {
        //it has not been added
        temp[0] = circle_vector[0];
        temp[1] = circle_vector[1];
        temp[2] = circle_vector[2];
        temp[3] = 1;
        
        map.insert(std::pair<string, Vec4f>(hash, temp));
    }
    return map;
    
}



/*
 This is the drawing function of the project and will attempt to gather and draw or store the found circles
 This is the only place in which the HoughCircles function is called.
 The reason why it takes in and void* is to get it to work with the openCV sliders which need that function prototype
 */
void drawHough(int, void*)
{
    Mat new_image = Mat::zeros( orig_src.size(), orig_src.type() );
    double c_thresh = contrast_threshy/20.;
    vector<Vec3f> circles;
    
    // Convert to grayscale
    cvtColor( orig_src, orig_gray, CV_BGR2GRAY );
    src_gray = orig_gray.clone();
    
    if (debugmode)
        printf("EDGE: %d\nCENTER: %d\nBLUR: %d\n\n", edge_threshy, center_threshy, blur_threshy * 2 + 1);
    
    // Blur input picture
    GaussianBlur( src_gray, src_gray, Size(blur_threshy * 2 + 1, blur_threshy * 2 + 1), 0, 0 );
    //imshow("after blur", src_gray);
    
    /// Apply the Hough Transform to find the circles
    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, cntr_distance, edge_threshy, center_threshy, min_circle_radius, max_circle_radius );
    Mat src_copy;
    if (debugmode)
        src_copy = orig_src.clone();
    else
        src_copy = src.clone();
    
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
            aggregated_map = hash_insert_modular(circles[i], aggregated_map, map_iterator, 1);
            //hash_insert(circles[i]);
            
            //this displays the cumulative circles onto the screen
            circle( src, center, radius, color, 2, 8, 0 );
        }
        
    }
    radii_vector = circles;
    
    circle_list.push_back( circles );
    
    if (debugmode) {
        imshow( "Hough Circle Transform Demo", src_copy );
        //std::cout << edge_threshy << " " << blur_threshy << " " << center_threshy << std::endl;
    }
}




/*
 This function takes in the current mat and then resizes it and returns the new resized mat.
 This does it in order to get a closer resolution size and value for the pictures that will be ran on
 */
Mat
set_image_resolution(Mat value){
    Mat small_image, resized_image;
    
    float percent = image_height/value.rows;
    
    resize(value, resized_image, Size(),percent, percent, INTER_LINEAR);
    return resized_image;
}

/*
 This is part of the logfile creation
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
        logfile << "Blur amount: "<< blur << std::endl;
        logfile << "Circle Center distance: "<< circle_centers << std::endl;
        logfile << "Canny edge threshold: "<< canny_threshold << std::endl;
        logfile << "Center threshold: "<< center_threshold << std::endl;
        logfile << "Color: " << temp_col<< std::endl;
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
        logfile << "Total_Circles: "<< total_circles << std::endl;
        logfile << "Total_Aggregated_Circles: "<< total_aggregated_circles << std::endl;
        logfile << "Total_Passes: "<< debug_passes_counter << std::endl;
        logfile << "Blur Range: "<< blur_low << "-" << blur_high << std::endl;
        logfile << "Edge Range: "<< edge_low << "-" << edge_high << std::endl;
        logfile << "Center Range: "<< cent_low << "-" << cent_high << std::endl;
        logfile << "----------------------------------------------------------------------\n";
        logfile << 'X' << '\t' << 'Y' << '\t' << 'R' << '\t' << "# of Circles" << std::endl;
        for (map_iterator = aggregated_map.begin(); map_iterator != aggregated_map.end() ; map_iterator++) {
            logfile << map_iterator->second[0] << '\t' << map_iterator->second[1] << '\t' <<
                map_iterator->second[2] << '\t' << map_iterator->second[3] << '\t' << '\n';
        }
        logfile.close();
    }
}


// this is the header for the logfile for the pass that has been ran.
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
 This function just outputs the circles to the logfile.
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


/*
 This is the main function of the program which generates all of the passes.
 It is possible to add more passes by adding more loops to the function and increasing the range of parameters that it takes in.
 */
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
    imshow( no_aggregation, src );
    imwrite(logfile_output + "no_aggregation.jpg", src);
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
 This just inserts the second map into the first map using the specified hashing function.
 occurence checks to see if the value needs to have more than the circle_occurence amount in the bucket.
 */

std::map<string, Vec4f> hash_loop( std::map<string, Vec4f> map_new, std::map<string, Vec4f> map_old, int hashing_function, bool occurence)
{
    for (map_iterator = map_old.begin(); map_iterator != map_old.end() ; map_iterator++) {
        Vec4f temp = map_iterator->second;
        if (occurence) {
            if (temp[3] >= circle_occurence) {
                Vec3f temp_3f;
                temp_3f[0] = temp[0];
                temp_3f[1] = temp[1];
                temp_3f[2] = temp[2];
                map_new = hash_insert_modular(temp_3f, map_new, map_iterator, hashing_function);
            }
        } else {
            Vec3f temp_3f;
            temp_3f[0] = temp[0];
            temp_3f[1] = temp[1];
            temp_3f[2] = temp[2];
            map_new = hash_insert_modular(temp_3f, map_new, map_iterator, hashing_function);
        }
        
    }
    return map_new;
}

/*
 hash_routine just goes through all of the currently avaliable hash functions in order to get rid of edge cases
 it then prints out the correctly aggregated values  and puts that back into the original map
 */
void hash_routine()
{
    std::map<string, Vec4f> aggregated_map_front;
    std::map<string, Vec4f> aggregated_map_back;
   
    aggregated_map_back = hash_loop(aggregated_map_back, aggregated_map, 2, true);
    
    aggregated_map_front = hash_loop(aggregated_map_front, aggregated_map_back, 3, false);
    aggregated_map_back.clear();

    aggregated_map_back = hash_loop(aggregated_map_back, aggregated_map_front, 4, false);
    aggregated_map_front.clear();
    
    aggregated_map_front = hash_loop(aggregated_map_front, aggregated_map_back, 5, false);
    aggregated_map_back.clear();
    
    aggregated_map_back = hash_loop(aggregated_map_back, aggregated_map_front, 6, false);
    aggregated_map_front.clear();
    
    aggregated_map_front = hash_loop(aggregated_map_front, aggregated_map_back, 7, false);
    aggregated_map_back.clear();
    
    aggregated_map_back = hash_loop(aggregated_map_back, aggregated_map_front, 8, false);
    aggregated_map_front.clear();
    
    
    
    Mat agg_src = orig_src.clone();
    for (map_iterator = aggregated_map_back.begin(); map_iterator != aggregated_map_back.end() ; map_iterator++) {
        
        Vec4f temp = map_iterator->second;

        Point center(cvRound(temp[0]), cvRound(temp[1]));
        int radius = cvRound(temp[2]);
        circle( agg_src, center, radius, Scalar(0,255, 0), 2, 8, 0 );
    }
    

    aggregated_map = aggregated_map_back;
    imshow( end_aggregation , agg_src);
    imwrite(logfile_output + "final_aggregation.jpg", agg_src);
    //imwrite(logfile_output + "circle_aggregate.jpg", agg_src);
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
            circle( agg_src, center, radius, Scalar(0,155, 0), 2, 8, 0 );
            total_aggregated_circles++;
        }
    }
    imshow(one_aggregation, agg_src);
    
    imwrite(logfile_output + "circle_aggregate.jpg", agg_src);
}

// Callback: when use picks 'File | Open' from main menu
void open_cb(Fl_Widget*, void*) {
    
    // Create the file chooser, and show it
    Fl_File_Chooser chooser(".",                        // directory
                            "*",                        // filter
                            Fl_File_Chooser::MULTI,     // chooser type
                            "Title Of Chooser");        // title
    chooser.show();
    
    // Block until user picks something.
    //     (The other way to do this is to use a callback())
    //
    while(chooser.shown())
    { Fl::wait(); }
    
    // User hit cancel?
    if ( chooser.value() == NULL )
    { fprintf(stderr, "(User hit 'Cancel')\n"); return; }
    
    // Print what the user picked
    //fprintf(stderr, "--------------------\n");
    //fprintf(stderr, "DIRECTORY: '%s'\n", chooser.directory());
    //fprintf(stderr, "    VALUE: '%s'\n", chooser.value());
    //fprintf(stderr, "    COUNT: %d files selected\n", chooser.count());
    
    // Multiple files? Show all of them
    //if ( chooser.count() > 1 ) {
    //    for ( int t=1; t<=chooser.count(); t++ ) {
    //        fprintf(stderr, " VALUE[%d]: '%s'\n", t, chooser.value(t));
    //    }
    //}
    
    // get src image, from file chosen
    orig_src = imread( chooser.value(), 1 );
    string temp = "Input File: \n";
    file_name = chooser.value();
    
    temp = temp + file_name;
    
    const char* fname = temp.c_str();
    
    message_buff->text(fname);
    
    // validate src image
    if( !orig_src.data )
           { exit(1); }
    
    // get data from src image
    original_row_amount = orig_src.rows;
    original_column_amount = orig_src.cols;
    
    // Resize image, for consistency
    orig_src = set_image_resolution(orig_src);
    src = orig_src.clone();
 
}

// Callback: when user picks 'Quit'
void quit_cb(Fl_Widget*, void*) {
    exit(0);
}

// Callback: set all of the values to default.
void default_cb(Fl_Widget*, void*) {
    max_blur_slider->value(20);
    min_blur_slider->value(10);
    max_edge_slider->value(300);
    min_edge_slider->value(100);
    max_cent_slider->value(80);
    min_cent_slider->value(80);
    ptol_slider->value(30);
    rtol_slider->value(30);
    occurence_slider->value(5);
}

// Callback: for when the run button is pushed
void run_cb(Fl_Widget*, void*) {
    
    bool run = true;
    string error_text = "";
    string error_style = "";
    
    // we are checking to see if the values are correct to be ran.
    if (max_blur_slider->value() < min_blur_slider->value()) {
        error_text = error_text + "Minimum blur value higher than maximum center value\n";
        error_style = error_style+"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
        run = false;
    }
    
    if (max_edge_slider->value() < min_edge_slider->value()) {
        error_text = error_text + "Minimum edge value higher than maximum center value\n";
        error_style = error_style+"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
        run = false;
    }
    
    if (max_cent_slider->value() < min_cent_slider->value()) {
        error_text = error_text + "Minimum center value higher than maximum center value\n";
        error_style = error_style+"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
        run = false;
    }
    if (file_name.compare("") == 0) {
        error_text = error_text + "No input file selected\n";
        error_style = error_style+"AAAAAAAAAAAAAAAAAAAAAA\n";
        run = false;
    }
    
    
    const char * c = error_text.c_str();
    const char * s = error_style.c_str();
    buff->text(c);
    sbuff->text(s);
    
    if(run)
    {
        // updating all of the global variables.
        blur_high = max_blur_slider->value();
        blur_low = min_blur_slider->value();
        edge_high = max_edge_slider->value();
        edge_low = min_edge_slider->value();
        cent_high = max_cent_slider->value();
        cent_low = min_cent_slider->value();
        pixel_tolerance = ptol_slider->value();
        radius_tolerance = rtol_slider->value();
        circle_occurence = occurence_slider->value();
        
        
        string window_name = "Hough Circle Transform Demo";
        if(debugmode){
            //printf("GETOPT ARGS\nBLUR:%d to %d\nEDGE:%d to %d\nCENT:%d to %d\n", blur_low, blur_high, edge_low, edge_high, cent_low, cent_high);
            // Create window and trackbars
            namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
            createTrackbar( "Hough Edge:", window_name, &edge_threshy, max_threshold, drawHough );
            createTrackbar( "Hough Center:", window_name, &center_threshy, max_threshold, drawHough );
            createTrackbar( "Gaussian Blur:", window_name, &blur_threshy, 31, drawHough );
            drawHough(0, 0);
            waitKey(0);
            cvDestroyAllWindows();
            std::cout << edge_threshy << " " << blur_threshy << " " << center_threshy << std::endl;
        }
        else {
            // Run circle detection, track timing also
            if(debugmode_passes)
                print_log_header(file_name, orig_src.rows, orig_src.cols, original_row_amount, original_column_amount);
            
            passes(edge_low, edge_high, blur_low, blur_high, cent_low, cent_high);
            
            
            draw_aggregate_list();
            
            hash_routine();
            
            waitKey(0);
            cvDestroyAllWindows();
            //imwrite(logfile_output + "circle_recog.jpg", src);
            
            //write_circle_list();
            //std::cout << "--------------------------"<<std::endl;
            //write_aggregate_list();
            if(!debugmode_passes)
                print_aggregate_logfile(file_name, orig_src.rows, orig_src.cols, original_row_amount, original_column_amount);
            // print new log file
            
        }
        
        //std::cout << debug_passes_counter << std::endl;
        //std::cout << pixel_tolerance << " " << radius_tolerance;
    }
}



// Callback: The debug mode callback checks to see what mode to run in the debug.
// it switches between none, sliders, or output pre aggregated circles to the logfile.
void debug_cb(Fl_Widget*, void* a) {
    int debug_flag = *((int*)(&a));
    
    if(debug_flag == 0)
    {
        debugmode = false;
        debugmode_passes = false;
    }
    else if(debug_flag == 1)
    {
        debugmode = true;
        debugmode_passes = false;
    }
    else if(debug_flag == 2)
    {
        debugmode_passes = true;
        debugmode = false;
    }
}

int main(int argc, char** argv)
{
    int c;
    char *token;
    
    //  The original GetOPS.
    while ((c = getopt (argc, argv, "o:b:e:c:r:p:")) != -1)
        switch (c)
    {
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
    
    // Generating the gui section
    Fl_Window win(600, 500, "Circle Image Recognition");
    Fl_Menu_Bar menubar(0,0,600,25);
    Fl_Menu_Item menutable[] = {
        //{"foo",0,0,0,FL_MENU_INACTIVE},
        {"&File", 0, 0, 0, FL_SUBMENU},
        {"&Open", 0, open_cb},
        {"&Quit", 0, quit_cb},
        {0},
        {"&Debug", 0, 0, 0, FL_SUBMENU},
        {"&None", 0, debug_cb, (void *)0, FL_MENU_RADIO|FL_MENU_VALUE},
        {"&Sliders", 0, debug_cb, (void *)1, FL_MENU_RADIO},
        {"&Logfile", 0, debug_cb, (void *)2, FL_MENU_RADIO},
        {0},
        {0}
    };
    
    (&menubar)->copy(menutable);

    int YP = 170;
    
    //set the text on the screen
    Fl_Multiline_Output *out = new Fl_Multiline_Output(20,130,0,0,"Pre Processing Sliders");
    out->labelsize(20);
    out->align(FL_ALIGN_RIGHT_TOP);
    
    
    // this section sets all of the sliders onto the screen.
    //x, y, width, height on screen
    max_blur_slider = new SliderInput(20,YP + 50,150,20,"Max Blur Amount");
    max_blur_slider->bounds(2,30);       // set min/max slider
    max_blur_slider->value(20);           // set initial value
    
    
    min_blur_slider = new SliderInput(20,YP,150,20,"Min Blur Amount");
    min_blur_slider->bounds(1,29);       // set min/max for slider
    min_blur_slider->value(10);           // set initial value
    
    max_edge_slider = new SliderInput(220,YP + 50,150,20,"Max Edge Amount");
    max_edge_slider->bounds(50,500);       // set min/max slider
    max_edge_slider->value(300);           // set initial value
    
    min_edge_slider = new SliderInput(220, YP,150,20,"Min Edge Amount");
    min_edge_slider->bounds(50,500);       // set min/max for slider
    min_edge_slider->value(100);           // set initial value
    
    max_cent_slider = new SliderInput(420, YP + 50,150,20,"Max Center Amount");
    max_cent_slider->bounds(30,150);       // set min/max slider
    max_cent_slider->value(80);           // set initial value
    
    min_cent_slider = new SliderInput(420,YP,150,20,"Min Center Amount");
    min_cent_slider->bounds(30,150);       // set min/max for slider
    min_cent_slider->value(80);           // set initial value
    
    
    Fl_Multiline_Output *out1 = new Fl_Multiline_Output(20,YP + 80,0,0,"Post Processing Sliders");
    out1->labelsize(20);
    out1->align(FL_ALIGN_RIGHT_TOP);
    
    ptol_slider = new SliderInput(20,YP + 120,150,20,"Pixel Aggregation Tolerance");
    ptol_slider->bounds(1,150);       // set min/max for slider
    ptol_slider->value(30);           // set initial value
    
    rtol_slider = new SliderInput(220,YP + 120,150,20,"Radius Aggregation Tolerance");
    rtol_slider->bounds(1,150);       // set min/max for slider
    rtol_slider->value(30);           // set initial value
    
    
    occurence_slider = new SliderInput(420,YP + 120,150,20,"Aggregation Threshold");
    occurence_slider->bounds(1,100);       // set min/max for slider
    occurence_slider->value(5);           // set initial value
    
    
    
    
    
    //Generate a new text buffer for the gui to write to.
    buff = new Fl_Text_Buffer();
    message_buff = new Fl_Text_Buffer();
    sbuff = new Fl_Text_Buffer();
    
    message_disp = disp = new Fl_Text_Display(20, 30, 550, 100);
    message_disp->buffer(message_buff);
    
    
    disp = new Fl_Text_Display(20, 340, 550 , 100, "Errors");
    int stable_size = sizeof(stable)/sizeof(stable[0]);
    disp->highlight_data(sbuff, stable, stable_size, 'A', 0, 0);
    disp->buffer(buff);

    
    //adding the buttons and button callbacks.
    Fl_Button *but2 = new Fl_Button(300,450,140,25,"Default");
    but2->callback(default_cb);
    win.add(but2);
    
    Fl_Button *but1 = new Fl_Button(450,450,140,25,"Run");
    but1->callback(run_cb);
    win.add(but1);
    
    win.show();

    return(Fl::run());
}

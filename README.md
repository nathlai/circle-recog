Circle Recognition Project

GENERAL USAGE NOTES
--------------------
    
    This project has only been tested on Mac computers running Xcode version 4.6

    This project uses openCV in order to process the images and look for circles using the hough transform
    
    It is necessary to have openCV installed and linked to the Xcode project in order to get the application to work
    
    For information regarding where to download Xcode for Mac and use it see https://developer.apple.com/xcode/
    
    For information regarding openCV and downloading it see http://opencv.org
    
    For information on how to set up opencv on the mac see http://tilomitra.com/opencv-on-mac-osx/


Installing in XCODE
--------------------

    In order to run this project in xcode, it is first necessary to create a new project.  From the selection menu, chose OS X application and then chose Command line tool.  Choose a name for the Product Name and set the Type to C++.  From here follow the directions specified in the article on setting up opencv on the mac.  These directions will link the opencv files to the current project.  Replace the default main.cpp file with our main.cpp file and this is all that is necessary to set up the project.  In order to run the project and get results, review the next section.


Using in XCODE
---------------

    In order to use this project in xcode, it is necessary to have opencv
    correctly linked to the project.  Once that is done, one needs to specify an
    image in order to detect the circles.  This is done by secting the product
    drop down menu, select scheme, and then chose edit.  From here we can select
    the command line values of the picture that we want to look through.
    Absolute pathing works the best.  This should be the first parameter. From
    our experience, we had to ensure that the compiler is set to the GCC
    compiler, versus the default Apple LLVM compiler. Otherwise, the Apple
    compiler with throw errors.
    
    Run the xcode program, and play with the slider bars in order to improve circle recognition accuracy.  When the window exits, a simple log file will be written.  This will be adjusted in future improvements.
    
    
    
Future improvements
--------------------
    Auto adjust image saturation and contrast for pre-processing
    
    Auto adjust image size in order to have more accurate results
    
    Auto adjust blur based off of image size and/or original resolution
    
    Create optimal calculations for finding real circles
    
    Delete obviously incorrect circles that go outside of known boundaries.
    
    Write used values to a log file in order to keep track of results.  Let users determine where the logfile is written to.
    
    
    
    

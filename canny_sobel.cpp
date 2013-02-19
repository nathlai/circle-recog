// Code snippet to observe what cvCanny and cvSobel does
    CvMat img = (CvMat)new_image;
    CvMat* edges = cvCreateMat( new_image.rows, new_image.cols, CV_8UC1 );
    cvCanny( &img, edges, MAX(edge_threshy/2,1), edge_threshy, 3 );
    Mat test = edges;
    imshow("test", test);

    CvMat* dx = cvCreateMat( new_image.rows, new_image.cols, CV_16SC1 );
    CvMat* dy = cvCreateMat( new_image.rows, new_image.cols, CV_16SC1 );
    CvMat gray = (CvMat)src_gray;
    cvSobel( &gray, dx, 1, 0, 3 );
    cvSobel( &gray, dy, 0, 1, 3 );
    Mat testx = dx;
    Mat testy = dy;
    imshow("testx", testx);
    imshow("testy", testy);

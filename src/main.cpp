#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <stdio.h>
#include <vector>

using namespace cv;
using namespace std;

/* SPECIFIC VARIABLE */
char* vdoFilename = "INSERT YOUR VIDEO PATH";
String bgFileName = "INSERT BACKGROUND PICTURE PATH";
char* logfileName = "INSERT LOG FILE PATH";
String outputVideoMask = "INSERT OUTPUT MASK VIDEO PATH";
String outputVideoDetection = "INSERT OUTPUT DETECTION VIDEO PATH";


/* DEFINE VARIABLES AND FUNCTIONS */
// GLOBAL&LOCAL VARIABLE
int fcount = 1, fWidth = 360, fHeight = 300;
double fps, format, totalf, frameH, frameW, fourcc;
int temp = 50;
double prevArea[5000], prevLen[5000], timer[5000], status[5000];
int noiseW = 10, noiseH = 10;
char* nextline = new char[100];
int pepW = 0,pepH = 0, objW = 0, objH_set = 0, objH_real = 0;
Mat prevImg[5000];
Scalar colorG = Scalar(0, 255, 0); //green
Scalar colorR = Scalar(0, 0, 255); //red
Scalar colorP = Scalar(255, 0, 255); //pink
bool saveBG = true;
std::stringstream time_screen;
string time_string;
string event_string;

// METHOD
int VDO_Acquisition();
void VDO_Detection(Mat background, Mat firstFrame,Mat currentFrame);
bool Event_Classification(Mat img_a, Mat img_b);
void Result_Representation();

void info()
{
    cout
    << "------------------------------------------------------------"  << endl
    << endl
    << "This program shows real time video from the CCTV and warn "   << endl
    << "in case of unattended objects or the object is stolen."  << endl
    << endl
    << "------------------------------------------------------------"  << endl
    << endl;
}

void vdoinfo(){
    cout << "VDO INFO : " << vdoFilename  <<endl;
    cout << " Frame rate (fps) : " << fps <<endl;
    cout << " Total frame (frame) : " << totalf <<endl;
    cout << " Time (Sec): " << ( totalf/fps) <<endl;
    //cout << "   Format : " << format <<endl;
    cout << " Frame hight : " << frameH <<endl;
    cout << " Frame width : " << frameW <<endl;
}

FILE * outputfile;
void logfile()
{
    char* logfileName_ = logfileName;
    outputfile = fopen(logfileName_,"w");
    // fputs("Writing this to a file.\n", outputfile);
}

int milisec_time = 0, sec_time = 0, min_time = 0;
void clocktime() {
    if (milisec_time == 60){
        milisec_time = 0;
        sec_time++;
        if(sec_time == 60){
            sec_time = 0;
            min_time++;
        }
    }
    if(milisec_time >= 0 && milisec_time < 10){
        if(sec_time >= 0 && sec_time < 10){
            if(min_time >= 0 && min_time < 10){
                cout << "Time: 0" << min_time << ":0" << sec_time << ":0" << milisec_time ;
                time_screen << "Time: 0" << min_time << ":0" << sec_time << ":0" << milisec_time ;
            }
            else{
                cout << "Time: " << min_time << ":0" << sec_time << ":0" << milisec_time ;
                time_screen << "Time: " << min_time << ":0" << sec_time << ":0" << milisec_time ;
            }
        }
        else{
            if(min_time >= 0 && min_time < 10){
                cout << "Time: 0" << min_time << ":" << sec_time << ":0" << milisec_time ;
                time_screen << "Time: 0" << min_time << ":" << sec_time << ":0" << milisec_time ;
            }
            else{
                cout << "Time: " << min_time << ":" << sec_time << ":0" << milisec_time;
                time_screen << "Time: " << min_time << ":" << sec_time << ":0" << milisec_time ;
            }
        }
    }
    else{
        if(sec_time >= 0 && sec_time < 10){
            if(min_time >= 0 && min_time < 10){
                cout << "Time: 0" << min_time << ":0" << sec_time << ":" << milisec_time ;
                time_screen << "Time: 0" << min_time << ":0" << sec_time << ":" << milisec_time ;
            }
            else{
                cout << "Time: " << min_time << ":0" << sec_time << ":" << milisec_time  ;
                time_screen << "Time: " << min_time << ":0" << sec_time << ":" << milisec_time ;
            }
        }
        else{
            if(min_time >= 0 && min_time < 10){
                cout << "Time: 0" << min_time << ":" << sec_time << ":" << milisec_time ;
                time_screen << "Time: 0" << min_time << ":" << sec_time << ":" << milisec_time ;
            }
            else{
                cout << "Time: " << min_time << ":" << sec_time << ":" << milisec_time ;
                time_screen << "Time: " << min_time << ":" << sec_time << ":" << milisec_time ;
            }
        }
    }
}

void defineWindow(){
    namedWindow("BACKGROUND",CV_WINDOW_AUTOSIZE);
    namedWindow("FIRST_FRAME",CV_WINDOW_AUTOSIZE);
    namedWindow("CURRENT_FRAME",CV_WINDOW_AUTOSIZE);
    namedWindow("bg");
    namedWindow("ff");
    namedWindow("cf");

    logfile();

    for (int i = 0; i < temp; i++)
    {prevArea[i] = 0; prevLen[i] = 0;  timer[i]=0; status[i]= 0;}
}

void releasewindow(){
    fclose(outputfile);
}
VideoWriter oVideoWriterMask;
VideoWriter oVideoWriterDetection;
void defineSaveVDO(){
    String outputVideoMask_=outputVideoMask;
    String outputVideoDetection_=outputVideoDetection;
    oVideoWriterMask.open(
        outputVideoMask_,
        CV_FOURCC('D', 'I', 'V', '3'),
        fps,
        Size((int)fWidth, (int)fHeight),
        true
    ); //initialize the VideoWriter object
    oVideoWriterDetection.open(
        outputVideoDetection_,
        CV_FOURCC('D', 'I', 'V', '3'),
        fps,
        Size((int)fWidth, (int)fHeight) , true
    ); //initialize the VideoWriter object
}

String fullbody_cascade_name = "../Xcode/RealProject/hogcascade_pedestrians.xml";
String fullbody_haarcascade_name = "../Xcode/RealProject/haarcascade_fullbody.xml";
String lowerbody_haarcascade_name = "../Xcode/RealProject/haarcascade_lowerbody.xml";
String upperbody_haarcascade_name = "../Xcode/RealProject/haarcascade_upperbody.xml";
String msc_upperbody_haarcascade_name = "../Xcode/RealProject/haarcascade_mcs_upperbody.xml";
CascadeClassifier fullbody_cascade;
CascadeClassifier fullbody_haarcascade;
CascadeClassifier lowerbody_haarcascade;
CascadeClassifier upperbody_haarcascade;
CascadeClassifier msc_upperbody_haarcascade;

void defineCascade(){
    if(!fullbody_cascade.load(fullbody_cascade_name)){
        printf("--(!)Error loading\n");
        exit(0);
    }
    if(!fullbody_haarcascade.load(fullbody_haarcascade_name)){
        printf("--(!)Error loading\n");
        exit(0);
    }
    if( !lowerbody_haarcascade.load(lowerbody_haarcascade_name)){
        printf("--(!)Error loading\n");
        exit(0);
    }
    if( !upperbody_haarcascade.load(upperbody_haarcascade_name)){
        printf("--(!)Error loading\n");
        exit(0);
    }
    if(!msc_upperbody_haarcascade.load(msc_upperbody_haarcascade_name)){
        printf("--(!)Error loading\n");
        exit(0);
    }
}
void showHistogram(Mat& img)
{
    int bins = 256;             // number of bins
    int nc = img.channels();    // number of channels
    vector<Mat> hist(nc);       // array for storing the histograms
    vector<Mat> canvas(nc);     // images for displaying the histogram
    int hmax[3] = {0,0,0};      // peak value for each histogram
    for (int i=0; i<hist.size(); i++)
        hist[i] = Mat::zeros(1, bins, CV_32SC1);
    for (int i=0; i<img.rows; i++)
    {
        for (int j=0; j<img.cols; j++)
        {
            for (int k=0; k<nc; k++)
            {
                uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<Vec3b>(i,j)[k];
                hist[k].at<int>(val) += 1;
            }
        }
    }
    for (int i=0; i<nc; i++)
    {
        for (int j=0; j<bins-1; j++)
            hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
    }
    const char* wname[3] = { "blue", "green", "red" };
    Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };
    for (int i = 0; i < nc; i++)
    {
        canvas[i] = Mat::ones(125, bins, CV_8UC3);
        for (int j=0, rows=canvas[i].rows; j<bins-1; j++)
        {
            line(
                 canvas[i],
                 Point(j, rows),
                 Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])),
                 nc == 1 ? Scalar(200,200,200) : colors[i],
                 1, 8, 0
                 );
        }
        imshow(nc == 1 ? "value" : wname[i], canvas[i]);
    }
}

void showHuman(std::vector<Rect> body, Mat img, Mat img_gray){
    for(size_t i=0; i<body.size(); i++){
        Point center( body[i].x + body[i].width*0.5, body[i].y + body[i].height*0.5 );
        ellipse(img, center, Size( body[i].width*0.5, body[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = img_gray( body[i] );
        std::vector<Rect> bags;

        //-- In each body, detect bags
        //eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        //
        for(size_t j=0; j<bags.size(); j++){
            Point center( body[i].x + bags[j].x + bags[j].width*0.5, body[i].y + bags[j].y + bags[j].height*0.5 );
            int radius = cvRound( (bags[j].width + bags[j].height)*0.25 );
            //circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
        //imshow("img",img);
    }
}
bool isHuman(Mat& img){
    std::vector<Rect> body;
    Mat img_gray;
    cvtColor(img, img_gray, CV_BGR2GRAY);
    equalizeHist(img_gray, img_gray);

    fullbody_cascade.detectMultiScale(
        img_gray, body, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30)
    );
    if(body.size() > 0){
        //showHuman(body, img, img_gray);
        //printf("  Can detect full\n");
        return true;
    }

    fullbody_haarcascade.detectMultiScale(
        img_gray, body, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30)
    );
    if(body.size() > 0){
        //showHuman(body, img, img_gray);
        //printf("  Can detect full haar\n");
        return true;
    }

    upperbody_haarcascade.detectMultiScale(
        img_gray, body, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30)
    );

    if(body.size() > 0){
        //showHuman(body, img, img_gray);
        //printf("  Can detect upper\n");
        return true;
    }

    msc_upperbody_haarcascade.detectMultiScale(
        img_gray, body, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30)
    );
    if(body.size() > 0){
        //showHuman(body, img, img_gray);
        //printf("  Can detect msc upper\n");
        return true;
    }

    lowerbody_haarcascade.detectMultiScale(
        img_gray, body, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30)
    );
    if(body.size() > 0){
        //showHuman(body, img, img_gray);
        //printf("  Can detect lower\n");
        return true;
    }
    else{
        //printf("  \n");
        return false;
    }
    return false;
}

void header(){
    info();
    defineCascade();
    defineWindow();
}
void footer(){
    releasewindow();
}

int main(){
    header();
    //Function
    /*1. VDO_Acquisition
     [input]
     [output]
     */
    //any error occur in this process
    if(VDO_Acquisition() == -1) printf("Error!!");
    /*2. VDO_Detection
     [input]
     [output]
     */
    /*3. Event_Classification
     [input]
     [output]
     */
    // if(!Event_Classification() == -1) printf("Error!!");
    /*4. Result_Representation
     [input]
     [output]
     */
    // any error occur in this process
    footer();
    return 0;
}


int VDO_Acquisition(){
    VideoCapture cap(vdoFilename); // open the video file for reading
    if ( !cap.isOpened() )  // if not success, exit program
    {
        cout << "Cannot open a video device or video file!\n" << endl;
        return -1;
    }
    fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    //format = cap.get(CV_CAP_PROP_FORMAT);
    totalf = cap.get(CV_CAP_PROP_FRAME_COUNT);
    frameH = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    frameW= cap.get(CV_CAP_PROP_FRAME_WIDTH);
    fourcc = cap.get(CV_CAP_PROP_FOURCC);
    vdoinfo();
    defineSaveVDO();
    Mat background, firstFrame, currentFrame;
    background  = imread(bgFileName, CV_LOAD_IMAGE_COLOR);  // Read the file
    while(1){
        Mat frame;
        bool bSuccess = cap.read(frame); // read a new frame from video
        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }
        currentFrame = frame;
        if( firstFrame.empty() ) {
            firstFrame = currentFrame;
        }
        // PROCESS:: Resizing
        Size s( fWidth, fHeight ); //Width: 360 Height: 300
        resize( firstFrame, firstFrame, s, 0, 0, CV_INTER_AREA );
        resize( currentFrame, currentFrame, s, 0, 0, CV_INTER_AREA );
        if( !background.empty() && saveBG){
            //vector that stores the compression parameters of the image
            vector<int> compression_params;
            //specify the compression technique
            compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
            compression_params.push_back(98); //specify the compression quality
            //write first image of the frame as background image to file
            bool bSuccess = imwrite(
                "BACKGROUND IMAGE PATH", firstFrame, compression_params
            );
            if (bSuccess){
                saveBG = false;
                cout << "\n[Saved background image to file.]\n" << endl;
            }
            else {
                cout << "ERROR : Failed to save the image" << endl;
                system("pause"); //wait for a key press
            }
            // Read the file
            // background  = imread(bgFileName, CV_LOAD_IMAGE_COLOR);
        }
        VDO_Detection(background, firstFrame, currentFrame);
        //imshow("BACKGROUND", background);
        //imshow("FIRST_FRAME", firstFrame);
        moveWindow("CURRENT_FRAME", 0, 0);
        imshow("CURRENT_FRAME", currentFrame);
        // wait for 'esc' key press for 30 ms.
        // If 'esc' key is pressed, break loop
        if(waitKey(30) == 27)
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
    }
    return 0;
}

void VDO_Detection(Mat background, Mat firstFrame, Mat currentFrame){
    printf("Frame#%d   ", fcount++);
    if(abs((int)fcount % (int)fps) == 0){
        milisec_time++;
    }
    // sprintf(nextline,"Frame#%d\n",fcount);
    fputs(nextline, outputfile);
    clocktime();
    printf("\n ");
    fprintf(outputfile, "\n  ");
    Mat a_blur, b_blur;
    blur(firstFrame, a_blur, cv::Size(4,4));
    blur(currentFrame, b_blur, cv::Size(4,4));
    Mat c;
    absdiff(b_blur, a_blur, c);
    // imshow("DIFFERENCE", c);
    std::vector<Mat> channels;
    split(c, channels);
    Mat d = Mat::zeros(c.size(), CV_8UC1);
    for (int i=0; i<channels.size(); i++)
    {
        Mat thresh;
        threshold(channels[i], thresh, 25 , 100, CV_THRESH_BINARY);
        /*//CV_THRESH_BINARY
         //CV_THRESH_BINARY_INV
         //CV_THRESH_MASK
         //CV_THRESH_OTSU
         //CV_THRESH_TOZERO
         //CV_THRESH_TOZERO_INV
         //CV_THRESH_TRUNC*/
        d |= thresh;
    }
    Mat kernel, e;
    getStructuringElement(MORPH_RECT, Size(10,10));
    morphologyEx(d, e, MORPH_CLOSE, kernel, Point(-1,-1));
    //imshow("MORPHOLOGY", e);
    // Find all contours
    Mat threshold_output;
    vector< vector< Point> > contour;
    vector<Vec4i> hierarchy;
    /// Detect edges using Threshold
    threshold( e, threshold_output, 50, 255, THRESH_BINARY );
    findContours(
        threshold_output,
        contour,
        hierarchy,
        CV_RETR_EXTERNAL,
        CV_CHAIN_APPROX_SIMPLE
    );
    vector<vector<Point> > contours_poly( contour.size() );
    vector<Rect> boundRect(contour.size());
    vector<Point2f>center(contour.size());
    vector<float>radius(contour.size());
    /// Get the moments
    vector<Moments> mu(contour.size());
    vector<Point2f> mc(contour.size());
    // Select only large enough contours
    vector< vector< Point> > objects;
    for (int i=0; i<contour.size(); i++)
    {
        approxPolyDP( Mat(contour[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        mu[i] = moments( contour[i], false );
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
        double area = contourArea(contour [i]);
        //if (area > 10000 && area < 90000)
        objects.push_back(contour [i]);
    }
    // Use the filtered blobs above to create a mask image to
    // extract the foreground object
    Mat mask =  Mat::zeros( threshold_output.size(), CV_8UC3);
    drawContours(mask, objects, -1, CV_RGB(255,255,255), -1);
    // Highlight the foreground object by darken the rest of the image
    Mat detect = currentFrame.clone();
    Mat drawing = Mat::zeros( mask.size(), CV_8UC3 );
    for (int i=0; i<contour.size(); i++)
    {
        int x = boundRect[i].x;         int y = boundRect[i].y;
        int w = boundRect[i].width;     int h = boundRect[i].height;
        double currArea = contourArea(contour[i]);
        double currLen = arcLength( contour[i], true );
        if(w>noiseW && h>noiseH){
            //-- starting object classification
            Mat cropImg;
            Rect myROI(x, y, w, h);
            cropImg = currentFrame(myROI);
            // first time //
            if(prevArea[i] == 0 && prevLen[i]==0){
                prevArea[i] = currArea; //cout << "prevArea["<<i<<"] " << prevArea[i] <<std:: endl;
                prevLen[i] = currLen; //cout << "prevLen["<<i<<"] " << prevLen[i]  <<std:: endl;
                if(status[i]==1) status[i] =1; //people
            }
            if( isHuman(cropImg) ){
                pepW = w;
                pepH = h;
                objH_set = pepH/2;
                prevArea[i] = currArea;
                prevLen[i] = currLen;
                status[i] = 1; //people
            }// end if( isHuman(cropImg) )
            else{
                objH_real = h;
                if(timer[i] < 5 && objH_real <= objH_set) { //object
                    status[i] = 2;
                    if(abs(ceil((int)prevArea[i])- ceil((int)currArea)) <= 5){
                        timer[i]++;
                        //-- starting event classification
                        Mat img_bg,img_ff,img_cf;
                        Rect myROI(x, y, w, h);
                        img_bg = background(myROI);
                        img_ff = firstFrame(myROI);
                        img_cf = currentFrame(myROI);
                        if( Event_Classification(img_bg ,img_ff )  ){
                            if( !Event_Classification(img_bg,img_cf )) {
                                //if(timer[i]>fps*4)
                                status[i] = 3; //unattended object
                            }
                        }
                        else{
                            status[i] = 4; //stolen object
                        }
                        // moveWindow("bg", 500, 500);
                        // imshow("bg",img_bg);
                        // moveWindow("ff", 200, 360);
                        // imshow("ff",img_ff);
                        // moveWindow("cf", 300, 360);
                        // imshow("cf",img_cf);
                        // -- end event classification
                    }
                }
                else{
                    if(status[i] != 2){
                        prevArea[i] = currArea;
                        prevLen[i] = currLen;
                    }
                    if(status[i] != 1) status[i] = -1; //unknow
                }
            }
            // end object classification
            if(status[i] == 1){
                putText(
                    detect,
                    "People",
                    cvPoint(boundRect[i].x, boundRect[i].y),
                    FONT_HERSHEY_COMPLEX_SMALL,
                    0.8,
                    cvScalar(200,200,250),
                    1,
                    CV_AA
                );
                /*
                rectangle(
                    currentFrame,
                    boundRect[i].tl(),
                    boundRect[i].br(),
                    colorG,
                    2,
                    8,
                    0
                )
                */
                printf("People:%.2f\n", (int)i);
                fprintf(outputfile, "People:%.2f\n",(int)i);
                event_string = "none";
            }
            else if(status[i] == 2){
                putText(
                    detect,
                    "Object",
                    cvPoint(boundRect[i].x, boundRect[i].y),
                    FONT_HERSHEY_COMPLEX_SMALL,
                    0.8,
                    cvScalar(200,200,250),
                    1,
                    CV_AA
                );
                rectangle(
                    detect,
                    boundRect[i].tl(),
                    boundRect[i].br(),
                    colorG,
                    2,
                    8,
                    0
                );
                printf( "Object:%.2f\n",(int)i);
                fprintf(outputfile,"Object:%.2f\n",(int)i);
                event_string = "none";
            }
            else if(status[i] == 3){
                putText(
                    detect,
                    "Unattended Object",
                    cvPoint(
                        boundRect[i].x,
                        boundRect[i].y
                    ),
                    FONT_HERSHEY_COMPLEX_SMALL,
                    0.8,
                    cvScalar(200,200,250),
                    1,
                    CV_AA
                );
                rectangle(
                    detect,
                    boundRect[i].tl(),
                    boundRect[i].br(),
                    colorR,
                    2,
                    8,
                    0
                );
                printf("Object:%.2f\n  Event:Unattended object", (int)i);
                fprintf(
                    outputfile,
                    "Object:%.2f\n   Event:Unattended object",
                    (int)i
                );
                event_string = "unattended";
            }
            else if(status[i] == 4){
                putText(
                    detect,
                    "Stolen Object",
                    cvPoint(boundRect[i].x, boundRect[i].y),
                    FONT_HERSHEY_COMPLEX_SMALL,
                    0.8,
                    cvScalar(200,200,250),
                    1,
                    CV_AA
                );
                rectangle(
                    detect,
                    boundRect[i].tl(),
                    boundRect[i].br(),
                    colorR,
                    2,
                    8,
                    0
                );
                printf("Object:%.2f\n  Event:Stolen object", (int)i);
                fprintf(
                    outputfile,
                    "Object:%.2f\n   Event:Stolen object",
                    (int)i
                );
                event_string = "stolen";
            }
            else{
                printf("Object:Unknown Event:Unknown");
                fprintf(outputfile, "Object:Unknown  Event:Unknown");
                event_string = "none";
                /*
                putText(
                    detect,
                    "Unknown.",
                    cvPoint(boundRect[i].x, boundRect[i].y),
                    FONT_HERSHEY_COMPLEX_SMALL,
                    0.8,
                    cvScalar(200,200,250),
                    1,
                    CV_AA
                );
                */
                // rectangle(  detect, boundRect[i].tl(), boundRect[i].br(), colorP, 2, 8, 0 );
            }
            drawContours(
                drawing,
                contour,
                i,
                colorG,
                2,
                8,
                hierarchy,
                0,
                Point()
            );
            circle(drawing, mc[i], 4, colorP, -1, 8, 0);
            printf("\n ");
            fprintf(outputfile, "\n  ");
        }
    }
    printf("\n");
    fprintf(outputfile, "\n");
    // imshow("MORPHOLOGY", e);
    // imshow("MASK",mask);
    oVideoWriterMask.write(mask);
    // imshow("CONTOUR",drawing);
    moveWindow("DETECTION", 360, 0);
    rectangle( detect,
              Point( -150,5 ),
              Point( 400,5),
              Scalar( 0, 0, 0 ),
              20,
              200 );
    time_string = time_screen.str();
    time_screen.str("");
    time_screen.clear();
    putText(
        detect,
        time_string,
        cvPoint(10,10),
        FONT_HERSHEY_COMPLEX_SMALL,
        0.5,
        cvScalar(255,255,255),
        1,
        CV_AA
    );
    putText(
        detect,
        "Event: " + event_string,
        cvPoint(220,10),
        FONT_HERSHEY_COMPLEX_SMALL,
        0.5,
        cvScalar(255,255,255),
        1,
        CV_AA
    );
    imshow("DETECTION",detect);
    oVideoWriterDetection.write(detect);
}
bool Event_Classification(Mat img_a, Mat img_b ){
    Mat a_blur, b_blur;
    blur(img_a, a_blur, cv::Size(4,4));
    blur(img_b, b_blur, cv::Size(4,4));
    Mat c;
    absdiff(b_blur, a_blur, c);
    std::vector<Mat> channels;
    split(c, channels);
    Mat d = Mat::zeros(c.size(), CV_8UC1);
    for (int i=0; i<channels.size(); i++)
    {
        Mat thresh;
        threshold(channels[i], thresh, 20 , 255, CV_THRESH_BINARY);
        d |= thresh;
    }
    Mat kernel, e;
    getStructuringElement(MORPH_RECT, Size(10,10));
    morphologyEx(d, e, MORPH_CLOSE, kernel, Point(-1,-1));
    // Find all contours
    Mat threshold_output;
    vector< vector< Point> > contour;
    vector<Vec4i> hierarchy;
    /// Detect edges using Threshold
    threshold(e, threshold_output, 50, 255, THRESH_BINARY );
    findContours(
        threshold_output,
        contour,
        hierarchy,
        CV_RETR_EXTERNAL,
        CV_CHAIN_APPROX_SIMPLE
    );
    vector<vector<Point> > contours_poly( contour.size() );
    vector<Rect> boundRect( contour.size() );
    vector<Point2f>center( contour.size() );
    vector<float>radius( contour.size() );
    /// Get the moments
    vector<Moments> mu(contour.size());
    vector<Point2f> mc( contour.size() );

    // Select only large enough contours
    vector< vector< Point> > objects;
    for (int i = 0; i < contour.size(); i++)
    {
        approxPolyDP( Mat(contour[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        mu[i] = moments( contour[i], false );
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );

        double area = contourArea(contour [i]);
        //if (area > 10000 && area < 90000)
        objects.push_back(contour [i]);
    }

    // Use the filtered blobs above to create a mask image to
    // extract the foreground object
    Mat mask =  Mat::zeros( threshold_output.size(), CV_8UC3);
    drawContours(mask, objects, -1, CV_RGB(255,255,255), -1);
    // Highlight the foreground object by darken the rest of the image
    int x, y, w, h;
    if(contour.size()>0){
        for (int i = 0; i < contour.size(); i++) {
            x = boundRect[i].x;         y = boundRect[i].y;
            w = boundRect[i].width;     h = boundRect[i].height;
            moveWindow("mk", 0, 360);
            imshow("mk", mask);
        }
        return false;
    }
    return true;
}

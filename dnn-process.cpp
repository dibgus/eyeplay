//#include <boost/python.hpp> //todo interop: send image from python, return contours
#include <dlib/dnn.h> //dynamic neural network module with mmod of dlib
#include <dlib/data_io> //reading the neural network definitions
#include <dlib/gui_widgets> //gui to display results
#include <dlib/opencv.h> //dlib opencv bindings
#include <cv2> //used for video stream

using namespace std;
using namespace dlib;
using namespace cv;

find_cards()
{
    net_type neuralnet;
    deserialize("machine-learned/detector-dnn.dat") >> neuralnet;
    //declare and load the neural net
    VideoCapture feed(0); //get the camera feed
    if(!feed.isOpened())
        return -1;
    image_window win; //where images will display. May be changed to cv's image display
    for (;;)
    {
        Mat frame;
        feed >> frame; //load current frame
        //cvtColor(frame, frame, BGR2RGB)
        cv_image<bgr_pixel> cimg(frame) //convert image from mat to dlib compatible format
        //might need to convert to rgb, in which case, uncomment above
        auto contours = neuralnet(frame) //detect contours with neuralnet
        win.clear_overlay(); //clear any previous bounding boxes
        win.set_image(frame); //set disp of current frame
        for (auto&& c : contours)
            win.add_overlay(c); //draw bounding boxes
    }

}

/**
BOOST_PYTHON_MODULE(detectcards)
{
    using namespace boost::python
    def("find_cards", find_cards)
}
*/
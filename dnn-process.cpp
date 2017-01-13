//#include <boost/python.hpp> //todo interop: send image from python, return contours
#include <dlib/dnn.h> //dynamic neural network module with mmod of dlib
#include <dlib/data_io.h> //reading the neural network definitions
#include <dlib/gui_widgets.h> //gui to display results
#include <dlib/opencv.h> //dlib opencv bindings
#include <opencv2/opencv.hpp> //used for video stream

using namespace std;
using namespace dlib;
using namespace cv;

void find_cards()
{
    using net_type = loss_mmod<con<1,6,6,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
    net_type neuralnet;
    dlib::deserialize("machine-learned/detector-dnn.dat") >> neuralnet;
    //declare and load the neural net
    cv::VideoCapture feed(0); //get the camera feed
    if(!feed.isOpened())
        return;
    dlib::image_window win; //where images will display. May be changed to cv's image display
    for (;;)
    {
        cv::Mat frame;
        feed >> frame; //load current frame
        //cvtColor(frame, frame, BGR2RGB)
        dlib::cv_image(frame) //convert image from mat to dlib compatible format
        //might need to convert to rgb, in which case, uncomment above
        auto contours = dlib::net(frame) //detect contours with neuralnet
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


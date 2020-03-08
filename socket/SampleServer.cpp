#include <stdlib.h>
#include <string>
#include <string.h>
#include <iostream>
#include <chrono>
#include <iomanip>

#ifdef WIN32
#include <conio.h>
#define _getch getch
#else
#include <termios.h>
#include <unistd.h>
#endif

#include "SocketServer.h"

using namespace ninebot_algo;
using namespace socket_algo;

#ifndef WIN32
/* reads from keypress, doesn't echo */
int getch(void) {
    struct termios oldattr, newattr;
    int ch;
    tcgetattr( STDIN_FILENO, &oldattr );
    newattr = oldattr;
    newattr.c_lflag &= ~( ICANON | ECHO );
    tcsetattr( STDIN_FILENO, TCSANOW, &newattr );
    ch = getchar();
    tcsetattr( STDIN_FILENO, TCSANOW, &oldattr );
    return ch;
}
#endif

void stepServer(SocketServer* server){

	int cnt_recv = 0;
	int cnt_float_rcv = 0;
	int cnt_image_send = 0;
	int cnt_err_send = 0;
	int cnt_err_rcv = 0;

	// Create variables 
	const int length_recv = 5;
	float* floats_recv = new float[length_recv];
	char* chars_recv = new char[length_recv];
	float* bounding_box = new float[length_recv];

	const int length_send = 3;
	float* floats_send = new float[length_send];
	char* char_send = new char[length_send];

	while (true) {
		std::cout << "--- server ---" << std::endl;

		if (server->isStopped())
			break;

		if (!server->isConnected()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			continue;
		}
		
		// Send the Image 
		cv::Mat color_image = cv::imread("test/input.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
	    if(! color_image.data )                              // Check for invalid input
	    {
	        std::cout <<  "Could not open or find the color_image" << std::endl ;
	    }

	    cv::resize(color_image, color_image, cv::Size(80,60));

		int send_info_test = server->sendImage(color_image,80,60);

		if (send_info_test < 0) {
			cnt_err_send++;
			std::cout << "send test image failed\n" << std::endl;
		}
		else {
			cnt_image_send++;
			std::cout << "sent test image #" << cnt_image_send << std::endl;
		}


		// Receive Bounding Box coordinates (5 floats)
		int rcv_info_test = server->recvFloats(floats_recv, length_recv);
		if (rcv_info_test < 0) {
			cnt_err_rcv++;
			std::cout << "rcv float failed\n" << std::endl;
		}
		else {
			cnt_float_rcv++;
			std::cout << "rcv float succeeded #" << cnt_float_rcv << std::endl;
			for (int i = 0; i < length_recv; i++)
			{
				bounding_box[i] = *(float*)&floats_recv[i];
				std::cout << "Coordinate #" << i << "  " << bounding_box[i] << std::endl;				
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}

	return;
}

int main(int argc, char** argv) {
	SocketServer server; //(true, true, 8081);

	std::thread send_thread = std::thread(stepServer, &server);
	
	int c;
	bool is_exit = false;
	while (c = getch()) {
		switch (c) {
		case 27:			// ESC 
			is_exit = true;
			break;
		default:
			break;
		}

		if (is_exit)
			break;
	}

	server.stopSocket();

	send_thread.join();

    return 0;
}

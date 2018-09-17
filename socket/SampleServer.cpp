/*! define the server to test class SocketServer.
* Filename: server.cpp
* Version: 0.20
* Algo team, Ninebot Inc., 2017
*/

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
	int cnt_float_send = 0;
	int cnt_image_send = 0;
	int cnt_err = 0;

	// Create variables 
	const int length_recv = 3;
	char* chars_recv = new char[length_recv];
	const int length_send = 3;
	float* floats_send = new float[length_send];

	while (true) {
		std::cout << "--- server ---" << std::endl;

		if (server->isStopped())
			break;

		if (!server->isConnected()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			continue;
		}

		// Float 
		floats_send[0] = (float)cnt_float_send + 0.1;
		floats_send[1] = (float)cnt_float_send + 0.2;
		floats_send[2] = (float)cnt_float_send + 0.3;
		std::cout << "send floats = (" << floats_send[0] << "," << floats_send[1] << "," << floats_send[2] << ")" << std::endl << std::endl;

		// Image 
		cv::Mat1w random_image(3,3);
		cv::randu(random_image, cv::Scalar(200), cv::Scalar(400));		
		std::cout << "send image = "<< std::endl << " "  << random_image << std::endl << std::endl;

		// Send
		int send_info_floats = server->sendFloats(floats_send, length_send);
		if (send_info_floats < 0) {
			cnt_err++;
			std::cout << "send chars failed\n" << std::endl;
		}
		else {
			cnt_float_send++;
			std::cout << "sent chars #" << cnt_float_send << std::endl;
		}

		int send_info_image = server->sendDepth(random_image);
		if (send_info_image < 0) {
			cnt_err++;
			std::cout << "send image failed\n" << std::endl;
		}
		else {
			cnt_image_send++;
			std::cout << "sent image #" << cnt_image_send << std::endl;
		}

		// int recv_info = server->recvChars(chars_recv,length_recv);
		// if (recv_info>0)
		// {
		// 	cnt_recv++;
		// 	std::cout << "policy_receive #" << cnt_recv << std::endl;
		// }
		// else {
		// 	cnt_err++;
		// 	std::cout << "received failed\n" << std::endl;
		// }

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

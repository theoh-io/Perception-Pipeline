/********************************************************************************* 
  *Copyright(C),Ninebot
  *Description: Executable file on server, including communicating and processing
**********************************************************************************/  

#include <stdlib.h>
#include <string>
#include <string.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <stdio.h>

#include "SocketClient.h"
#include "alog.h"

#ifdef WIN32
#include <conio.h>
#define _getch getch
#else
#include <termios.h>
#include <unistd.h>
#endif

// #define IP_ADDRESS "128.179.136.242"

#define IP_ADDRESS "127.0.0.1"

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

void stepClient(SocketClient* client) {

	// Misc 
	int cnt_send = 0;
	int cnt_sent = 0;
	int cnt_err = 0;

	// Create variables 
	const int length_recv = 3;
	char* chars_recv = new char[length_recv];
	const int length_send = 3;
	char* chars_send = new char[length_send];
	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.

	while (true) {
		std::cout << "--- client ---" << std::endl;

		if (!client->isConnected()) {
			client->stopSocket();
			break;
		}

		//Receive
		// int recv_info = client->recvChars(chars_recv,length_recv);
		cv::Mat image;
		image.setTo(cv::Scalar(0));
		int recv_info = client->recvImage(image,3,3,1);
		std::cout << "recv image = "<< std::endl << " "  << image << std::endl;

		if (recv_info < 0){
			std::cout << "recv failed\n" << std::endl;
			client->stopSocket();
			break;
		}
		else {
			cnt_send++;
			std::cout << "received #" << cnt_send << std::endl;

			// cv::Mat img_resize;
			// cv::resize(image, img_resize, cv::Size(100, 100));
		 	// cv::imshow( "Display window", image );                   // Show our image inside it.		
		}

		/* compute command */

		// int send_info = client->sendChars(chars_send, length_send);
		// if (send_info>0)
		// {
		// 	cnt_sent++;
		// 	std::cout << "sent command #" << cnt_sent << std::endl;
		// }
		// else {
		// 	cnt_err++;
		// 	std::cout << "sent failed" << std::endl;
		// }

		if (client->isStopped())
			break;

	}

	// Delete variable 
	delete chars_recv;
	delete chars_send;

	return;
}

int main(int argc, char** argv) {
 
 	char file_output[] = "../example/output_file.txt"; 

    //Set IP
    SocketClient client(IP_ADDRESS,8081);

	//SocketClient client;
	if (!client.initSocket()) {
		std::cout << "Disconnect Server " << std::endl;
		return -1;
	}

	//set thread
	std::thread receive_thread = std::thread(stepClient, &client);
	
	// char c 
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

		if (is_exit || client.isStopped())
			break;
	}

	client.stopSocket();

	receive_thread.join();

    return 0;
}

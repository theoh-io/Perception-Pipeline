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

// #define IP_ADDRESS "128.179.163.19"
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
	int cnt_char_recv = 0;
	int cnt_image_recv = 0;
	int cnt_sent = 0;
	int cnt_err = 0;

	// Create variables 
	const int length_recv = 3;
	float* floats_recv = new float[length_recv];
	const int length_send = 3;
	char* chars_send = new char[length_send];
	cv::namedWindow( "Test window", cv::WINDOW_AUTOSIZE );// Create a window for display.

	std::string foldername = "images";
	std::string cmd_str_mk = "mkdir \"" + foldername + "\"";
	system(cmd_str_mk.c_str());
	ALOGD("Command %s was executed. ", cmd_str_mk.c_str());	

	while (true) {
		std::cout << "--- client ---" << std::endl;

		if (!client->isConnected()) {
			client->stopSocket();
			break;
		}

		//Receive
		// int recv_floats_info = client->recvFloats(floats_recv,length_recv);
		// if (recv_floats_info < 0){
		// 	std::cout << "recv char failed\n" << std::endl;
		// 	client->stopSocket();
		// 	break;
		// }	
		// else {
		// 	cnt_char_recv++;
		// 	std::cout << "received char #" << cnt_char_recv << std::endl;
		// 	std::cout << "received floats = (" << floats_recv[0] << "," << floats_recv[1] << "," << floats_recv[2] << ")" << std::endl << std::endl;
		// }

		cv::Mat test_image;
		int recv_test_info = client->receiveImage(test_image,128,96);

		if (recv_test_info < 0){
			std::cout << "recv failed\n" << std::endl;
			client->stopSocket();
			break;
		}	
		else {
			cnt_image_recv++;
			std::cout << "received #" << cnt_image_recv << std::endl;
			cv::imshow( "Test window", test_image );                   // Show our depth inside it.		
		 	cv::waitKey(10);
		 	// cv::imwrite( foldername + "/recv"+ std::to_string(cnt_image_recv) + ".jpg", test_image);
		}

		if (client->isStopped())
			break;

	}

	// Delete variable 
	delete floats_recv;
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

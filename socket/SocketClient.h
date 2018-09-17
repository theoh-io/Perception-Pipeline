#ifndef SOCKET_CLIENT_H_
#define SOCKET_CLIENT_H_

#include <string>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>

class SocketClient {
public:
	SocketClient(std::string host = "127.0.0.1", int port = 8081);
	~SocketClient();

	bool initSocket();
	bool disconnectSocket();
	bool isConnected();
	bool isStopped();
	bool stopSocket();
	int sendLine(std::string line);
	std::string receiveLine();
	int sendChars(const char* send_chars, const int length);
	int recvChars(char* recv_chars, const int length);

	// call example
	void sendMessage();
	void receiveMessage();
	void sendChars();
	void receiveChars();
	int recvImage(cv::Mat& image, int height, int width, int channels);

private:

	std::string _host;
	int _port;
	int _socket;
	int _init_success;

	std::mutex _mutex_socket_connected;
	std::mutex _mutex_socket_stopped;
	bool _socket_connected;
	bool _socket_stopped;
};

#endif 

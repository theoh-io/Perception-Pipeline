#ifndef ALOG_H_
#define ALOG_H_

#ifndef ANDROID_NDK
#define ALOGI(...) printf(__VA_ARGS__); printf("\n")
#define ALOGD(...) printf(__VA_ARGS__); printf("\n")
#define ALOGE(...) printf(__VA_ARGS__); printf("\n")
#define ALOGW(...) printf(__VA_ARGS__); printf("\n")
#else
#include <android/log.h>
#define LOG_TAG		"ninebot_algo:Socket"
#define ALOGI(...) __android_log_print (ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGD(...) __android_log_print (ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print (ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGW(...) __android_log_print (ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#endif

#endif
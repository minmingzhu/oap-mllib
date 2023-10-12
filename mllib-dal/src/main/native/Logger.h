#pragma once

#include <cstdarg>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;
namespace logger {
extern  std::mutex logMutex;
class Logger {
std::ofstream logFile;
public:
    static Logger& getInstance() {
        static std::once_flag flag;
        static Logger instance;
        std::call_once(flag, [] {
            instance = Logger();
        });
        return instance;
    }

    void printLogToFile(const char *format, ...);
    void closeFile();
private:
    Logger() {
        char* path = std::getenv("SPARKJOB_CONFIG_DIR");
        if (path != nullptr) {
         std::cout << "SPARKJOB_CONFIG_DIR Directory: " << path << std::endl;
        } else {
         std::cout << "SPARKJOB_CONFIG_DIR environment variable not found." << std::endl;
        }
        auto filePath = fs::path(path) / fs::path("training_breakdown");
        std::cout << "file path: "<< filePath << std::endl;
        logFile.open(filePath, std::ios::app);
    }
};

// message type for print functions
enum MessageType {
    DEBUG = 0,
    ASSERT = 1,
    INFO = 2,
    NONE = 3,
    WARN = 4,
    ERROR = 5
};

int print(MessageType message_type, const std::string &msg);
int print(MessageType message_type, const char *format, ...);
int println(MessageType message_type, const char *format, ...);
int println(MessageType message_type, const std::string &msg);

int printerr(MessageType message_type, const std::string &msg);
int printerr(MessageType message_type, const char *format, ...);
int printerrln(MessageType message_type, const char *format, ...);
int printerrln(MessageType message_type, const std::string &msg);

}; // namespace logger

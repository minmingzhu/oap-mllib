#include <iomanip>
#include <tuple>
#include <iostream>

#include "Logger.h"

namespace logger {
std::mutex logMutex;

class LoggerLevel {
  public:
    int level;
    LoggerLevel() {
        level = 2;
        if (const char *env_p = std::getenv("OAP_MLLIB_LOGGER_CPP_LEVEL")) {
            level = atoi(env_p);
        }
        if (level > 5 || level < 0 || level == 3) {
            level = 2;
        }
    }
    int get_level() { return level; }
} logger_level;

std::tuple<std::string, bool> get_prefix(MessageType message_type) {
    std::string prefix;
    bool isLoggerEnabled = false;
    if (message_type >= logger_level.get_level()) {
        isLoggerEnabled = true;
    }
    switch (message_type) {
    case NONE:
        break;
    case INFO:
        prefix = "[INFO]";
        break;
    case WARN:
        prefix = "[WARNING]";
        break;
    case ERROR:
        prefix = "[ERROR]";
        break;
    case DEBUG:
        prefix = "[DEBUG]";
        break;
    case ASSERT:
        prefix = "[ASSERT]";
        break;
    default:
        break;
    }

    return {prefix + " ", isLoggerEnabled};
}

int print2streamFromArgs(MessageType message_type, FILE *stream,
                         const char *format, va_list args) {
    // print prefix
    auto [prefix, enable] = get_prefix(message_type);
    if (!enable)
        return 0;
    fprintf(stream, "%s", prefix.c_str());

    // print message
    int ret = vfprintf(stream, format, args);
    fflush(stream);

    return ret;
}

int print2streamFromArgsln(MessageType message_type, FILE *stream,
                           const char *format, va_list args) {
    // print prefix
    auto [prefix, enable] = get_prefix(message_type);
    if (!enable)
        return 0;
    fprintf(stream, "%s", prefix.c_str());

    // print message
    int ret = vfprintf(stream, format, args);
    fflush(stream);
    fprintf(stream, "\n");
    fflush(stream);

    return ret;
}

int print2stream(MessageType message_type, FILE *stream, const char *format,
                 ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgs(message_type, stream, format, args);
    va_end(args);

    return ret;
}

int print2streamln(MessageType message_type, FILE *stream, const char *format,
                   ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgsln(message_type, stream, format, args);
    va_end(args);

    return ret;
}

int print(MessageType message_type, const std::string &msg) {
    int ret = print2stream(message_type, stdout, msg.c_str());
    return ret;
}

int print(MessageType message_type, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgs(message_type, stdout, format, args);
    va_end(args);
    return ret;
}

int println(MessageType message_type, const std::string &msg) {
    int ret = print2streamln(message_type, stdout, msg.c_str());
    return ret;
}

int println(MessageType message_type, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgsln(message_type, stdout, format, args);
    va_end(args);
    return ret;
}

int printerr(MessageType message_type, const std::string &msg) {
    int ret = print2stream(message_type, stderr, msg.c_str());
    return ret;
}

int printerr(MessageType message_type, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgs(message_type, stderr, format, args);
    va_end(args);
    return ret;
}

int printerrln(MessageType message_type, const std::string &msg) {
    int ret = print2streamln(message_type, stderr, msg.c_str());
    return ret;
}

int printerrln(MessageType message_type, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgsln(message_type, stderr, format, args);
    va_end(args);
    return ret;
}


void printLogToFile(const char *format, ...) {
     std::cout << "printLogToFile "<< std::endl;
     char* path = std::getenv("spark.oap.mllib.record.output.path");
     auto filePath = fs::path(path) / fs::path("training_breakdown");

     std::cout << "file path: "
          << filePath << std::endl;
     std::ofstream logFile(filePath);
     std::lock_guard<std::mutex> lock(logMutex);
     va_list args;
     va_start(args, format);
     std::ostringstream formattedMessage;
     while (*format != '\0') {
         if (*format == '%' && *(format + 1) == 'd') {
             int intValue = va_arg(args, int);
             formattedMessage << intValue;
             format += 2;
         } else if (*format == '%' && *(format + 1) == 'f') {
             double floatValue = va_arg(args, double);
             formattedMessage << floatValue;
             format += 2;
         } else {
             formattedMessage << *format++;
         }
     }
     va_end(args);
     logFile << formattedMessage.str() << std::endl;
     logFile.close();
}

}; // namespace logger

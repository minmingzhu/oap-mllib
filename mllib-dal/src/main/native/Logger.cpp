#include <iomanip>
#include <tuple>

#include "Logger.h"

namespace logger {
std::tuple<std::string, bool> get_prefix(MessageType message_type) {
    std::string prefix;
    bool enable{true};
    switch (message_type) {
    case NONE:
        break;
    case INFO:
        prefix = "[INFO   ]";
        break;
    case WARN:
        prefix = "[WARNING]";
        break;
    case ERROR:
        prefix = "[ERROR  ]";
        break;
    case DEBUG:
        prefix = "[DEBUG  ]";
        break;
    case ASSERT:
        prefix = "[ASSERT ]";
        break;
    default:
        break;
    }
    return {prefix + " ", enable};
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
    int ret = print2stream(message_type, stdout, msg.c_str());
    fprintf(stdout, "\n");
    return ret;
}

int println(MessageType message_type, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgs(message_type, stdout, format, args);
    va_end(args);
    fprintf(stdout, "\n");
    return ret;
}

int printerr(MessageType message_type, const std::string &msg) {
    int ret = print2stream(message_type, stdout, msg.c_str());
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
    int ret = print2stream(message_type, stderr, msg.c_str());
    fprintf(stderr, "\n");
    return ret;
}

int printerrln(MessageType message_type, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = print2streamFromArgs(message_type, stderr, format, args);
    va_end(args);
    fprintf(stderr, "\n");
    return ret;
}

int print(MessageType message_type, const oneapi::dal::table &table) {
    auto [prefix, enable] = get_prefix(message_type);
    if (!enable)
        return 0;
    FILE *output = stdout;

    auto arr = oneapi::dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();
    if (table.get_row_count() <= 10) {
        for (std::int64_t i = 0; i < table.get_row_count(); i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                fprintf(output, "%s", prefix.c_str());
                fprintf(output, "%10f", x[i * table.get_column_count() + j]);
            }
            fprintf(output, "\n");
        }
    } else {
        for (std::int64_t i = 0; i < 5; i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                fprintf(stdout, "%s", prefix.c_str());
                fprintf(output, "%10f", x[i * table.get_column_count() + j]);
            }
            fprintf(output, "\n");
        }
        fprintf(output, "%s", prefix.c_str());
        fprintf(output, "...%ld lines skipped...\n",
                (table.get_row_count() - 10));
        for (std::int64_t i = table.get_row_count() - 5;
             i < table.get_row_count(); i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                fprintf(stdout, "%s", prefix.c_str());
                fprintf(output, "%10f", x[i * table.get_column_count() + j]);
            }
            fprintf(output, "\n");
        }
    }
    return 0;
}
}; // namespace logger

// Writing JSON strings.
#pragma once

#include <cstdio>
#include <string>

namespace owe {

// s as the contents of a JSON string literal (quotes, backslashes and control characters escaped).
inline std::string jsonEscape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            r += '\\';
            r += c;
        } else if (c == '\n') {
            r += "\\n";
        } else if ((unsigned char)c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            r += buf;
        } else {
            r += c;
        }
    }
    return r;
}

}  // namespace owe

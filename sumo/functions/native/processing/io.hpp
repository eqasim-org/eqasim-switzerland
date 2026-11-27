#pragma once
#include <algorithm>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/io/buffered.h>
#include <arrow/ipc/api.h>
#include <charconv>
#include <cmath>
#include <cstring>
#include <deque>
#include <exception>
#include <expat.h>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <zlib.h>

namespace sumo_native {
using Attrs = std::map<std::string, std::string>;
inline void check(const arrow::Status &status) {
    if (!status.ok())
        throw std::runtime_error(status.ToString());
}
template <class T> T take(arrow::Result<T> result) {
    check(result.status());
    return std::move(result).ValueOrDie();
}
inline double numeric(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
        value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        value.remove_suffix(1);
    if (!value.empty() && value.front() == '+')
        value.remove_prefix(1);
    double result;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() || !std::isfinite(result))
        throw std::runtime_error("Invalid finite numeric value: " + std::string(value));
    return result;
}
inline std::string number(double value, int precision = 12) {
    char data[128];
    auto end = std::to_chars(data, data + sizeof(data), value, std::chars_format::general, precision);
    if (end.ec != std::errc())
        throw std::runtime_error("Cannot format number");
    return {data, end.ptr};
}
inline std::string shortest(double value) {
    char data[128];
    auto end = std::to_chars(data, data + sizeof(data), value);
    if (end.ec != std::errc())
        throw std::runtime_error("Cannot format number");
    return {data, end.ptr};
}
inline std::string python_float(double value) {
    auto text = shortest(value);
    auto e = text.find('e');
    if (e != std::string::npos) {
        int exponent = std::stoi(text.substr(e + 1));
        if (exponent >= -4 && exponent < 16) {
            bool negative = text[0] == '-';
            auto digits = text.substr(negative ? 1 : 0, e - (negative ? 1 : 0));
            auto point = digits.find('.');
            if (point != std::string::npos)
                digits.erase(point, 1);
            int position = exponent + 1;
            if (position <= 0)
                text = "0." + std::string(-position, '0') + digits;
            else if (position >= int(digits.size()))
                text = digits + std::string(position - digits.size(), '0') + ".0";
            else
                text = digits.substr(0, position) + "." + digits.substr(position);
            if (negative)
                text = "-" + text;
        }
    }
    if (text.find_first_of(".e") == std::string::npos)
        text += ".0";
    return text;
}
inline std::string get(const Attrs &attrs, const std::string &key, std::string fallback = "") {
    auto found = attrs.find(key);
    return found == attrs.end() ? fallback : found->second;
}
inline std::string json_string(std::string_view value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        default:
            if (c < 32) {
                const char *h = "0123456789abcdef";
                out += "\\u00";
                out += h[c >> 4];
                out += h[c & 15];
            } else
                out += char(c);
        }
    }
    return out + '"';
}
inline std::string json_attrs(const Attrs &attrs) {
    std::string out = "{";
    bool first = true;
    for (const auto &pair : attrs) {
        if (!first)
            out += ',';
        first = false;
        out += json_string(pair.first) + ':' + json_string(pair.second);
    }
    return out + '}';
}
class Json {
    std::string_view data;
    size_t cursor = 0;
    unsigned hex4() {
        unsigned result = 0;
        for (int i = 0; i < 4; ++i) {
            if (cursor == data.size())
                fail();
            char c = data[cursor++];
            int n = c >= '0' && c <= '9'   ? c - '0'
                    : c >= 'a' && c <= 'f' ? c - 'a' + 10
                    : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                           : -1;
            if (n < 0)
                fail();
            result = result * 16 + n;
        }
        return result;
    }
    static void utf8(std::string &out, unsigned c) {
        if (c < 0x80)
            out += char(c);
        else if (c < 0x800) {
            out += char(0xc0 | (c >> 6));
            out += char(0x80 | (c & 63));
        } else if (c < 0x10000) {
            out += char(0xe0 | (c >> 12));
            out += char(0x80 | ((c >> 6) & 63));
            out += char(0x80 | (c & 63));
        } else {
            out += char(0xf0 | (c >> 18));
            out += char(0x80 | ((c >> 12) & 63));
            out += char(0x80 | ((c >> 6) & 63));
            out += char(0x80 | (c & 63));
        }
    }

  public:
    explicit Json(std::string_view value) : data(value) {}
    [[noreturn]] void fail() const {
        throw std::runtime_error("Malformed native route JSON at byte " + std::to_string(cursor));
    }
    void space() {
        while (cursor < data.size() &&
               (data[cursor] == ' ' || data[cursor] == '\t' || data[cursor] == '\r' || data[cursor] == '\n'))
            ++cursor;
    }
    bool consume(char c) {
        space();
        if (cursor < data.size() && data[cursor] == c) {
            ++cursor;
            return true;
        }
        return false;
    }
    void expect(char c) {
        if (!consume(c))
            fail();
    }
    std::string string() {
        expect('"');
        std::string out;
        while (cursor < data.size()) {
            unsigned char c = data[cursor++];
            if (c == '"')
                return out;
            if (c < 32)
                fail();
            if (c != '\\') {
                out += char(c);
                continue;
            }
            if (cursor == data.size())
                fail();
            char code = data[cursor++];
            switch (code) {
            case '"':
            case '\\':
            case '/':
                out += code;
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u': {
                unsigned cp = hex4();
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    if (cursor + 2 > data.size() || data.substr(cursor, 2) != "\\u")
                        fail();
                    cursor += 2;
                    unsigned low = hex4();
                    if (low < 0xdc00 || low > 0xdfff)
                        fail();
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                } else if (cp >= 0xdc00 && cp <= 0xdfff)
                    fail();
                utf8(out, cp);
                break;
            }
            default:
                fail();
            }
        }
        fail();
    }
    void skip() {
        space();
        if (cursor == data.size())
            fail();
        char c = data[cursor];
        if (c == '"') {
            string();
            return;
        }
        if (c == '[' || c == '{') {
            ++cursor;
            char end = c == '[' ? ']' : '}';
            if (consume(end))
                return;
            do {
                if (c == '{') {
                    string();
                    expect(':');
                }
                skip();
            } while (consume(','));
            expect(end);
            return;
        }
        size_t begin = cursor;
        while (cursor < data.size() &&
               std::string_view(",]} \t\r\n").find(data[cursor]) == std::string_view::npos)
            ++cursor;
        if (begin == cursor)
            fail();
    }
    Attrs attrs() {
        Attrs result;
        expect('{');
        if (!consume('}')) {
            do {
                auto key = string();
                expect(':');
                result[key] = string();
            } while (consume(','));
            expect('}');
        }
        finish();
        return result;
    }
    void finish() {
        space();
        if (cursor != data.size())
            fail();
    }
    std::vector<std::string> selected_path(int choice) {
        expect('[');
        int index = 0;
        std::vector<std::string> result;
        bool found = false;
        if (!consume(']')) {
            do {
                if (index++ == choice) {
                    expect('[');
                    expect('[');
                    if (!consume(']')) {
                        do {
                            result.push_back(string());
                        } while (consume(','));
                        expect(']');
                    }
                    expect(',');
                    skip();
                    expect(']');
                    found = true;
                } else
                    skip();
            } while (consume(','));
            expect(']');
        }
        finish();
        if (!found || result.empty())
            throw std::runtime_error("Missing selected assignment path");
        return result;
    }
};
inline std::string xml_escape(std::string_view text) {
    std::string result;
    for (unsigned char c : text) {
        switch (c) {
        case '&':
            result += "&amp;";
            break;
        case '<':
            result += "&lt;";
            break;
        case '>':
            result += "&gt;";
            break;
        case '"':
            result += "&quot;";
            break;
        case '\n':
            result += "&#10;";
            break;
        case '\r':
            result += "&#13;";
            break;
        case '\t':
            result += "&#9;";
            break;
        default:
            if (c < 32)
                throw std::runtime_error("Invalid XML control character");
            result += char(c);
        }
    }
    return result;
}
inline std::string xml_start(const std::string &tag, const Attrs &attrs) {
    std::string result = "<" + tag;
    for (const auto &pair : attrs)
        result += ' ' + pair.first + "=\"" + xml_escape(pair.second) + '"';
    return result + '>';
}
inline std::vector<std::string> tokens(std::string_view value) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at < value.size()) {
        while (at < value.size() && std::isspace(static_cast<unsigned char>(value[at])))
            ++at;
        size_t start = at;
        while (at < value.size() && !std::isspace(static_cast<unsigned char>(value[at])))
            ++at;
        if (at > start)
            out.emplace_back(value.substr(start, at - start));
    }
    return out;
}
struct Xml {
    XML_Parser parser;
    std::function<void(std::string, const Attrs &)> start;
    std::function<void(std::string)> end;
    std::exception_ptr error;
    bool stop = false;
    Xml() : parser(XML_ParserCreate(nullptr)) {
        if (!parser)
            throw std::bad_alloc();
        XML_SetUserData(parser, this);
        XML_SetElementHandler(parser, on_start, on_end);
        // Accept ordinary XML declarations without fetching external DTD resources.
        XML_SetParamEntityParsing(parser, XML_PARAM_ENTITY_PARSING_NEVER);
    }
    ~Xml() {
        XML_ParserFree(parser);
    }
    static void on_start(void *user, const char *tag, const char **attrs) {
        auto &self = *static_cast<Xml *>(user);
        try {
            Attrs values;
            for (int i = 0; attrs[i]; i += 2)
                values[attrs[i]] = attrs[i + 1];
            if (self.start)
                self.start(tag, values);
        } catch (...) {
            self.error = std::current_exception();
            XML_StopParser(self.parser, XML_FALSE);
        }
    }
    static void on_end(void *user, const char *tag) {
        auto &self = *static_cast<Xml *>(user);
        try {
            if (self.end)
                self.end(tag);
            if (self.stop)
                XML_StopParser(self.parser, XML_FALSE);
        } catch (...) {
            self.error = std::current_exception();
            XML_StopParser(self.parser, XML_FALSE);
        }
    }
    void read(const std::string &path) {
        gzFile input = gzopen(path.c_str(), "rb");
        if (!input)
            throw std::runtime_error("Cannot open XML: " + path);
        gzbuffer(input, 1 << 20);
        std::vector<char> buffer(1 << 20);
        try {
            while (!stop) {
                int size = gzread(input, buffer.data(), buffer.size());
                if (size < 0) {
                    int code;
                    throw std::runtime_error(gzerror(input, &code));
                }
                int status = XML_Parse(parser, buffer.data(), size, size == 0);
                if (error)
                    std::rethrow_exception(error);
                if (status == XML_STATUS_ERROR && !stop)
                    throw std::runtime_error("XML parse error in " + path + ": " +
                                             XML_ErrorString(XML_GetErrorCode(parser)));
                if (!size)
                    break;
            }
        } catch (...) {
            gzclose(input);
            throw;
        }
        if (gzclose(input) != Z_OK)
            throw std::runtime_error("Error closing XML input: " + path);
    }
};
struct IpcInput {
    std::shared_ptr<arrow::io::MemoryMappedFile> file;
    std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader;
    explicit IpcInput(const std::string &path) {
        file = take(arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ));
        reader = take(arrow::ipc::RecordBatchFileReader::Open(file));
    }
};
inline std::string_view string_at(const std::shared_ptr<arrow::Array> &array, int64_t i) {
    if (array->IsNull(i))
        throw std::runtime_error("Unexpected null string");
    switch (array->type_id()) {
    case arrow::Type::STRING:
        return static_cast<arrow::StringArray &>(*array).GetView(i);
    case arrow::Type::LARGE_STRING:
        return static_cast<arrow::LargeStringArray &>(*array).GetView(i);
    case arrow::Type::STRING_VIEW:
        return static_cast<arrow::StringViewArray &>(*array).GetView(i);
    default:
        throw std::runtime_error("Expected string column");
    }
}
struct IpcOutput {
    std::shared_ptr<arrow::io::FileOutputStream> file;
    std::shared_ptr<arrow::io::BufferedOutputStream> buffer;
    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
    explicit IpcOutput(const std::string &path, std::shared_ptr<arrow::Schema> schema) {
        file = take(arrow::io::FileOutputStream::Open(path));
        buffer = take(arrow::io::BufferedOutputStream::Create(1 << 20, arrow::default_memory_pool(), file));
        writer = take(arrow::ipc::MakeFileWriter(buffer, schema));
    }
    void write(const std::shared_ptr<arrow::RecordBatch> &batch) {
        check(writer->WriteRecordBatch(*batch));
    }
    void close() {
        check(writer->Close());
        check(buffer->Close());
    }
};
inline std::shared_ptr<arrow::Array> strings(const std::vector<std::string> &data) {
    arrow::StringBuilder builder;
    check(builder.Reserve(data.size()));
    for (const auto &value : data)
        check(builder.Append(value));
    return take(builder.Finish());
}
inline std::shared_ptr<arrow::Array> doubles(const std::vector<double> &data) {
    arrow::DoubleBuilder builder;
    check(builder.AppendValues(data));
    return take(builder.Finish());
}
inline std::shared_ptr<arrow::Array> ints64(const std::vector<int64_t> &data) {
    arrow::Int64Builder builder;
    check(builder.AppendValues(data));
    return take(builder.Finish());
}
inline std::shared_ptr<arrow::Array> ints32(const std::vector<int32_t> &data) {
    arrow::Int32Builder builder;
    check(builder.AppendValues(data));
    return take(builder.Finish());
}
struct Options {
    std::map<std::string, std::string> values;
    int64_t batch = 10000, bytes = 16 << 20;
    int threads = 1, xml_threads = 1;
    explicit Options(int argc, char **argv) {
        if (argc < 2)
            throw std::runtime_error("Missing native command");
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 == argc)
                throw std::runtime_error("Missing option value");
            values[argv[i]] = argv[i + 1];
        }
        if (values.count("--batch"))
            batch = std::stoll(values["--batch"]);
        if (values.count("--bytes"))
            bytes = std::stoll(values["--bytes"]);
        if (values.count("--threads"))
            threads = std::stoi(values["--threads"]);
        if (values.count("--xml-threads"))
            xml_threads = std::stoi(values["--xml-threads"]);
        if (batch < 1 || bytes < 1 || threads < 1 || xml_threads < 1)
            throw std::runtime_error("Batch and thread limits must be positive");
    }
    std::string at(const std::string &name) const {
        auto it = values.find("--" + name);
        if (it == values.end())
            throw std::runtime_error("Missing --" + name);
        return it->second;
    }
};
template <class Function> void parallel(size_t count, int workers, Function operation) {
    std::vector<std::future<void>> pending;
    workers = std::min<int>(workers, count);
    if (workers <= 1) {
        for (size_t i = 0; i < count; ++i)
            operation(i);
        return;
    }
    for (int w = 0; w < workers; ++w)
        pending.push_back(std::async(std::launch::async, [&, w] {
            for (size_t i = w; i < count; i += workers)
                operation(i);
        }));
    for (auto &future : pending)
        future.get();
}
inline std::string read_text(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot read " + path);
    return {std::istreambuf_iterator<char>(input), {}};
}
inline void write_text(const std::string &path, const std::string &text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
    if (!output)
        throw std::runtime_error("Cannot write " + path);
}
} // namespace sumo_native

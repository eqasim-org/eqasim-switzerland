// Bulk Arrow IPC -> SUMO trip XML. No Python row iteration or XML objects.
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <zlib.h>
#include <charconv>
#include <cmath>
#include <fstream>
#include <future>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

template<class T> T checked(arrow::Result<T> result) {
    if (!result.ok()) throw std::runtime_error(result.status().ToString());
    return std::move(result).ValueOrDie();
}

std::string_view text(const std::shared_ptr<arrow::Array>& values, int64_t row) {
    switch (values->type_id()) {
        case arrow::Type::STRING: return static_cast<arrow::StringArray&>(*values).GetView(row);
        case arrow::Type::LARGE_STRING: return static_cast<arrow::LargeStringArray&>(*values).GetView(row);
        case arrow::Type::BINARY: return static_cast<arrow::BinaryArray&>(*values).GetView(row);
        case arrow::Type::LARGE_BINARY: return static_cast<arrow::LargeBinaryArray&>(*values).GetView(row);
        case arrow::Type::STRING_VIEW: return static_cast<arrow::StringViewArray&>(*values).GetView(row);
        case arrow::Type::BINARY_VIEW: return static_cast<arrow::BinaryViewArray&>(*values).GetView(row);
        default: throw std::runtime_error("Expected string/binary Arrow column");
    }
}

void escaped(std::string& out, std::string_view value) {
    if (value.find("\xEF\xBF\xBE") != std::string_view::npos || value.find("\xEF\xBF\xBF") != std::string_view::npos)
        throw std::runtime_error("Invalid XML Unicode character");
    for (unsigned char c : value) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\n': out += "&#10;"; break;
            case '\r': out += "&#13;"; break;
            case '\t': out += "&#9;"; break;
            default:
                if (c < 32) throw std::runtime_error("Invalid XML control character");
                out += static_cast<char>(c);
        }
    }
}

void attribute(std::string& out, const char* name, std::string_view value) {
    out += ' '; out += name; out += "=\""; escaped(out, value); out += '"';
}

void parameter(std::string& out, const char* key, std::string_view value) {
    out += "<param"; attribute(out, "key", key); attribute(out, "value", value); out += "/>";
}

std::string number(double value, int precision = 0) {
    char buffer[384];
    auto result = precision ? std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general, precision)
                            : std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::fixed);
    if (result.ec != std::errc()) throw std::runtime_error("Cannot format numeric XML value");
    return std::string(buffer, result.ptr);
}

std::string compress_block(std::string data) {
    z_stream stream{};
    if (deflateInit2(&stream, 1, Z_DEFLATED, 15+16, 8, Z_DEFAULT_STRATEGY)!=Z_OK)
        throw std::runtime_error("Cannot initialize gzip compressor");
    std::string result(deflateBound(&stream,data.size()), '\0');
    stream.next_in=reinterpret_cast<Bytef*>(data.data()); stream.avail_in=data.size();
    stream.next_out=reinterpret_cast<Bytef*>(result.data()); stream.avail_out=result.size();
    int status=deflate(&stream,Z_FINISH); auto size=stream.total_out; deflateEnd(&stream);
    if(status!=Z_STREAM_END) throw std::runtime_error("gzip block compression failed");
    result.resize(size); return result;
}

class Output {
    gzFile gzip = nullptr;
    std::ofstream plain;
    int workers;
    bool parallel=false;
    std::deque<std::future<std::string>> pending;
    void flush_one() {
        auto value=pending.front().get(); pending.pop_front();
        plain.write(value.data(),value.size());
        if(!plain) throw std::runtime_error("Error writing gzip block");
    }
public:
    explicit Output(const std::string& path, int threads=1):workers(threads) {
        if (path.size() >= 3 && path.substr(path.size() - 3) == ".gz") {
            if(workers>1){ parallel=true; plain.open(path,std::ios::binary);
                if(!plain) throw std::runtime_error("Cannot open gzip output"); return; }
            gzip = gzopen(path.c_str(), "wb1");
            if (!gzip) throw std::runtime_error("Cannot open gzip output");
            gzbuffer(gzip, 1024 * 1024);
        } else {
            plain.open(path, std::ios::binary);
            if (!plain) throw std::runtime_error("Cannot open XML output");
        }
    }
    ~Output() { if (gzip) gzclose(gzip); }
    void write(std::string_view data) {
        if(parallel) {
            pending.push_back(std::async(std::launch::async,compress_block,std::string(data)));
            if(pending.size()>=static_cast<size_t>(workers)) flush_one();
            return;
        }
        if (gzip) {
            if (gzwrite(gzip, data.data(), static_cast<unsigned>(data.size())) != static_cast<int>(data.size()))
                throw std::runtime_error("Error writing gzip XML");
        } else {
            plain.write(data.data(), data.size());
            if (!plain) throw std::runtime_error("Error writing XML");
        }
    }
    void finish() {
        while(!pending.empty()) flush_one();
        if (gzip) { auto file = gzip; gzip = nullptr; if (gzclose(file) != Z_OK) throw std::runtime_error("Error closing gzip XML"); }
        else { plain.close(); if (!plain) throw std::runtime_error("Error closing XML"); }
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) throw std::runtime_error("Usage: write_trips trips.arrow vehicle-types.xml output.xml[.gz]");
        auto input = checked(arrow::io::MemoryMappedFile::Open(argv[1], arrow::io::FileMode::READ));
        auto reader = checked(arrow::ipc::RecordBatchFileReader::Open(input));
        std::ifstream prelude(argv[2], std::ios::binary);
        if (!prelude) throw std::runtime_error("Cannot open vehicle type definitions");
        std::string header((std::istreambuf_iterator<char>(prelude)), std::istreambuf_iterator<char>());
        int workers=argc==5?std::stoi(argv[4]):1;
        if(workers<1 || workers>256) throw std::runtime_error("Invalid XML compression thread count");
        Output output(argv[3],workers);
        output.write("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<routes>\n");
        output.write(header);
        std::string buffer; buffer.reserve(2 * 1024 * 1024);
        int64_t count = 0;
        for (int batch_index = 0; batch_index < reader->num_record_batches(); ++batch_index) {
            auto batch = checked(reader->ReadRecordBatch(batch_index));
            auto column = [&](const char* name) {
                auto values = batch->GetColumnByName(name);
                if (!values) throw std::runtime_error(std::string("Missing Arrow column: ") + name);
                return values;
            };
            auto ids = column("id"), departs = column("depart"), origins = column("origin"), destinations = column("destination");
            auto people = column("person_id"), indices = column("trip_index"), sources = column("source_trip_id");
            auto modes = column("mode"), references = column("reference"), loads = column("loads_network"), xml = column("xml");
            for (int64_t row = 0; row < batch->num_rows(); ++row) {
                if (!xml->IsNull(row)) buffer.append(text(xml, row));
                else {
                    if (ids->IsNull(row) || departs->IsNull(row) || origins->IsNull(row) || destinations->IsNull(row)
                        || people->IsNull(row) || indices->IsNull(row) || modes->IsNull(row) || loads->IsNull(row))
                        throw std::runtime_error("Missing required population trip value");
                    double depart = static_cast<arrow::DoubleArray&>(*departs).Value(row);
                    if (!std::isfinite(depart) || depart < 0) throw std::runtime_error("Invalid vehicle departure");
                    buffer += "<trip"; attribute(buffer, "id", text(ids, row)); attribute(buffer, "depart", number(depart));
                    attribute(buffer, "fromLonLat", text(origins, row)); attribute(buffer, "toLonLat", text(destinations, row));
                    attribute(buffer, "type", "sumo_query_car"); buffer += '>';
                    parameter(buffer, "person_id", text(people, row)); parameter(buffer, "trip_index", text(indices, row));
                    if (!sources->IsNull(row)) parameter(buffer, "source_trip_id", text(sources, row));
                    parameter(buffer, "mode", text(modes, row));
                    if (!references->IsNull(row)) {
                        double value = static_cast<arrow::DoubleArray&>(*references).Value(row);
                        if (std::isfinite(value)) parameter(buffer, "source_travel_time_s", number(value, 15));
                    }
                    parameter(buffer, "_sumo_loads_network", static_cast<arrow::BooleanArray&>(*loads).Value(row) ? "true" : "false");
                    buffer += "</trip>";
                }
                if (buffer.size() >= 1024 * 1024) { output.write(buffer); buffer.clear(); }
                if (++count % 1000000 == 0) std::cerr << "SUMO native XML: " << count << " trips written\n";
            }
        }
        output.write(buffer); output.write("\n</routes>\n"); output.finish();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SUMO native XML error: " << error.what() << '\n'; return 1;
    }
}

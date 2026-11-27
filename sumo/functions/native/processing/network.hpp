#pragma once
#include "io.hpp"
namespace sumo_native {
struct Edge {
    std::string id;
    double length = 0;
    std::vector<uint64_t> lanes;
    uint64_t allowed = 0;
};
struct Connection {
    std::string from, to;
    int from_lane = 0, to_lane = 0;
    uint64_t allowed = 0;
};
inline uint64_t pair_key(int32_t a, int32_t b) {
    return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
}
struct Network {
    std::vector<Edge> edges;
    std::unordered_map<std::string, int32_t> ids;
    std::vector<std::string> classes{"passenger"};
    std::unordered_map<std::string, uint64_t> types;
    std::unordered_map<uint64_t, uint64_t> transitions;
    explicit Network(const std::string &net, const std::string &definitions) {
        Xml vtypes;
        vtypes.start = [&](std::string tag, const Attrs &attrs) {
            if (tag != "vType")
                return;
            auto cls = get(attrs, "vClass", "passenger");
            auto found = std::find(classes.begin(), classes.end(), cls);
            if (found == classes.end()) {
                if (classes.size() == 64)
                    throw std::runtime_error("Too many native vehicle classes");
                classes.push_back(cls);
                found = classes.end() - 1;
            }
            types[get(attrs, "id")] = uint64_t(1) << std::distance(classes.begin(), found);
        };
        // Definitions are fragments; use a temporary parser feed with a routes wrapper.
        std::string text = "<routes>" + read_text(definitions) + "</routes>";
        if (XML_Parse(vtypes.parser, text.data(), text.size(), XML_TRUE) == XML_STATUS_ERROR) {
            if (vtypes.error)
                std::rethrow_exception(vtypes.error);
            throw std::runtime_error("Invalid vehicle type definitions");
        }
        std::vector<Connection> connections;
        Edge current;
        bool active = false;
        Xml xml;
        xml.start = [&](std::string tag, const Attrs &attrs) {
            if (tag == "edge") {
                active = get(attrs, "function") != "internal";
                if (active) {
                    current = Edge();
                    current.id = get(attrs, "id");
                }
            } else if (tag == "lane" && active) {
                if (current.lanes.empty())
                    current.length = numeric(get(attrs, "length"));
                auto mask = permissions(attrs);
                current.lanes.push_back(mask);
                current.allowed |= mask;
            } else if (tag == "connection") {
                auto from = get(attrs, "from"), to = get(attrs, "to");
                if (!from.empty() && !to.empty() && from[0] != ':' && to[0] != ':')
                    connections.push_back({from, to, std::stoi(get(attrs, "fromLane", "0")),
                                           std::stoi(get(attrs, "toLane", "0")), permissions(attrs)});
            }
        };
        xml.end = [&](std::string tag) {
            if (tag == "edge" && active) {
                if (!current.lanes.empty()) {
                    if (ids.count(current.id))
                        throw std::runtime_error("Duplicate network edge " + current.id);
                    if (edges.size() >= size_t(INT32_MAX))
                        throw std::runtime_error("Too many network edges");
                    ids[current.id] = edges.size();
                    edges.push_back(std::move(current));
                }
                active = false;
            }
        };
        xml.read(net);
        for (const auto &c : connections) {
            auto a = ids.find(c.from), b = ids.find(c.to);
            if (a == ids.end() || b == ids.end())
                throw std::runtime_error("Connection references absent road edge");
            const auto &first = edges[a->second];
            const auto &last = edges[b->second];
            if (c.from_lane < 0 || c.to_lane < 0 || size_t(c.from_lane) >= first.lanes.size() ||
                size_t(c.to_lane) >= last.lanes.size())
                throw std::runtime_error("Connection has an invalid lane index");
            transitions[pair_key(a->second, b->second)] |=
                first.lanes[c.from_lane] & last.lanes[c.to_lane] & c.allowed;
        }
    }
    uint64_t permissions(const Attrs &attrs) const {
        auto allow = tokens(get(attrs, "allow")), disallow = tokens(get(attrs, "disallow"));
        bool specified = attrs.count("allow");
        uint64_t result = 0;
        auto has = [](const std::vector<std::string> &items, const std::string &key) {
            return std::find(items.begin(), items.end(), key) != items.end();
        };
        for (size_t i = 0; i < classes.size(); ++i)
            if ((!specified || has(allow, "all") || has(allow, classes[i])) && !has(disallow, "all") &&
                !has(disallow, classes[i]))
                result |= uint64_t(1) << i;
        return result;
    }
    uint64_t vehicle_class(const Attrs &attrs) const {
        auto found = types.find(get(attrs, "type"));
        return found == types.end() ? 1 : found->second;
    }
    void validate(const std::vector<int32_t> &route, const Attrs &attrs) const {
        if (route.empty())
            throw std::runtime_error("Empty route for " + get(attrs, "id"));
        uint64_t cls = vehicle_class(attrs);
        for (size_t i = 0; i < route.size(); ++i) {
            auto edge = route[i];
            if (edge < 0 || size_t(edge) >= edges.size())
                throw std::runtime_error("Route " + get(attrs, "id") + " references absent edges");
            if (!(edges[edge].allowed & cls))
                throw std::runtime_error("Route " + get(attrs, "id") +
                                         " uses an edge forbidden to its vehicle class");
            if (i) {
                auto found = transitions.find(pair_key(route[i - 1], edge));
                if (found == transitions.end())
                    throw std::runtime_error("Route " + get(attrs, "id") +
                                             " has a forbidden/disconnected transition");
                if (!(found->second & cls))
                    throw std::runtime_error("Route " + get(attrs, "id") +
                                             " has a turn forbidden to its vehicle class");
            }
        }
    }
    void write_edges(const std::string &path, const Options &opts) const {
        auto schema =
            arrow::schema({arrow::field("id", arrow::utf8()), arrow::field("length", arrow::float64())});
        IpcOutput output(path, schema);
        std::vector<std::string> names;
        std::vector<double> lengths;
        size_t bytes = 0;
        auto flush = [&] {
            if (names.empty())
                return;
            output.write(arrow::RecordBatch::Make(schema, names.size(), {strings(names), doubles(lengths)}));
            names.clear();
            lengths.clear();
            bytes = 0;
        };
        for (const auto &edge : edges) {
            names.push_back(edge.id);
            lengths.push_back(edge.length);
            bytes += edge.id.size() + 8;
            if (names.size() >= size_t(opts.batch) || bytes >= size_t(opts.bytes))
                flush();
        }
        flush();
        output.close();
    }
};
inline double position(const std::string &text, double length) {
    if (text == "max" || text == "last")
        return length;
    if (text == "base")
        return 0;
    double value = numeric(text);
    if (value < 0)
        value += length;
    if (value < -1e-6 || value > length + 1e-6)
        throw std::runtime_error("Vehicle position outside edge length");
    return std::clamp(value, 0., length);
}
struct EdgeTable {
    std::vector<std::string> ids;
    std::vector<double> lengths;
    std::unordered_map<std::string, int32_t> lookup;
    explicit EdgeTable(const std::string &path) {
        IpcInput input(path);
        for (int b = 0; b < input.reader->num_record_batches(); ++b) {
            auto batch = take(input.reader->ReadRecordBatch(b));
            auto names = batch->GetColumnByName("id");
            auto lens = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("length"));
            if (!names || !lens)
                throw std::runtime_error("Invalid edge IPC schema");
            for (int64_t row = 0; row < batch->num_rows(); ++row) {
                auto name = std::string(string_at(names, row));
                if (lookup.count(name))
                    throw std::runtime_error("Duplicate IPC edge ID");
                lookup[name] = ids.size();
                ids.push_back(std::move(name));
                lengths.push_back(lens->Value(row));
            }
        }
    }
};
} // namespace sumo_native

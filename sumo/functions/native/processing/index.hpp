#pragma once
#include "io.hpp"
namespace sumo_native {
inline void index_xml(const Options &opts) {
    auto source_schema =
        arrow::schema({arrow::field("id", arrow::utf8()), arrow::field("payload", arrow::utf8())});
    auto assignment_schema = arrow::schema(
        {arrow::field("attributes", arrow::utf8()), arrow::field("choices", arrow::utf8()),
         arrow::field("probabilities", arrow::utf8()), arrow::field("alternative_count", arrow::int64())});
    IpcOutput sources(opts.at("sources"), source_schema),
        assignments(opts.at("assignments"), assignment_schema);
    std::vector<std::string> ids, payloads, attrs, choices, probabilities;
    std::vector<int64_t> counts;
    size_t bytes = 0, total = 0;
    std::string types;
    Attrs current, params;
    int depth = 0, type_depth = 0;
    bool trip = false;
    const std::vector<std::string> excluded = {"from",       "to",      "via",  "fromLonLat", "toLonLat",
                                               "viaLonLat",  "fromXY",  "toXY", "viaXY",      "fromJunction",
                                               "toJunction", "fromTaz", "toTaz"};
    auto flush_sources = [&] {
        if (ids.empty())
            return;
        sources.write(arrow::RecordBatch::Make(source_schema, ids.size(), {strings(ids), strings(payloads)}));
        ids.clear();
        payloads.clear();
        bytes = 0;
    };
    Xml source;
    source.start = [&](std::string tag, const Attrs &values) {
        ++depth;
        if (type_depth) {
            types += xml_start(tag, values);
            return;
        }
        if (depth == 2 && tag == "vType") {
            type_depth = depth;
            types += xml_start(tag, values);
            return;
        }
        if (depth == 2 && tag == "trip") {
            trip = true;
            current = values;
            params.clear();
            for (auto &key : excluded)
                current.erase(key);
        } else if (trip && depth == 3 && tag == "param") {
            params[get(values, "key")] = get(values, "value");
        } else if (depth == 2 && tag != "routes")
            throw std::runtime_error("Native source inputs require trips and vTypes; found " + tag);
    };
    source.end = [&](std::string tag) {
        if (type_depth) {
            types += "</" + tag + ">";
            if (depth == type_depth)
                type_depth = 0;
        } else if (trip && tag == "trip") {
            if (get(current, "id").empty())
                throw std::runtime_error("Source trip has no ID");
            auto data =
                "{\"attributes\":" + json_attrs(current) + ",\"parameters\":" + json_attrs(params) + "}";
            ids.push_back(current.at("id"));
            bytes += data.size() + ids.back().size();
            payloads.push_back(std::move(data));
            trip = false;
            ++total;
            if (ids.size() >= size_t(opts.batch) || bytes >= size_t(opts.bytes))
                flush_sources();
            if (total % 1000000 == 0)
                std::cout << "SUMO native indexing: " << total << " source trips\n" << std::flush;
        }
        --depth;
    };
    source.read(opts.at("trips"));
    flush_sources();
    sources.close();
    write_text(opts.at("types"), types);
    std::cout << "SUMO native indexing: source complete, " << total << " trips\n" << std::flush;
    total = 0;
    bytes = 0;
    depth = 0;
    bool vehicle = false;
    std::string paths, weights;
    int64_t alternatives = 0;
    auto flush_assignments = [&] {
        if (attrs.empty())
            return;
        assignments.write(arrow::RecordBatch::Make(
            assignment_schema, attrs.size(),
            {strings(attrs), strings(choices), strings(probabilities), ints64(counts)}));
        attrs.clear();
        choices.clear();
        probabilities.clear();
        counts.clear();
        bytes = 0;
    };
    Xml assignment;
    assignment.start = [&](std::string tag, const Attrs &values) {
        ++depth;
        if (tag == "vehicle") {
            if (vehicle)
                throw std::runtime_error("Nested assignment vehicle");
            vehicle = true;
            current = values;
            paths = "[";
            weights = "[";
            alternatives = 0;
        } else if (vehicle && tag == "route") {
            double probability = numeric(get(values, "probability", "1"));
            auto edges = tokens(get(values, "edges"));
            if (probability < 0 || edges.empty())
                throw std::runtime_error("Invalid assigned route distribution for " + get(current, "id"));
            if (probability) {
                if (alternatives++) {
                    paths += ',';
                    weights += ',';
                }
                paths += "[[";
                for (size_t i = 0; i < edges.size(); ++i) {
                    if (i)
                        paths += ',';
                    paths += json_string(edges[i]);
                }
                auto p = shortest(probability);
                paths += "]," + p + "]";
                weights += p;
            }
        }
    };
    assignment.end = [&](std::string tag) {
        if (tag == "vehicle" && vehicle) {
            if (!alternatives)
                throw std::runtime_error("No positive assigned route probability for " + get(current, "id"));
            auto encoded = json_attrs(current);
            paths += ']';
            weights += ']';
            bytes += encoded.size() + paths.size() + weights.size();
            attrs.push_back(std::move(encoded));
            choices.push_back(std::move(paths));
            probabilities.push_back(std::move(weights));
            counts.push_back(alternatives);
            vehicle = false;
            ++total;
            if (attrs.size() >= size_t(opts.batch) || bytes >= size_t(opts.bytes))
                flush_assignments();
            if (total % 1000000 == 0)
                std::cout << "SUMO native indexing: " << total << " assigned vehicles\n" << std::flush;
        }
        --depth;
    };
    assignment.read(opts.at("assigned"));
    flush_assignments();
    assignments.close();
    std::cout << "SUMO native indexing complete: " << total << " assigned vehicles\n" << std::flush;
}
} // namespace sumo_native

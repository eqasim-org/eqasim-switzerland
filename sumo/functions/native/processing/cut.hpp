#pragma once
// Reuse the same fractional-cut arithmetic as the Python reference's native kernel.
#include "../route_kernel.cpp"
#include "routes.hpp"
namespace sumo_native {
struct Piece {
    int32_t target;
    double start, end, length;
};
inline void cut(const Options &opts) {
    EdgeTable source(opts.at("edges"));
    Network region(opts.at("network"), opts.at("types"));
    region.write_edges(opts.at("output-edges"), opts);
    std::vector<std::vector<Piece>> by_edge(source.ids.size());
    IpcInput mapping(opts.at("mapping"));
    for (int b = 0; b < mapping.reader->num_record_batches(); ++b) {
        auto batch = take(mapping.reader->ReadRecordBatch(b));
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            auto from = source.lookup.find(std::string(string_at(batch->GetColumnByName("source"), i)));
            auto to = region.ids.find(std::string(string_at(batch->GetColumnByName("target"), i)));
            if (from == source.lookup.end() || to == region.ids.end())
                throw std::runtime_error("Regional mapping references an absent edge");
            auto value = [&](const std::string &name) {
                return std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName(name))->Value(i);
            };
            Piece p{to->second, value("start"), value("end"), value("length")};
            if (!(std::isfinite(p.start) && std::isfinite(p.end) && std::isfinite(p.length) && p.start >= 0 &&
                  p.start < p.end && p.end <= 1 && p.length > 0))
                throw std::runtime_error("Invalid regional edge fragment");
            by_edge[from->second].push_back(p);
        }
    }
    std::vector<int64_t> offsets{0};
    std::vector<Piece> pieces;
    std::vector<double> starts, ends, lengths;
    std::vector<std::vector<int64_t>> target_pieces(region.edges.size());
    for (const auto &edge : by_edge) {
        for (const auto &p : edge) {
            target_pieces[p.target].push_back(pieces.size());
            pieces.push_back(p);
            starts.push_back(p.start);
            ends.push_back(p.end);
            lengths.push_back(p.length);
        }
        offsets.push_back(pieces.size());
    }
    by_edge.clear();
    by_edge.shrink_to_fit();
    std::vector<std::pair<int64_t, int64_t>> pairs;
    for (const auto &transition : region.transitions) {
        int32_t a = transition.first >> 32, b = uint32_t(transition.first);
        for (auto from : target_pieces[a])
            for (auto to : target_pieces[b])
                pairs.emplace_back(from, to);
    }
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    std::vector<int64_t> allowed_from, allowed_to;
    for (const auto &pair : pairs) {
        allowed_from.push_back(pair.first);
        allowed_to.push_back(pair.second);
    }
    IpcInput input(opts.at("routes"));
    RouteWriter output(opts.at("output"), opts);
    int64_t source_count = 0, outside = 0, repeated = 0, interpolated = 0;
    for (int b = 0; b < input.reader->num_record_batches(); ++b) {
        auto batch = take(input.reader->ReadRecordBatch(b));
        check_route_batch(batch);
        std::vector<int64_t> selected;
        for (int64_t row = 0; row < batch->num_rows(); ++row) {
            ++source_count;
            auto view = list_at(batch->GetColumnByName("edges"), row);
            auto values = std::static_pointer_cast<arrow::Int32Array>(view.values);
            bool inside = false;
            for (int64_t i = 0; i < view.length; ++i) {
                int32_t edge = values->Value(view.offset + i);
                if (edge < 0 || size_t(edge) >= source.ids.size())
                    throw std::runtime_error("Regional input references unknown source edge index");
                if (offsets[edge + 1] > offsets[edge])
                    inside = true;
            }
            if (inside)
                selected.push_back(row);
            else
                ++outside;
        }
        // Native batches can contain many outside routes; only retained candidates are materialized.
        for (size_t begin = 0; begin < selected.size(); begin += opts.batch) {
            size_t count = std::min<size_t>(opts.batch, selected.size() - begin);
            std::vector<std::vector<Route>> results(count);
            std::vector<int64_t> interpolations(count, 0);
            parallel(count, opts.threads, [&](size_t index) {
                auto vehicle = read_route(batch, selected[begin + index]);
                auto attrs = Json(vehicle.attributes).attrs();
                auto params = Json(vehicle.parameters).attrs();
                std::vector<int64_t> route(vehicle.edges.begin(), vehicle.edges.end());
                if (route.empty() || vehicle.exits.size() != route.size())
                    throw std::runtime_error("Missing route edge exit times");
                int64_t capacity = 0;
                for (auto edge : route)
                    capacity += offsets[edge + 1] - offsets[edge];
                std::vector<Fragment> fragments(capacity);
                double first = position(get(attrs, "departPos", "0"), source.lengths[route.front()]),
                       last = position(get(attrs, "arrivalPos", "max"), source.lengths[route.back()]);
                double clock = vehicle.depart;
                for (auto exit : vehicle.exits) {
                    if (!std::isfinite(exit) || exit < clock)
                        throw std::runtime_error("Invalid route edge exit times");
                    clock = exit;
                }
                int64_t used = cut_route(route.size(), route.data(), vehicle.exits.data(), vehicle.depart,
                                         first, last, source.lengths.data(), offsets.data(), starts.data(),
                                         ends.data(), lengths.data(), allowed_from.data(), allowed_to.data(),
                                         allowed_from.size(), fragments.data());
                if (used < 0)
                    throw std::runtime_error("Invalid endpoint positions for " + vehicle.id);
                if (!used)
                    return;
                int64_t visits = fragments[used - 1].visit + 1;
                int64_t at = 0;
                while (at < used) {
                    int64_t stop = at + 1;
                    while (stop < used && fragments[stop].visit == fragments[at].visit)
                        ++stop;
                    auto attributes = attrs, parameters = params;
                    const auto &entry = fragments[at];
                    const auto &exit = fragments[stop - 1];
                    Route result;
                    result.id = vehicle.id + (visits > 1 ? "__cut_" + std::to_string(entry.visit) : "");
                    result.depart = numeric(number(entry.begin));
                    result.cost = exit.end - entry.begin;
                    attributes["id"] = result.id;
                    attributes["depart"] = number(entry.begin);
                    attributes["departPos"] = number(std::max(0., entry.depart_pos));
                    attributes["arrivalPos"] = number(exit.arrival_pos);
                    bool estimated = false;
                    for (int64_t i = at; i < stop; ++i) {
                        auto &part = fragments[i];
                        result.edges.push_back(pieces[part.piece].target);
                        result.exits.push_back(part.end);
                        result.distance += part.distance;
                        estimated |= part.interpolated;
                    }
                    if (attributes.count("routeLength"))
                        attributes["routeLength"] = number(result.distance);
                    if (entry.new_entry) {
                        attributes["departLane"] = "best";
                        attributes["departSpeed"] = "max";
                    }
                    if (exit.new_exit) {
                        attributes.erase("arrivalLane");
                        attributes.erase("arrivalSpeed");
                    }
                    if (attributes.count("arrival"))
                        attributes["arrival"] = number(exit.end);
                    parameters["sumo_parent_vehicle_id"] = vehicle.id;
                    parameters["sumo_cut_visit"] = std::to_string(entry.visit);
                    parameters["sumo_original_depart"] = python_float(vehicle.depart);
                    parameters["sumo_boundary_timing"] = estimated ? "interpolated" : "edge_exit_times";
                    result.attributes = json_attrs(attributes);
                    result.parameters = json_attrs(parameters);
                    region.validate(result.edges, attributes);
                    if (estimated)
                        ++interpolations[index];
                    results[index].push_back(std::move(result));
                    at = stop;
                }
            });
            for (size_t i = 0; i < count; ++i) {
                if (results[i].empty())
                    ++outside;
                if (results[i].size() > 1)
                    ++repeated;
                interpolated += interpolations[i];
                for (auto &route : results[i])
                    output.append(std::move(route));
            }
        }
        if (source_count % 100000 < batch->num_rows())
            std::cout << "SUMO native regional cutting: " << source_count << " source routes, "
                      << output.count << " visits written\n"
                      << std::flush;
    }
    output.close();
    std::string report = "{\"source_count\":" + std::to_string(source_count) +
                         ",\"count\":" + std::to_string(output.count) +
                         ",\"outside_or_discarded\":" + std::to_string(outside) +
                         ",\"repeated_visits\":" + std::to_string(repeated) +
                         ",\"interpolated_visits\":" + std::to_string(interpolated) + "}\n";
    write_text(opts.at("report"), report);
    std::cout << "SUMO native regional cutting complete: " << source_count << " routes, " << output.count
              << " visits\n"
              << std::flush;
}
} // namespace sumo_native

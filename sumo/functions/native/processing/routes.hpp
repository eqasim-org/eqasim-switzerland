#pragma once
#include "network.hpp"
namespace sumo_native {
struct Route {
    std::string id, attributes, parameters;
    double depart = 0, cost = 0, distance = 0;
    std::vector<int32_t> edges;
    std::vector<double> exits;
};
inline std::shared_ptr<arrow::Schema> route_schema() {
    return arrow::schema(
        {arrow::field("id", arrow::utf8()), arrow::field("depart", arrow::float64()),
         arrow::field("attributes", arrow::utf8()), arrow::field("parameters", arrow::utf8()),
         arrow::field("edges", arrow::list(arrow::int32())),
         arrow::field("exit_times", arrow::list(arrow::float64())), arrow::field("cost", arrow::float64()),
         arrow::field("distance", arrow::float64())});
}
class RouteWriter {
    IpcOutput output;
    std::vector<Route> rows;
    const Options &opts;
    size_t bytes = 0;

  public:
    int64_t count = 0;
    RouteWriter(const std::string &path, const Options &settings)
        : output(path, route_schema()->WithMetadata(arrow::key_value_metadata(
                           {"sumo_format", "sumo_edges", "sumo_types"},
                           {"sumo-routes-v1",
                            settings.values.count("--metadata-edges")
                                ? settings.at("metadata-edges")
                                : (settings.values.count("--output-edges") ? settings.at("output-edges")
                                                                           : settings.at("edges")),
                            settings.at("types")}))),
          opts(settings) {}
    void append(Route route) {
        bytes += route.id.size() + route.attributes.size() + route.parameters.size() +
                 route.edges.size() * 12 + 32;
        rows.push_back(std::move(route));
        ++count;
        if (rows.size() >= size_t(opts.batch) || bytes >= size_t(opts.bytes))
            flush();
    }
    void flush() {
        if (rows.empty())
            return;
        std::vector<std::string> ids, attrs, params;
        std::vector<double> depart, cost, distance;
        auto edge_values = std::make_shared<arrow::Int32Builder>();
        auto exit_values = std::make_shared<arrow::DoubleBuilder>();
        arrow::ListBuilder edges(arrow::default_memory_pool(), edge_values),
            exits(arrow::default_memory_pool(), exit_values);
        for (const auto &route : rows) {
            ids.push_back(route.id);
            attrs.push_back(route.attributes);
            params.push_back(route.parameters);
            depart.push_back(route.depart);
            cost.push_back(route.cost);
            distance.push_back(route.distance);
            check(edges.Append());
            check(exits.Append());
            check(edge_values->AppendValues(route.edges));
            check(exit_values->AppendValues(route.exits));
        }
        output.write(arrow::RecordBatch::Make(route_schema(), rows.size(),
                                              {strings(ids), doubles(depart), strings(attrs), strings(params),
                                               take(edges.Finish()), take(exits.Finish()), doubles(cost),
                                               doubles(distance)}));
        rows.clear();
        bytes = 0;
    }
    void close() {
        flush();
        output.close();
    }
};
struct ListView {
    std::shared_ptr<arrow::Array> values;
    int64_t offset, length;
};
inline void check_route_batch(const std::shared_ptr<arrow::RecordBatch> &batch) {
    for (const auto &key : {"id", "attributes", "parameters"}) {
        auto column = batch->GetColumnByName(key);
        if (!column || column->null_count() ||
            (column->type_id() != arrow::Type::STRING && column->type_id() != arrow::Type::LARGE_STRING &&
             column->type_id() != arrow::Type::STRING_VIEW))
            throw std::runtime_error("Invalid route IPC string column: " + std::string(key));
    }
    for (const auto &key : {"depart", "cost", "distance"}) {
        auto column = batch->GetColumnByName(key);
        if (!column || column->null_count() || column->type_id() != arrow::Type::DOUBLE)
            throw std::runtime_error("Invalid route IPC numeric column: " + std::string(key));
    }
    for (const auto &key : {"edges", "exit_times"}) {
        auto column = batch->GetColumnByName(key);
        if (!column || column->null_count() ||
            (column->type_id() != arrow::Type::LIST && column->type_id() != arrow::Type::LARGE_LIST))
            throw std::runtime_error("Invalid route IPC list column: " + std::string(key));
        auto item = static_cast<const arrow::BaseListType &>(*column->type()).value_type();
        if (item->id() != (std::string(key) == "edges" ? arrow::Type::INT32 : arrow::Type::DOUBLE))
            throw std::runtime_error("Invalid route IPC list value type: " + std::string(key));
    }
}
inline ListView list_at(const std::shared_ptr<arrow::Array> &column, int64_t row) {
    if (!column || column->IsNull(row))
        throw std::runtime_error("Null route list");
    if (column->type_id() == arrow::Type::LIST) {
        auto &array = static_cast<arrow::ListArray &>(*column);
        return {array.values(), array.value_offset(row), array.value_length(row)};
    }
    if (column->type_id() == arrow::Type::LARGE_LIST) {
        auto &array = static_cast<arrow::LargeListArray &>(*column);
        return {array.values(), array.value_offset(row), array.value_length(row)};
    }
    throw std::runtime_error("Expected route list column");
}
inline Route read_route(const std::shared_ptr<arrow::RecordBatch> &batch, int64_t row) {
    Route out;
    out.id = string_at(batch->GetColumnByName("id"), row);
    out.depart = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("depart"))->Value(row);
    out.attributes = string_at(batch->GetColumnByName("attributes"), row);
    out.parameters = string_at(batch->GetColumnByName("parameters"), row);
    out.cost = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("cost"))->Value(row);
    out.distance =
        std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("distance"))->Value(row);
    auto edges = list_at(batch->GetColumnByName("edges"), row),
         times = list_at(batch->GetColumnByName("exit_times"), row);
    auto ev = std::static_pointer_cast<arrow::Int32Array>(edges.values);
    auto tv = std::static_pointer_cast<arrow::DoubleArray>(times.values);
    if (edges.length != times.length)
        throw std::runtime_error("Unaligned edge exit times");
    for (int64_t i = 0; i < edges.length; ++i) {
        out.edges.push_back(ev->Value(edges.offset + i));
        out.exits.push_back(tv->Value(times.offset + i));
    }
    return out;
}

struct GroupTable {
    IpcInput input;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    std::vector<std::string_view> choices;
    explicit GroupTable(const std::string &path) : input(path) {
        for (int b = 0; b < input.reader->num_record_batches(); ++b) {
            auto batch = take(input.reader->ReadRecordBatch(b));
            auto gids = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("gid"));
            auto data = batch->GetColumnByName("choices");
            for (int64_t i = 0; i < batch->num_rows(); ++i) {
                if (gids->Value(i) != int64_t(choices.size() + 1))
                    throw std::runtime_error("Noncontiguous route group IDs");
                choices.push_back(string_at(data, i));
            }
            batches.push_back(std::move(batch));
        }
    }
    std::vector<std::string> selected(int64_t gid, int choice) const {
        if (gid <= 0 || gid > int64_t(choices.size()))
            throw std::runtime_error("Unknown assignment group");
        return Json(choices[gid - 1]).selected_path(choice);
    }
};
struct Selected {
    IpcInput input;
    int batch_index = 0;
    int64_t row = 0;
    std::shared_ptr<arrow::RecordBatch> batch;
    explicit Selected(const std::string &path) : input(path) {
        advance_batch();
    }
    void advance_batch() {
        while (!batch || row == batch->num_rows()) {
            if (batch_index == input.reader->num_record_batches()) {
                batch = nullptr;
                return;
            }
            batch = take(input.reader->ReadRecordBatch(batch_index++));
            row = 0;
        }
    }
    bool valid() const {
        return bool(batch);
    }
    double departure() const {
        return std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("depart"))->Value(row);
    }
    struct Item {
        std::string id, attrs, params;
        double depart;
        int64_t gid;
        int choice;
    };
    Item next() {
        Item out{std::string(string_at(batch->GetColumnByName("id"), row)),
                 std::string(string_at(batch->GetColumnByName("attributes"), row)),
                 std::string(string_at(batch->GetColumnByName("parameters"), row)),
                 departure(),
                 std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("gid"))->Value(row),
                 std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName("choice"))->Value(row)};
        ++row;
        advance_batch();
        return out;
    }
};
inline std::shared_ptr<arrow::Schema> metrics_schema() {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (auto key : {"trip_id", "source_trip_id", "person_id", "trip_index", "source_mode"})
        fields.push_back(arrow::field(key, arrow::utf8()));
    for (auto key : {"departure_time_s", "reference_travel_time_s", "loaded_travel_time_s", "distance_m"})
        fields.push_back(arrow::field(key, arrow::float64()));
    fields.push_back(arrow::field("loads_network", arrow::boolean()));
    return arrow::schema(fields);
}
struct MetricWriter {
    IpcOutput output;
    std::vector<Route> rows;
    const Options &opts;
    size_t bytes = 0;
    MetricWriter(const std::string &path, const Options &settings)
        : output(path, metrics_schema()), opts(settings) {}
    void append(const Route &route) {
        Route compact;
        compact.id = route.id;
        compact.depart = route.depart;
        compact.cost = route.cost;
        compact.distance = route.distance;
        compact.parameters = route.parameters;
        bytes += compact.id.size() + compact.parameters.size() + 32;
        rows.push_back(std::move(compact));
        if (rows.size() >= size_t(opts.batch) || bytes >= size_t(opts.bytes))
            flush();
    }
    void flush() {
        if (rows.empty())
            return;
        auto schema = metrics_schema();
        std::vector<std::shared_ptr<arrow::ArrayBuilder>> builders;
        for (const auto &field : schema->fields())
            builders.push_back(take(arrow::MakeBuilder(field->type())));
        for (const auto &route : rows) {
            auto params = Json(route.parameters).attrs();
            int index = 0;
            for (auto key : {"", "source_trip_id", "person_id", "trip_index", "mode"}) {
                auto *builder = static_cast<arrow::StringBuilder *>(builders[index].get());
                if (index == 0)
                    check(builder->Append(route.id));
                else if (params.count(key))
                    check(builder->Append(params[key]));
                else
                    check(builder->AppendNull());
                ++index;
            }
            check(static_cast<arrow::DoubleBuilder *>(builders[5].get())->Append(route.depart));
            auto *reference = static_cast<arrow::DoubleBuilder *>(builders[6].get());
            if (params.count("source_travel_time_s"))
                check(reference->Append(numeric(params.at("source_travel_time_s"))));
            else
                check(reference->AppendNull());
            check(
                static_cast<arrow::DoubleBuilder *>(builders[7].get())->Append(numeric(number(route.cost))));
            check(static_cast<arrow::DoubleBuilder *>(builders[8].get())
                      ->Append(numeric(number(route.distance))));
            check(static_cast<arrow::BooleanBuilder *>(builders[9].get())
                      ->Append(get(params, "_sumo_loads_network") == "true"));
        }
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        for (auto &builder : builders)
            arrays.push_back(take(builder->Finish()));
        output.write(arrow::RecordBatch::Make(metrics_schema(), rows.size(), arrays));
        rows.clear();
        bytes = 0;
    }
    void close() {
        flush();
        output.close();
    }
};
inline void realize(const Options &opts) {
    Network network(opts.at("network"), opts.at("types"));
    network.write_edges(opts.at("edges"), opts);
    GroupTable groups(opts.at("groups"));
    Selected selected(opts.at("selected"));
    RouteWriter routes(opts.at("output"), opts);
    MetricWriter metrics(opts.at("metrics"), opts);
    std::vector<double> costs(network.edges.size(), std::numeric_limits<double>::quiet_NaN());
    double begin = 0, end = 0, previous_end = -std::numeric_limits<double>::infinity(),
           previous_depart = -std::numeric_limits<double>::infinity();
    std::string previous_id;
    bool active = false;
    Xml weights;
    auto process = [&] {
        while (selected.valid() && selected.departure() < end) {
            std::vector<Selected::Item> items;
            size_t bytes = 0;
            while (selected.valid() && selected.departure() < end && items.size() < size_t(opts.batch) &&
                   bytes < size_t(opts.bytes)) {
                auto item = selected.next();
                if (!std::isfinite(item.depart) || item.depart < begin || item.depart < 0)
                    throw std::runtime_error("Loaded edge costs do not cover departure " +
                                             shortest(item.depart));
                if (item.depart < previous_depart ||
                    (item.depart == previous_depart && item.id < previous_id))
                    throw std::runtime_error("Selected demand is not sorted by departure and ID");
                previous_depart = item.depart;
                previous_id = item.id;
                if (item.gid <= 0 || item.gid > int64_t(groups.choices.size()))
                    throw std::runtime_error("Unknown assignment group");
                bytes += item.id.size() + item.attrs.size() + item.params.size() +
                         8 * groups.choices[item.gid - 1].size();
                items.push_back(std::move(item));
            }
            std::vector<Route> result(items.size());
            parallel(items.size(), opts.threads, [&](size_t i) {
                const auto &item = items[i];
                auto attrs = Json(item.attrs).attrs();
                auto params = Json(item.params).attrs();
                if (get(attrs, "id") != item.id || numeric(get(attrs, "depart")) != item.depart)
                    throw std::runtime_error("Selected route identity/departure does not match metadata");
                Route route;
                route.id = item.id;
                route.depart = item.depart;
                route.attributes = item.attrs;
                params["sumo_timing"] = "macroscopic_departure_slice";
                route.parameters = json_attrs(params);
                auto path = groups.selected(item.gid, item.choice);
                for (const auto &name : path) {
                    auto found = network.ids.find(name);
                    if (found == network.ids.end())
                        throw std::runtime_error("Route " + item.id + " references absent edge " + name);
                    route.edges.push_back(found->second);
                }
                network.validate(route.edges, attrs);
                double first =
                    position(get(attrs, "departPos", "0"), network.edges[route.edges.front()].length);
                double last =
                    position(get(attrs, "arrivalPos", "max"), network.edges[route.edges.back()].length);
                double clock = route.depart;
                for (size_t j = 0; j < route.edges.size(); ++j) {
                    int edge = route.edges[j];
                    double length = network.edges[edge].length, travel = costs[edge];
                    if (!std::isfinite(travel))
                        throw std::runtime_error("No loaded travel time for assigned edge " +
                                                 network.edges[edge].id);
                    if (length <= 0 || travel <= 0)
                        throw std::runtime_error("Invalid loaded edge length or travel time");
                    double low = j == 0 ? first : 0, high = j + 1 == route.edges.size() ? last : length;
                    if (high < low)
                        throw std::runtime_error("Arrival precedes departure position for " + item.id);
                    route.distance += high - low;
                    clock += travel * (high - low) / length;
                    route.exits.push_back(clock);
                }
                route.cost = clock - route.depart;
                result[i] = std::move(route);
            });
            for (auto &route : result) {
                metrics.append(route);
                routes.append(std::move(route));
            }
            if (routes.count % 100000 < opts.batch)
                std::cout << "SUMO native realization: " << routes.count << " vehicles timed/written\n"
                          << std::flush;
        }
    };
    weights.start = [&](std::string tag, const Attrs &attrs) {
        if (tag == "interval") {
            if (active)
                throw std::runtime_error("Nested edge-cost interval");
            begin = numeric(get(attrs, "begin"));
            end = numeric(get(attrs, "end"));
            if (!(previous_end <= begin && begin < end))
                throw std::runtime_error("Loaded edge-cost intervals must be ordered and nonoverlapping");
            std::fill(costs.begin(), costs.end(), std::numeric_limits<double>::quiet_NaN());
            active = true;
        } else if (tag == "edge" && active) {
            double cost = numeric(get(attrs, "traveltime"));
            if (cost <= 0)
                throw std::runtime_error("Invalid loaded travel time for edge " + get(attrs, "id"));
            auto edge = network.ids.find(get(attrs, "id"));
            if (edge != network.ids.end())
                costs[edge->second] = cost;
        }
    };
    weights.end = [&](std::string tag) {
        if (tag == "interval" && active) {
            process();
            active = false;
            previous_end = end;
            std::cout << "SUMO native realization: loaded interval [" << begin << ", " << end << "); "
                      << routes.count << " written\n"
                      << std::flush;
            if (!selected.valid())
                weights.stop = true;
        }
    };
    weights.read(opts.at("weights"));
    if (selected.valid())
        throw std::runtime_error("Loaded edge costs do not cover every departure");
    routes.close();
    metrics.close();
    write_text(opts.at("report"),
               "{\"vehicles\":" + std::to_string(routes.count) + ",\"format\":\"arrow\"}\n");
}
} // namespace sumo_native

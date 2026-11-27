#pragma once
#include "routes.hpp"
namespace sumo_native {
inline std::string gzip_block(std::string data) {
  z_stream stream{};
  if (deflateInit2(&stream, 1, Z_DEFLATED, 31, 8, Z_DEFAULT_STRATEGY) != Z_OK)
    throw std::runtime_error("Cannot initialize gzip compressor");
  std::string result(deflateBound(&stream, data.size()), '\0');
  stream.next_in = reinterpret_cast<Bytef *>(data.data());
  stream.avail_in = data.size();
  stream.next_out = reinterpret_cast<Bytef *>(result.data());
  stream.avail_out = result.size();
  int status = deflate(&stream, Z_FINISH);
  auto size = stream.total_out;
  deflateEnd(&stream);
  if (status != Z_STREAM_END)
    throw std::runtime_error("Cannot compress XML block");
  result.resize(size);
  return result;
}
class XmlOutput {
  std::ofstream output;
  std::string buffer;
  std::deque<std::future<std::string>> pending;
  const Options &opts;
  bool gzip;
  void collect() {
    auto data = pending.front().get();
    pending.pop_front();
    output.write(data.data(), data.size());
    if (!output)
      throw std::runtime_error("Cannot write regional XML output");
  }

public:
  XmlOutput(const std::string &path, const Options &options)
      : output(path, std::ios::binary), opts(options),
        gzip(path.size() >= 3 && path.substr(path.size() - 3) == ".gz") {
    if (!output)
      throw std::runtime_error("Cannot open regional XML output");
    buffer.reserve(opts.bytes);
  }
  void append(std::string_view text) {
    buffer.append(text);
    if (buffer.size() >= size_t(opts.bytes))
      flush();
  }
  void flush() {
    if (buffer.empty())
      return;
    if (gzip) {
      pending.push_back(
          std::async(std::launch::async, gzip_block, std::move(buffer)));
      if (pending.size() >= size_t(opts.xml_threads))
        collect();
      buffer = std::string();
    } else {
      output.write(buffer.data(), buffer.size());
      buffer.clear();
      if (!output)
        throw std::runtime_error("Cannot write regional XML output");
    }
  }
  void close() {
    flush();
    while (!pending.empty())
      collect();
    output.close();
    if (!output)
      throw std::runtime_error("Cannot close regional XML output");
  }
};
inline bool retain_route(const std::string &id, double fraction,
                         const std::string &seed) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : seed + ":" + id) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ULL;
  hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebULL;
  hash ^= hash >> 31;
  return double(hash >> 11) / 9007199254740992.0 < fraction;
}
inline std::string route_xml(const Route &route, const EdgeTable &edges,
                             const Network &network, double retention = 0,
                             const std::string &seed = "0") {
  auto attrs = Json(route.attributes).attrs();
  auto params = Json(route.parameters).attrs();
  if (get(attrs, "id") != route.id ||
      numeric(get(attrs, "depart")) != route.depart)
    throw std::runtime_error(
        "Route identity/departure does not match metadata");
  network.validate(route.edges, attrs);
  if (retain_route(route.id, retention, seed))
    params.emplace("device.rerouting.pre-period", "0");
  std::string xml = xml_start("vehicle", attrs) + "<route edges=\"";
  for (size_t i = 0; i < route.edges.size(); ++i) {
    if (i)
      xml += ' ';
    auto edge = route.edges[i];
    if (edge < 0 || size_t(edge) >= edges.ids.size())
      throw std::runtime_error("Unknown regional edge index");
    xml += xml_escape(edges.ids[edge]);
  }
  xml += "\" exitTimes=\"";
  for (size_t i = 0; i < route.exits.size(); ++i) {
    if (i)
      xml += ' ';
    xml += number(route.exits[i]);
  }
  xml += "\" cost=\"" + number(route.cost) + "\" routeLength=\"" +
         number(route.distance) + "\"/>";
  for (const auto &pair : params)
    xml += "<param key=\"" + xml_escape(pair.first) + "\" value=\"" +
           xml_escape(pair.second) + "\"/>";
  return xml + "</vehicle>\n";
}
inline void export_xml(const Options &opts) {
  EdgeTable edges(opts.at("edges"));
  Network network(opts.at("network"), opts.at("types"));
  if (edges.ids.size() != network.edges.size())
    throw std::runtime_error("Regional edge dictionary does not match network");
  for (size_t i = 0; i < edges.ids.size(); ++i)
    if (edges.ids[i] != network.edges[i].id)
      throw std::runtime_error(
          "Regional edge dictionary order does not match network");
  IpcInput input(opts.at("routes"));
  XmlOutput output(opts.at("output"), opts);
  output.append("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<routes>\n");
  output.append(read_text(opts.at("types")));
  double retention = opts.values.count("--initial-route-retention")
                         ? numeric(opts.at("initial-route-retention"))
                         : 0;
  if (retention < 0 || retention > 1)
    throw std::runtime_error(
        "Initial route retention must be between zero and one");
  std::string seed =
      opts.values.count("--retention-seed") ? opts.at("retention-seed") : "0";
  int64_t written = 0;
  double previous = -std::numeric_limits<double>::infinity();
  std::string previous_id;
  for (int b = 0; b < input.reader->num_record_batches(); ++b) {
    auto batch = take(input.reader->ReadRecordBatch(b));
    check_route_batch(batch);
    for (int64_t start = 0; start < batch->num_rows(); start += opts.batch) {
      size_t count = std::min<int64_t>(opts.batch, batch->num_rows() - start);
      std::vector<std::string> result(count);
      parallel(count, opts.threads, [&](size_t i) {
        result[i] = route_xml(read_route(batch, start + i), edges, network,
                              retention, seed);
      });
      for (size_t i = 0; i < count; ++i) {
        auto id =
            std::string(string_at(batch->GetColumnByName("id"), start + i));
        double depart = std::static_pointer_cast<arrow::DoubleArray>(
                            batch->GetColumnByName("depart"))
                            ->Value(start + i);
        if (depart < previous || (depart == previous && id < previous_id))
          throw std::runtime_error("Regional routes are not sorted");
        previous = depart;
        previous_id = id;
        output.append(result[i]);
        ++written;
      }
      if (written % 100000 < opts.batch)
        std::cout << "SUMO native XML: " << written
                  << " regional vehicles written\n"
                  << std::flush;
    }
  }
  output.append("</routes>\n");
  output.close();
  std::cout << "SUMO native XML complete: " << written << " vehicles\n"
            << std::flush;
}
} // namespace sumo_native

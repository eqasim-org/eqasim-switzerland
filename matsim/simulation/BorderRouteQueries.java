import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.GZIPOutputStream;
import org.matsim.api.core.v01.*;
import org.matsim.api.core.v01.network.*;
import org.matsim.api.core.v01.population.Person;
import org.matsim.vehicles.Vehicle;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.network.turnRestrictions.DisallowedNextLinksUtils;
import org.matsim.core.router.util.*;
import org.eqasim.core.components.routing.speedy.MemoryEfficientSpeedyALTFactory;

/** Small source-launch helper using the configured eqasim JAR's car/turn-aware router.
 * It reads a local query batch, returns metrics, and never writes the scenario network.
 */
public class BorderRouteQueries {
    private static Set<String> inside;

    public static void main(String[] args) throws Exception {
        // Arguments: network, inside-node IDs, query CSV, result CSV.gz, full/exterior.
        inside = new HashSet<>(Files.readAllLines(Path.of(args[1])));
        Network network = NetworkUtils.readNetwork(args[0]);
        for (Link link : new ArrayList<>(network.getLinks().values())) {
            if (!link.getAllowedModes().contains("car")) {
                network.removeLink(link.getId());
                continue;
            }
            // The router's restrictions context has no mode filter of its own.
            // Retain car bans, including multi-link sequences, but not bus/truck bans.
            link.setAllowedModes(Set.of("car"));
            var bans = NetworkUtils.getDisallowedNextLinks(link);
            if (bans != null) {
                for (String mode : new ArrayList<>(bans.getAsMap().keySet())) {
                    if (!mode.equals("car")) bans.removeDisallowedLinkSequences(mode);
                }
            }
        }
        if (args[4].equals("exterior")) isolateExterior(network);
        else if (!args[4].equals("full")) throw new IllegalArgumentException(args[4]);
        for (Node node : new ArrayList<>(network.getNodes().values())) {
            if (node.getInLinks().isEmpty() && node.getOutLinks().isEmpty()) network.removeNode(node.getId());
        }
        DisallowedNextLinksUtils.clean(network);
        // Removing Swiss roads intentionally disconnects some foreign approaches.
        // Reject impossible directed pairs before ALT explores a whole component.
        DirectedReachability reachability = new DirectedReachability(network);
        TravelTime time = (link, t, person, vehicle) -> link.getLength() / link.getFreespeed();
        TravelDisutility cost = new TravelDisutility() {
            public double getLinkTravelDisutility(Link link, double t, Person person, Vehicle vehicle) {
                return getLinkMinimumTravelDisutility(link);
            }
            public double getLinkMinimumTravelDisutility(Link link) {
                return link.getLength() / link.getFreespeed();
            }
        };
        LeastCostPathCalculator router = new MemoryEfficientSpeedyALTFactory()
                .createPathCalculator(network, cost, time);
        try (var input = Files.newBufferedReader(Path.of(args[2]));
             var output = new PrintWriter(new GZIPOutputStream(Files.newOutputStream(Path.of(args[3]))))) {
            if (!"query_id,start_link,end_link".equals(input.readLine())) {
                throw new IllegalArgumentException("Unexpected border-route query schema");
            }
            output.println("query_id,found,time_s,distance_m,entries,exits,restriction_violations");
            String line;
            int count = 0, disconnected = 0;
            while ((line = input.readLine()) != null) {
                String[] query = line.split(",", -1);
                Link from = network.getLinks().get(Id.createLinkId(query[1]));
                Link to = network.getLinks().get(Id.createLinkId(query[2]));
                boolean reachable = from != null && to != null && reachability.canReach(from, to);
                if (!reachable) disconnected++;
                var path = !reachable ? null
                        : router.calcLeastCostPath(from, to, 28800, null, null);
                if (path == null) {
                    output.println(query[0] + ",false,0,0,0,0,0");
                } else {
                    var links = new ArrayList<Link>();
                    links.add(from);
                    links.addAll(path.links);
                    if (!from.equals(to) || !path.links.isEmpty()) links.add(to);
                    record(output, query[0], links);
                }
                if (++count % 1000 == 0) {
                    output.flush();
                    System.out.println("Border-route queries: " + count
                            + " (disconnected pairs rejected: " + disconnected + ")");
                }
            }
            System.out.println("Border-route queries completed: " + count
                    + "; disconnected pairs rejected: " + disconnected);
        }
    }

    /** Directed reachability on a compact graph of strongly connected components.
     * Turn restrictions are deliberately ignored here: this is a necessary test,
     * not a replacement router. Only a proven absence of a path rejects a query.
     */
    static final class DirectedReachability {
        private final Map<Id<Node>, Integer> nodeIndices = new HashMap<>();
        private final int[] component, offsets, targets;
        // Bound memory even when survey origins occupy many separate components.
        private final Map<Integer, BitSet> reachableComponents = new LinkedHashMap<>(64, .75f, true) {
            protected boolean removeEldestEntry(Map.Entry<Integer, BitSet> entry) {
                return size() > 64;
            }
        };

        DirectedReachability(Network network) {
            int n = network.getNodes().size(), m = network.getLinks().size(), index = 0;
            for (Node node : network.getNodes().values()) nodeIndices.put(node.getId(), index++);
            int[] outgoing = new int[n + 1], incoming = new int[n + 1];
            for (Link link : network.getLinks().values()) {
                outgoing[nodeIndices.get(link.getFromNode().getId()) + 1]++;
                incoming[nodeIndices.get(link.getToNode().getId()) + 1]++;
            }
            for (int i = 1; i <= n; i++) {
                outgoing[i] += outgoing[i - 1];
                incoming[i] += incoming[i - 1];
            }
            int[] forward = new int[m], reverse = new int[m];
            int[] outCursor = outgoing.clone(), inCursor = incoming.clone();
            for (Link link : network.getLinks().values()) {
                int from = nodeIndices.get(link.getFromNode().getId());
                int to = nodeIndices.get(link.getToNode().getId());
                forward[outCursor[from]++] = to;
                reverse[inCursor[to]++] = from;
            }

            // Iterative Kosaraju avoids recursion overflow on national networks.
            boolean[] seen = new boolean[n];
            int[] stack = new int[n], order = new int[n], cursor = outgoing.clone();
            int ordered = 0;
            for (int start = 0; start < n; start++) {
                if (seen[start]) continue;
                int top = 0;
                stack[0] = start;
                seen[start] = true;
                while (top >= 0) {
                    int node = stack[top];
                    if (cursor[node] < outgoing[node + 1]) {
                        int next = forward[cursor[node]++];
                        if (!seen[next]) {
                            seen[next] = true;
                            stack[++top] = next;
                        }
                    } else {
                        order[ordered++] = node;
                        top--;
                    }
                }
            }
            component = new int[n];
            Arrays.fill(component, -1);
            int components = 0;
            for (int i = n - 1; i >= 0; i--) {
                int start = order[i];
                if (component[start] >= 0) continue;
                int size = 1;
                stack[0] = start;
                component[start] = components;
                while (size > 0) {
                    int node = stack[--size];
                    for (int edge = incoming[node]; edge < incoming[node + 1]; edge++) {
                        int next = reverse[edge];
                        if (component[next] < 0) {
                            component[next] = components;
                            stack[size++] = next;
                        }
                    }
                }
                components++;
            }

            // Collapse each component; one-way connections between components
            // must remain usable (different SCCs do not imply no directed path).
            offsets = new int[components + 1];
            for (int node = 0; node < n; node++) {
                for (int edge = outgoing[node]; edge < outgoing[node + 1]; edge++) {
                    if (component[node] != component[forward[edge]]) offsets[component[node] + 1]++;
                }
            }
            for (int i = 1; i <= components; i++) offsets[i] += offsets[i - 1];
            targets = new int[offsets[components]];
            cursor = offsets.clone();
            for (int node = 0; node < n; node++) {
                for (int edge = outgoing[node]; edge < outgoing[node + 1]; edge++) {
                    int from = component[node], to = component[forward[edge]];
                    if (from != to) targets[cursor[from]++] = to;
                }
            }
            System.out.println("Border-route reachability: " + n + " nodes, " + components + " directed components");
        }

        boolean canReach(Link from, Link to) {
            if (from.getId().equals(to.getId())) return true;
            // MATSim routes between the end of the start link and the beginning
            // of the destination link; adjacent links need no intermediate edge.
            int start = component[nodeIndices.get(from.getToNode().getId())];
            int end = component[nodeIndices.get(to.getFromNode().getId())];
            if (start == end) return true;
            BitSet reached = reachableComponents.get(start);
            if (reached == null) {
                reached = new BitSet(offsets.length - 1);
                int[] stack = new int[offsets.length - 1];
                int size = 1;
                stack[0] = start;
                reached.set(start);
                while (size > 0) {
                    int node = stack[--size];
                    for (int edge = offsets[node]; edge < offsets[node + 1]; edge++) {
                        int next = targets[edge];
                        if (!reached.get(next)) {
                            reached.set(next);
                            stack[size++] = next;
                        }
                    }
                }
                reachableComponents.put(start, reached);
            }
            return reached.get(end);
        }
    }

    private static void isolateExterior(Network network) {
        // An approach must remain outside CH until its selected entry (and vice
        // versa on return). Split crossing endpoints into private dead ends so
        // even two border links sharing a Swiss node cannot shortcut through CH.
        for (Link link : new ArrayList<>(network.getLinks().values())) {
            boolean fromInside = inside.contains(link.getFromNode().getId().toString());
            boolean toInside = inside.contains(link.getToNode().getId().toString());
            if (fromInside && toInside) {
                network.removeLink(link.getId());
            } else if (fromInside != toInside) {
                Node old = fromInside ? link.getFromNode() : link.getToNode();
                Node terminal = network.getFactory().createNode(
                        Id.createNodeId("borderQuery:" + link.getId()), old.getCoord());
                network.addNode(terminal);
                inside.add(terminal.getId().toString());
                // Remove/reinsert updates MATSim's node adjacency maps too.
                network.removeLink(link.getId());
                if (fromInside) link.setFromNode(terminal); else link.setToNode(terminal);
                network.addLink(link);
            }
        }
    }

    private static void record(PrintWriter output, String id, List<Link> links) {
        int entries = 0, exits = 0, violations = 0;
        double distance = 0, time = 0;
        for (int i = 0; i < links.size(); i++) {
            Link link = links.get(i);
            distance += link.getLength();
            time += link.getLength() / link.getFreespeed();
            boolean from = inside.contains(link.getFromNode().getId().toString());
            boolean to = inside.contains(link.getToNode().getId().toString());
            if (from != to) { if (to) entries++; else exits++; }
            if (i > 0 && !links.get(i - 1).getToNode().equals(link.getFromNode())) {
                throw new IllegalStateException("Disconnected border route " + id);
            }
            var bans = NetworkUtils.getDisallowedNextLinks(link);
            if (bans == null) continue;
            for (var sequence : bans.getDisallowedLinkSequences("car")) {
                if (i + sequence.size() >= links.size()) continue;
                boolean match = true;
                for (int j = 0; j < sequence.size(); j++) {
                    match &= sequence.get(j).equals(links.get(i + j + 1).getId());
                }
                if (match) violations++;
            }
        }
        output.println(id + ",true," + time + "," + distance + "," + entries + "," + exits + "," + violations);
    }
}

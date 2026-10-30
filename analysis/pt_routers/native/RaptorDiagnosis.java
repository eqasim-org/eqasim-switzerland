// Counterfactual PT diagnostics against the exact eqasim/MATSim runtime JAR.
// Loads existing scenario inputs read-only; no pipeline or simulation is run.
import java.nio.file.*;
import java.util.*;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.SequenceWriter;
import com.fasterxml.jackson.dataformat.csv.CsvMapper;
import com.fasterxml.jackson.dataformat.csv.CsvSchema;
import org.eqasim.core.components.raptor.EqasimRaptorConfigGroup;
import org.eqasim.core.components.raptor.EqasimRaptorUtils;
import org.matsim.api.core.v01.*;
import org.matsim.api.core.v01.network.*;
import org.matsim.api.core.v01.population.*;
import org.matsim.core.config.*;
import org.matsim.core.network.*;
import org.matsim.core.network.io.MatsimNetworkReader;
import org.matsim.core.router.*;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.core.utils.geometry.CoordUtils;
import org.matsim.facilities.FacilitiesUtils;
import org.matsim.pt.routes.TransitPassengerRoute;
import org.matsim.pt.transitSchedule.api.TransitScheduleReader;
import org.matsim.vehicles.MatsimVehicleReader;
import ch.sbb.matsim.config.SwissRailRaptorConfigGroup;
import ch.sbb.matsim.routing.pt.raptor.*;

public class RaptorDiagnosis {
    private static final double SUPPRESS_DIRECT_WALK = 1_000_000.;

    @JsonIgnoreProperties(ignoreUnknown=true)
    public static class Request {
        public String trip_id, cohort;
        public double origin_x, origin_y, destination_x, destination_y, departure_time, euclidean_distance_km;
        public boolean expected_java;
    }
    public static class Result {
        public String trip_id, cohort, variant, status, error, pt_modes, origin_link, destination_link;
        public boolean expected_java, available, same_snapped_link;
        public double search_radius, direct_walk_factor, transfer_walk_limit, departure_time, euclidean_distance_km;
        public double origin_snap_m, destination_snap_m, route_time_min=Double.NaN, route_distance_km=Double.NaN;
        public double walk_time_min=0, in_vehicle_time_min=0, estimated_direct_walk_min;
        public int boardings, origin_stops_400, destination_stops_400, origin_stops_1500, destination_stops_1500;
    }
    record Variant(String name, boolean originalCoordinates, double radius, double walkFactor) {}

    public static void main(String[] args) throws Exception {
        if(args.length<4)throw new IllegalArgumentException("config.xml network.xml.gz requests.csv output.csv [access radius]");
        Config config=ConfigUtils.createConfig(new SwissRailRaptorConfigGroup());
        ConfigUtils.loadConfig(config,args[0]);
        config.setContext(org.matsim.core.utils.io.IOUtils.resolveFileOrResource(args[0]));
        System.err.println("DIAG: loading network");
        Network network=NetworkUtils.createNetwork();
        new MatsimNetworkReader(network).readFile(args[1]);
        System.err.println("DIAG: network loaded: "+network.getNodes().size()+" nodes; "+network.getLinks().size()+" links");
        Scenario scenario=ScenarioUtils.createScenario(config);
        new TransitScheduleReader(scenario).readURL(ConfigGroup.getInputFileURL(config.getContext(),config.transit().getTransitScheduleFile()));
        if(config.transit().getVehiclesFile()!=null)
            new MatsimVehicleReader(scenario.getTransitVehicles()).readURL(ConfigGroup.getInputFileURL(config.getContext(),config.transit().getVehiclesFile()));
        System.err.println("DIAG: building RAPTOR data");
        SwissRailRaptorData data=SwissRailRaptorData.create(scenario.getTransitSchedule(),scenario.getTransitVehicles(),
            RaptorUtils.createStaticConfig(config),network,new OccupancyData());
        NetworkUtils.getNearestLink(network,network.getNodes().values().iterator().next().getCoord());
        List<Request> requests=new ArrayList<>();
        CsvMapper csv=new CsvMapper();
        try(MappingIterator<Request> input=csv.readerFor(Request.class).with(CsvSchema.emptySchema().withHeader()).readValues(Path.of(args[2]).toFile())) {
            input.forEachRemaining(requests::add);
        }
        double radius=config.transitRouter().getSearchRadius();
        double factor=config.transitRouter().getDirectWalkFactor();
        double larger=args.length>4?Double.parseDouble(args[4]):1500.;
        List<Variant> variants=List.of(
            new Variant("baseline",false,radius,factor),
            new Variant("original_coordinates",true,radius,factor),
            new Variant("access_1500",false,larger,factor),
            new Variant("suppress_direct_walk",false,radius,SUPPRESS_DIRECT_WALK),
            new Variant("original_coordinates_access_1500",true,larger,factor),
            new Variant("access_1500_suppress_direct_walk",false,larger,SUPPRESS_DIRECT_WALK),
            new Variant("original_coordinates_access_1500_suppress_direct_walk",true,larger,SUPPRESS_DIRECT_WALK));
        if(args.length>5 && args[5].equals("followup"))variants=List.of(
            new Variant("baseline",false,radius,factor),
            new Variant("original_coordinates_access_1500_suppress_direct_walk",true,larger,SUPPRESS_DIRECT_WALK),
            new Variant("original_coordinates_access_3000",true,2*larger,factor),
            new Variant("original_coordinates_access_3000_suppress_direct_walk",true,2*larger,SUPPRESS_DIRECT_WALK));
        CsvSchema schema=csv.schemaFor(Result.class).withHeader();
        try(SequenceWriter writer=csv.writer(schema).writeValues(Path.of(args[3]).toFile())) {
            for(Variant variant:variants) {
                RaptorParameters parameters=EqasimRaptorUtils.createParameters(config,
                    ConfigUtils.addOrGetModule(config,EqasimRaptorConfigGroup.class),scenario.getTransitSchedule(),false);
                // Builders lock the shared Config. Person parameters remain independent
                // and are what the stop finder / route selector actually consume.
                parameters.setSearchRadius(variant.radius());
                parameters.setDirectWalkFactor(variant.walkFactor());
                var builder=new SwissRailRaptor.Builder(data,config).with(person->parameters);
                if(!ConfigUtils.addOrGetModule(config,SwissRailRaptorConfigGroup.class).getModeToModeTransferPenaltyParameterSets().isEmpty())
                    builder.with(new ModeSpecificTransferCostCalculator());
                SwissRailRaptor router=builder.build();
                int valid=0;
                for(Request request:requests) {
                    Result result=route(request,variant,router,parameters,data,network,scenario,
                        config.transitRouter().getMaxBeelineWalkConnectionDistance());
                    writer.write(result);
                    if(result.available)++valid;
                }
                writer.flush();
                System.err.println("DIAG: "+variant.name()+": "+valid+" / "+requests.size()+" valid PT results");
            }
        }
    }

    static Result route(Request request,Variant variant,SwissRailRaptor router,RaptorParameters parameters,
                        SwissRailRaptorData data,Network network,Scenario scenario,double transferLimit) {
        Result result=new Result();
        result.trip_id=request.trip_id;result.cohort=request.cohort;result.expected_java=request.expected_java;
        result.variant=variant.name();result.search_radius=variant.radius();result.direct_walk_factor=variant.walkFactor();
        result.transfer_walk_limit=transferLimit;result.departure_time=request.departure_time;
        result.euclidean_distance_km=request.euclidean_distance_km;
        Coord origin=new Coord(request.origin_x,request.origin_y), destination=new Coord(request.destination_x,request.destination_y);
        Link originLink=NetworkUtils.getNearestLink(network,origin), destinationLink=NetworkUtils.getNearestLink(network,destination);
        result.origin_link=originLink.getId().toString();result.destination_link=destinationLink.getId().toString();
        result.same_snapped_link=originLink==destinationLink;
        result.origin_snap_m=CoordUtils.calcEuclideanDistance(origin,originLink.getCoord());
        result.destination_snap_m=CoordUtils.calcEuclideanDistance(destination,destinationLink.getCoord());
        var from=variant.originalCoordinates()?FacilitiesUtils.wrapLinkAndCoord(originLink,origin):new LinkWrapperFacility(originLink);
        var to=variant.originalCoordinates()?FacilitiesUtils.wrapLinkAndCoord(destinationLink,destination):new LinkWrapperFacility(destinationLink);
        result.estimated_direct_walk_min=CoordUtils.calcEuclideanDistance(from.getCoord(),to.getCoord())/parameters.getBeelineWalkSpeed()/60;
        result.origin_stops_400=data.findNearbyStops(from.getCoord().getX(),from.getCoord().getY(),400).size();
        result.destination_stops_400=data.findNearbyStops(to.getCoord().getX(),to.getCoord().getY(),400).size();
        result.origin_stops_1500=data.findNearbyStops(from.getCoord().getX(),from.getCoord().getY(),1500).size();
        result.destination_stops_1500=data.findNearbyStops(to.getCoord().getX(),to.getCoord().getY(),1500).size();
        try {
            List<? extends PlanElement> elements=router.calcRoute(DefaultRoutingRequest.withoutAttributes(from,to,request.departure_time,null));
            if(elements==null||elements.isEmpty()) {result.status="EMPTY";return result;}
            double arrival=request.departure_time,distance=0.;Set<String> modes=new TreeSet<>();
            for(PlanElement element:elements)if(element instanceof Leg leg) {
                if(leg.getTravelTime().isUndefined()||leg.getRoute()==null) {result.status="MISSING_LEG_TIME_OR_ROUTE";return result;}
                double start=leg.getDepartureTime().isDefined()?leg.getDepartureTime().seconds():arrival;
                arrival=Math.max(arrival,start)+leg.getTravelTime().seconds();distance+=leg.getRoute().getDistance();
                if(leg.getRoute() instanceof TransitPassengerRoute transit) {
                    ++result.boardings;
                    double waiting=transit.getBoardingTime().seconds()-leg.getDepartureTime().seconds();
                    result.in_vehicle_time_min+=(transit.getTravelTime().seconds()-waiting)/60;
                    modes.add(scenario.getTransitSchedule().getTransitLines().get(transit.getLineId()).getRoutes().get(transit.getRouteId()).getTransportMode());
                } else if(leg.getMode().contains("walk"))result.walk_time_min+=leg.getTravelTime().seconds()/60;
            }
            result.route_time_min=(arrival-request.departure_time)/60;result.route_distance_km=distance/1000;
            result.pt_modes=String.join(" + ",modes);
            if(!Double.isFinite(arrival)||!Double.isFinite(distance))result.status="NON_FINITE_TIME_OR_DISTANCE";
            else if(result.boardings>0) {result.available=true;result.status="PT";}
            else if(origin.equals(destination)) {result.available=true;result.status="ZERO_DISTANCE_WALK";}
            else result.status="WALK_ONLY";
        } catch(Exception error) {result.status="ERROR";result.error=error.getClass().getSimpleName()+": "+error.getMessage();}
        return result;
    }
}

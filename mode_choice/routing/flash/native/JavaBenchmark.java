import java.nio.file.*;
import java.util.*;
import java.io.*;
import org.matsim.api.core.v01.*;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.core.population.routes.RouteUtils;
import org.matsim.pt.transitSchedule.api.*;
import org.matsim.core.router.LinkWrapperFacility;
import ch.sbb.matsim.routing.pt.raptor.*;

// A matched timetable benchmark against the actual Java RAPTOR shipped in eqasim.
// Matched earliest-arrival costs, not the production Swiss fare/utility model.
public class JavaBenchmark {
 public static void main(String[] args) throws Exception {
  var config=ConfigUtils.createConfig();config.transit().setUseTransit(true);
  config.transitRouter().setMaxBeelineWalkConnectionDistance(0.1);
  config.transitRouter().setSearchRadius(1);config.transitRouter().setExtensionRadius(1);
  var scenario=ScenarioUtils.createScenario(config);var net=scenario.getNetwork();var nf=net.getFactory();
  var schedule=scenario.getTransitSchedule();var sf=schedule.getFactory();
  var stopLinks=new HashMap<String,org.matsim.api.core.v01.network.Link>();
  for(String row:Files.readAllLines(Path.of(args[0]))) {
   if(row.startsWith("stop\t"))continue;var v=row.split("\t");
   var node=nf.createNode(Id.createNodeId(v[0]),new Coord(Double.parseDouble(v[1]),Double.parseDouble(v[2])));net.addNode(node);
   var link=nf.createLink(Id.createLinkId("stop-"+v[0]),node,node);link.setLength(0);link.setFreespeed(10);link.setCapacity(1000);link.setNumberOfLanes(1);net.addLink(link);stopLinks.put(v[0],link);
   var stop=sf.createTransitStopFacility(Id.create(v[0],TransitStopFacility.class),node.getCoord(),false);stop.setLinkId(link.getId());schedule.addStopFacility(stop);
  }
  var trips=new LinkedHashMap<String,List<String[]>>();
  for(String row:Files.readAllLines(Path.of(args[1]))) {if(row.startsWith("trip\t"))continue;var v=row.split("\t");trips.computeIfAbsent(v[0],k->new ArrayList<>()).add(v);}
  int edge=0;
  for(var entry:trips.entrySet()) {
   var events=entry.getValue();double start=Double.parseDouble(events.getFirst()[3]);
   var stops=new ArrayList<TransitRouteStop>();var path=new ArrayList<Id<org.matsim.api.core.v01.network.Link>>();
   org.matsim.api.core.v01.network.Link previous=null;
   for(var event:events) {
    var link=stopLinks.get(event[1]);
    if(previous!=null) {
     var connection=nf.createLink(Id.createLinkId("edge-"+edge++),previous.getToNode(),link.getFromNode());
     var a=previous.getCoord();var b=link.getCoord();connection.setLength(Math.hypot(a.getX()-b.getX(),a.getY()-b.getY()));connection.setFreespeed(10);connection.setCapacity(1000);connection.setNumberOfLanes(1);net.addLink(connection);path.add(connection.getId());
    }
    path.add(link.getId());
    stops.add(sf.createTransitRouteStop(schedule.getFacilities().get(Id.create(event[1],TransitStopFacility.class)),Double.parseDouble(event[2])-start,Double.parseDouble(event[3])-start));previous=link;
   }
   var route=sf.createTransitRoute(Id.create(entry.getKey(),TransitRoute.class),RouteUtils.createLinkNetworkRouteImpl(path.getFirst(),path.subList(1,path.size()-1),path.getLast()),stops,"bus");
   var departure=sf.createDeparture(Id.create("dep",Departure.class),start);route.addDeparture(departure);
   var line=sf.createTransitLine(Id.create(entry.getKey(),TransitLine.class));line.addRoute(route);schedule.addTransitLine(line);
  }
  var transfers=new ArrayList<String[]>();
  for(String row:Files.readAllLines(Path.of(args[4]))) {
   var v=row.split("\\t");transfers.add(v);
   schedule.getMinimalTransferTimes().set(Id.create(v[0],TransitStopFacility.class),Id.create(v[1],TransitStopFacility.class),Double.parseDouble(v[2]));
  }
  var data=SwissRailRaptorData.create(schedule,null,RaptorUtils.createStaticConfig(config),net,null);
  var parameters=RaptorUtils.createParameters(config);
  parameters.setMarginalUtilityOfTravelTime_utl_s("pt",-1);
  parameters.setMarginalUtilityOfTravelTime_utl_s("walk",-1);
  parameters.setDirectWalkFactor(1e9);
  parameters.setMarginalUtilityOfTravelTime_utl_s("bus",-1);
  parameters.setMarginalUtilityOfWaitingPt_utl_s(-1);
  parameters.setTransferPenaltyFixCostPerTransfer(0);
  parameters.setTransferPenaltyPerTravelTimeHour(0);
  parameters.setTransferPenaltyMinimum(0);parameters.setTransferPenaltyMaximum(0);
  RaptorStopFinder finder=(from,to,person,departure,attributes,params,raptorData,direction)->{
   var found=new ArrayList<InitialStop>();
   // The wrapper's link IDs are stop-<native stop index>.
   String id=(direction==RaptorStopFinder.Direction.ACCESS?from:to).getLinkId().toString().substring(5);
   found.add(new InitialStop(schedule.getFacilities().get(Id.create(id,TransitStopFacility.class)),0,0,0,"walk"));
   for(var transfer:transfers) {
    boolean access=direction==RaptorStopFinder.Direction.ACCESS;
    if(transfer[access?0:1].equals(id)) {
     var stop=schedule.getFacilities().get(Id.create(transfer[access?1:0],TransitStopFacility.class));
     double seconds=Double.parseDouble(transfer[2]);
     found.add(new InitialStop(stop,seconds,seconds,0,"walk"));
    }
   }
   return found;
  };
  var router=new SwissRailRaptor.Builder(data,config).with(person->parameters).with(finder).build();
  var rows=Files.readAllLines(Path.of(args[2]));
  // Warm up JIT and timetable structures before recording query-only time.
  for(int i=0;i<100;i++) {var v=rows.get(i%rows.size()).split("\t");router.calcRoute(org.matsim.core.router.DefaultRoutingRequest.withoutAttributes(new LinkWrapperFacility(stopLinks.get(v[1])),new LinkWrapperFacility(stopLinks.get(v[2])),Double.parseDouble(v[3]),null));}
  long begin=System.nanoTime();
  try(var out=new PrintWriter(args[3])) {
   out.println("trip_id,total_time_min,in_vehicle_time_min,transfers,distance_km");
   for(String row:rows) {
    var v=row.split("\t");double time=Double.parseDouble(v[3]);
    var elements=router.calcRoute(org.matsim.core.router.DefaultRoutingRequest.withoutAttributes(new LinkWrapperFacility(stopLinks.get(v[1])),new LinkWrapperFacility(stopLinks.get(v[2])),time,null));
    double best=Double.POSITIVE_INFINITY,iv=0,dist=0;int changes=0;
    double arrival=time,t=0,d=0;int n=0;
    if(elements!=null) for(var element:elements) {
     if(element instanceof org.matsim.api.core.v01.population.Leg leg) {
      arrival=Math.max(arrival,leg.getDepartureTime().seconds()+leg.getTravelTime().seconds());
      if(leg.getRoute() instanceof org.matsim.pt.routes.TransitPassengerRoute route) {
       n++;t+=route.getTravelTime().seconds()-(route.getBoardingTime().seconds()-leg.getDepartureTime().seconds());d+=route.getDistance();
      }
     }
    }
    if(n>0){best=arrival-time;iv=t;dist=d;changes=n-1;}
    out.println(v[0]+","+best/60+","+iv/60+","+changes+","+dist/1000);
   }
  }
  System.err.println("QUERY_SECONDS="+(System.nanoTime()-begin)/1e9);
 }
}

// Adapter for FLASH-TB (MIT). Upstream is compiled separately, never vendored.
#include <fstream>
#include <chrono>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <exception>
#include <omp.h>
#include "DataStructures/GTFS/Data.h"
#include "DataStructures/Intermediate/Data.h"
#include "DataStructures/TripBased/Data.h"
#include "Algorithms/TripBased/Preprocessing/ULTRABuilderTransitive.h"
#include "Algorithms/TripBased/Preprocessing/ARCFlagTBBuilder.h"
#include "Algorithms/TripBased/Query/ARCTransitiveQuery.h"
#include "Algorithms/TripBased/Query/TransitiveQuery.h"
#include "Helpers/String/String.h"

double distance(const RAPTOR::Data& d, Vertex a, Vertex b) {
    const auto x=d.stopData[a].coordinates, y=d.stopData[b].coordinates;
    const double pi=acos(-1), lat1=x.latitude*pi/180, lat2=y.latitude*pi/180;
    const double u=sin((lat2-lat1)/2),v=sin((y.longitude-x.longitude)*pi/360);
    return 6371000*2*asin(std::min(1.0,sqrt(u*u+cos(lat1)*cos(lat2)*v*v)));
}
double legDistance(const RAPTOR::Data& d,const RAPTOR::JourneyLeg& leg) {
    if(!leg.usesRoute) return distance(d,leg.from,leg.to);
    // Recover the boarded segment from its stop-event times, including loops.
    size_t begin=d.firstStopIdOfRoute[leg.routeId], end=d.firstStopIdOfRoute[leg.routeId+1];
    size_t length=end-begin;
    const size_t first=d.firstStopEventOfRoute[leg.routeId];
    const size_t trips=(d.firstStopEventOfRoute[leg.routeId+1]-first)/length;
    for(size_t i=0;i<length;++i) {
        if(d.stopIds[begin+i]!=leg.from)continue;
        size_t low=0,high=trips;
        while(low<high) {size_t mid=(low+high)/2;
            if(d.stopEvents[first+mid*length+i].departureTime<leg.departureTime)low=mid+1;else high=mid;
        }
        for(size_t trip=low;trip<trips && d.stopEvents[first+trip*length+i].departureTime==leg.departureTime;++trip) {
            double total=0;
            for(size_t j=i+1;j<length;++j) {
                total+=distance(d,d.stopIds[begin+j-1],d.stopIds[begin+j]);
                if(d.stopIds[begin+j]==leg.to && d.stopEvents[first+trip*length+j].arrivalTime==leg.arrivalTime)return total;
            }
        }
    }
    return distance(d,leg.from,leg.to);
}
// The timetable is shared read-only; every worker owns its mutable query labels.
// Pools persist across requests. Chunk output is concatenated in input order.
constexpr size_t QUERY_CHUNK_SIZE = 128;
struct Request {size_t id, from, to; int departure;};

template<class Query> class QueryPool {
public:
    QueryPool(TripBased::Data& data, int threads): data(data), queries(threads) {
        std::vector<std::exception_ptr> errors(threads);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int worker=0;worker<threads;++worker) {
            try {queries[worker]=std::make_unique<Query>(data);}
            catch(...) {errors[worker]=std::current_exception();}
        }
        for(const auto& error:errors)if(error)std::rethrow_exception(error);
    }
    void serve(const std::string& input,const std::string& output) {
        std::ifstream in(input);std::ofstream out(output);
        in.imbue(std::locale::classic());out.imbue(std::locale::classic());
        if(!in||!out)throw std::runtime_error("Cannot open routing request or result file");
        std::vector<Request> requests;Request request;
        while(in>>request.id>>request.from>>request.to>>request.departure) {
            if(request.from>=data.numberOfStops()||request.to>=data.numberOfStops())
                throw std::runtime_error("Invalid stop index");
            requests.push_back(request);
        }
        if(!in.eof())throw std::runtime_error("Malformed routing request");
        out<<"query\tjourney\tleg\tfrom\tto\tdeparture\tarrival\tvehicle\tdistance_m\troute_id\tgtfs_route_type\n";
        const size_t chunks=(requests.size()+QUERY_CHUNK_SIZE-1)/QUERY_CHUNK_SIZE;
        std::vector<std::string> results(chunks);
        std::vector<std::exception_ptr> errors(queries.size());
        int activeThreads=1;
        auto begin=std::chrono::steady_clock::now();
        #pragma omp parallel num_threads(queries.size())
        {
            const int worker=omp_get_thread_num();
            auto& query=*queries[worker];
            #pragma omp single
            activeThreads=omp_get_num_threads();
            #pragma omp for schedule(dynamic,1)
            for(size_t chunk=0;chunk<chunks;++chunk) {
                try {
                    std::ostringstream buffer;buffer.imbue(std::locale::classic());buffer<<std::setprecision(15);
                    const size_t end=std::min(requests.size(),(chunk+1)*QUERY_CHUNK_SIZE);
                    for(size_t index=chunk*QUERY_CHUNK_SIZE;index<end;++index) {
                        const auto& r=requests[index];
                        query.run(StopId(r.from),r.departure,StopId(r.to));size_t journeyIndex=0;
                        for(const auto& journey:query.getJourneys()) {size_t legIndex=0;
                            for(const auto& leg:journey) {
                                // Tagged IDs format with separators; export their raw values.
                                buffer<<r.id<<'\t'<<journeyIndex<<'\t'<<legIndex++<<'\t'<<leg.from.value()<<'\t'<<leg.to.value()<<'\t'
                                    <<(leg.departureTime+(leg.usesRoute&&data.raptorData.implicitDepartureBufferTimes?data.raptorData.stopData[leg.from].minTransferTime:0))
                                    <<'\t'<<leg.arrivalTime<<'\t'<<leg.usesRoute<<'\t'<<legDistance(data.raptorData,leg)
                                    <<'\t'<<(leg.usesRoute?int(leg.routeId):-1)<<'\t'<<(leg.usesRoute?data.raptorData.routeData[leg.routeId].type:-1)<<'\n';
                            }
                            ++journeyIndex;
                        }
                    }
                    results[chunk]=buffer.str();
                } catch(...) {errors[worker]=std::current_exception();}
            }
        }
        for(const auto& error:errors)if(error)std::rethrow_exception(error);
        for(const auto& result:results)out<<result;
        out.flush();
        if(!out)throw std::runtime_error("Failed to write routing results");
        std::cerr<<"QUERY_THREADS="<<activeThreads<<"\nQUERY_COUNT="<<requests.size()
                 <<"\nQUERY_SECONDS="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<std::endl;
    }
private:
    TripBased::Data& data;
    std::vector<std::unique_ptr<Query>> queries;
};
int queryThreads(const char* value) {
    const int threads=std::stoi(value);
    if(threads<1)throw std::runtime_error("Query threads must be positive");
    return threads;
}
int main(int argc,char** argv) {try {
    if(argc<3)throw std::runtime_error("prepare GTFS/ YYYYMMDD binary threads cells | query binary requests.tsv output.tsv [tb]");
    std::string command=argv[1];
    if(command=="prepare") {
        if(argc!=7)throw std::runtime_error("Invalid prepare arguments");
        auto gtfs=GTFS::Data::FromGTFS(std::string(argv[2])+"/");
        for(auto& stop:gtfs.stops)stop.name=stop.stopId;
        int day=stringToDay(argv[3]);auto inter=Intermediate::Data::FromGTFS(gtfs,day,day,false,false);
        if(inter.trips.empty())throw std::runtime_error("No active GTFS trips on the selected date");
        inter.makeTransitiveStopGraph();auto raptor=RAPTOR::Data::FromIntermediate(inter);
        if(raptor.numberOfStops()<2||raptor.numberOfTrips()==0)throw std::runtime_error("No active transit service");
        TripBased::Data data(raptor);int threads=std::stoi(argv[5]);
        TripBased::ULTRABuilderTransitive<false> builder(data);builder.computeShortcuts(ThreadPinning(threads,1));
        Graph::move(std::move(builder.getStopEventGraph()),data.stopEventGraph);
        // Geographic longitude stripes: deterministic partition without an external partitioner.
        int cells=std::min(std::stoi(argv[6]),int(data.numberOfStops()));
        if(cells<2)throw std::runtime_error("At least two partition cells required");
        std::vector<size_t> order(data.numberOfStops());std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return data.raptorData.stopData[a].coordinates.longitude<data.raptorData.stopData[b].coordinates.longitude;});
        for(size_t i=0;i<order.size();++i)data.raptorData.stopData[order[i]].partition=i*cells/order.size();
        data.raptorData.numberOfPartitions=cells;
        TripBased::ARCFlagTBBuilder flags(data,threads,1);flags.computeARCFlags(false);data.serialize(argv[4]);
        std::ofstream transfers(std::string(argv[4])+".transfers.tsv");
        transfers.imbue(std::locale::classic());
        for(auto from:data.raptorData.transferGraph.vertices())for(auto edge:data.raptorData.transferGraph.edgesFrom(from))
            transfers<<from.value()<<'\t'<<data.raptorData.transferGraph.get(ToVertex,edge).value()<<'\t'<<data.raptorData.transferGraph.get(TravelTime,edge)<<'\n';
        std::ofstream stops(std::string(argv[4])+".stops.tsv");stops.imbue(std::locale::classic());stops<<"stop\tlat\tlon\tgtfs_stop_id\n"<<std::setprecision(15);
        for(auto stop:data.stops()) {const auto& c=data.raptorData.stopData[stop].coordinates;stops<<stop.value()<<'\t'<<c.latitude<<'\t'<<c.longitude<<'\t'<<data.raptorData.stopData[stop].name<<'\n';}
    } else if(command=="query") {
        if(argc<5)throw std::runtime_error("Invalid query arguments");TripBased::Data data(argv[2]);
        const int threads=argc>6?queryThreads(argv[6]):1;omp_set_dynamic(0);
        if(argc>5 && std::string(argv[5])=="tb") {QueryPool<TripBased::TransitiveQuery<>> pool(data,threads);pool.serve(argv[3],argv[4]);}
        else {QueryPool<TripBased::ARCTransitiveQuery<>> pool(data,threads);pool.serve(argv[3],argv[4]);}
    } else if(command=="serve") {
        TripBased::Data data(argv[2]);const int threads=argc>3?queryThreads(argv[3]):1;omp_set_dynamic(0);
        QueryPool<TripBased::ARCTransitiveQuery<>> pool(data,threads);
        std::string input,output;
        while(std::cin>>std::quoted(input)>>std::quoted(output)) {
            pool.serve(input,output);std::cout<<"FLASH_DONE"<<std::endl;
        }
    } else throw std::runtime_error("Unknown command");
    return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}}

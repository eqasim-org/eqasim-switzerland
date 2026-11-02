// One-pass, bounded-memory XML extraction. XML entities are decoded by Expat.
// Driver/link pairs are sorted in fixed-size runs rather than kept in RAM.
#include <expat.h>
#include <zlib.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
using Id = uint32_t;
constexpr Id UNKNOWN = numeric_limits<Id>::max();
const array<string, 6> STATIONS = {"2391498", "2927869", "2391502", "1289297", "1289295", "106813"};

string attr(const char** values, const char* name, const string& fallback="") {
    for (int i=0; values[i]; i+=2) if (!strcmp(values[i], name)) return values[i+1];
    return fallback;
}
string clean(string value) {
    for (char& c : value) if (c=='\t' || c=='\n' || c=='\r') c=' ';
    return value;
}
double seconds(const string& text) {
    if (text.empty() || text=="undefined") return -1;
    if (text.find(':')==string::npos) return stod(text);
    int h=0,m=0; double s=0; char c;
    istringstream in(text); in >> h >> c >> m >> c >> s;
    return h*3600.+m*60.+s;
}
struct Parser {
    virtual void start(const char*, const char**) {}
    virtual void end(const char*) {}
    virtual void text(const char*, int) {}
    virtual bool finished() const {return false;}
    virtual ~Parser() = default;
};
void parse(const string& path, Parser& reader) {
    gzFile stream=gzopen(path.c_str(),"rb");
    if (!stream) throw runtime_error("Cannot open "+path);
    gzbuffer(stream, 1<<20);
    XML_Parser xml=XML_ParserCreate(nullptr);
    XML_SetUserData(xml,&reader);
    XML_SetElementHandler(xml, [](void* p,const char* t,const char** a){static_cast<Parser*>(p)->start(t,a);},
                         [](void* p,const char* t){static_cast<Parser*>(p)->end(t);});
    XML_SetCharacterDataHandler(xml,[](void* p,const char* s,int n){static_cast<Parser*>(p)->text(s,n);});
    // External DTDs are unnecessary for MATSim data and are never fetched.
    vector<char> buffer(1<<20); uint64_t total=0; int n=0;
    while ((n=gzread(stream,buffer.data(),buffer.size()))>0) {
        if (XML_Parse(xml,buffer.data(),n,false)==XML_STATUS_ERROR)
            throw runtime_error(string("Invalid XML: ")+XML_ErrorString(XML_GetErrorCode(xml)));
        total+=n;
        if(reader.finished()) break;
    }
    int error=0; const char* message=gzerror(stream,&error);
    if (n<0 || (error!=Z_OK && error!=Z_STREAM_END)) throw runtime_error(message);
    if (!reader.finished() && XML_Parse(xml,nullptr,0,true)==XML_STATUS_ERROR) throw runtime_error("Incomplete XML");
    XML_ParserFree(xml); gzclose(stream);
    cerr << "Finished XML: " << total << " uncompressed bytes\n";
}

struct Activity {string type, x, y, municipality;};
struct Leg {string mode, vehicle, route; double departure=-1; Id sequence=0; Activity origin,destination;};
struct Plans : Parser {
    ofstream persons, trips, routes;
    map<string,string> attributes;
    vector<Leg> legs;
    Activity current_activity, origin;
    vector<size_t> pending;
    string person, attribute_name, attribute_text, attribute_target, route_text;
    Id index=0;
    int depth=0, person_depth=0, plan_depth=0, activity_depth=0, leg_depth=0, attribute_depth=0;
    bool selected=false, route=false, network_route=false, work=false, education=false;
    Leg leg;
    Plans(const string& out):persons(out+"/persons.tsv"),trips(out+"/trips.tsv"),routes(out+"/planned_corridor.tsv") {
        persons << "idx\tperson\tage\tsex\tcanton\tsubpopulation\tis_external_fr\tis_crossborder\tis_freight\temployed\tstudent\thome_x\thome_y\thas_work_activity\thas_education_activity\n";
        trips << "person_idx\tsequence\tmode\tvehicle\tdeparture\torigin_purpose\torigin_x\torigin_y\torigin_municipality\tdestination_purpose\tdestination_x\tdestination_y\tdestination_municipality\tstation_mask\n";
        routes << "person_idx\tsequence\tlinks\n";
    }
    string value(initializer_list<const char*> keys, const string& fallback="") {
        for (const char* k:keys) if(attributes.count(k)) return clean(attributes[k]);
        return fallback;
    }
    void start(const char* tag,const char** a) override {
        ++depth;
        if (!strcmp(tag,"person")) {
            person=attr(a,"id"); person_depth=depth; attributes.clear(); legs.clear();
            pending.clear(); origin={}; work=education=false;
        } else if (!person.empty() && !strcmp(tag,"plan")) {
            selected=attr(a,"selected")=="yes"; plan_depth=depth;
        } else if (selected && !strcmp(tag,"activity")) {
            activity_depth=depth; current_activity={attr(a,"type"),attr(a,"x"),attr(a,"y"),""};
        } else if (selected && !strcmp(tag,"leg")) {
            leg_depth=depth; leg={}; leg.mode=attr(a,"mode"); leg.departure=seconds(attr(a,"dep_time"));
            network_route=false;
        } else if (selected && leg_depth && !strcmp(tag,"route")) {
            route=true; route_text.clear(); network_route=attr(a,"type")=="links";
            leg.vehicle=attr(a,"vehicleRefId");
        } else if (!person.empty() && !strcmp(tag,"attribute")) {
            attribute_depth=depth; attribute_name=attr(a,"name"); attribute_text.clear();
            attribute_target=(depth==person_depth+2 ? "person" : activity_depth ? "activity" : "");
        }
    }
    void text(const char* data,int size) override {
        if (attribute_depth) attribute_text.append(data,size);
        else if (route && network_route) route_text.append(data,size);
    }
    void end(const char* tag) override {
        if (!strcmp(tag,"attribute")) {
            if (attribute_target=="person") attributes[attribute_name]=attribute_text;
            if (attribute_target=="activity" && attribute_name=="municipalityId") current_activity.municipality=attribute_text;
            attribute_depth=0; attribute_target.clear();
        } else if (!strcmp(tag,"route")) {route=false; if(network_route) leg.route=std::move(route_text);
        } else if (selected && !strcmp(tag,"leg")) {
            if (network_route) {leg.sequence=legs.size();leg.origin=origin;legs.push_back(std::move(leg));pending.push_back(legs.size()-1);}
            leg_depth=0;
        } else if (selected && !strcmp(tag,"activity")) {
            // Ignore access/egress interaction activities when describing trip purpose.
            if (current_activity.type.find("interaction")==string::npos) {
                for (size_t k:pending) legs[k].destination=current_activity;
                pending.clear(); origin=current_activity;
                work|=origin.type=="work"; education|=origin.type.rfind("education",0)==0;
                if (origin.type=="home" && !attributes.count("home_x") && !attributes.count("home_coordinate_x")) {
                    attributes["home_x"]=origin.x;attributes["home_y"]=origin.y;
                }
            }
            activity_depth=0;
        } else if (!strcmp(tag,"plan")) {selected=false;plan_depth=0;
        } else if (!strcmp(tag,"person")) {
            persons << index << '\t' << clean(person) << '\t' << value({"age"}) << '\t' << value({"sex"}) << '\t'
                    << value({"cantonName","cantonId"}) << '\t' << value({"subpopulation"},"ordinary") << '\t'
                    << value({"isExternalFR"},"false") << '\t' << value({"isCrossBorder"},"false") << '\t'
                    << value({"isFreight"},"false") << '\t' << value({"employed","is_employed"}) << '\t'
                    << value({"is_student","isStudent"}) << '\t' << value({"home_x","home_coordinate_x"}) << '\t'
                    << value({"home_y","home_coordinate_y"}) << '\t' << work << '\t' << education << '\n';
            for (const auto& l:legs) {
                unsigned mask=0; istringstream tokens(l.route);string id;
                while(tokens>>id) for(unsigned k=0;k<STATIONS.size();++k) if(id==STATIONS[k])mask|=1u<<k;
                trips << index << '\t' << l.sequence << '\t' << clean(l.mode) << '\t' << clean(l.vehicle) << '\t' << l.departure << '\t'
                      << clean(l.origin.type) << '\t' << l.origin.x << '\t' << l.origin.y << '\t' << l.origin.municipality << '\t'
                      << clean(l.destination.type) << '\t' << l.destination.x << '\t' << l.destination.y << '\t'
                      << l.destination.municipality << '\t' << mask << '\n';
                if(mask) routes << index << '\t' << l.sequence << '\t' << clean(l.route) << '\n';
            }
            if (++index%100000==0) cerr << "Plans: " << index << " persons\n";
            person.clear(); person_depth=0;
        }
        --depth;
    }
};

struct Stats {
    uint32_t entries=0, initial=0, car=0, freight=0, transit=0, bike=0, other=0, stuck=0;
    double travel_time=0;
    uint32_t completed_links=0, padding=0;
    uint32_t hourly[32]={};
};
static_assert(sizeof(Stats)==176,"Python binary layout must match");
struct Journey {
    Id person=UNKNOWN,sequence=0,last=UNKNOWN,start_link=UNKNOWN;
    string vehicle, mode;
    double start=0,entered=0;
    uint8_t category=4,mask=0;
    bool last_initial=false;
    vector<Id> route;
    vector<pair<Id,double>> stations;
};
struct Events : Parser {
    unordered_map<string,Id> links, people;
    unordered_set<Id> freight_people;
    unordered_map<string,Journey> active;
    unordered_map<Id,Id> sequences;
    vector<Stats> stats;
    vector<uint64_t> pairs;
    vector<string> runs;
    unordered_map<Id,unsigned> station_links;
    ofstream journeys;
    string out;
    uint64_t count=0, unknown=0, unstaged=0, entries=0;
    double max_time=0, cutoff=86400;
    bool window_finished=false;
    bool finished() const override {return window_finished;}
    Events(const string& directory, double end_time):journeys(directory+"/corridor_journeys.tsv"),out(directory),cutoff(end_time) {
        ifstream ids(out+"/link_ids.txt"); string id;
        while(getline(ids,id)) links.emplace(id,links.size());
        stats.resize(links.size());
        for(unsigned k=0;k<STATIONS.size();++k) {
            auto it=links.find(STATIONS[k]);if(it==links.end())throw runtime_error("Station link not in network");
            station_links[it->second]=k;
        }
        ifstream persons(out+"/persons.tsv");getline(persons,id);
        while(getline(persons,id)) {
            vector<string> fields;istringstream row(id);string field;while(getline(row,field,'\t'))fields.push_back(field);
            Id person=stoul(fields[0]);people.emplace(fields[1],person);
            if(fields.size()>8 && fields[8]=="true")freight_people.insert(person);
        }
        journeys << "person_idx\tsequence\tvehicle\tmode\tstart\tend\tstart_link\tend_link\tstatus\tstation_mask\tstation_times\tlinks\n";
        pairs.reserve(8*1024*1024);
        cerr << "Event index: " << links.size() << " links, " << people.size() << " persons\n";
    }
    void flush() {
        if(pairs.empty())return;
        sort(pairs.begin(),pairs.end()); pairs.erase(unique(pairs.begin(),pairs.end()),pairs.end());
        string path=out+"/members_run_"+to_string(runs.size())+".bin";
        ofstream f(path,ios::binary);f.write(reinterpret_cast<const char*>(pairs.data()),pairs.size()*8);
        runs.push_back(path);pairs.clear();
        cerr << "Membership run " << runs.size() << "; events " << count << "; simulation hour " << max_time/3600 << '\n';
    }
    Id link(const string& id) {auto i=links.find(id);return i==links.end()?UNKNOWN:i->second;}
    uint8_t category(const string& vehicle,const string& mode,const string& person) {
        if (person.rfind("pt_",0)==0 || vehicle.rfind("veh_",0)==0) return 2;
        if (mode=="bike")return 3;
        if (mode=="car")return 0;
        if (mode=="truck" || mode=="truck_8" || mode=="truck_12" || mode=="truck_18" || mode=="truck_26" || mode=="truck_40")return 1;
        return 4;
    }
    void visit(Journey& j,Id k,double time,bool initial=false) {
        if(k==UNKNOWN)return;
        auto& s=stats[k];
        if(initial)++s.initial;
        else {
            ++s.entries;++entries;
            if(j.category==0)++s.car;else if(j.category==1)++s.freight;else if(j.category==2)++s.transit;else if(j.category==3)++s.bike;else ++s.other;
            ++s.hourly[min(31,max(0,int(time/3600)))];
        }
        // Unique users are based on actual LinkEnter events, matching passage totals.
        if(!initial && j.person!=UNKNOWN && j.category!=2 && j.category!=3) {
            pairs.push_back((uint64_t(k)<<32)|j.person);
            if(pairs.size()>=8*1024*1024)flush();
        }
        j.route.push_back(k); j.last=k;j.entered=time;j.last_initial=initial;
        auto station=station_links.find(k);
        if(!initial && station!=station_links.end()) {j.mask|=1u<<station->second;j.stations.emplace_back(k,time);}
    }
    void finish(Journey& j,double time,const string& status) {
        if(!j.mask)return;
        journeys << j.person << '\t' << j.sequence << '\t' << j.vehicle << '\t' << j.mode << '\t' << j.start << '\t' << time << '\t'
                 << j.start_link << '\t' << j.last << '\t' << status << '\t' << unsigned(j.mask) << '\t';
        for(size_t i=0;i<j.stations.size();++i){if(i)journeys<<',';journeys<<j.stations[i].first<<':'<<j.stations[i].second;}
        journeys << '\t';for(size_t i=0;i<j.route.size();++i){if(i)journeys<<' ';journeys<<j.route[i];}journeys << '\n';
    }
    void start(const char* tag,const char** a) override {
        if(strcmp(tag,"event") || window_finished)return;
        double time=stod(attr(a,"time","0"));
        // Standard MATSim event files are chronological. Agent profiles use the
        // half-open calendar day; the separate reference CSV retains its own
        // inclusive-midnight convention for the traffic-volume display.
        if(cutoff>0 && time>=cutoff) {window_finished=true;max_time=cutoff;return;}
        ++count; string type=attr(a,"type");
        if(type!="entered link" && type!="left link" && type!="vehicle enters traffic" && type!="vehicle leaves traffic" && type!="stuckAndAbort" && type!="stuck")return;
        max_time=max(max_time,time);
        string vehicle=attr(a,"vehicle");
        if(type=="vehicle enters traffic") {
            string person=attr(a,"person"),mode=attr(a,"networkMode");
            auto p=people.find(person); Journey j;j.person=p==people.end()?UNKNOWN:p->second;
            j.vehicle=vehicle;j.mode=mode;j.start=time;j.start_link=link(attr(a,"link"));j.category=category(vehicle,mode,person);
            if(freight_people.count(j.person))j.category=1;
            j.sequence=j.person==UNKNOWN?0:sequences[j.person]++;
            if(active.count(vehicle)) {finish(active[vehicle],time,"restarted");++unstaged;}
            active[vehicle]=std::move(j);visit(active[vehicle],active[vehicle].start_link,time,true);
        } else if(type=="entered link") {
            auto it=active.find(vehicle);
            if(it==active.end()) {Journey j;j.vehicle=vehicle;j.category=category(vehicle,"","");active[vehicle]=std::move(j);it=active.find(vehicle);++unknown;}
            visit(it->second,link(attr(a,"link")),time);
        } else if(type=="left link") {
            auto it=active.find(vehicle);Id k=link(attr(a,"link"));
            if(it!=active.end() && k!=UNKNOWN && it->second.last==k && !it->second.last_initial && it->second.category!=3 && time>=it->second.entered) {
                stats[k].travel_time+=time-it->second.entered;++stats[k].completed_links;
            }
        } else if(type=="vehicle leaves traffic") {
            auto it=active.find(vehicle);if(it!=active.end()) {finish(it->second,time,"completed");active.erase(it);}
        } else {
            Id k=link(attr(a,"link"));if(k!=UNKNOWN)++stats[k].stuck;
            string person=attr(a,"person");
            for(const string& candidate:{vehicle,person+":car",person+":truck"}) {
                auto it=active.find(candidate);if(it!=active.end()){finish(it->second,time,"stuck");active.erase(it);break;}
            }
        }
        if(count%5000000==0)cerr<<"Events: "<<count<<"; simulation hour "<<time/3600<<"; active vehicles "<<active.size()<<'\n';
    }
    void complete() {
        for(auto& item:active)finish(item.second,max_time,"unfinished");
        flush();
        ofstream binary(out+"/link_stats.bin",ios::binary);binary.write(reinterpret_cast<const char*>(stats.data()),stats.size()*sizeof(Stats));binary.close();
        // Merge bounded sorted runs, emitting each (link, driver) exactly once.
        struct Item {uint64_t value;size_t run;bool operator>(const Item& rhs)const{return value>rhs.value;}};
        priority_queue<Item,vector<Item>,greater<Item>> queue;
        vector<ifstream> streams;streams.reserve(runs.size());
        for(size_t i=0;i<runs.size();++i){streams.emplace_back(runs[i],ios::binary);uint64_t v;if(streams.back().read(reinterpret_cast<char*>(&v),8))queue.push({v,i});}
        vector<uint64_t> offsets(stats.size()+1,0);ofstream members(out+"/link_members.bin",ios::binary);
        uint64_t previous=~uint64_t(0),unique_pairs=0;
        while(!queue.empty()) {
            auto item=queue.top();queue.pop();
            if(item.value!=previous) {
                Id k=item.value>>32,p=Id(item.value);members.write(reinterpret_cast<char*>(&p),4);++offsets[k+1];++unique_pairs;previous=item.value;
            }
            uint64_t next;if(streams[item.run].read(reinterpret_cast<char*>(&next),8))queue.push({next,item.run});
        }
        for(size_t i=1;i<offsets.size();++i)offsets[i]+=offsets[i-1];
        ofstream o(out+"/link_member_offsets.bin",ios::binary);o.write(reinterpret_cast<const char*>(offsets.data()),offsets.size()*8);
        for(auto& s:streams)s.close();for(const auto& path:runs)filesystem::remove(path);
        ofstream meta(out+"/events_summary.json");
        meta<<"{\"events\":"<<count<<",\"link_entries\":"<<entries<<",\"unique_link_driver_pairs\":"<<unique_pairs
            <<",\"entries_without_vehicle_start\":"<<unknown<<",\"restarted_journeys\":"<<unstaged
            <<",\"active_vehicles_at_end\":"<<active.size()<<",\"last_event_time\":"<<max_time<<"}\n";
        cerr<<"Completed "<<count<<" events; "<<unique_pairs<<" exact unique link/driver pairs\n";
    }
};

#include "purposes.h"

int main(int argc,char** argv) {
    try {
        if(argc!=4 && argc!=5)throw runtime_error("Usage: extract plans|events INPUT OUTPUT_DIRECTORY [EVENT_END_TIME=86400]");
        filesystem::create_directories(argv[3]);string mode=argv[1];
        if(mode=="plans") {Plans p(argv[3]);parse(argv[2],p);}
        else if(mode=="events") {Events e(argv[3],argc==5?stod(argv[4]):86400);parse(argv[2],e);e.complete();}
        else if(mode=="purposes") {PurposeEvents e(argv[3],argc==5?stod(argv[4]):86400);parse(argv[2],e);e.complete();}
        else throw runtime_error("Unknown extraction mode");
    } catch(const exception& e) {cerr<<e.what()<<'\n';return 1;}
    return 0;
}

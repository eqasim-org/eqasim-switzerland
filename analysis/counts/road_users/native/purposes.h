// Stream actual road passages and join only their corresponding selected-plan
// network leg. Counts are sparse: no journey routes or event XML trees are held.
struct PurposeEvents : Parser {
    struct Plan {string vehicle, mode; uint16_t origin=0, destination=0;};
    struct Vehicle {uint8_t group=6; uint16_t origin=0, destination=0; bool excluded=false;};
    unordered_map<string,Id> links, people;
    vector<uint8_t> groups;
    vector<vector<Plan>> plans;
    unordered_map<Id,Id> sequences;
    unordered_map<string,Vehicle> active;
    unordered_map<uint64_t,uint64_t> counts;
    vector<string> labels={"Unknown"};
    unordered_map<string,uint16_t> label_ids={{"Unknown",0},{"",0}};
    string out;
    bool done=false;
    uint64_t events=0, passages=0, unmatched=0;
    double cutoff;
    bool finished() const override {return done;}

    static vector<string> fields(const string& line) {
        vector<string> result; size_t start=0, end;
        while((end=line.find('\t',start))!=string::npos) {result.push_back(line.substr(start,end-start));start=end+1;}
        result.push_back(line.substr(start));return result;
    }
    uint16_t purpose(const string& text) {
        auto found=label_ids.find(text);if(found!=label_ids.end())return found->second;
        if(labels.size()>=4096)throw runtime_error("Too many activity purposes");
        uint16_t id=labels.size();labels.push_back(text);label_ids[text]=id;return id;
    }
    PurposeEvents(const string& directory,double end):out(directory),cutoff(end) {
        string line;
        ifstream ids(out+"/link_ids.txt");while(getline(ids,line))links.emplace(line,links.size());
        ifstream persons(out+"/persons.tsv");getline(persons,line);
        while(getline(persons,line)) {auto f=fields(line);people.emplace(f.at(1),stoul(f.at(0)));}
        ifstream g(out+"/person_groups.bin",ios::binary);
        groups=vector<uint8_t>(istreambuf_iterator<char>(g),{});
        if(groups.size()!=people.size())throw runtime_error("Person group index does not match exported people");
        plans.resize(people.size());
        ifstream trips(out+"/trips.tsv");getline(trips,line);
        while(getline(trips,line)) {
            auto f=fields(line);Id p=stoul(f.at(0)),s=stoul(f.at(1));
            if(p>=plans.size() || s!=plans[p].size())throw runtime_error("Non-contiguous selected-plan leg index");
            plans[p].push_back({f.at(3),f.at(2),purpose(f.at(5)),purpose(f.at(9))});
        }
        cerr<<"Purpose index: "<<people.size()<<" people, "<<labels.size()<<" activity types\n";
    }
    void start(const char* tag,const char** a) override {
        if(strcmp(tag,"event") || done)return;
        double time=stod(attr(a,"time","0"));if(time>=cutoff){done=true;return;}
        ++events;string type=attr(a,"type"),vehicle=attr(a,"vehicle");
        if(type=="vehicle enters traffic") {
            string person=attr(a,"person"),mode=attr(a,"networkMode");Vehicle v;
            v.excluded=person.rfind("pt_",0)==0 || vehicle.rfind("veh_",0)==0 || mode=="bike";
            auto found=people.find(person);
            if(found!=people.end()) {
                Id p=found->second,s=sequences[p]++;v.group=groups.at(p);
                // A mismatch stays Unknown. Do not borrow another trip's purpose
                // or all activities from a person's daily plan.
                if(s<plans[p].size() && plans[p][s].vehicle==vehicle && plans[p][s].mode==mode) {
                    v.origin=plans[p][s].origin;v.destination=plans[p][s].destination;
                }
            }
            active[vehicle]=v;
        } else if(type=="entered link" && time>=0) {
            auto link=links.find(attr(a,"link"));if(link==links.end())return;
            auto found=active.find(vehicle);Vehicle v;
            if(found!=active.end())v=found->second;
            else v.excluded=vehicle.rfind("veh_",0)==0;
            if(v.excluded)return;
            uint64_t key=(uint64_t(link->second)<<32)|(uint64_t(v.group)<<24)|(uint64_t(v.origin)<<12)|v.destination;
            ++counts[key];++passages;if(!v.origin || !v.destination)++unmatched;
        } else if(type=="vehicle leaves traffic")active.erase(vehicle);
        else if(type=="stuckAndAbort" || type=="stuck") {
            string person=attr(a,"person");
            for(const string& candidate:{vehicle,person+":car",person+":truck"})if(active.erase(candidate))break;
        }
        if(events%10000000==0)cerr<<"Purpose events: "<<events<<"; simulation hour "<<time/3600<<'\n';
    }
    void complete() {
        ofstream result(out+"/purpose_counts.tsv");result<<"link_idx\tgroup_id\torigin\tdestination\tpassages\n";
        for(const auto& item:counts)result<<(item.first>>32)<<'\t'<<((item.first>>24)&255)<<'\t'
            <<((item.first>>12)&4095)<<'\t'<<(item.first&4095)<<'\t'<<item.second<<'\n';
        ofstream names(out+"/purpose_labels.tsv");names<<"idx\tlabel\n";
        for(size_t i=0;i<labels.size();++i)names<<i<<'\t'<<labels[i]<<'\n';
        cerr<<"Purpose passages: "<<passages<<"; unknown origin/destination: "<<unmatched<<"; sparse cells: "<<counts.size()<<'\n';
    }
};

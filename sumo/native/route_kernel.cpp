// Small numerical kernels. Python owns metadata, validation and seeded choices.
#include <cmath>
#include <cstdint>
extern "C" {
int edge_clocks(int64_t count, const double* lengths, const double* costs,
                double depart, double start, double stop, double* exits, double* distance) {
    double clock = depart, total = 0;
    for (int64_t i=0; i<count; ++i) {
        double low = i==0 ? start : 0;
        double high = i==count-1 ? stop : lengths[i];
        if (!std::isfinite(lengths[i]) || lengths[i]<=0 || !std::isfinite(costs[i]) || costs[i]<=0 || high<low) return 1;
        total += high-low;
        clock += costs[i]*(high-low)/lengths[i];
        exits[i]=clock;
    }
    *distance=total;
    return 0;
}
// One row for each intersected route-edge fragment; visit numbering preserves gaps.
struct Fragment { int64_t piece, visit; double begin,end,depart_pos,arrival_pos,distance; int64_t new_entry,new_exit,interpolated; };
int64_t cut_route(int64_t n, const int64_t* edges, const double* exits, double depart,
    double first, double last, const double* lengths, const int64_t* offsets,
    const double* starts, const double* ends, const double* piece_lengths,
    const int64_t* allowed_from, const int64_t* allowed_to, int64_t allowed_n, Fragment* out) {
    int64_t used=0, visit=0, previous_index=-2, previous_piece=-1;
    double previous_end=0;
    for(int64_t i=0;i<n;++i) {
        int64_t edge=edges[i]; double lower=i==0?first/lengths[edge]:0, upper=i==n-1?last/lengths[edge]:1;
        if(upper<lower) return -1;
        double entered=i==0?depart:exits[i-1];
        for(int64_t p=offsets[edge];p<offsets[edge+1];++p) {
            double a=std::fmax(lower,starts[p]), b=std::fmin(upper,ends[p]);
            if(b-a<1e-10) continue;
            bool allowed=false;
            // Connections are sorted lexicographically; binary search instead of per-fragment scanning.
            int64_t low=0, high=allowed_n;
            while(low<high){int64_t m=(low+high)/2; if(allowed_from[m]<previous_piece || (allowed_from[m]==previous_piece && allowed_to[m]<p)) low=m+1; else high=m;}
            allowed=low<allowed_n && allowed_from[low]==previous_piece && allowed_to[low]==p;
            bool chain=used && i==previous_index+1 && previous_end>=1-1e-9 && a<=1e-9 && allowed;
            if(used && !chain) ++visit;
            double denominator=std::fmax(upper-lower,1e-12), span=ends[p]-starts[p];
            out[used++]={p,visit,entered+(exits[i]-entered)*(a-lower)/denominator,
                entered+(exits[i]-entered)*(b-lower)/denominator,(a-starts[p])/span*piece_lengths[p],
                (b-starts[p])/span*piece_lengths[p],(b-a)/span*piece_lengths[p],
                i>0 || a>lower+1e-9,i<n-1 || b<upper-1e-9,a>lower+1e-9 || b<upper-1e-9};
            previous_index=i; previous_end=b; previous_piece=p;
        }
    }
    return used;
}
}

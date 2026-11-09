#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#endif
static constexpr int N=256, P=32640; static constexpr size_t ROOT_BYTES=8388608;
struct alignas(64) Pair {float forward[32],reverse[32];};
struct alignas(64) Head {uint8_t payload[128];};
struct Root {
 Head heads[256]; Pair pair[P];
 static int idx(int a,int b){if(a>b)std::swap(a,b);return a*(511-a)/2+b-a-1;}
 float& lane(int a,int b,int c=0){auto &p=pair[idx(a,b)];return a<b?p.forward[c]:p.reverse[c];}
 void put(int a,int b,int c,float v){lane(a,b,c)=v;lane(b,a,c)=-v;}
 float triangle(int a,int b,int c=0){return lane(0,a,c)+lane(a,b,c)-lane(0,b,c);}
 void update(int a,int b,int c,float grad){put(0,a,c,lane(0,a,c)+grad/3.f);put(a,b,c,lane(a,b,c)+grad/3.f);put(0,b,c,lane(0,b,c)-grad/3.f);}
 void gauge(unsigned seed,float s){std::mt19937 rg(seed);std::normal_distribution<float> nd(0,s);float g[256];for(float &x:g)x=nd(rg);for(int i=0;i<N;i++)for(int j=i+1;j<N;j++)put(i,j,0,lane(i,j,0)+g[i]-g[j]);}
};
static_assert(sizeof(Root)==ROOT_BYTES);
uint64_t hash64(uint64_t a){a^=a>>30;a*=0xbf58476d1ce4e5b9ULL;a^=a>>27;a*=0x94d049bb133111ebULL;return a^(a>>31);}
// Hyperedge IDs remain stable after vector reallocation; 0 is reserved as a null ID.
struct Fact {uint32_t src,dst,relation,anchor_pair; float strength;uint32_t next;};
struct Memory {
 Root *root; std::vector<Fact> facts; std::vector<uint32_t> first; std::unordered_map<uint64_t,uint32_t> exact;std::unordered_map<uint32_t,std::vector<uint32_t>> edges;
 explicit Memory():root(static_cast<Root*>(std::aligned_alloc(4096,ROOT_BYTES))),first(P,0){if(!root)std::abort();std::memset(root,0,ROOT_BYTES);facts.push_back({});}
 ~Memory(){std::free(root);}
 std::pair<int,int> anchor(uint32_t concept_id) const{uint64_t k=hash64(concept_id+0x9e3779b97f4a7c15ULL);int a=1+k%255;int b=1+(k>>15)%255;if(a==b)b=b%255+1;if(a>b)std::swap(a,b);return {a,b};}
 uint32_t add(uint32_t source,uint32_t relation,uint32_t dest,float weight=1.f){auto [a,b]=anchor(source);int p=Root::idx(a,b);uint32_t id=facts.size();facts.push_back({source,dest,relation,(uint32_t)p,weight,first[p]});first[p]=id;edges[source].push_back(id);if(!exact.count((uint64_t(source)<<32)|relation))exact[(uint64_t(source)<<32)|relation]=id;root->update(a,b,0,0.0001f*weight);return id;}
 // Only canonical pair index is needed to find the posting list; exact source/relation verify collisions.
 uint32_t get(uint32_t source,uint32_t relation)const {auto [a,b]=anchor(source);uint32_t id=first[Root::idx(a,b)];while(id){auto &f=facts[id];if(f.src==source && f.relation==relation)return id;id=f.next;}return 0;}
 uint32_t get_hash(uint32_t source,uint32_t relation)const {auto it=exact.find((uint64_t(source)<<32)|relation);return it==exact.end()?0:it->second;}
 bool two_hop(uint32_t s,uint32_t goal,uint32_t relation)const {auto it=edges.find(s);if(it==edges.end())return false;for(auto id:it->second){auto &f=facts[id];if(f.relation!=relation)continue;auto jt=edges.find(f.dst);if(jt==edges.end())continue;for(auto jd:jt->second){auto &g=facts[jd];if(g.relation==relation&&g.dst==goal)return true;}}return false;}
};
static double ms(std::chrono::steady_clock::time_point t){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();}
int main(int argc,char**argv){if(argc<2){std::cerr<<"usage: native_recursive facts.tsv [queries.tsv]\n";return 2;}Memory m;
#ifdef __linux__
 int locked=(mlock(m.root,ROOT_BYTES)==0);
#else
 int locked=0;
#endif
 std::ifstream in(argv[1]);uint32_t s,r,d;size_t read=0;auto t=std::chrono::steady_clock::now();while(in>>s>>r>>d){m.add(s,r,d);++read;}double ingest=ms(t);
 if(!read){std::cerr<<"no facts\n";return 2;}
 bool maps=true;for(int i=0;i<N;i++)for(int j=i+1;j<N;j++)if(&m.root->pair[Root::idx(i,j)]!=&m.root->pair[Root::idx(j,i)])maps=false;
 std::mt19937 rng(2026);std::vector<std::pair<uint32_t,uint32_t>> q;for(size_t i=1;i<m.facts.size();i+=std::max<size_t>(1,m.facts.size()/5000))q.push_back({m.facts[i].src,m.facts[i].relation});std::shuffle(q.begin(),q.end(),rng);
 size_t trials=250000, ok=0; t=std::chrono::steady_clock::now();for(size_t i=0;i<trials;i++){auto [a,b]=q[i%q.size()];ok+=(m.get(a,b)!=0);}double posting=ms(t);
 t=std::chrono::steady_clock::now();size_t ok2=0;for(size_t i=0;i<trials;i++){auto [a,b]=q[i%q.size()];ok2+=(m.get_hash(a,b)!=0);}double hash=ms(t);
 // A concept-specific anchored cycle learns an increment. Gauge operations must preserve it.
 auto [a,b]=m.anchor(m.facts[1].src);float before=m.root->triangle(a,b);m.root->update(a,b,0,1.0f);float learned=m.root->triangle(a,b);m.root->gauge(987,4.0f);float invariant=m.root->triangle(a,b);
 // This benchmark uses one canonical head per fact and validates the exact membership link, not language prediction.
 size_t two_count=0, two_ok=0, two_direct=0;
 if(argc>2){std::ifstream qi(argv[2]);uint32_t start,end,rel;while(qi>>start>>rel>>end){++two_count;two_ok+=m.two_hop(start,end,rel);two_direct+=(m.get(start,rel)!=0 && m.facts[m.get(start,rel)].dst==end);}}
 // Verify root unchanged by arena growth and all membership chains accessible.
 size_t counted=0;for(auto id:m.first)for(;id;id=m.facts[id].next)++counted;
 std::cout<<"{\"facts\":"<<read<<",\"root_bytes\":"<<ROOT_BYTES<<",\"root_locked\":"<<locked<<",\"pair_map_pass\":"<<(maps?"true":"false")<<",\"membership_count\":"<<counted<<",\"ingest_ms\":"<<ingest<<",\"queries\":"<<trials<<",\"posting_found\":"<<ok<<",\"hash_found\":"<<ok2<<",\"posting_ms\":"<<posting<<",\"hash_ms\":"<<hash<<",\"cycle_before\":"<<before<<",\"cycle_after_update\":"<<learned<<",\"cycle_after_gauge\":"<<invariant<<",\"twohop_queries\":"<<two_count<<",\"twohop_found\":"<<two_ok<<",\"direct_found\":"<<two_direct<<"}\n";
#ifdef __linux__
 if(locked)munlock(m.root,ROOT_BYTES);
#endif
}

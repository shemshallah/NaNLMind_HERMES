#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <random>
#ifdef __linux__
#include <sys/mman.h>
#endif
constexpr int N=256; constexpr size_t NC=32640, ROOT=8388608;
struct alignas(64) Pair {float f[32],r[32];};
struct alignas(64) Head {uint8_t bytes[128];};
struct Mem {
 Head head[N]; Pair pair[NC];
 static int idx(int i,int j){if(i>j)std::swap(i,j);return i*(511-i)/2+(j-i-1);}
 float& edge(int i,int j,int layer=0){Pair &p=pair[idx(i,j)];return i<j?p.f[layer]:p.r[layer];}
 void put(int i,int j,float v,int layer=0){edge(i,j,layer)=v;edge(j,i,layer)=-v;}
 float flux(int a,int b,int layer=0){return edge(0,a,layer)+edge(a,b,layer)-edge(0,b,layer);}
 void write(int a,int b,float v,int layer=0){float old=flux(a,b,layer);put(a,b,edge(a,b,layer)+v-old,layer);}
};
static_assert(sizeof(Mem)==ROOT);
extern "C" {
void* ep_new(){void* p=std::aligned_alloc(4096,ROOT);if(p)std::memset(p,0,ROOT);return p;}
void ep_free(void*p){std::free(p);}
int ep_pair_count(){return NC;}
size_t ep_root_size(){return ROOT;}
void ep_set(void*p,int a,int b,float v){if(!p||a<1||b>255||a>=b)return;((Mem*)p)->write(a,b,v);}
float ep_get(void*p,int a,int b){if(!p||a<1||b>255||a>=b)return NAN;return ((Mem*)p)->flux(a,b);}
float ep_raw(void*p,int a,int b){if(!p||a<1||b>255||a>=b)return NAN;return ((Mem*)p)->edge(a,b);}
void ep_set_layer(void*p,int layer,int a,int b,float v){if(!p||layer<0||layer>1||a<1||b>255||a>=b)return;((Mem*)p)->write(a,b,v,layer);}
float ep_get_layer(void*p,int layer,int a,int b){if(!p||layer<0||layer>1||a<1||b>255||a>=b)return NAN;return ((Mem*)p)->flux(a,b,layer);}
void ep_gauge(void*p,uint32_t seed,float scale){if(!p)return; Mem&m=*(Mem*)p;std::mt19937 gen(seed);std::normal_distribution<float> dist(0,scale);float g[256];for(float&v:g)v=dist(gen);
for(int layer=0;layer<2;layer++)for(int i=0;i<256;i++)for(int j=i+1;j<256;j++)m.put(i,j,m.edge(i,j,layer)+g[i]-g[j],layer);}
int ep_audit(void*p){if(!p)return 0;Mem&m=*(Mem*)p;for(int i=0;i<256;i++)for(int j=i+1;j<256;j++){
 if(&m.pair[Mem::idx(i,j)]!=&m.pair[Mem::idx(j,i)])return 0;
 if(m.edge(i,j)!=-m.edge(j,i)||m.edge(i,j,1)!=-m.edge(j,i,1))return 0;
 }return 1;}
int ep_save(void*p,const char* f){FILE*h=fopen(f,"wb");if(!h)return 0;bool ok=fwrite("NANEP040",1,8,h)==8 && fwrite(p,1,ROOT,h)==ROOT;fclose(h);return ok;}
int ep_load(void*p,const char*f){FILE*h=fopen(f,"rb");if(!h)return 0;char magic[8];bool ok=fread(magic,1,8,h)==8 && memcmp(magic,"NANEP040",8)==0 && fread(p,1,ROOT,h)==ROOT;fclose(h);return ok;}
}

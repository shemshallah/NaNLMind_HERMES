// Minimal 8MiB resident mirror-pair episodic copy-memory test.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdio>
#ifdef __linux__
#include <sys/mman.h>
#endif
static constexpr size_t SIZE=8388608;
struct alignas(64) Pair {float f[32],r[32];};
struct alignas(64) Head {uint8_t b[128];};
struct Root{Head h[256];Pair p[32640];
 static int idx(int a,int b){if(a>b)std::swap(a,b);return a*(511-a)/2+b-a-1;}
 float& e(int a,int b){Pair &c=p[idx(a,b)];return a<b?c.f[0]:c.r[0];}
 void put(int a,int b,float v){e(a,b)=v;e(b,a)=-v;}
 static void coord(int slot,int &a,int &b){a=1;while(slot>=255-a){slot-=255-a;++a;}b=a+1+slot;}
 void store(int slot,uint8_t v){int a,b;coord(slot,a,b);put(a,b,float(v));}
 int read(int slot){int a,b;coord(slot,a,b);return std::clamp(int(std::lround(e(0,a)+e(a,b)-e(0,b))),0,255);}
};
static_assert(sizeof(Root)==SIZE);
extern "C" {
void* memory_new(){void* p=std::aligned_alloc(4096,SIZE);if(p)std::memset(p,0,SIZE);return p;}
void memory_free(void*p){std::free(p);}
void memory_store(void*p,int slot,int v){((Root*)p)->store(slot,(uint8_t)v);}
int memory_read(void*p,int slot){return ((Root*)p)->read(slot);}
int memory_raw(void*p,int slot){int a,b;Root::coord(slot,a,b);return std::clamp(int(std::lround(((Root*)p)->e(a,b))),0,255);}
void memory_gauge(void*p,int seed,float sigma){auto&r=*(Root*)p;std::mt19937 gen(seed);std::normal_distribution<float>dist(0,sigma);float g[256];for(auto &v:g)v=dist(gen);for(int a=0;a<256;a++)for(int b=a+1;b<256;b++)r.put(a,b,r.e(a,b)+g[a]-g[b]);}
}

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
static constexpr size_t SIZE=8388608;
struct alignas(64) Head {unsigned char v[128];};
struct alignas(64) Pair {float f[32],r[32];};
struct Root {Head h[256];Pair p[32640];
 static int index(int a,int b){if(a>b)std::swap(a,b);return a*(511-a)/2+b-a-1;}
 float &edge(int a,int b){auto&cell=p[index(a,b)];return a<b?cell.f[0]:cell.r[0];}
 void put(int a,int b,float v){edge(a,b)=v;edge(b,a)=-v;}
 static void pair_for_slot(int s,int &a,int&b){a=1;while(s>=255-a){s-=255-a;a++;}b=a+1+s;}
 void store(int s,float v){int a,b;pair_for_slot(s,a,b);put(a,b,v);}
 float read(int s){int a,b;pair_for_slot(s,a,b);return edge(0,a)+edge(a,b)-edge(0,b);}
 float raw(int s){int a,b;pair_for_slot(s,a,b);return edge(a,b);}
};
static_assert(sizeof(Root)==SIZE);
extern "C" {
 void* mem_new(){void*p=std::aligned_alloc(4096,SIZE);if(p)std::memset(p,0,SIZE);return p;}
 void mem_free(void*p){std::free(p);}
 void mem_write(void*p,int s,float v){if(p && s>=0 && s<32385)((Root*)p)->store(s,v);}
 float mem_read(void*p,int s){return (p && s>=0 && s<32385)?((Root*)p)->read(s):NAN;}
 float mem_raw(void*p,int s){return (p && s>=0 && s<32385)?((Root*)p)->raw(s):NAN;}
 void mem_gauge(void*p,int seed,float sigma){if(!p)return;auto&r=*(Root*)p;std::mt19937 gen(seed);std::normal_distribution<float>dist(0,sigma);float g[256];for(auto&x:g)x=dist(gen);for(int a=0;a<256;a++)for(int b=a+1;b<256;b++)r.put(a,b,r.edge(a,b)+g[a]-g[b]);}
 int mem_snapshot(void*p,const char*fn){if(!p||!fn)return 0;auto*f=std::fopen(fn,"wb");if(!f)return 0;auto n=std::fwrite(p,1,SIZE,f);int ok=std::fclose(f);return n==SIZE && ok==0;}
 int mem_restore(void*p,const char*fn){if(!p||!fn)return 0;auto*f=std::fopen(fn,"rb");if(!f)return 0;auto n=std::fread(p,1,SIZE,f);int rest=std::fgetc(f);std::fclose(f);return n==SIZE && rest==EOF;}
}

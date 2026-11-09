#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdio>
#ifdef __linux__
#include <sys/mman.h>
#endif
static constexpr int N=256, D=16, K=16;
static constexpr size_t PAIRS=32640, ROOT=8388608;
struct alignas(64) Pair{float f[32],r[32];};
struct alignas(64) Head{unsigned char bytes[128];};
struct Memory {Head heads[256];Pair pair[PAIRS];
  static int idx(int i,int j){if(i>j)std::swap(i,j);return i*(511-i)/2+j-i-1;}
  float &edge(int i,int j,int c){Pair &p=pair[idx(i,j)];return i<j?p.f[c]:p.r[c];}
  void skew(int i,int j,int c,float v){edge(i,j,c)=v;edge(j,i,c)=-v;}
  float flux(int d,int c){int a=1+2*d,b=a+1;return edge(0,a,c)+edge(a,b,c)-edge(0,b,c);}
  void add_gradient(int d,int c,float amount){int a=1+2*d,b=a+1;
    skew(0,a,c,edge(0,a,c)+amount/3);
    skew(a,b,c,edge(a,b,c)+amount/3);
    skew(0,b,c,edge(0,b,c)-amount/3);
  }
};
static_assert(sizeof(Memory)==ROOT);
extern "C" {
void* nan_new(int lock){void *p=std::aligned_alloc(4096,ROOT);if(!p)return nullptr;std::memset(p,0,ROOT);
#ifdef __linux__
 if(lock) mlock(p,ROOT);
#endif
 return p;}
void nan_free(void* p){if(!p)return;
#ifdef __linux__
 munlock(p,ROOT);
#endif
 std::free(p);}
void nan_get(void* ptr,float *dst){auto& m=*(Memory*)ptr;for(int c=0;c<K;c++)for(int d=0;d<D;d++)dst[c*D+d]=m.flux(d,c);}
void nan_apply(void* ptr,const float *grad,float lr){auto &m=*(Memory*)ptr;for(int c=0;c<K;c++)for(int d=0;d<D;d++)m.add_gradient(d,c,-lr*grad[d*K+c]);}
void nan_predict(void* ptr,const float *x,float *out){auto& m=*(Memory*)ptr;for(int c=0;c<K;c++){double v=0;for(int d=0;d<D;d++)v+=x[d]*m.flux(d,c);out[c]=(float)v;}}
// All fitted weights reside in the physical 8MiB pair lanes; no detached neural parameter array.
double nan_epoch(void* ptr,const float* xs,const int* ys,int n,float lr,unsigned int seed){auto& m=*(Memory*)ptr;
 std::vector<int> ix(n);for(int i=0;i<n;i++)ix[i]=i;std::mt19937 rng(seed);std::shuffle(ix.begin(),ix.end(),rng);
 double total=0;float logit[K],grad[K];
 for(int t:ix){const float *x=xs+(size_t)t*D;float mx=-1e30f;
  for(int c=0;c<K;c++){double v=0;for(int d=0;d<D;d++)v+=x[d]*m.flux(d,c);logit[c]=v;mx=std::max(mx,logit[c]);}
  double den=0;for(int c=0;c<K;c++){grad[c]=std::exp(logit[c]-mx);den+=grad[c];}
  total+= -logit[ys[t]]+mx+std::log(den);
  for(int c=0;c<K;c++)grad[c]=float(grad[c]/den-(c==ys[t]));
  for(int c=0;c<K;c++)for(int d=0;d<D;d++)m.add_gradient(d,c,-lr*grad[c]*x[d]);
 }
 return total/n;
}
void nan_gauge(void* ptr,int seed,float sigma){auto& m=*(Memory*)ptr;std::mt19937 rng(seed);std::normal_distribution<float> normal(0,sigma);
 for(int c=0;c<K;c++){float g[N];for(float &v:g)v=normal(rng);
  for(int d=0;d<D;d++){int a=1+2*d,b=a+1;
   m.skew(0,a,c,m.edge(0,a,c)+g[0]-g[a]);
   m.skew(0,b,c,m.edge(0,b,c)+g[0]-g[b]);
   m.skew(a,b,c,m.edge(a,b,c)+g[a]-g[b]);
  }
 }
}
void nan_raw_predict(void* ptr,const float* x,float* out){auto& m=*(Memory*)ptr;for(int c=0;c<K;c++){double v=0;for(int d=0;d<D;d++){int a=1+2*d;v+=x[d]*m.edge(a,a+1,c);}out[c]=v;}}
int nan_snapshot(void* ptr,const char* path){FILE*f=fopen(path,"wb");if(!f)return 0;const char magic[8]={'N','A','N','C','Y','C','3','0'};int ok=fwrite(magic,1,8,f)==8&&fwrite(ptr,1,ROOT,f)==ROOT;fclose(f);return ok;}
int nan_restore(void*ptr,const char*path){FILE*f=fopen(path,"rb");if(!f)return 0;char magic[8];int ok=fread(magic,1,8,f)==8&&memcmp(magic,"NANCYC30",8)==0&&fread(ptr,1,ROOT,f)==ROOT;fclose(f);return ok;}
int nan_audit(void*ptr){auto&m=*(Memory*)ptr;for(int i=0;i<N;i++)for(int j=i+1;j<N;j++){
 if(&m.pair[Memory::idx(i,j)]!=&m.pair[Memory::idx(j,i)])return 0;
 for(int c=0;c<K;c++)if(m.edge(i,j,c)!=-m.edge(j,i,c))return 0;
 }return 1;}
}

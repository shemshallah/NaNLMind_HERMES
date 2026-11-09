#include <array>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#endif

constexpr uint32_t N=256;
constexpr size_t HEAD_BYTES=128,PAIR_BYTES=256;
constexpr size_t PAIRS=N*(N-1)/2;
constexpr size_t ROOT_BYTES=N*HEAD_BYTES+PAIRS*PAIR_BYTES;
struct alignas(64) Pair {float forward[32];float reverse[32];};
static_assert(sizeof(Pair)==PAIR_BYTES);
struct alignas(64) Head {uint8_t state[HEAD_BYTES];};
static_assert(sizeof(Head)==HEAD_BYTES);
struct Root {
  Head heads[N];Pair pairs[PAIRS];
  static uint32_t id(uint32_t i,uint32_t j){
    if(i==j||i>=N||j>=N)std::abort();
    if(i>j)std::swap(i,j);
    return i*(2*N-i-1)/2 + (j-i-1);
  }
  Pair& cell(uint32_t i,uint32_t j){return pairs[id(i,j)];}
  float& direction(uint32_t i,uint32_t j){
    auto& p=cell(i,j);return (i<j?p.forward[0]:p.reverse[0]);
  }
  void store_skew(uint32_t i,uint32_t j,float x){
    direction(i,j)=x;direction(j,i)=-x;
  }
  float circulation(uint32_t i,uint32_t j){
    return direction(0,i)+direction(i,j)+direction(j,0);
  }
  // Scalar supervised gradient for a triangle readout, where dLoss/dFlux is supplied.
  void gradient_step(uint32_t i,uint32_t j,float grad,float learning_rate){
    const float delta=-learning_rate*grad;
    store_skew(0,i,direction(0,i)+delta);
    store_skew(i,j,direction(i,j)+delta);
    store_skew(j,0,direction(j,0)+delta);
  }
};
static_assert(sizeof(Root)==ROOT_BYTES);
int main(){
  auto raw=std::aligned_alloc(4096,ROOT_BYTES);if(!raw)return 2;
  std::memset(raw,0,ROOT_BYTES);
  auto* root=reinterpret_cast<Root*>(raw);
  const auto base=root;
  size_t checked=0;
  for(uint32_t i=0;i<N;i++)for(uint32_t j=i+1;j<N;j++){
    auto& p=root->cell(i,j);
    if(&p!=&root->cell(j,i)||&root->direction(i,j)!=&p.forward[0]||&root->direction(j,i)!=&p.reverse[0])return 3;
    checked++;
  }
  if(checked!=PAIRS||root!=base)return 4;
  std::mt19937 mt(2029);std::normal_distribution<float> normal(0,1);
  double max_gauge=0,max_skew=0;
  for(int q=0;q<100;q++){
    std::array<float,N> g;for(auto& x:g)x=normal(mt)*3;
    for(int k=0;k<48;k++){
      const uint32_t a=1+2*k,b=2+2*k;
      const float semantic=normal(mt);
      root->store_skew(0,a,g[0]-g[a]);
      root->store_skew(0,b,g[0]-g[b]);
      root->store_skew(a,b,semantic+g[a]-g[b]);
      max_gauge=std::max(max_gauge,std::abs(double(root->circulation(a,b))-semantic));
      max_skew=std::max(max_skew,std::abs(double(root->direction(b,a)+root->direction(a,b))));
    }
  }
  // Verify gradient of L=(circulation-target)^2/2 using finite differences.
  const uint32_t a=3,b=4;const float target=.42f;
  auto loss=[&](){const double e=root->circulation(a,b)-target;return .5*e*e;};
  auto& v=root->direction(a,b);const float before=v;
  const float delta=1e-3f;
  root->store_skew(a,b,before+delta);const double hi=loss();
  root->store_skew(a,b,before-delta);const double lo=loss();
  root->store_skew(a,b,before);
  const double grad_fd=(hi-lo)/(2*delta);const double grad=root->circulation(a,b)-target;
  const double grad_error=std::abs(grad_fd-grad);
  const double old_loss=loss();root->gradient_step(a,b,float(grad),.1f);const double new_loss=loss();
  // Time 500K triangle reads, keeping root allocation fixed. Not an Android benchmark.
  volatile double sink=0;
  auto begin=std::chrono::steady_clock::now();
  for(int k=0;k<500000;k++){
    int x=1+2*(k%48),y=x+1;
    sink=sink+root->circulation(x,y);
  }
  const auto end=std::chrono::steady_clock::now();
  const double millis=std::chrono::duration<double,std::milli>(end-begin).count();
  // On Linux, locking may fail due to resource limits; never conflate alloc with lock.
  int lock_status=-2;
#ifdef __linux__
  lock_status=mlock(root,ROOT_BYTES);
  if(lock_status==0)munlock(root,ROOT_BYTES);
#endif
  const bool ok=max_gauge<3e-6&&max_skew==0&&grad_error<2e-3&&new_loss<old_loss;
  std::cout << "{\"root_bytes\":"<<ROOT_BYTES<<",\"mirror_pairs\":"<<checked
    <<",\"max_gauge_error\":"<<max_gauge<<",\"max_reverse_error\":"<<max_skew
    <<",\"finite_difference_error\":"<<grad_error<<",\"loss_before_step\":"<<old_loss
    <<",\"loss_after_step\":"<<new_loss<<",\"reads\":500000,\"read_milliseconds\":"<<millis
    <<",\"mlock_status\":"<<lock_status<<",\"passed\":"<<(ok?"true":"false")<<"}\n";
  std::free(raw);return ok?0:5;
}

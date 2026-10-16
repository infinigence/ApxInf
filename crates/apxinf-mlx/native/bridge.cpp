#include "bridge.h"
#include <mlx/mlx.h>
#include <mlx/compile.h>
#include <mlx/fast.h>
#include <mlx/version.h>
#include <mlx/memory.h>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
namespace mx = mlx::core;
namespace {
thread_local std::string last_error;
using A = mx::array;
using S = mx::Stream;
using F = std::function<std::vector<A>(const std::vector<A>&)>;
using K = mx::fast::CustomKernelFunction;
#define TRY try {
#define CATCH } catch (const std::exception& e) { last_error = e.what(); return -1; } catch (...) { last_error = "unknown MLX exception"; return -1; } return 0;
mx::Dtype dtype(int32_t d) { switch(d) {case 0:return mx::float32;case 1:return mx::float16;case 2:return mx::bfloat16;case 3:return mx::int32;case 4:return mx::uint32;case 5:return mx::bool_;default:throw std::invalid_argument("unsupported MLX ABI dtype");} }
int32_t dtype_code(mx::Dtype d) {for(int32_t i=0;i<6;++i)if(dtype(i)==d)return i;throw std::invalid_argument("MLX dtype outside bridge contract");}
mx::Shape shape(const int32_t* values,size_t n) {mx::Shape out;for(size_t i=0;i<n;++i){if(values[i]<0)throw std::invalid_argument("negative shape extent");out.push_back(values[i]);}return out;}
std::vector<A> arrays(const void* const* handles,size_t n) {std::vector<A> out;out.reserve(n);for(size_t i=0;i<n;++i){if(!handles[i])throw std::invalid_argument("null array");out.push_back(*static_cast<const A*>(handles[i]));}return out;}
void publish(const std::vector<A>& values,void** out,size_t count) {if(values.size()!=count)throw std::invalid_argument("output count differs from contract");std::vector<std::unique_ptr<A>> owned;for(const auto& v:values)owned.emplace_back(std::make_unique<A>(v));for(size_t i=0;i<count;++i)out[i]=owned[i].release();}
template<class T>A upload(const uint8_t* bytes,size_t len,mx::Shape sh,mx::Dtype dt){std::vector<T> owned(len/sizeof(T));if(len)std::memcpy(owned.data(),bytes,len);return A(owned.begin(),sh,dt);}
}
extern "C" {
const char* apx_mlx_error(void){return last_error.c_str();}
int apx_mlx_stream_new(int32_t gpu,int32_t index,void** out){TRY
 if(std::string(mx::version())!="0.31.2")throw std::runtime_error("linked MLX runtime differs from pinned 0.31.2 SDK");
 mx::Device d(gpu?mx::Device::gpu:mx::Device::cpu,index);
 if(!mx::is_available(d))throw std::runtime_error("requested MLX device unavailable");
 // Reuse MLX's thread-local device stream: MLX 0.31.2 has no individual
 // stream-destruction API, so new_stream per model would retain global queues.
 *out=new S(mx::default_stream(d));CATCH}
void apx_mlx_stream_free(void* p){delete static_cast<S*>(p);}
int apx_mlx_sync(void* p){TRY mx::synchronize(*static_cast<S*>(p));CATCH}
int apx_mlx_memory_stats(size_t* active,size_t* cache,size_t* peak){TRY *active=mx::get_active_memory();*cache=mx::get_cache_memory();*peak=mx::get_peak_memory();CATCH}
int apx_mlx_reset_peak_memory(void){TRY mx::reset_peak_memory();CATCH}
int apx_mlx_clear_cache(void){TRY mx::clear_cache();CATCH}
int apx_mlx_array_new(void*,const int32_t* dims,size_t rank,int32_t dt,const uint8_t* bytes,size_t len,void** out){TRY
 auto sh=shape(dims,rank);size_t count=1;for(auto d:sh){if(d&&count>SIZE_MAX/size_t(d))throw std::overflow_error("shape overflow");count*=d;}
 if(count>SIZE_MAX/mx::size_of(dtype(dt))||count*mx::size_of(dtype(dt))!=len)throw std::invalid_argument("upload byte length mismatch");
 A result=[&](){switch(dt){case 0:return upload<float>(bytes,len,sh,dtype(dt));case 1:return upload<mx::float16_t>(bytes,len,sh,dtype(dt));case 2:return upload<mx::bfloat16_t>(bytes,len,sh,dtype(dt));case 3:return upload<int32_t>(bytes,len,sh,dtype(dt));case 4:return upload<uint32_t>(bytes,len,sh,dtype(dt));default:throw std::invalid_argument("bool host upload unsupported");}}();*out=new A(std::move(result));CATCH}
int apx_mlx_array_clone(const void* p,void** out){TRY *out=new A(*static_cast<const A*>(p));CATCH}
void apx_mlx_array_free(void* p){delete static_cast<A*>(p);}
int apx_mlx_array_info(const void* p,int32_t* dims,size_t capacity,size_t* rank,int32_t* dt){TRY
 const auto& a=*static_cast<const A*>(p);*rank=a.ndim();*dt=dtype_code(a.dtype());if(capacity<a.ndim())throw std::invalid_argument("shape output too short");for(size_t i=0;i<a.ndim();++i)dims[i]=a.shape(i);CATCH}
int apx_mlx_array_read(void* s,const void* p,uint8_t* bytes,size_t len){TRY
 auto a=mx::contiguous(*static_cast<const A*>(p),false,*static_cast<S*>(s));if(a.nbytes()!=len)throw std::invalid_argument("download byte length mismatch");mx::eval(a);mx::synchronize(*static_cast<S*>(s));if(len)std::memcpy(bytes,a.data<uint8_t>(),len);CATCH}
int apx_mlx_eval(void* s,const void* const* p,size_t n){TRY mx::eval(arrays(p,n));mx::synchronize(*static_cast<S*>(s));CATCH}
int apx_mlx_op(void* stream,int32_t op,const void* const* inputs,size_t count,const int32_t* ints,size_t nints,const float* floats,size_t nfloats,void** out){TRY
 auto s=*static_cast<S*>(stream);auto a=arrays(inputs,count);std::vector<int> iv(ints,ints+nints);
 auto i=[&](size_t n){if(n>=nints)throw std::invalid_argument("missing integer argument");return ints[n];};
 auto f=[&](size_t n){if(n>=nfloats)throw std::invalid_argument("missing scalar argument");return floats[n];};
 A result=[&]() -> A {switch(op){
 case 0:return mx::astype(a.at(0),dtype(i(0)),s);
 case 1:return mx::reshape(a.at(0),shape(ints,nints),s);
 case 2:return mx::transpose(a.at(0),iv,s);
 case 3:return mx::contiguous(a.at(0),false,s);
 case 4:{auto rank=a.at(0).ndim();if(nints!=2*rank)throw std::invalid_argument("slice rank mismatch");return mx::slice(a[0],shape(ints,rank),shape(ints+rank,rank),s);}
 case 5:return mx::concatenate(a,i(0),s);
 case 6:return mx::broadcast_to(a.at(0),shape(ints,nints),s);
 case 7:return mx::matmul(a.at(0),a.at(1),s);
 case 8:return mx::add(a.at(0),a.at(1),s);
 case 9:return mx::multiply(a.at(0),a.at(1),s);
 case 10:return mx::divide(a.at(0),a.at(1),s);
 case 11:return mx::power(a.at(0),a.at(1),s);
 case 12:return mx::rsqrt(a.at(0),s);
 case 13:return mx::exp(a.at(0),s);
 case 14:return mx::sigmoid(a.at(0),s);
 case 15:return mx::softmax(a.at(0),i(0),true,s);
 case 16:return mx::sum(a.at(0),std::vector<int>(iv.begin()+1,iv.end()),i(0)!=0,s);
 case 17:return mx::mean(a.at(0),std::vector<int>(iv.begin()+1,iv.end()),i(0)!=0,s);
 case 18:return mx::arange(f(0),f(1),f(2),dtype(i(0)),s);
 case 19:return mx::take(a.at(0),a.at(1),i(0),s);
 case 20:return mx::slice_update(a.at(0),a.at(1),a.at(2),iv,s);
 case 21:return mx::argmax(a.at(0),i(0),false,s);
 case 22:return mx::subtract(a.at(0),a.at(1),s);
 case 23:return mx::negative(a.at(0),s);
 case 24:return mx::cos(a.at(0),s);
 case 25:return mx::sin(a.at(0),s);
 case 26:return mx::less_equal(a.at(0),a.at(1),s);
 case 27:return mx::where(a.at(0),a.at(1),a.at(2),s);
 case 28:return mx::zeros(shape(ints+1,nints-1),dtype(i(0)),s);
 case 29:return mx::fast::scaled_dot_product_attention(a.at(0),a.at(1),a.at(2),f(0),i(0)?"causal":"",a.size()>3?std::optional<A>(a[3]):std::nullopt,std::nullopt,s);
 case 30:return mx::fast::rope(a.at(0),i(0),i(1)!=0,f(0),f(1),a.at(1),std::nullopt,s);
 case 31:return mx::quantized_matmul(a.at(0),a.at(1),a.at(2),a.at(3),i(0)!=0,i(1),i(2),"affine",s);
 case 32:{size_t rank=i(0);if(nints!=2+2*rank)throw std::invalid_argument("strides rank mismatch");mx::Strides strides;for(size_t j=0;j<rank;++j)strides.push_back(i(1+rank+j));return mx::as_strided(a.at(0),shape(ints+1,rank),strides,i(1+2*rank),s);}
 case 33:return mx::copy(a.at(0),s);
 case 34:return mx::isfinite(a.at(0),s);
 case 35:return mx::max(a.at(0),i(0),true,s);
 case 36:return mx::equal(a.at(0),a.at(1),s);
 case 37:return mx::nan_to_num(a.at(0),f(0),f(1),f(2),s);
 case 38:return mx::fast::rms_norm(a.at(0),a.at(1),f(0),s);
 case 39:return mx::dequantize(a.at(0),a.at(1),a.at(2),i(0),i(1),"affine",std::nullopt,std::nullopt,s);
 default:throw std::invalid_argument("unknown MLX bridge operation");}}();
 *out=new A(std::move(result));CATCH}
int apx_mlx_compile_new(apx_mlx_callback cb,void* ctx,size_t outputs,int32_t shapeless,void** out){TRY
 // MLX treats presence as disabled, including values such as "0". Required
 // compiled execution must never silently receive its original eager body.
 if(std::getenv("MLX_DISABLE_COMPILE"))throw std::runtime_error("MLX_DISABLE_COMPILE is set; required compiled execution is unavailable");
 F body=[cb,ctx,outputs](const std::vector<A>& input){std::vector<const void*> in;for(const auto& a:input)in.push_back(&a);std::vector<void*> handles(outputs,nullptr);
 int status=cb(ctx,in.data(),in.size(),handles.data(),outputs);std::vector<std::unique_ptr<A>> owned;for(auto h:handles)owned.emplace_back(static_cast<A*>(h));
 if(status!=0)throw std::runtime_error("Rust MLX trace callback failed");std::vector<A> result;for(auto& h:owned){if(!h)throw std::runtime_error("trace callback returned null");result.push_back(*h);}return result;};
 *out=new F(mx::compile(body,shapeless!=0));CATCH}
void apx_mlx_compile_free(void* p){delete static_cast<F*>(p);}
int apx_mlx_compile_call(void* p,const void* const* inputs,size_t n,void** outputs,size_t count){TRY auto result=(*static_cast<F*>(p))(arrays(inputs,n));publish(result,outputs,count);CATCH}
int apx_mlx_quantize(void* s,const void* input,int32_t group,int32_t bits,void** outputs){TRY auto result=mx::quantize(*static_cast<const A*>(input),group,bits,"affine",std::nullopt,*static_cast<S*>(s));publish(result,outputs,3);CATCH}
int apx_mlx_metal_new(const char* name,const char* source,const char* header,const char* const* ins,size_t ni,const char* const* outs,size_t no,void** out){TRY
 std::vector<std::string> input_names,output_names;for(size_t i=0;i<ni;++i)input_names.emplace_back(ins[i]);for(size_t i=0;i<no;++i)output_names.emplace_back(outs[i]);*out=new K(mx::fast::metal_kernel(name,input_names,output_names,source,header,true,false));CATCH}
void apx_mlx_metal_free(void* p){delete static_cast<K*>(p);}
int apx_mlx_metal_call(void* stream,void* kernel,const void* const* inputs,size_t ni,const int32_t* dims,const size_t* ranks,const int32_t* dtypes,size_t no,const int32_t* grid,const int32_t* group,int32_t templ,void** outputs){TRY
 std::vector<mx::Shape> shapes;std::vector<mx::Dtype> types;size_t offset=0;for(size_t j=0;j<no;++j){shapes.push_back(shape(dims+offset,ranks[j]));offset+=ranks[j];types.push_back(dtype(dtypes[j]));}
 std::vector<std::pair<std::string,mx::fast::TemplateArg>> templates;if(templ>=0)templates.push_back({"T",dtype(templ)});
 auto result=(*static_cast<K*>(kernel))(arrays(inputs,ni),shapes,types,{grid[0],grid[1],grid[2]},{group[0],group[1],group[2]},templates,std::nullopt,false,*static_cast<S*>(stream));publish(result,outputs,no);CATCH}
}

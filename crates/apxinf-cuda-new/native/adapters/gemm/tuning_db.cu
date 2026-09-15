#include "internal.h"
#include <fstream>
#include <filesystem>
#include <unistd.h>
namespace apxinf::gemm {
static std::string path(const std::string& dir,const std::string& key){
 uint64_t h=14695981039346656037ULL;for(unsigned char c:key){h^=c;h*=1099511628211ULL;}
 std::ostringstream s;s<<dir<<'/'<<std::hex<<h<<".recipe";return s.str();
}
std::string read_recipe(const std::string& dir,const std::string& key){
 if(dir.empty())return {};std::ifstream f(path(dir,key));std::string stored,recipe;
 if(!std::getline(f,stored)||stored!=key||!std::getline(f,recipe))return {};return recipe;
}
void write_recipe(const std::string& dir,const std::string& key,const std::string& recipe){
 if(dir.empty())return;std::error_code error;std::filesystem::create_directories(dir,error);if(error)return;
 auto dest=path(dir,key),tmp=dest+"."+std::to_string(getpid())+".tmp";
 {std::ofstream f(tmp);f<<key<<'\n'<<recipe<<'\n';if(!f)return;}
 std::filesystem::rename(tmp,dest,error);
}
}

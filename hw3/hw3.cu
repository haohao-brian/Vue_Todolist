// nvcc -O3 -std=c++17 -arch=sm_70 hw3_gpu.cu /work/b10502010/pp25/hw3/lodepng/lodepng.cpp -o hw3_gpu
// (adjust -I paths if your lodepng.h is elsewhere)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <lodepng.h>

// ---------- lightweight vector math (double3) ----------
struct d3 { double x,y,z; };
__host__ __device__ inline d3 make_d3(double x=0,double y=0,double z=0){ return {x,y,z}; }
__host__ __device__ inline d3 operator+(d3 a,d3 b){ return make_d3(a.x+b.x,a.y+b.y,a.z+b.z); }
__host__ __device__ inline d3 operator-(d3 a,d3 b){ return make_d3(a.x-b.x,a.y-b.y,a.z-b.z); }
__host__ __device__ inline d3 operator*(d3 a,double s){ return make_d3(a.x*s,a.y*s,a.z*s); }
__host__ __device__ inline d3 operator*(double s,d3 a){ return a*s; }
__host__ __device__ inline d3 operator/(d3 a,double s){ return make_d3(a.x/s,a.y/s,a.z/s); }
__host__ __device__ inline double dot(d3 a,d3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline d3 cross(d3 a,d3 b){
    return make_d3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__host__ __device__ inline d3 pow3c(d3 v, d3 e){
    return make_d3( pow(v.x, e.x), pow(v.y, e.y), pow(v.z, e.z) );
}
__host__ __device__ inline double length(d3 a){ return sqrt(dot(a,a)); }
__host__ __device__ inline d3 normalize(d3 a){ double L=length(a); return L>0? a/L : a; }
__host__ __device__ inline double clampd(double x,double lo,double hi){ return fmin(fmax(x,lo),hi); }
__host__ __device__ inline d3 clamp3(d3 v,double lo,double hi){ return make_d3(clampd(v.x,lo,hi),clampd(v.y,lo,hi),clampd(v.z,lo,hi)); }
__host__ __device__ inline d3 pow3(d3 v,double e){ return make_d3(pow(v.x,e),pow(v.y,e),pow(v.z,e)); }
__host__ __device__ inline d3 add3(d3 a,d3 b){ return make_d3(a.x+b.x,a.y+b.y,a.z+b.z); }
__host__ __device__ inline d3 mul3(d3 a,d3 b){ return make_d3(a.x*b.x,a.y*b.y,a.z*b.z); }

// ---------- constants / params ----------
constexpr double PI = 3.1415926535897932384626433832795;
__constant__ int   d_AA;                 // anti-aliasing
__constant__ double d_power, d_md_iter, d_ray_step, d_shadow_step, d_step_limiter, d_ray_multiplier, d_bailout, d_eps, d_FOV, d_far_plane;
__constant__ int d_width, d_height;
__constant__ d3 d_cam_pos, d_target_pos;

// ---------- palette ----------
__device__ d3 pal(double t, d3 a, d3 b, d3 c, d3 d){
    // a + b * cos(2*pi*(c*t + d))
    double kx = 2.0*PI*(c.x*t + d.x);
    double ky = 2.0*PI*(c.y*t + d.y);
    double kz = 2.0*PI*(c.z*t + d.z);
    return add3(a, mul3(b, make_d3(cos(kx), cos(ky), cos(kz))));
}

// ---------- mandelbulb DE ----------
__device__ double md(d3 p, double &trap){
    d3 v = p;
    double dr = 1.0;
    double r  = length(v);
    trap = r;

    for(int i=0;i<(int)d_md_iter;++i){
        // spherical
        double theta = atan2(v.y, v.x) * d_power;
        double phi   = asin(fmax(fmin(v.z / (r + 1e-16), 1.0), -1.0)) * d_power;

        dr = d_power * pow(r, d_power - 1.0) * dr + 1.0;

        double rp = pow(r, d_power);
        double cth = cos(theta), sth = sin(theta), cph = cos(phi), sph = sin(phi);
        v = add3(p, make_d3(rp * cth * cph, rp * cph * sth, -rp * sph));

        trap = fmin(trap, r);
        r = length(v);
        if(r > d_bailout) break;
    }
    return 0.5 * log(r) * r / (dr + 1e-16);
}

// rotation by +90° around X: (x, y', z') with matrix [[1,0,0],[0,cos,-sin],[0,sin,cos]]
__device__ d3 rotX90(d3 p){
    double c = cos(PI/2.0), s = sin(PI/2.0);
    return make_d3(p.x, c*p.y - s*p.z, s*p.y + c*p.z);
}

// map with ID/trap
__device__ double map_id(d3 p, double &trap, int &ID){
    d3 rp = rotX90(p);
    ID = 1;
    return md(rp, trap);
}
__device__ double map_only(d3 p){
    double t; int id;
    return map_id(p,t,id);
}

// soft shadow (Inigo Quilez style)
__device__ double softshadow(d3 ro, d3 rd, double k){
    double res = 1.0;
    double t = 0.0;
    for(int i=0;i<(int)d_shadow_step;++i){
        double h = map_only(add3(ro, rd*t));
        res = fmin(res, k * h / fmax(t, 1e-6));
        if(res < 0.02) return 0.02;
        t += fmin(fmax(h, 0.001), d_step_limiter);
        if(t > d_far_plane) break;
    }
    return clampd(res, 0.02, 1.0);
}

// normal via central differences
__device__ d3 calcNor(d3 p){
    double e = d_eps;
    double dx = map_only(make_d3(p.x+e, p.y, p.z)) - map_only(make_d3(p.x-e, p.y, p.z));
    double dy = map_only(make_d3(p.x, p.y+e, p.z)) - map_only(make_d3(p.x, p.y-e, p.z));
    double dz = map_only(make_d3(p.x, p.y, p.z+e)) - map_only(make_d3(p.x, p.y, p.z-e));
    return normalize(make_d3(dx,dy,dz));
}

// sphere tracing
__device__ double trace(d3 ro, d3 rd, double &trap, int &ID){
    double t = 0.0;
    for(int i=0;i<(int)d_ray_step;++i){
        double len = map_id(add3(ro, rd*t), trap, ID);
        if(fabs(len) < d_eps || t > d_far_plane) break;
        t += len * d_ray_multiplier;
    }
    return (t < d_far_plane) ? t : -1.0;
}

// ---------- kernel ----------
__global__ void render_kernel(unsigned char* rgba){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= d_width || y >= d_height) return;

    // AA accumulation
    double accR=0, accG=0, accB=0;
    for(int m=0;m<d_AA;++m){
        for(int n=0;n<d_AA;++n){
            // pixel sub-sample
            double px = (double)x + (double)m / (double)d_AA;
            double py = (double)y + (double)n / (double)d_AA;

            // screen to uv in [-aspect,+aspect]x[-1,1], y flipped
            double aspect = (double)d_width / (double)d_height;
            double u = (- (double)d_width + 2.0*px) / (double)d_height;
            double v = (- (double)d_height + 2.0*py) / (double)d_height;
            v = -v; // flip

            // camera basis
            d3 ro = d_cam_pos;
            d3 ta = d_target_pos;
            d3 cf = normalize(ta - ro);
            d3 cs = normalize(cross(cf, make_d3(0,1,0)));
            d3 cu = normalize(cross(cs, cf));
            d3 rd = normalize(cs*u + cu*v + cf*d_FOV);

            // march
            double trap; int objID;
            double t = trace(ro, rd, trap, objID);

            d3 col = make_d3(0,0,0);
            if(t >= 0.0){
                d3 pos = ro + rd*t;
                d3 nr  = calcNor(pos);
                d3 sd  = normalize(d_cam_pos); // light direction like original
                d3 sc  = make_d3(1.0, 0.9, 0.717);

                // base color via orbit trap palette
                d3 base = pal(trap - 0.4,
                              make_d3(0.5,0.5,0.5),
                              make_d3(0.5,0.5,0.5),
                              make_d3(1.0,1.0,1.0),
                              make_d3(0.0,0.1,0.2));

                // lighting
                double amb = (0.7 + 0.3*nr.y) * (0.2 + 0.8*clampd(0.05*log(fmax(trap,1e-6)),0.0,1.0));
                double sdw = softshadow(pos + nr*0.001, sd, 16.0);
                double dif = clampd(dot(sd, nr), 0.0, 1.0) * sdw;
                d3  hal = normalize(sd - rd);
                double spe = pow(clampd(dot(nr, hal), 0.0, 1.0), 32.0) * dif;

                d3 lin = make_d3(0,0,0);
                lin = lin + make_d3(0.3,0.3,0.3) * (0.05 + 0.95*amb);
                lin = add3(lin, sc * (0.8*dif));

                col = mul3(base, lin);
                // fake SSS & specular
                //col = pow3(col,  make_d3(0.7,0.9,1.0).x); // scalar pow on each; simple: use x
		col = pow3c(col, make_d3(0.7, 0.9, 1.0));
                col = add3(col, make_d3(spe*0.8, spe*0.8, spe*0.8));
            }
            // gamma 2.2
            col = clamp3(make_d3(pow(col.x, 0.4545), pow(col.y, 0.4545), pow(col.z, 0.4545)), 0.0, 1.0);

            accR += col.x; accG += col.y; accB += col.z;
        }
    }
    double inv = 1.0 / (double)(d_AA*d_AA);
    unsigned char R = (unsigned char)(255.0 * accR * inv);
    unsigned char G = (unsigned char)(255.0 * accG * inv);
    unsigned char B = (unsigned char)(255.0 * accB * inv);

    size_t idx = 4ull * ( (size_t)y * (size_t)d_width + (size_t)x );
    rgba[idx+0]=R; rgba[idx+1]=G; rgba[idx+2]=B; rgba[idx+3]=255;
}

// ---------- host glue ----------
static void check(cudaError_t e, const char* msg){
    if(e!=cudaSuccess){ fprintf(stderr, "CUDA %s: %s\n", msg, cudaGetErrorString(e)); std::exit(2); }
}

int main(int argc, char** argv){
    // args: x1 y1 z1  x2 y2 z2  W H  out.png
    if(argc != 10){
        fprintf(stderr, "Usage: %s x1 y1 z1 x2 y2 z2 W H out.png\n", argv[0]);
        return 1;
    }
    d3 cam = make_d3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    d3 tar = make_d3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    int W = atoi(argv[7]);
    int H = atoi(argv[8]);
    std::string out = argv[9];

    // params (mirror your CPU defaults)
    int   AA = 3;
    double power=8.0, md_iter=24, ray_step=10000, shadow_step=1500, step_limiter=0.2,
           ray_multiplier=0.1, bailout=2.0, eps=0.0005, FOV=1.5, far_plane=100.0;

    // push constants
    check(cudaMemcpyToSymbol(d_AA, &AA, sizeof(int)), "cpy d_AA");
    check(cudaMemcpyToSymbol(d_power, &power, sizeof(double)), "cpy power");
    check(cudaMemcpyToSymbol(d_md_iter, &md_iter, sizeof(double)), "cpy md_iter");
    check(cudaMemcpyToSymbol(d_ray_step, &ray_step, sizeof(double)), "cpy ray_step");
    check(cudaMemcpyToSymbol(d_shadow_step, &shadow_step, sizeof(double)), "cpy sh_step");
    check(cudaMemcpyToSymbol(d_step_limiter, &step_limiter, sizeof(double)), "cpy step_lim");
    check(cudaMemcpyToSymbol(d_ray_multiplier, &ray_multiplier, sizeof(double)), "cpy ray_mul");
    check(cudaMemcpyToSymbol(d_bailout, &bailout, sizeof(double)), "cpy bailout");
    check(cudaMemcpyToSymbol(d_eps, &eps, sizeof(double)), "cpy eps");
    check(cudaMemcpyToSymbol(d_FOV, &FOV, sizeof(double)), "cpy fov");
    check(cudaMemcpyToSymbol(d_far_plane, &far_plane, sizeof(double)), "cpy far");
    check(cudaMemcpyToSymbol(d_width, &W, sizeof(int)), "cpy W");
    check(cudaMemcpyToSymbol(d_height, &H, sizeof(int)), "cpy H");
    check(cudaMemcpyToSymbol(d_cam_pos, &cam, sizeof(d3)), "cpy cam");
    check(cudaMemcpyToSymbol(d_target_pos, &tar, sizeof(d3)), "cpy tar");

    // device buffer
    size_t bytes = (size_t)W * (size_t)H * 4;
    unsigned char* d_rgba=nullptr;
    check(cudaMalloc(&d_rgba, bytes), "malloc d_rgba");

    dim3 blk(16,16);
    dim3 grd( (W+blk.x-1)/blk.x, (H+blk.y-1)/blk.y );
    render_kernel<<<grd, blk>>>(d_rgba);
    check(cudaGetLastError(), "launch");
    check(cudaDeviceSynchronize(), "sync");

    // copy back
    std::vector<unsigned char> h_rgba(bytes);
    check(cudaMemcpy(h_rgba.data(), d_rgba, bytes, cudaMemcpyDeviceToHost), "memcpy D2H");
    cudaFree(d_rgba);

    // write PNG
    unsigned error = lodepng_encode32_file(out.c_str(), h_rgba.data(), W, H);
    if(error){
        fprintf(stderr, "PNG error %u: %s\n", error, lodepng_error_text(error));
        return 3;
    }
    printf("Wrote %s (%dx%d)\n", out.c_str(), W, H);
    return 0;
}


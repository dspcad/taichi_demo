#include <thread>
#include <chrono>
#include <iostream>
#include "glm/glm.hpp"
#include "glm/ext.hpp"
#include "taichi/aot_demo/framework.hpp"
#include "taichi/aot_demo/vulkan/interop/cross_device_copy.hpp"

using namespace ti::aot_demo;

static std::string get_aot_file_dir(TiArch arch) {
    switch(arch) {
        case TI_ARCH_VULKAN: {
            return "8_fireworks/assets/fireworks_vulkan";
        }
        case TI_ARCH_X64: {
            return "8_fireworks/assets/fireworks_x64";
        }
        case TI_ARCH_CUDA: {
            return "8_fireworks/assets/fireworks_cuda";
        }
        default: {
            throw std::runtime_error("Unrecognized arch");
        }
    }
}



struct App8_fireworks : public App {
  ti::Runtime runtime_;
  ti::AotModule module_;
  TiArch arch_;

  ti::Kernel k_draw_;
  ti::Texture canvas_;
  ti::NdArray<float> _ndarrayInputTexture;
  ti::NdArray<float> _ndarrayiTime;

  float *_timeBuffer;


  std::unique_ptr<GraphicsTask> draw_points;

  virtual AppConfig cfg() const override final {
    AppConfig out {};
    out.app_name = "8_fireworks";
    out.framebuffer_width = 800;
    out.framebuffer_height = 600;
    return out;
  }

  virtual void initialize(TiArch arch) override final{
    if(arch != TI_ARCH_VULKAN && arch != TI_ARCH_OPENGL) {
        std::cout << "4_texture_fractal only supports vulkan, opengl backend" << std::endl;
        exit(0);
    }

    arch_ = arch;
    
    // 1. Create runtime
    GraphicsRuntime& runtime_ = F_->runtime();
    //if(arch_ == TI_ARCH_VULKAN) {
    //    // Reuse the vulkan runtime from renderer framework
    //    runtime_ = ti::Runtime(arch_, F_->runtime(), false);;
    //} else {
    //    runtime_ = ti::Runtime(arch_);
    //}
    
    // 2. Load AOT module
#ifdef TI_AOT_DEMO_ANDROID_APP
    std::vector<uint8_t> tcm;
    F_->asset_mgr().load_file("E8_fireworks.tcm", tcm);
    module_ = runtime_.create_aot_module(tcm);
#else
    auto aot_file_path = get_aot_file_dir(arch_);
    module_ = runtime_.load_aot_module(aot_file_path);
#endif
    
    // 3. Load kernels
    k_draw_ = module_.get_kernel("draw");

    _ndarrayInputTexture = runtime_.allocate_ndarray<float>({800, 600}, {4}, true);
    canvas_ = runtime_.allocate_texture2d(800, 600, TI_FORMAT_RGBA32F, TI_NULL_HANDLE);
    _ndarrayiTime        = runtime_.allocate_ndarray<float>({1}, {1}, true);

    draw_points = runtime_.draw_texture(canvas_).build();


    k_draw_[0] = canvas_;
    k_draw_[1] = _ndarrayInputTexture;
    k_draw_[2] = _ndarrayiTime;

    _timeBuffer = reinterpret_cast<float*>(_ndarrayiTime.map());
    *_timeBuffer = 0;
    if (_timeBuffer) { _ndarrayiTime.unmap();}


    // 7. Run initialization kernels
    Renderer& renderer = F_->renderer();
    renderer.set_framebuffer_size(800, 600);

    std::cout << "initialized!" << std::endl;
  }
  virtual bool update() override final {
    // 8. Run compute kernels
    
    k_draw_.launch();
    _timeBuffer = reinterpret_cast<float*>(_ndarrayiTime.map());
    *_timeBuffer = ((*_timeBuffer) + 0.01);
    std::cout << "stepped! (fps=" << F_->fps() << ")" << "    iTime: " << *_timeBuffer << std::endl;
    if (_timeBuffer) { _ndarrayiTime.unmap(); }


    return true;
  }
  virtual void render() override final {
    Renderer& renderer = F_->renderer();
    renderer.enqueue_graphics_task(*draw_points);
  }
};

std::unique_ptr<App> create_app() {
  return std::make_unique<App8_fireworks>();
}

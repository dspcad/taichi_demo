import argparse
import taichi as ti
from taichi.math import *
import numpy as np
import math
import tempfile
import os

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str)
args = parser.parse_args()

def get_save_dir(name, arch):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(curr_dir, f"{name}_{arch}")
def get_archive_path():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(curr_dir, f"../../framework/android/app/src/main/assets/E8_fireworks.tcm")

if args.arch == "cuda":
    arch = ti.cuda
    platform = None
elif args.arch == "x64":
    arch = ti.x64
    platform = None
elif args.arch == "vulkan":
    arch = ti.vulkan
    platform = None
elif args.arch == "android-vulkan":
    arch = ti.vulkan
    platform = "android"
else:
    assert False


ti.init(arch=arch, offline_cache=False, default_fp=ti.f32)

W = 800
H = 600

RES  = (W, H)
RES_ = (H, W)

NUM_EXPLOSIONS = 8
NUM_PARTICLES = 70
output_texture  = ti.Vector.ndarray(4, dtype=ti.f32, shape=RES)
#output_texture = ti.Texture(ti.Format.rgba32f, RES)
#output_texture = ti.Vector.field(4, dtype=ti.f32, shape=RES)
print(output_texture.shape)
iTime          = ti.ndarray(dtype=ti.f32, shape=(1))

#image = ti.Vector.field(4, dtype=ti.f32, shape=RES)
NUM_EXPLOSIONS

# Noise functions by Dave Hoskins 
MOD3 = vec3(.1031,.11369,.13787)

@ti.func 
def S(x, y, z):
	return smoothstep(x, y, z)

@ti.func 
def B(x, y, z, w):
	return S(x-z, x+z, w)*S(y+z, y-z, w)

@ti.func
def saturate(x):
	return clamp(x, 0., 1.)

@ti.func
def hash31(p): 
	p3 = fract(p * MOD3)
	p3 += dot(p3, p3.yzx + 19.19)
	return fract(vec3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x))

@ti.func
def hash12(p):
	p3 = fract(vec3(p.xyx) * MOD3)
	p3 += dot(p3, p3.yzx + 19.19)
	return fract((p3.x + p3.y) * p3.z)

@ti.func
def circ(uv, pos, size): 
	uv -= pos    
	size *= size
	return S(size*1.1, size, dot(uv, uv))

@ti.func
def light(uv, pos, size): 
	uv -= pos
	size *= size
	return size/dot(uv, uv)

@ti.func
def explosion(uv, p, seed, t):
	col = vec3(0.)
	en = hash31(seed)
	baseCol = en
	for i in range(NUM_PARTICLES):
		n = hash31(i)-.5
	   
		startP = p-vec2(0., t*t*.1)        
		endP = startP+normalize(n.xy)*n.z
		
		pt = 1.-pow(t-1., 2.)
		pos = mix(p, endP, pt)    
		size = mix(.01, .005, S(0., .1, pt))
		size *= S(1., .1, pt)
		
		sparkle = (sin((pt+n.z)*100.)*.5+.5)
		sparkle = pow(sparkle, pow(en.x, 3.)*50.)*mix(0.01, .01, en.y*n.y)
	  
		# size += sparkle*B(.6, 1., .1, t)
		size += sparkle*B(en.x, en.y, en.z, t)
		
		col += baseCol*light(uv, pos, size)
	
	return col
@ti.func 
def N(h):
	return fract(sin(vec4(6,9,1,0)*h) * 9e2)


@ti.func
def Rainbow(c,iTime : ti.types.ndarray(ndim=1)):
	
	t=iTime[0]
	
	avg = (c.r+c.g+c.b)/3.
	c = avg + (c-avg)*sin(vec3(0., .333, .666)+t)
	
	c += sin(vec3(.4, .3, .3)*t + vec3(1.1244,3.43215,6.435))*vec3(.4, .1, .5)
	
	return c






@ti.kernel
def draw(output_texture: ti.types.ndarray(dtype=ti.math.vec4, ndim=2),
        #input_texture_nd: ti.types.ndarray(ndim=2), 
        iTime: ti.types.ndarray(ndim=1)):
    for i, j in output_texture:
        uv = vec2(i/RES[0], j/RES[1])
        uv.x -= .5
        uv.x *= RES[0]/RES[1]
	   
        n = hash12(uv+10.)
		#t = iTime[None]*.5
        t = iTime[0]*.5
	   
        c = vec3(0.)
	   
        for k in range(NUM_EXPLOSIONS):
            et = t+k*1234.45235
            idx = floor(et)
            et -= idx
		   
            p = hash31(idx).xy
            p.x -= .5
            p.x *= 1.6
            c += explosion(uv, p, idx, et)
	  
        c = Rainbow(c,iTime)
        #output_texture.store(ti.Vector([i, j]), vec4(c, 1.))
        output_texture[i, j] = vec4(c, 1.)
	   
        #image[i, j] = vec4(c, 1.)
		#image[i, j] = vec4(1.,0.,1., 1.)


#if __name__ == "__main__":
#    # Serialize!
#    mod = ti.aot.Module(arch)
#
#    mod.add_kernel(draw)
#
#
#    if platform == "android":
#        mod.archive(get_archive_path())
#    else:
#        save_dir = get_save_dir("fireworks", args.arch)
#        os.makedirs(save_dir, exist_ok=True)
#        mod.save(save_dir)


if __name__ == "__main__":
   gui = ti.GUI('Fireworks', res=RES)
   # canvas    = gui.get_canvas()
   iTime[0] = 0.0
   while gui.running:
       print(iTime[0])
       iTime[0] += 0.01
       draw(output_texture, iTime)
       gui.set_image(output_texture.to_numpy())
       gui.show()


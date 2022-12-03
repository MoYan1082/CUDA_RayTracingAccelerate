## CUDA_RayTracingAccelerate

效果演示（渲染时间为`9s`）：

![example](./images/figure1.png)

### :computer:开发环境

- OS：ubuntu20.04
- Cuda：11.3
- GPU：RTX 2080 Ti 

### :hammer_and_wrench:如何使用

```shell
git clone https://github.com/MoYan1082/CUDA_RayTracingAccelerate.git
cd CUDA_RayTracingAccelerate
mkdir build
cmake ..
make
./main
```

渲染出来的图片路径为：`build/figure.ppm`。

### :milky_way:Features

- 物体：圆球。
- 材质：漫反射、镜面反射、折射。
- BVH：方差建树、非递归搜索。
- 随机数：每个像素以及每次采样都使用不同的随机数种子 `curand_uniform_double(curandState&)`。



参考：[RayTracingTheNextWeek](http://raytracing.github.io/books/RayTracingTheNextWeek.html)，[Asixa/ALightGPU](https://github.com/Asixa/ALightGPU).


#ifndef GPU_RAY_TRACER_IMAGE_HPP_
#define GPU_RAY_TRACER_IMAGE_HPP_

template <typename T>
auto make_image(int width, int height)
{
    return [data = new T[width * height], width](int h, int w) mutable -> T & {
        return data[h * width + w];
    };
}

#endif

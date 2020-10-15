#pragma once

template <typename T>
auto make_image(int width, int height)
{
    return [data = new T[width * height], width](int h, int w) mutable -> T & {
        return data[h * width + w];
    };
}

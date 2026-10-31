# Parallelized Prewitt Operator (CUDA)

Edge detection on grayscale images with the 3×3 Prewitt operator, implemented
as a sequential C++ baseline and as a CUDA kernel, with execution times
compared.

## How it works

- **Sequential** (`sequential.c`, C++): convolves every interior pixel with
  the horizontal and vertical Prewitt kernels, then writes the gradient
  magnitude clamped to 255.
- **CUDA** (`parallel.cu`): one GPU thread per pixel on a 2D grid of 16×16
  blocks. The kernels and image dimensions live in `__constant__` memory.
  Kernel time is measured with CUDA events, and end-to-end time (kernel and
  device-to-host copy) with `gettimeofday`.
- **Block-size sweep** (`blocksizeOpt.cu`): times the kernel across block
  sizes on the high-resolution image and plots the results.

## Running

The notebook targets **Google Colab with a GPU runtime**. Open
`PrewittOperator.ipynb` and run the cells in order: they install OpenCV,
write each source file with `%%writefile`, then compile and run it:

```bash
g++  -o sequential sequential.c $(pkg-config --cflags --libs opencv4)
nvcc -o parallel   parallel.cu  $(pkg-config --cflags --libs opencv4)
```

Input images: `Flower.jpg` (small) and `FlowerHighResolution.jpg`
(3872×2592).

## Known limitations

- The vertical-gradient sum accumulates into the horizontal one (`GxSum`
  instead of `GySum`), so the output is |Gx + Gy| rather than
  √(Gx² + Gy²).
- The block-size sweep includes sizes above CUDA's 1,024-threads-per-block
  limit. Those launches fail silently, so their timings are not meaningful.
- The sequential and CUDA versions run on different images, so their times
  are not directly comparable.

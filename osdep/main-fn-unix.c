#include "main-fn.h"

// <stdlib.h> first: __GLIBC__ is only defined once a glibc header has
// been included, and main-fn.h includes none — without this the guard
// below is silently false and the mallopt calls compile to nothing.
#include <stdlib.h>
#include <limits.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

int main(int argc, char *argv[])
{
    // Portable-build RAM-ratchet fix (replaces exporting MALLOC_ARENA_MAX=2
    // from the launcher): bound per-thread arena churn so the arenas created by
    // the per-file demux/segmented-http threads are REUSED rather than
    // multiplying, so RSS plateaus over a long playlist instead of climbing by
    // roughly one arena per file.
    //
    // Deliberately does NOT set M_TRIM_THRESHOLD: doing so disables the dynamic
    // mmap-threshold adaptation in glibc, so every large video buffer is mmapped
    // and munmapped on free. Because libplacebo imports host pointers into the
    // GPU, unmapping a buffer the GPU still holds oopses the kernel inside the
    // GPU driver (amdgpu_hmm_invalidate_gfx). See apply_malloc_tuning.sh.
    #if defined(__GLIBC__)
    mallopt(M_ARENA_MAX, 2);
    // Never hand memory back to the kernel. libplacebo (>= ~v7.365) imports host
    // pointers straight into the GPU for frame uploads, so a page glibc unmaps may
    // still be registered with the GPU; the driver then oopses in its MMU-notifier
    // callback (amdgpu_hmm_invalidate_gfx) and wedges the machine in-kernel.
    //   - M_MMAP_THRESHOLD at its ceiling keeps frame-sized buffers on the heap
    //     instead of in mmap segments that get munmapped on free.
    //   - M_TRIM_THRESHOLD huge stops the heap itself being trimmed back.
    // Cost: RSS plateaus rather than shrinking. That is the correct price.
    mallopt(M_MMAP_THRESHOLD, 32 * 1024 * 1024);
    mallopt(M_TRIM_THRESHOLD, INT_MAX);
    #endif
    return mpv_main(argc, argv);
}

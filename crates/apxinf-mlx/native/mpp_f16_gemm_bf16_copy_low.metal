// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// High half prologue (128 threads): copy the low half's output into columns [0, NW).
    {
        const int r = int(thread_index_in_threadgroup) / 8;
        const int c = int(threadgroup_position_in_grid.x) * 64 +
                      (int(thread_index_in_threadgroup) % 8) * 8;
        const int base = int(threadgroup_position_in_grid.y) * 64;
        for (int i = r; i < 64; i += 16) {
            const int row = base + i;
            if (row < T) {
                *reinterpret_cast<device uint4*>(out + row * N + c) =
                    *reinterpret_cast<const device uint4*>(lo + row * NW + c);
            }
        }
    }

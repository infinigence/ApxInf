// Adapted from EngineTailor campaign/clef-flash9b-m1pro 4030047b (C14). Copyright 2026 Haiyan Qin. MIT; see NOTICE.
// s28 high half prologue: copy the low half's output into columns [0, NW).
    {
        const int r = int(thread_index_in_threadgroup) / 8;
        const int c = int(threadgroup_position_in_grid.x) * 64 +
                      (int(thread_index_in_threadgroup) % 8) * 8;
        const int base = int(threadgroup_position_in_grid.y) * 64;
        for (int i = r; i < 64; i += 8) {
            const int row = base + i;
            if (row < T) {
                *reinterpret_cast<device uint4*>(out + row * N + c) =
                    *reinterpret_cast<const device uint4*>(lo + row * NW + c);
            }
        }
    }

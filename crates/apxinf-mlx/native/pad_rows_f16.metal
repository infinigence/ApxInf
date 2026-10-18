// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// Cast T rows of width D to FP16 and zero the padded rows.

    const uint i = thread_position_in_grid.x;
    out[i] = i < T * D ? half(float(h[i])) : half(0);

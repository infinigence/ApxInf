// Adapted from EngineTailor feat/clef-g20-engine 6edc065c (g20). Copyright 2026 Haiyan Qin. MIT; see NOTICE.
// Cast T rows of width D to FP16 and zero the padded rows.

    const uint i = thread_position_in_grid.x;
    out[i] = i < T * D ? half(float(h[i])) : half(0);

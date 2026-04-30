def trace_candidate(candidate, digits=4):
    return {
        "range_bin": int(candidate.range_bin),
        "doppler_bin": int(candidate.doppler_bin),
        "angle_bin": int(candidate.angle_bin),
        "range_m": round(float(candidate.range_m), digits),
        "angle_deg": round(float(candidate.angle_deg), 3),
        "x_m": round(float(candidate.x_m), digits),
        "y_m": round(float(candidate.y_m), digits),
        "rdi_peak": round(float(candidate.rdi_peak), digits),
        "rai_peak": round(float(candidate.rai_peak), digits),
        "score": round(float(candidate.score), digits),
    }


def trace_candidates(candidates, limit=12):
    return [trace_candidate(candidate) for candidate in list(candidates or [])[: int(limit)]]


def trace_reject(reject_reasons, reason):
    reject_reasons[reason] = int(reject_reasons.get(reason, 0)) + 1


def compute_TEW(projection_lower, projection_upper, width_lower, width_upper, width_peak):
    return (projection_lower/width_lower + projection_upper/width_upper)*width_peak / 2

def compute_TEW_Zero_Upper_Scatter(projection_lower, projection_upper, width_lower, width_upper, width_peak):
    return (projection_lower/width_lower)*width_peak / 2
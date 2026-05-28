def recommend_grid_params(ncell, nprocs):
    """
    ncell: tuple (nx, ny, nz)
    nprocs: int, number of MPI ranks

    Returns (best_blocking_factor, best_max_grid_size, summary)
    """
    import math

    nx, ny, nz = ncell

    # Candidate grid sizes and blocking factors
    candidates = [8, 12, 16, 24, 32]
    blocking_candidates = [4, 8, 16]

    best_score = float('inf')
    best = None

    for bf in blocking_candidates:
        for mgs in candidates:
            if mgs < bf:
                continue  # AMReX requires this

            # Number of boxes in each dimension
            bx = math.ceil(nx / mgs)
            by = math.ceil(ny / mgs)
            bz = math.ceil(nz / mgs)

            nboxes = bx * by * bz

            # Target is: about 2–4 boxes per processor
            target_boxes_min = 2 * nprocs
            target_boxes_max = 4 * nprocs

            # Penalize deviations
            if nboxes < nprocs:
                score = 1e9  # too few boxes
            else:
                # score based on deviation from target box count
                if nboxes < target_boxes_min:
                    score = target_boxes_min - nboxes
                elif nboxes > target_boxes_max:
                    score = (nboxes - target_boxes_max) * 0.5
                else:
                    score = 0  # ideal range

            # Also penalize tiny boxes (overhead)
            cells_per_box = (nx/ bx) * (ny/ by) * (nz/ bz)
            if cells_per_box < 2000:
                score += 1000  # small penalty

            if score < best_score:
                best_score = score
                best = (bf, mgs, nboxes, int(cells_per_box))

    bf, mgs, nboxes, cpb = best
    summary = (
        f"Recommended blocking_factor = {bf}\n"
        f"Recommended max_grid_size   = {mgs}\n"
        f"Grid boxes produced = {nboxes}\n"
        f"Average cells per box = {cpb}\n"
        f"Boxes per rank = {nboxes / nprocs:.2f}"
    )

    return bf, mgs, summary


# Example usage:
if __name__ == "__main__":
    ncell = (128, 64, 128)
    nprocs = 256
    bf, mgs, summary = recommend_grid_params(ncell, nprocs)
    print(summary)
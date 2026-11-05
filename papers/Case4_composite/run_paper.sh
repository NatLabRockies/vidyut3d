#!/usr/bin/env bash
# Case 4 (composite geometry): the four wall treatments of the paper's
# convergence figure, each on grids 32 ... 512, into runs/<config>/<n>/.
#
#   fixes_off     all safeguards off, offset fit on (as in the paper runs)
#   fixes_on      safeguards on, wall value at the foot of the normal
#   gec3d_setting isotropic 1/d coefficient (the setting of the 3D GEC run)
#   ray_perface   ray-hit wall value + per-face wall lookup (inputs2d default,
#                 the configuration the paper recommends)
#
# Every switch is spelled out so that a configuration does not depend on the
# defaults in inputs2d. The log line "IB reconstruction: ..." records the
# reconstruction and prob.corner_fallback of each run.
#
# Usage: ./run_paper.sh [config ...]     (default: all four)
# Then:  python3 get_error.py runs/ray_perface
#        python3 convergence_figure_paper.py
set -e
here=$(cd "$(dirname "$0")" && pwd)
RES=${RES:-"32 64 128 256 512"}
common="vidyut.freeze_etemp=1"
cfgs=("$@"); [ ${#cfgs[@]} -eq 0 ] && cfgs=(fixes_off fixes_on gec3d_setting ray_perface)
for c in "${cfgs[@]}"; do
    case $c in
        fixes_off)     x="prob.corner_fallback=0 prob.d_floor_frac=0 prob.face_fallback=0 prob.proj_switch=0 prob.perface_fallback=0 prob.ray_dirichlet=0" ;;
        fixes_on)      x="prob.corner_fallback=1 prob.d_floor_frac=0.5 prob.face_fallback=1 prob.proj_switch=0 prob.perface_fallback=0 prob.ray_dirichlet=0" ;;
        gec3d_setting) x="prob.corner_fallback=1 prob.d_floor_frac=0.5 prob.face_fallback=1 prob.proj_switch=1 prob.perface_fallback=0 prob.ray_dirichlet=0" ;;
        ray_perface)   x="prob.corner_fallback=1 prob.d_floor_frac=0.5 prob.face_fallback=1 prob.proj_switch=0 prob.perface_fallback=1 prob.ray_dirichlet=1" ;;
        *) echo "unknown config $c"; exit 1 ;;
    esac
    "$here/run_config.sh" "$c" "$common $x" "$RES"
done

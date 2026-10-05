"""
Simulate the SCExAO pupil

This is the 2024 SCExAO pupil script (miles_pupil_2024.py) edited to describe
the pupil as seen on the internal calibration source in 2026. PROVISIONAL: the
changes were fitted to focal-plane images taken on the internal source on
three dates in 2026 and agree with one pupil-camera image; they have not been
confirmed by the instrument team.

Unchanged from 2024: the primary, the secondary and the four spiders.

Changed:
    mask          ONE dead-actuator mask, 0.40 m across, at (1.74, 1.40) m,
                  on the first-quadrant spider. The 2024 script has two masks
                  of 0.632 m and a thin support spider for the second; that
                  second mask and its spider are gone.
    defect        NEW. One opaque disc, 0.25 m across, at (2.30, 0.14) m, in
                  the open pupil. The images fit a ~0.7 um optical-path bump
                  (a stuck actuator) equally well; this script is amplitude
                  only.
    illumination  NEW, OPTIONAL, off by default. On the internal source the
                  pupil is lit by a gaussian beam: amplitude
                  exp(-q/2 * |r - r0|^2) with q = 0.073 /m^2 (1/e^2 intensity
                  radius 5.2 m on a 3.9 m pupil radius). r0 = (0.44, 0.79) m
                  on one date and about 0.4 m further out on two others.
                  This belongs to the internal source's beam, not to the
                  pupil: it changed between dates and presumably does not
                  apply on sky. Pass illum_q=ILLUM_Q to switch it on, and
                  treat r0 as a nuisance parameter, not a constant.

Focal-plane images fix the positions only up to a half turn of all features
together. With actuators=False, defect=False the output is identical to the
2024 script's with actuators=False.
"""
from argparse import ArgumentParser

import hcipy as hp
import numpy as np
from astropy.io import fits

__all__ = ["generate_pupil"]

## constants
PUPIL_DIAMETER = 7.92  # m
OBSTRUCTION_DIAMETER = 2.403  # m
INNER_RATIO = OBSTRUCTION_DIAMETER / PUPIL_DIAMETER
SPIDER_WIDTH = 0.1735  # m
SPIDER_OFFSET = 0.639  # m, spider intersection offset
SPIDER_ANGLE = 51.75  # deg
# PUPIL_ANGLE = -39  # deg
ACTUATOR_DIAMETER = 0.40  # m
ACTUATOR_OFFSET = (1.74, 1.40)  # (x, y), m
DEFECT_DIAMETER = 0.25  # m
DEFECT_OFFSET = (2.30, 0.14)  # (x, y), m
ILLUM_Q = 0.073  # 1/m^2, internal source only: amplitude = exp(-ILLUM_Q / 2 * |r - r0|^2)
ILLUM_OFFSET = (0.44, 0.79)  # r0 (x, y), m, internal source only

## command-line arg parsing
parser = ArgumentParser()
parser.add_argument("filename", help="output FITS filename")
parser.add_argument("-n",
                    type=int,
                    default=256,
                    help="size of pixel grid. Defaults to %(default)d")
parser.add_argument(
    "-o",
    "--outer",
    default=1,
    type=float,
    help=
    "Outer diameter of pupil as a fraction of the true pupil diameter. Defaults to %(default).02f",
)
parser.add_argument(
    "-i",
    "--inner",
    default=INNER_RATIO,
    type=float,
    help=
    "Inner diameter of pupil as a fraction of the true pupil diameter. Defaults to %(default).02f",
)
parser.add_argument(
    "-s",
    "--scale",
    default=1,
    type=float,
    help=
    "Scale factor for spiders and bad DM actuator masks as a fraction of their true size. Defaults to %(default).02f",
)
parser.add_argument(
    "-r",
    "--rotate",
    default=0,
    type=float,
    help="Pupil offset angle in degrees. Defaults to %(default).01f",
)
parser.add_argument(
    "-f",
    "--oversample-factor",
    default=8,
    type=int,
    help=
    "Oversample factor for supersampled evaluation of the pupil grid. Defaults to %(default)d",
)
parser.add_argument(
    "--no-spiders",
    action="store_false",
    dest="spiders",
    help="Don't create spider obstructions.",
)
parser.add_argument(
    "--no-actuators",
    action="store_false",
    dest="actuators",
    help="Don't create bad actuator mask obstructions.",
)

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


def field_combine(field1, field2):
    return lambda grid: field1(grid) * field2(grid)


def generate_pupil(
    n: int = 256,
    outer: float = 1,
    inner: float = INNER_RATIO,
    scale: float = 1,
    angle: float = 0,
    oversample: int = 8,
    spiders: bool = True,
    actuators: bool = True,
    pupil_grid = None,
    defect: bool = True,
    illum_q: float = 0,
    illum_offset = ILLUM_OFFSET,
):
    f"""
    Generate a SCExAO pupil parametrically.

    Parameters
    ----------
    n : int, optional
        Grid size in pixels. Default is 256
    outer : float, optional
        Outer pupil diameter as a fraction of the true diameter. Default is 1.0
    inner : float, optional
        Diameter of central obstruction as a fraction of the true diameter. Default is {INNER_RATIO:.03f}
    scale : float, optional
        Scale factor for over-sizing spiders and actuator masks. Default is 1.0
    angle : float, optional
        Pupil rotation angle, in degrees. Default is 0
    oversample : int, optional
        Oversample factor for supersampling the pupil grid. Default is 8
    spiders : bool, optional
        Add spiders to pupil. Default is True
    actuators : bool, optional
        Add bad actuator mask. Default is True
    defect : bool, optional
        Add the small opaque defect in the open pupil. Default is True
    illum_q : float, optional
        Gaussian illumination parameter in 1/m^2. Default is 0, uniform illumination; the internal source fitted {ILLUM_Q}
    illum_offset : (float, float), optional
        Centre (x, y) of the gaussian illumination in meters. Default is {ILLUM_OFFSET}

    Notes
    -----
    The smallest element in this pupil is the spiders, which are approximately {SPIDER_WIDTH*1e3:.1f} mm wide. This is about 2.2\% of the telescope diameter, which means you need to have a miinimum of ~46 pixels across the aperture to sample this element.

    With gaussian illumination the output is an amplitude, not a binary transmission mask.

    """
    pupil_diameter = PUPIL_DIAMETER * outer
    # make grid over full diameter so undersized pupils look undersized
    max_diam = PUPIL_DIAMETER if outer <= 1 else pupil_diameter

    if pupil_grid is None:
        grid = hp.make_pupil_grid(n, diameter=max_diam)
    else:
        grid = pupil_grid

    # This sets us up with M1+M2, just need to add spiders and DM masks
    pupil_field = hp.make_obstructed_circular_aperture(pupil_diameter, inner)

    # add spiders to field generator
    if spiders:
        spider_width = SPIDER_WIDTH * scale
        sint = np.sin(np.deg2rad(SPIDER_ANGLE))
        cost = np.cos(np.deg2rad(SPIDER_ANGLE))

        # spider in quadrant 1
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (SPIDER_OFFSET, 0),  # start
                (cost * pupil_diameter + SPIDER_OFFSET,
                 sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 2
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (-SPIDER_OFFSET, 0),  # start
                (-cost * pupil_diameter - SPIDER_OFFSET,
                 sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 3
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (-SPIDER_OFFSET, 0),  # start
                (-cost * pupil_diameter - SPIDER_OFFSET,
                 -sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 4
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (SPIDER_OFFSET, 0),  # start
                (cost * pupil_diameter + SPIDER_OFFSET,
                 -sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )

    # add actuator masks to field generator
    if actuators:
        # circular mask
        actuator_diameter = ACTUATOR_DIAMETER * scale
        actuator_mask_1 = hp.make_obstruction(
            hp.circular_aperture(diameter=actuator_diameter,
                                 center=ACTUATOR_OFFSET))
        pupil_field = field_combine(pupil_field, actuator_mask_1)

    # add defect to field generator
    if defect:
        defect_mask = hp.make_obstruction(
            hp.circular_aperture(diameter=DEFECT_DIAMETER,
                                 center=DEFECT_OFFSET))
        pupil_field = field_combine(pupil_field, defect_mask)

    rotated_pupil_field = hp.make_rotated_aperture(pupil_field,
                                                   np.deg2rad(angle))

    pupil = hp.evaluate_supersampled(rotated_pupil_field, grid, oversample)

    # gaussian illumination, evaluated at the pixel centers
    if illum_q:
        sint = np.sin(np.deg2rad(angle))
        cost = np.cos(np.deg2rad(angle))
        x0 = cost * illum_offset[0] - sint * illum_offset[1]
        y0 = sint * illum_offset[0] + cost * illum_offset[1]
        pupil = pupil * np.exp(-0.5 * illum_q *
                               ((grid.x - x0)**2 + (grid.y - y0)**2))
    return pupil


def main():
    args = parser.parse_args()
    pupil = generate_pupil(
        args.n,
        outer=args.outer,
        inner=args.inner,
        scale=args.scale,
        angle=args.rotate,
        spiders=args.spiders,
        actuators=args.actuators,
    )
    hdr = fits.Header()
    hdr["PUPDIAM"] = PUPIL_DIAMETER * args.outer, "m, diameter of pupil"
    hdr["OBSTDIAM"] = (
        PUPIL_DIAMETER * args.inner,
        "m, diameter of secondary obstruction",
    )
    if args.spiders:
        hdr["SPIDWDTH"] = SPIDER_WIDTH * args.scale, "m, width of support spiders"
        hdr["SPIDOFF"] = SPIDER_OFFSET, "m, offset of support spiders"
        hdr["SPIDANG"] = args.rotate, "m, angle of spiders with respect to the pupil"
    if args.actuators:
        hdr["ACTDIAM"] = ACTUATOR_DIAMETER * args.scale, "m, diameter of actuator masks"
    fits.writeto(args.filename, pupil, overwrite=True)


if __name__ == "__main__":
    main()
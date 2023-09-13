import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
import matplotlib.patches as ptc
from time import time

from track_data import center_sample_points
from track_data import left_boundary_points
from track_data import right_boundary_points

center_points = center_sample_points.center_points
left_boundary_points = left_boundary_points.left_boundary_points
right_boundary_points = right_boundary_points.right_boundary_points


def simulate(
    cat_states,
    cat_controls,
    t,
    step_horizon,
    N,
    reference,
    rob_diam,
    cat_real_states,
    save=False,
):
    def init():
        return (path, horizon, circle, direction)

    def animate(i):
        # get variables
        x = cat_real_states[i][0]
        y = cat_real_states[i][1]
        th = cat_real_states[i][2]

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        x_new = cat_states[0, :, i]
        y_new = cat_states[1, :, i]
        horizon.set_data(x_new, y_new)

        # update current_state
        circle.set_center((x, y))

        x_dir = x+(rob_diam/2)*cos(th)
        y_dir = y+(rob_diam/2)*sin(th)
        direction.set_data((x,x_dir), (y,y_dir))

        # pc = find_projection_to_center((x, y), center_points)

        # update projection to center line
        # pc_scatter.set_offsets(pc)

        # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)

        return (path, horizon, circle,direction)

    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("With Safety Filter")

    # create lines:
    #   path
    (path,) = ax.plot([], [], "k", linewidth=2)
    #   horizon
    (horizon,) = ax.plot([], [], "x-g", alpha=0.5)

    (direction, ) = ax.plot([],[], "#FF0000" )

    # init axis
    plt.axis([-2, 20, -2, 20])

    # Track

        # Track
    for i in range(len(center_points)):
        if i != len(center_points) - 1:
            xc, yc = center_points[i]
            xc_next, yc_next = center_points[i + 1]
            plt.plot([xc, xc_next], [yc, yc_next], "#D3D3D3")

    for i in range(len(left_boundary_points)):
        if i != len(left_boundary_points) - 1:
            xc, yc = left_boundary_points[i]
            xc_next, yc_next = left_boundary_points[i + 1]
            plt.plot([xc, xc_next], [yc, yc_next], "#000000")

    for i in range(len(right_boundary_points)):
        if i != len(right_boundary_points) - 1:
            xc, yc = right_boundary_points[i]
            xc_next, yc_next = right_boundary_points[i + 1]
            plt.plot([xc, xc_next], [yc, yc_next], "#000000")

    # Projection to the center line
    # pc_scatter = plt.scatter(0, 0, color="red", marker="o")

    # Intiial cirlce in dashed outline line style
    circle = ptc.Circle(
        (reference[0], reference[1]),
        rob_diam / 2,
        edgecolor="r",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(circle)

    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=len(t),
        interval=step_horizon * 100,
        blit=True,
        repeat=True,
    )
    plt.show()

    if save == True:
        sim.save("./animation" + str(time()) + ".gif", writer="ffmpeg", fps=30)

    return

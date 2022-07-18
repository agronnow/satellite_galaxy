from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from satellite_init.orbit.satellite_orbit import satellite_orbit

def animate_orbit(xlim, ylim, zlim, interval_data, interval_frame, mass=1.25e12, name="Draco", evol=True, save=True):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    timetxt = ax.text2D(0.85, 0.85, '', transform=ax.transAxes, fontsize=14)

    def update(num, data, orbit, time):
        orbit.set_data(data[0, :2, :num]) 
        orbit.set_3d_properties(data[0, 2, :num])
        timetxt.set_text("Time: %.1f Gyr" % time[num])

    orbit=satellite_orbit(mass=mass,evol=evol)
    orbit.calc_orbit(name)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_zlabel('z [kpc]')

    fig.set_size_inches(10,10)

    ax.scatter(0,0,0,c='C1')
    ax.text(0,0,0,'GC',fontsize=15)
    orb,=ax.plot(orbit.rads[:1,0,0].flatten(),orbit.rads[:1,1,0].flatten(),orbit.rads[:1,2,0].flatten())
    ani=animation.FuncAnimation(fig, update, range(2,len(orbit.ts),interval_data), fargs=(orbit.rads.T[:,:,::-1],orb, orbit.ts[::-1]), interval=interval_frame, blit=False)
    
    if save:
        FFwriter = animation.FFMpegWriter(fps=30,codec="libx264",bitrate=2000)
        ani.save('animation.mp4', writer = FFwriter, dpi=100)
        plt.show()
    else:
        plt.show()

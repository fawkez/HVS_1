# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:40:55 2020

@author: Sill Verberne
"""

#main program file
import numpy as np
import functions as fn
import pars
import matplotlib.pyplot as plt
import time as tm
import matplotlib.lines as mlines
#import ITFAstyle #This is a file I used that makes the plots look nice =)
from matplotlib.animation import FuncAnimation
import astropy.units as u
# import gala.potential as gp
# import gala.dynamics as gd
import glob
from PIL import Image

start_time = tm.time()

#Parameters ----
epsilon = 0.000000001 #allowed error with the timestep used
tfinal = 1*0.7*1e9 #end time (seconds)
plot = True #Set true if you want to see the orbits of the simulated pebbles
alpha = 1.5  #multiplication of dt when the error is very small
animation = False
#---------------

def simulation():
    st_time = tm.time()

    #assign the initial positions, velocities, masses
    xarr, varr, marr = fn.init_3body()

    #declarations
    time = 0 #start time
    dt = 0.0001*pars.yr #initial timestep guess
    times = [0]
    #start main loop
    if plot:
        time_ = [0]
        #compute the total energy, used for verification
        etot0 = fn.e_tot(xarr, varr, marr)
        etot_ = [etot0]
        error_ = [0]
        x_0 = [xarr[0,0]]
        y_0 = [xarr[1,0]]
        x_1 = [xarr[0,1]]
        y_1 = [xarr[1,1]]
        x_2 = [xarr[0,2]]
        y_2 = [xarr[1,2]]
    V1 = np.sqrt(varr[:,1][0]**2 + varr[:,1][1]**2)/10**5
    V2 = np.sqrt(varr[:,2][0]**2 + varr[:,2][1]**2)/10**5
    print ('V1 = ', V1)
    print ('V2 = ', V2)
    Epot1 = - pars.gN * marr[0] * marr[1] / np.sqrt(xarr[:,1][0]**2+xarr[:,1][1]**2)
    Epot2 = - pars.gN * marr[0] * marr[2] / np.sqrt(xarr[:,2][0]**2+xarr[:,2][1]**2)
    Etot1 = Epot1 + 0.5 * pars.star1_mass * (V1*10**5)**2
    Etot2 = Epot2 + 0.5 * pars.star2_mass * (V2*10**5)**2
    print ('Etot star 1 = ', Etot1)
    print ('Etot star 2 = ', Etot2)

    while time<tfinal:
        # calculate forces, update positions and velocities this involves
        # calling the function that computes the accelerations
        err = 1
        while err > epsilon: #Only go on if the error is sufficiently small
            #Check for dt
            xarr_dt, varr_dt = fn.Runge_Kutta(xarr, varr, marr, dt)
            varr_dt_copy = varr_dt.copy()
            xarr_dt_copy = xarr_dt.copy()
            #Check for dt/2
            xarr_dt2_, varr_dt2_ = fn.Runge_Kutta(xarr, varr, marr, dt/2)
            xarr_dt2, varr_dt2 = fn.Runge_Kutta(xarr_dt2_, varr_dt2_, marr, \
                                                dt/2)
            varr_dt2_copy = varr_dt2.copy()
            xarr_dt2_copy = xarr_dt2.copy()
            #Determine the error
            err = max(np.sum(abs((varr_dt_copy - varr_dt2_copy)/varr_dt_copy),\
                             axis=0))
            if err > epsilon: #if the error is to large: dt-> dt/2
                dt *= 0.5
                continue
            elif err < 0.1*epsilon and dt < 20000: #if the error is very small: dt -> alpha*dt
                dt *= alpha
                continue
            else: #update if the error is sufficiently small
                xarr = xarr_dt_copy
                varr = varr_dt_copy
                time+= dt
                # print (dt)

        if plot:
            x_0.append(xarr[0,0])
            x_1.append(xarr[0,1])
            x_2.append(xarr[0,2])
            y_0.append(xarr[1,0])
            y_1.append(xarr[1,1])
            y_2.append(xarr[1,2])
            times.append(time)

            time_.append(time)
            etot = fn.e_tot(xarr, varr, marr)
            etot_.append(etot)
            error = (etot - etot0)/etot0
            error_.append(error)


        

    print ('Runtime is ' + str(tm.time() - start_time) + ' seconds')
    V1 = np.sqrt(varr[:,1][0]**2 + varr[:,1][1]**2)/10**5
    V2 = np.sqrt(varr[:,2][0]**2 + varr[:,2][1]**2)/10**5
    print ('V1 = ', V1)
    print ('V2 = ', V2)
    Epot1 = - pars.gN * marr[0] * marr[1] / np.sqrt(xarr[:,1][0]**2+xarr[:,1][1]**2)
    Epot2 = - pars.gN * marr[0] * marr[2] / np.sqrt(xarr[:,2][0]**2+xarr[:,2][1]**2)
    Etot1 = Epot1 + 0.5 * pars.star1_mass * (V1*10**5)**2
    Etot2 = Epot2 + 0.5 * pars.star2_mass * (V2*10**5)**2
    Emax = max([Etot1, Etot2])
    print ('Etot star 1 = ', Etot1)
    print ('Etot star 2 = ', Etot2)

    # Determine if the star is unbound (i.e. an HVS)
    # mw = gp.MilkyWayPotential()
    # if Emax == Etot1 and Emax > 0:
    #     w0 = gd.PhaseSpacePosition(pos=[x_1[-1], y_1[-1], 0] * u.cm,
    #                        vel=[varr[:,1][0], varr[:,1][1], 0] * u.cm/u.s)
    #     orbit = mw.integrate_orbit(w0, dt=0.001*u.kyr, t1=0, t2=10*u.Myr)
    #     orbit.plot(['x', 'y'])
    #     plt.scatter(0,0, marker='x', color='red')
    #     plt.title('10 Myr integration')
    #     # plt.savefig('Integrated_orbit.png')
    #     # plt.show()
    #     print ('Velocity parameters after integration: ', orbit.vel_components())
    # if Emax == Etot2 and Emax > 0:
    #     w0 = gd.PhaseSpacePosition(pos=[x_2[-1], y_2[-1], 0] * u.cm,
    #                        vel=[varr[:,2][0], varr[:,2][1], 0] * u.cm/u.s)
    #     orbit = mw.integrate_orbit(w0, dt=0.001*u.kyr, t1=0, t2=10*u.Myr)
    #     orbit.plot(['x', 'y'])
    #     plt.scatter(0,0, marker='x', color='red')
    #     plt.title('10 Myr integration')
    #     # plt.savefig('Integrated_orbit.png')
    #     # plt.show()
    #     print ('Velocity parameters after integration: ', orbit.v_x[-1].to(u.km/u.s), orbit.v_y[-1].to(u.km/u.s), orbit.v_z[-1].to(u.km/u.s))
    #     print ('Velocity parameters after integration: ', orbit.v_x[1].to(u.km/u.s), orbit.v_y[1].to(u.km/u.s), orbit.v_z[1].to(u.km/u.s))

    print (len(x_0))
    #Only use 1 in 10 data points
    x_0 = np.array(x_0)#[::10]
    x_1 = np.array(x_1)#[::10]
    x_2 = np.array(x_2)#[::10]
    y_0 = np.array(y_0)#[::10]
    y_1 = np.array(y_1)#[::10]
    y_2 = np.array(y_2)#[::10]
    t_per = np.argmin((x_0-x_1)**2 + (y_0+y_1)**2)

        # Save data to a CSV file
    data = np.column_stack((times, x_0, y_0, x_1, y_1, x_2, y_2))
    np.savetxt('simulation_data.csv', data, delimiter=',', 
                header='time,x_0,y_0,x_1,y_1,x_2,y_2', comments='')

    if animation:
        last_t = 0
        # for i in range(50, len(x_0)):
        #     my_dpi=96
        #     plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
        #     print (i)
        #     t_scale = 300000
        #     dist = np.sqrt((x_1[i]/pars.au)**2 + (y_1[i]/pars.au)**2)
        #     a_line = 500000/570
        #     b_line = 2000 - 30*a_line
        #     def fun(dist):
        #         return a_line*dist + b_line
        #     t_scale = fun(dist)
        #     if times[i] - last_t < t_scale:
        #         print ('skipped')
        #         continue
        #     range_x_min = np.amin([x_1[i-50:int(i+1)], x_2[i-50:int(i+1)], x_0[i-50:int(i+1)]])/pars.au
        #     range_x_max = np.amax([x_1[i-50:int(i+1)], x_2[i-50:int(i+1)], x_0[i-50:int(i+1)]])/pars.au
        #     range_y_max = np.amax([y_1[i-50:int(i+1)], y_2[i-50:int(i+1)], y_0[i-50:int(i+1)]])/pars.au
        #     range_y_min = np.amin([y_1[i-50:int(i+1)], y_2[i-50:int(i+1)], y_0[i-50:int(i+1)]])/pars.au
        #     width_max = max([abs(range_y_max-range_y_min), abs(range_x_max-range_x_min)])
        #     # print (range_x_min, range_x_max, width_max)
        #     # print (-0.6*width_max + (range_x_max+range_x_min)/2, 0.6*width_max + (range_x_max+range_x_min)/2)
        #     # print (g)
        #     plt.xlim(-0.6*width_max + (range_x_max+range_x_min)/2, 0.6*width_max + (range_x_max+range_x_min)/2)
        #     plt.ylim(-0.6*width_max + (range_y_max+range_y_min)/2, 0.6*width_max + (range_y_max+range_y_min)/2)
        #     plt.scatter(x_1[i]/pars.au, y_1[i]/pars.au, color='blue', s=25, label='Star 1')
        #     plt.scatter(x_2[i]/pars.au, y_2[i]/pars.au, color='red', s=25, label='Star 2')
        #     plt.plot(x_1[i-50:int(i+1)]/pars.au, y_1[i-50:int(i+1)]/pars.au, color='blue')
        #     plt.plot(x_2[i-50:int(i+1)]/pars.au, y_2[i-50:int(i+1)]/pars.au, color='red')
        #     plt.scatter(x_0[i]/pars.au, y_0[i]/pars.au, color='black', s=100, label='Black hole')
        #     # plt.gca().set_aspect('equal', adjustable='box')
        #     times_day = str(round(float(times[i]-times[t_per])/(3600*24), 2))
        #     plt.annotate('time=%s days' %times_day, xy=(0.02,0.02), xycoords='figure fraction')
        #     plt.annotate('a_bin=%.2f au' %np.sqrt(((x_1[i]-x_2[i])/pars.au)**2+((y_1[i]-y_2[i])/pars.au)**2), xy=(0.9,0.02), xycoords='figure fraction')
        #     plt.ylabel('y [AU]')
        #     plt.xlabel('x [AU]')
        #     plt.gca().yaxis.set_label_coords(-0.15, 0.5)
        #     # plt.tight_layout(h_pad=0.05, w_pad=0.05)
        #     # plt.subplots_adjust(left=0, right=0.95, top=0.95, bottom=0)
        #     plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.05))
        #     plt.savefig('/home/sill/Documents/lame_gif/lame'+str(i)+'.JPG')
        #     last_t = times[i]
        # print (g)

        # def make_gif(frame_folder):
        #     list_ = list(glob.glob(f"{frame_folder}/*.JPG"))
        #     list_sort = [int(i[34:-4]) for i in list_]
        #     zipped_lists = zip(list_sort, list_)
        #     sorted_pairs = sorted(zipped_lists)
        #     tuples = zip(*sorted_pairs)
        #     list_, list_sort = [ list(tuple) for tuple in  tuples]
        #     print (list_sort)
        #     frames = [Image.open(image) for image in list_sort]
        #     frame_one = frames[0]
        #     frame_one.save("/home/sill/Documents/lame_gif/HVS_ejection.gif", format="GIF", append_images=frames,
        #                save_all=True, disposal=2, duration=50, loop=0, optimize=True)
        # make_gif('/home/sill/Documents/lame_gif/')

        # last_t = 0
        # from celluloid import Camera
        # camera = Camera(plt.figure())
        # for i in range(len(x_0)):
        #     print (i)
        #     if times[i] - last_t < 300000:
        #         print ('skipped')
        #         continue
        #     plt.scatter(x_0[i]/pars.au, y_0[i]/pars.au, color='black', s=15)
        #     plt.scatter(x_1[i]/pars.au, y_1[i]/pars.au, color='blue', s=25)
        #     plt.scatter(x_2[i]/pars.au, y_2[i]/pars.au, color='red', s=25)
        #     plt.plot(x_1[:i]/pars.au, y_1[:i]/pars.au, color='blue')
        #     plt.plot(x_2[:i]/pars.au, y_2[:i]/pars.au, color='red')
        #     # plt.xlim(-i, i)
        #     plt.gca().set_aspect('equal', adjustable='box')
        #     times_day = str(round(float(times[i])/(3600*24), 2))
        #     plt.annotate('time=%s days' %times_day, xy=(0,0), xycoords='figure fraction')
        #     plt.ylabel('au')
        #     plt.xlabel('au')
        #     # plt.autoscale()
        #     last_t = times[i]
        #     camera.snap()
        # anim = camera.animate(blit=True)
        # anim.save('HVS2.gif')

    if plot:
        fig, ax = plt.subplots(2,1, sharex=True, \
                               gridspec_kw={'height_ratios':[2,1]})
        fig.subplots_adjust(hspace=0)
        ax[0].set_title('Total energy conservation check')
        ax[0].plot(time_, etot_, label='Total energy')
        ax[1].plot(time_, error_, label='Error')
        ax[0].set_ylabel('Etot (erg)')
        ax[1].set_ylabel('Etotal error')
        ax[1].set_xlabel('Time (s)')

        plt.figure(figsize=(10,5))
        plt.scatter(np.array(x_0)/pars.au, np.array(y_0)/pars.au, color='black', label='Black hole')
        plt.plot(np.array(x_1)/pars.au, np.array(y_1)/pars.au, 'b', label='Star 1')
        plt.plot(np.array(x_2)/pars.au, np.array(y_2)/pars.au, 'r', label="Star 2")
        plt.xlabel('x [AU]')
        plt.ylabel('y [AU]')
        # plt.ylim(-1.1*pars.au, 1.1*pars.au)
        # plt.xlim(-1.1*pars.au, 1.1*pars.au)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        # plt.savefig('orbit.png', dpi=500)
        plt.show()


if __name__ == '__main__':
    simulation()

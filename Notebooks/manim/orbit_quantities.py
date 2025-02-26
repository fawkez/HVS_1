import numpy as np
import math
from numpy.linalg import norm
import astropy
from astropy.table import Table
import astropy.units as u

G=astropy.constants.G.to(u.Msun**-1 * u.AU**3 * u.yr**-2).value

def acc(m0, m1, m2, r_0, r_1, r_2):
    acc_0 = -G*m1*(r_0-r_1)/norm(r_0-r_1)**3 - G*m2*(r_0-r_2)/norm(r_0-r_2)**3
    acc_1 = -G*m0*(r_1-r_0)/norm(r_1-r_0)**3 - G*m2*(r_1-r_2)/norm(r_1-r_2)**3
    acc_2 = -G*m0*(r_2-r_0)/norm(r_2-r_0)**3 - G*m1*(r_2-r_1)/norm(r_2-r_1)**3
    
    return acc_0, acc_1, acc_2, 

def binary_E_L(m1, m2, r1x, r1y, r1z, v1x, v1y, v1z, r2x, r2y, r2z, v2x, v2y, v2z):
    mu = m1*m2/(m1+m2)
    rx = r2x-r1x
    ry = r2y-r1y
    rz = r2z-r1z
    vx = v2x-v1x
    vy = v2y-v1y
    vz = v2z-v1z
    E = 0.5*mu*(vx**2+vy**2+vz**2)-G*m1*m2/np.sqrt(rx**2+ry**2+rz**2)
    Lx = mu*(ry*vz-rz*vy)
    Ly = mu*(rz*vx-rx*vz)
    Lz = mu*(rx*vy-ry*vx)
    
    return E, Lx, Ly, Lz


def CM_r(m1, m2, r1x, r1y, r1z, r2x, r2y, r2z):
    rcx = (m1*r1x+m2*r2x)/(m1+m2)
    rcy = (m1*r1y+m2*r2y)/(m1+m2)
    rcz = (m1*r1z+m2*r2z)/(m1+m2)
    
    return rcx, rcy, rcz


def CM_v(m1, m2, v1x, v1y, v1z, v2x, v2y, v2z):
    vcx = (m1*v1x+m2*v2x)/(m1+m2)
    vcy = (m1*v1y+m2*v2y)/(m1+m2)
    vcz = (m1*v1z+m2*v2z)/(m1+m2)
    
    return vcx, vcy, vcz


def energy(m0, m1, m2, r0x, r0y, r0z, v0x, v0y, v0z, r1x, r1y, r1z, v1x, v1y, v1z, r2x, r2y, r2z, v2x, v2y, v2z):
    K0 = 0.5*m0*(v0x**2+v0y**2+v0z**2)
    U0 = -G*m0*m1/np.sqrt((r0x-r1x)**2+(r0y-r1y)**2+(r0z-r1z)**2)-G*m0*m2/np.sqrt((r0x-r2x)**2+(r0y-r2y)**2+(r0z-r2z)**2)
  
    K1 = 0.5*m1*(v1x**2+v1y**2+v1z**2)
    U1 = -G*m1*m0/np.sqrt((r1x-r0x)**2+(r1y-r0y)**2+(r1z-r0z)**2)-G*m1*m2/np.sqrt((r1x-r2x)**2+(r1y-r2y)**2+(r1z-r2z)**2)
  
    K2 = 0.5*m2*(v2x**2+v2y**2+v2z**2)
    U2 = -G*m2*m0/np.sqrt((r2x-r0x)**2+(r2y-r0y)**2+(r2z-r0z)**2)-G*m2*m1/np.sqrt((r2x-r1x)**2+(r2y-r1y)**2+(r2z-r1z)**2)

    Etot=K0+K1+K2+0.5*(U0+U1+U2)
  
    return Etot


def CM_distance(m0, m1, m2, r0x, r0y, r0z, r1x, r1y, r1z, r2x, r2y, r2z):
    rcx, rcy, rcz = CM_r(m1, m2, r1x, r1y, r1z, r2x, r2y, r2z)
    d = np.sqrt((r0x-rcx)**2+(r0y-rcy)**2+(r0z-rcz)**2)

    return d


def inertial_frame(m0, m1, m2, ab, eb, i, Omega, omega, phi, rp, ec, f):
    #Positions and velocities of binary CM and third body
    M=m0+m1+m2
    R=rp*(1+ec)/(1+ec*np.cos(f))*np.array([np.cos(f), np.sin(f), 0])
    r0=-(m1+m2)/M*R
    rc=+m0/M*R

    V=(G*M/(rp*(1+ec)))**0.5*np.array([-np.sin(f), ec+np.cos(f), 0])
    v0=-V*(m1+m2)/M
    vc=V*m0/(M)

    #Binary frame
    r=ab*(1-eb**2)/(1+eb*np.cos(phi))*np.array([np.cos(phi), np.sin(phi), 0])
    r1b=-(m2/(m1+m2))*r
    r2b=+(m1/(m1+m2))*r

    v=(G*(m1+m2)/(ab*(1-eb**2)))**0.5*np.array([-np.sin(phi), (np.cos(phi)+eb), 0])
    v1b=-(m2/(m1+m2))*v
    v2b=+(m1/(m1+m2))*v

    #Transformation to the inertial frame I,J,K
    I=np.array([np.cos(Omega)*np.cos(omega)-np.sin(Omega)*np.cos(i)*np.sin(omega), -np.cos(Omega)*np.sin(omega)-  np.sin(Omega)*np.cos(i)*np.cos(omega), np.sin(Omega)*np.sin(i)])
    J=np.array([np.sin(Omega)*np.cos(omega)+np.cos(Omega)*np.cos(i)*np.sin(omega), -np.sin(Omega)*np.sin(omega)+np.cos(Omega)*np.cos(i)*np.cos(omega), -np.cos(Omega)*np.sin(i)])
    K=np.array([np.sin(i)*np.sin(omega), np.sin(i)*np.cos(omega), np.cos(i)])

    #Positions and velocities in the inertial frame
    r1=rc+np.array([np.dot(r1b, I), np.dot(r1b, J), np.dot(r1b, K)])
    r2=rc+np.array([np.dot(r2b, I), np.dot(r2b, J), np.dot(r2b, K)])

    v1=vc+np.array([np.dot(v1b, I), np.dot(v1b, J), np.dot(v1b, K)])
    v2=vc+np.array([np.dot(v2b, I), np.dot(v2b, J), np.dot(v2b, K)])

    return r0, v0, r1, v1, r2, v2

def orb_parameters(m0, m1, m2, r0x, r0y, r0z, v0x, v0y, v0z, r1x, r1y, r1z, v1x, v1y, v1z, r2x, r2y, r2z, v2x, v2y, v2z):
    mu = m1*m2/(m1+m2)
  
    Eb, Lbx, Lby, Lbz = binary_E_L(m1, m2, r1x, r1y, r1z, v1x, v1y, v1z, r2x, r2y, r2z, v2x, v2y, v2z)
    ab=-G*m1*m2/(2*Eb)
    eb=np.sqrt(1+(2*Eb*(Lbx**2+Lby**2+Lbz**2))/(G**2*m1**2*m2**2*mu))
    c_phi=(ab*(1-eb**2)/(eb*np.sqrt((r2x-r1x)**2+(r2y-r1y)**2+(r2z-r1z)**2))-1/eb)
    s_phi=1/eb*(np.sqrt(1+2*eb*c_phi+eb**2))*((r2x-r1x)*(v2x-v1x)+(r2y-r1y)*(v2y-v1y)+(r2z-r1z)*(v2z-v1z))/(np.sqrt((r2x-r1x)**2+(r2y-r1y)**2+(r2z-r1z)**2)*np.sqrt((v2x-v1x)**2+(v2y-v1y)**2+(v2z-v1z)**2))
    phi=np.arctan2(s_phi,c_phi)

    rx=(r2x-r1x)/np.sqrt((r2x-r1x)**2+(r2y-r1y)**2+(r2z-r1z)**2)
    ry=(r2y-r1y)/np.sqrt((r2x-r1x)**2+(r2y-r1y)**2+(r2z-r1z)**2)
    rz=(r2z-r1z)/np.sqrt((r2x-r1x)**2+(r2y-r1y)**2+(r2z-r1z)**2)

    z1=Lbx/np.sqrt(Lbx**2+Lby**2+Lbz**2)
    z2=Lby/np.sqrt(Lbx**2+Lby**2+Lbz**2)
    z3=Lbz/np.sqrt(Lbx**2+Lby**2+Lbz**2)

    x1=rx*np.cos(phi)+(ry*z3-z2*rz)*np.sin(phi)
    x2=ry*np.cos(phi)+(rz*z1-z1*rx)*np.sin(phi)
    x3=rz*np.cos(phi)+(rx*z2-z1*ry)*np.sin(phi)

    y1=z2*x3-z3*x2
    y2=z3*x1-z1*x3
    y3=z1*x2-z2*x1

    ib=np.arctan2(np.sqrt(z1**2+z2**2),z3)
    omega=np.arctan2(x3,y3)
    Omega=np.arctan2(z1,-z2)

    rcx, rcy, rcz = CM_r(m1, m2, r1x, r1y, r1z, r2x, r2y, r2z)
    vcx, vcy, vcz = CM_v(m1, m2, v1x, v1y, v1z, v2x, v2y, v2z)
    
    Ec, Lcx, Lcy, Lcz = binary_E_L(m0, m1+m2, r0x, r0y, r0z, v0x, v0y, v0z, rcx, rcy, rcz, vcx, vcy, vcz)

    muc=m0*(m1+m2)/(m0+m1+m2)
    ec=np.sqrt(1+(2*Ec*(Lcx**2+Lcy**2+Lcz**2))/(G**2*m0**2*(m1+m2)**2*muc))
    rp=-G*m0*(m1+m2)*(1-ec)/(2*Ec)
    c_f=(rp*(1+ec)/(ec*np.sqrt((r0x-rcx)**2+(r0y-rcy)**2+(r0z-rcz)**2))-1/ec)
    s_f=1/ec*(np.sqrt(1+2*ec*c_f+ec**2))*((r0x-rcx)*(v0x-vcx)+(r0y-rcy)*(v0y-vcy)+(r0z-rcz)*(v0z-vcz))/(np.sqrt((r0x-rcx)**2+(r0y-rcy)**2+(r0z-rcz)**2)*np.sqrt((v0x-vcx)**2+(v0y-vcy)**2+(v0z-vcz)**2))
    f=np.arctan2(s_f,c_f)
    
    return ab, eb, ib, omega, Omega, rp, ec, Eb, Ec, phi, f 



def initial_time(m0, m1, m2, ab, rp, ec, N):
  #Tidal radius
  rt=ab*(m0/(m1+m2))**(1/3)

  #Period of the binary 
  P=2*np.pi*np.sqrt(ab**3/(G*(m1+m2)))
    
  #Time at semi-latus rectum and at tidal radius
  if ec>1:
    c_z_t=1/ec*(1-rt*(ec-1)/rp)
    z_t=np.arccosh(c_z_t)
    t_rt=np.sqrt(rp**3/(G*(m0+m1+m2)*(ec-1)**3))*(ec*np.sinh(z_t)-z_t)

    s_z_p2=np.sqrt(ec**2-1)
    c_z_p2=ec
    z_p2=np.arctanh(s_z_p2/c_z_p2)
    t_p2=np.sqrt(rp**3/(G*(m0+m1+m2)*(ec-1)**3))*(ec*s_z_p2-z_p2)

  elif ec<1:
    c_eta_t=1/ec*(1-rt*(1-ec)/rp)
    eta_t=np.arccos(c_eta_t)
    t_rt=np.sqrt(rp**3/(G*(m0+m1+m2)*(1-ec)**3))*(eta_t-ec*np.sin(eta_t))

    s_eta_p2=np.sqrt(1-ec**2)
    c_eta_p2=ec
    eta_p2=np.arctan2(s_eta_p2,c_eta_p2)
    t_p2=np.sqrt(rp**3/(G*(m0+m1+m2)*(1-ec)**3))*(eta_p2-ec*s_eta_p2)
    
  else:
    c_ft=2*rp/rt-1
    t2_ft=(1-c_ft)/(1+c_ft)   #tan^2(f_t/2)
    t_p2=np.sqrt(rp**3/(G*(m0+m1+m2)))*np.sqrt(32)/3
    t_rt=np.sqrt(rp**3/(G*(m0+m1+m2)))*np.sqrt(2)*(t2_ft+t2_ft**3/3)    

  #Initial time
  if ((rt/rp >= 1) and (t_rt>=t_p2)):
    t_0=-t_rt-N*P
  else:
    t_0=-t_p2-N*P

  return t_0



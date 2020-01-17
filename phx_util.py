import numpy as np
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.orbit import Orbit
from galpy.df import streamdf, streamgapdf
from streampepperdf import streampepperdf
from galpy.util import bovy_conversion, bovy_coords
import astropy.units as u
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord


R0,V0= 8., 220.
# Coordinate transformation routines
#_RAPHX= dasdf/180.*np.pi
#_DECPHX=asdf/180.*np.pi

obs = Orbit([27.60969888973802*u.deg,
  -43.54155350527743*u.deg,
  17.624735997461556*u.kpc,
  2.19712949699425679*(u.mas/u.yr),
  -0.5240686072157521*(u.mas/u.yr),
  19.93208180482703*(u.km/u.s)], \
                    radec=True,ro=8,vo=220, \
                    solarmotion=[-11.1,24.,7.25])

def setup_phxmodel(leading=False,
                    timpact=None,
                    hernquist=True,
                    age=1.5,
                    singleImpact=False,
                    length_factor=1.,
                    obs = obs,
                    sigvmod = .23,
                    **kwargs):
    #obs= Orbit([229.018,-0.124,23.2,-2.296,-2.257,-58.7],
    #           radec=True,ro=R0,vo=V0,
    #           solarmotion=[-11.1,24.,7.25])
    aAI= actionAngleIsochroneApprox(pot=MWPotential2014,b=.5)
    sigv= sigvmod*(5./age) #km/s, adjust for diff. age
    if timpact is None:
        sdf= streamdf(sigv/V0,progenitor=obs,
                      pot=MWPotential2014,aA=aAI,
                      leading=leading,nTrackChunks=11,
                      tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                      ro=R0,vo=V0,R0=R0,
                      vsun=[-11.1,V0+24.,7.25])
    elif singleImpact:
        sdf= streamgapdf(sigv/V0,progenitor=obs,
                         pot=MWPotential2014,aA=aAI,
                         leading=leading,nTrackChunks=11,
                         tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                         ro=R0,vo=V0,R0=R0,
                         vsun=[-11.1,V0+24.,7.25],
                         timpact= 0.3/bovy_conversion.time_in_Gyr(V0,R0),
                         spline_order=3,
                         hernquist=hernquist,
                         impact_angle=0.7,
                         impactb=0.,
                         GM= 10.**-2./bovy_conversion.mass_in_1010msol(V0,R0),
                         rs= 0.625/R0,
                         subhalovel=np.array([6.82200571,132.7700529,14.4174464])/V0,
                         **kwargs)
    else:
        sdf= streampepperdf(sigv/V0,progenitor=obs,
                            pot=MWPotential2014,aA=aAI,
                            leading=leading,nTrackChunks=101,
                            tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                            ro=R0,vo=V0,R0=R0,
                            vsun=[-11.1,V0+24.,7.25],
                            timpact=timpact,
                            spline_order=1,
                            hernquist=hernquist,
                            length_factor=length_factor)
    sdf.turn_physical_off()
    return sdf


def sp_stream_samples(sp, nsample =10000, lb=True, massexp=-2, GMmod=1., massrange = [6,9], cutoff =5., ratemod = 1., do_sample=False):
    massexp=massexp
    sample_GM= lambda: GMmod*powerlaw_wcutoff(massrange,cutoff)
    rate_range= np.arange(massrange[0]+0.5,massrange[1]+0.5,1)
    rate = ratemod*np.sum([dNencdm(sp,10.**r,Xrs=5.,plummer=False,rsfac=1.,sigma=120.) for r in rate_range])
    sample_rs= lambda x: rs(x*bovy_conversion.mass_in_1010msol(V0,R0)*10.**10.,plummer=False,rsfac=1.)
    ns= 0
    #print(rate)
    sp.simulate(rate=rate,sample_GM=sample_GM,sample_rs=sample_rs,Xrs=3.,sigma=120./220.)

    if do_sample==True:
        sp_sample= sp.sample(n=nsample,lb=lb)
        spc = SkyCoord( \
            sp_sample[0]*u.deg, \
            sp_sample[1]*u.deg, \
            distance=sp_sample[2]*u.kpc, \
            radial_velocity=sp_sample[3]*u.km/u.s, \
            pm_l_cosb=sp_sample[4]*u.mas/u.yr, \
            pm_b=sp_sample[5]*u.mas/u.yr, \
            frame='galactic')
        spphx = spc.transform_to(Phoenix)
        return sp_sample, spphx, spc


def parse_times(times,age):
    if 'sampling' in times:
        nsam= int(times.split('sampling')[0])
        return [float(ti)/bovy_conversion.time_in_Gyr(V0,R0)
                for ti in np.arange(1,nsam+1)/(nsam+1.)*age]
    return [float(ti)/bovy_conversion.time_in_Gyr(V0,R0)
            for ti in times.split(',')]
def parse_mass(mass):   
    return [float(m) for m in mass.split(',')]

# Functions to sample
def nsubhalo(m):
    return 0.3*(10.**6.5/m)
def rs(m,plummer=False,rsfac=1.):
    if plummer:
        return 1.62*rsfac/R0*(m/10.**8.)**0.5
    else:
        return 1.05*rsfac/R0*(m/10.**8.)**0.5
def dNencdm(sdf_pepper,m,Xrs=3.,plummer=False,rsfac=1.,sigma=120.):
    return sdf_pepper.subhalo_encounters(\
        sigma=sigma/V0,nsubhalo=nsubhalo(m),
        bmax=Xrs*rs(m,plummer=plummer,rsfac=rsfac))
def powerlaw_wcutoff(massrange,cutoff):
    accept= False
    while not accept:
        prop= (10.**-(massrange[0]/2.)+(10.**-(massrange[1]/2.)\
                         -10.**(-massrange[0]/2.))\
                   *np.random.uniform())**-2.
        if np.random.uniform() < np.exp(-10.**cutoff/prop):
            accept= True
    return prop/bovy_conversion.mass_in_msol(V0,R0)



class Phoenix(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Phoenix stream, as described in
        Balbinot et al. 2016

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    Lambda : `~astropy.coordinates.Angle`, optional, must be keyword
        The longitude-like angle corresponding to the direction along Phoenix.
    Beta : `~astropy.coordinates.Angle`, optional, must be keyword
        The latitude-like angle corresponding to the direction perpendicular to Phoenix.
    distance : `Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.
    pm_Lambda_cosBeta : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion along the stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_Beta : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in Declination for this object (``pm_ra_cosdec`` must
        also be given).
    radial_velocity : :class:`~astropy.units.Quantity`, optional, must be keyword
        The radial velocity of this object.

    """

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'Lambda'),
            coord.RepresentationMapping('lat', 'Beta'),
            coord.RepresentationMapping('distance', 'distance')]
    }

PHX_PHI = (-29.698) * u.degree # Euler angles from Balbanot et al. 2016 - psi was unspecified
PHX_THETA = (+72.247) * u.degree
PHX_PSI = (0.) * u.degree

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(PHX_PHI, "z")
C = rotation_matrix(PHX_THETA, "x")
B = rotation_matrix(PHX_PSI, "z")
A = np.diag([1.,1.,-1.])
PHX_MATRIX = matrix_product(A, B, C, D)

@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Phoenix)
def galactic_to_phoenix():
    """ Compute the transformation matrix from galactic to
        heliocentric Phoenix coordinates.
    """
    return PHX_MATRIX

@frame_transform_graph.transform(coord.StaticMatrixTransform, Phoenix, coord.Galactic)
def phoenix_to_galactic():
    """ Compute the transformation matrix from heliocentric Phoenix coordinates to
        galactic.
    """
    return matrix_transpose(PHX_MATRIX)

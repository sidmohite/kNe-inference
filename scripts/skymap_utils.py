#!/usr/bin/env python
"""
A module for handling skymaps and associated utilities.

Classes:

    Skymap_Probability
"""

__author__ = 'Siddharth Mohite'

import numpy as np
from scipy.stats import norm, truncnorm
from scipy.integrate import quad
import healpy as hp


class Skymap_Probability():
    """
    Ingests skymaps to acquire marginal distance distributions.

    Attributes
    ----------
    skymap_file : str
        Path to the fits.gz skymap file for an event.
    field_coords_array : array
        The array Cartesian coordinates of the corners of each field of view.
    nside : int
        Number (a power of 2) representing the resolution of the skymap.
    prob : array
        Array of probabilities for every pixels in the skymap.
    distmu : array
        Array of mean distances (Mpc) of the marginalised distance
        distribution for each pixel.
    distsigma : array
        Array of standard deviations (Mpc) of the marginalised distance
        distribution for each pixel.
    distnorm : array
        Array of normalization factors for the marginalised distance
        distribution for each pixel.

    Methods
    -------
    pix_in_fields():
        Returns the array of pixels contained within the fields specifed by
        the field_coords_array.
    calculate_field_probs():
        Returns the total probability contained within a field by summing over
        pixels.
    construct_margdist_distribution(ipix_field,field_prob,dmax=3000):
        Returns the marginalised probability density for pixels in a field.

    Usage
    -----
    skymap_prob = Skymap_Probability(skymap_fits_file,field_coords_array)
    """

    def __init__(self, skymap_fits_file, field_coords_array):
        """
        Instantiates class that handles skymap probability.

        Parameters:
        -----------
            skymap_fits_file : str
                Path to the fits.gz skymap file for an event.
            field_coords_array : array
                The array Cartesian coordinates of the corners of each field
                of view. Shape = (nfields, 4, 3).
        """
        print("Ingesting skymap:"+skymap_fits_file)
        self.skymap_file = skymap_fits_file
        self.field_coords_array = field_coords_array
        prob, distmu, distsigma, distnorm = hp.read_map(
                                                       self.skymap_file,
                                                       field=range(4))
        npix = len(prob)
        self.nside = hp.npix2nside(npix)
        self.prob = prob
        self.distmu = distmu
        self.distsigma = distsigma
        self.distnorm = distnorm

    def pix_in_fields(self):
        """
        Returns the array of pixel indices contained within the fields.
        """
        return np.array([
                       hp.query_polygon(self.nside, coords) for coords in
                       self.field_coords_array])

    def calculate_field_probs(self):
        """
        Returns the total probability contained within each field.
        """
        ipix_fields = self.pix_in_fields()
        return np.array([self.prob[ipix].sum() for ipix in ipix_fields])

    def construct_margdist_distribution(
                                       self, ipix_field, field_prob,
                                       dmax=3000):
        """
        Returns the approximate probability density for distance marginalised
        over pixels in a field.

        Parameters
        ----------
            ipix_field : array
                Array of pixel indices contributing to each field.
            field_prob : array
                Array of total field probabilites.
            dmax : float , optional
                Maximum distance (in Mpc) for the distance
                distribution(default=3000).

        Returns
        -------
            approx_dist_pdf : scipy.stats.rv_continuous.pdf object
                The probability density function (pdf) of the distance over
                the given field, approximated as a normal distribution.
        """
        dp_dr = lambda r: np.sum(
            self.prob[ipix_field] * r**2 * self.distnorm[ipix_field] *
            norm(self.distmu[ipix_field], self.distsigma[ipix_field]).pdf(r))\
            / field_prob
        mean = quad(lambda x: x * dp_dr(x), 0, dmax)[0]
        sd = np.sqrt(quad(lambda x: x**2 * dp_dr(x), 0, dmax)[0] - mean**2)
        approx_dist_pdf = truncnorm((0-mean)/sd, (dmax-mean)/sd, mean, sd).pdf
        return approx_dist_pdf

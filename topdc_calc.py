# Author: Gisell Osorio

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline

C0 = 3  # Velocity of light in vacuum in um/(10^-14 s)


class topdc_calc:
    def __init__(self, wf, neff_triplet, wp, neff_pump, *wp_cutoff):
        """
        A class for computing various physical quantities related to degenerate third-order
        parametric down-conversion (TOPDC) in optical fibers or waveguides.

        This class models the phase-matching conditions and joint spectral properties of triplet
        photon generation. It uses the pump and triplet modes refractive index data to
        compute wavevector amplitudes, group velocities, dispersion, phase mismatch, the joint
        spectral amplitude (JSA), the single photon reduced density matrix rho, mode decomposition of
        rho, and purity.


        Attributes
        ----------
        wf : float or np.ndarray
            Frequency array (or single float) for the triplet mode [in 10^14 rad/s].

        neff_triplet : float or np.ndarray
            Effective refractive index data for the triplet mode, corresponding to `wf`.

        wp : float or np.ndarray
            Frequency array (or single float) for the pump mode [in 10^14 rad/s].

        neff_pump : float or np.ndarray
            Effective refractive index data for the pump mode, corresponding to `wp`.

        wp_cutoff : float, optional
            Pump mode cutoff frequency [in 10^14 rad/s]. If provided, any calculation below this
            frequency will yield NaN values in the output arrays to account for mode cutoff.

        Methods
        -------
        neff_p(w0)
            Interpolates and returns the effective index for the pump at frequency w0.

        neff_t(w0)
            Interpolates and returns the effective index for the triplet at frequency w0.

        kp(w0)
            Computes the wavevector amplitude for the pump mode in units 1/um.

        kf(w0)
            Computes the wavevector amplitude for the triplet mode in units 1/um.

        vg_p(w0)
            Calculates the group velocity for the pump mode using numerical differentiation in units 10^8 m/s.

        vg_f(w0)
            Calculates the group velocity for the triplet mode using numerical differentiation in units 10^8 m/s.

        beta2_p(w0)
            Calculates group velocity dispersion (GVD) for the pump mode in units of 10^2 fs/um.

        beta2_f(w0)
            Calculates group velocity dispersion (GVD) for the triplet mode in units of 10^2 fs/um.

        delk0(w0_f)
            Computes the phase mismatch \Delta k0 = k_p(3 * w0_f) - 3 * k_f(w0_f).

        gvm(w0_f)
            Calculates the group velocity mismatch between the pump and triplet at central frequencies.

        w_pm(wi)
            Solves for the phase-matching frequency w0_f where delk0(w0_f) = 0 using `fsolve`.

        w_gvm(wi)
            Solves for the frequency where group velocity mismatch gvm(w0_f) = 0 using `fsolve`.

        pm_bw(L, w_i)
            Estimates the bandwidth of the phase-matching if the factorability conditions are met.

        jsa(w1, w2, w3, L, wp0, sigmap)
            Computes the joint spectral amplitude (JSA) of the triplet state.

        jsa_filter(w1, w2, w3, L, wp0, sigmap,wlow,whigh)
            Computes the filtered joint spectral amplitude (JSA) of the triplet state.

        rho(w1, w2, w3, L, wp0, sigmap)
            Computes the joint spectral amplitude (JSA) of the triplet state, its 2D projection, the single-photon
            reduced density matrix, eigenvalues, eigenmodes, and the spectral purity.

        rho_filter(w1, w2, w3, L, wp0, sigmap, wlow, whigh)
            Computes the joint spectral amplitude (JSA) of the triplet state, filtering below wlow and above whigh,
            its 2D projection, the single-photon reduced density matrix, eigenvalues, eigenmodes, and the spectral purity.

        rho_filter_auto( L, wp0, sigmap, wlow, whigh)
            Automatically computes the joint spectral amplitude (JSA) of the triplet state, filtering below wlow
            and above whigh, its 2D projection, the single-photon reduced density matrix, eigenvalues, eigenmodes, and
            the spectral purity.

        topdc_scaled_auto(self, L, wp0, wlow, whigh):
            Automatically computes the joint spectral amplitude (JSA) of the triplet state, filtering below wlow and
            above whigh, its 2D projection, the single-photon reduced density matrix, eigenvalues, eigenmodes, and the
            spectral purity. The frequencies are scaled by the phase-matching bandwidth.
        """
        self.wf = wf
        self.neff_triplet = neff_triplet
        self.wp = wp
        self.neff_pump = neff_pump
        self.wp_cutoff = wp_cutoff

    def neff_p(self, w0):
        """
        Returns the effective refractive index of the pump mode at given frequency value(s)
        using cubic spline interpolation.

        Parameters
        ----------
        w0 : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) at which to evaluate the pump mode's
            effective refractive index.

        Returns
        -------
        neff_p(w0) : np.ndarray
            Interpolated effective refractive index values corresponding to `w0`.
            Will contain `NaN` for frequency values below the pump mode cutoff
            (`wp_cutoff`), if provided. The shape of the output matches the shape of `w0`.
        """
        mask = ~np.isnan(self.neff_pump)
        wm = self.wp[mask]
        neffm = self.neff_pump[mask]
        if any(self.wp_cutoff):
            pmin_i = np.argmin(wm < self.wp_cutoff[0])
            neff = CubicSpline(wm[pmin_i:], neffm[pmin_i:], extrapolate=False)
            return neff(w0)
        else:
            neff = CubicSpline(wm, neffm, extrapolate=False)
            return neff(w0)

    def neff_f(self, w0):
        """
        Returns the effective refractive index of the triplet mode at given frequency value(s)
        using cubic spline interpolation.

        Parameters
        ----------
        w0 : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) at which to evaluate the triplet mode's
            effective refractive index.

        Returns
        -------
        neff_f(w0) : np.ndarray
            Interpolated effective refractive index values corresponding to `w0`.
            The shape of the output matches the shape of `w0`.
        """

        # Note: Considering no cutoff for the triplet mode
        mask = ~np.isnan(self.neff_triplet)
        wm = self.wf[mask]
        neffm = self.neff_triplet[mask]
        neff = CubicSpline(wm, neffm, extrapolate=False)
        return neff(w0)

    def kp(self, w0):
        """
        Computes the wavevector amplitude of the pump mode at given frequency value(s) in units 1/um.

        The wavevector is calculated using the formula:
            k(w0) = w0 * neff_p(w0) / C0

        where `neff_p(w0)` is obtained from the `neff_p` method, and `C0` is the speed of
        light in vacuum expressed in um / (10^-14 s).

        Parameters
        ----------
        w0 : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) at which to compute the pump wavevector.

        Returns
        -------
        kp(w0) : float or np.ndarray
            Pump mode wavevector values corresponding to the input frequency `w0`.
            Units: 1/um
            Returns `NaN` for frequency values below the cutoff, if a cutoff (`wp_cutoff`)
            was defined. The shape of the output matches the shape of `w0`.
        """

        kp = w0 * self.neff_p(w0) / C0
        return kp

    def kf(self, w0):
        """
        Computes the wavevector amplitude of the triplet mode at given frequency value(s) in units 1/um.

        The wavevector is calculated using the formula:
            k(w0) = w0 * neff_f(w0) / C0

        where `neff_p(w0)` is obtained from the `neff_p` method, and `C0` is the speed of
        light in vacuum expressed in um / (10^-14 s).

        Parameters
        ----------
        w0 : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) at which to compute the pump wavevector.

        Returns
        -------
        kf(w0) : float or np.ndarray
            Pump mode wavevector values corresponding to the input frequency `w0`.
            Units: 1/um
            The shape of the output matches the shape of `w0`.
        """
        kf = w0 * self.neff_f(w0) / C0
        return kf

    def vg_p(self, w0):
        """
        Calculates the group velocity of the pump mode at given frequency value(s) in units 10^8 m/s.

        The group velocity is computed as:
        1 / vg_P = d kp(w)/d w, where kp(w) = w * neff_P(w) / C0

        where `neff_p(w)` is obtained from the `neff_p` method, and `C0` is the speed of
        light in vacuum expressed in um / (10^-14 s).

        The derivative dk(w)/dw is estimated using numerical differentiation of the
        interpolated effective refractive index. Interpolation is performed with a cubic spline
        over the pump mode data provided at initialization.

        Parameters
        ----------
        w0 : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) at which to evaluate the group velocity.

        Returns
        -------
        vg_p(w0) : np.ndarray
            Group velocity of the pump mode at the specified frequency value(s).
            Units: [10^8 m/s]
            If a pump cutoff frequency (`wp_cutoff`) was provided, the output will contain
            `NaN` values for input frequencies below the cutoff. The shape of the output
            matches the shape of `w0`.
        """
        mask = ~np.isnan(self.neff_pump)
        wm = self.wp[mask]
        neffm = self.neff_pump[mask]
        if any(self.wp_cutoff):
            pmin_i = np.argmin(wm < self.wp_cutoff[0])
            neff = CubicSpline(wm[pmin_i:], neffm[pmin_i:], extrapolate=False)
            beta1 = neff(w0) / C0 + neff(w0, 1) * w0 / C0
            vg = 1 / beta1
            return vg
        else:
            neff = CubicSpline(wm, neffm, extrapolate=False)
            beta1 = neff(w0) / C0 + neff(w0, 1) * w0 / C0
            vg = 1 / beta1
            return np.array(vg)

    def vg_f(self, w0):
        """
        Calculates the group velocity of the triplet mode at given frequency value(s) in units 10^8 m/s.

        The group velocity is computed as:
        1 / vg = dk(w)/dw, where k(w) = w * neff(w) / C0

        The derivative dk(w)/dw is estimated using numerical differentiation of the
        interpolated effective refractive index. Interpolation is performed with a cubic spline
        over the triplet mode data provided at initialization.

        Parameters
        ----------
        w0 : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) at which to evaluate the group velocity.

        Returns
        -------
        vg_f(w0) : np.ndarray
            Group velocity of the triplet mode at the specified frequency value(s).
            Units: [10^8 m/s]
            The shape of the output matches the shape of `w0`.
        """

        # Note: Considering no cutoff for the triplet mode
        mask = ~np.isnan(self.neff_triplet)
        wm = self.wf[mask]
        neffm = self.neff_triplet[mask]
        neff = CubicSpline(wm, neffm, extrapolate=False)
        beta1 = neff(w0) / C0 + neff(w0, 1) * w0 / C0
        vg = 1 / beta1
        return np.array(vg)

    def beta2_p(self, w0):
        """
        Calculates the group velocity dispersion (GVD) of the pump mode at given frequency value(s) in units of 10^2 fs^2/µm. .

        The GVD is defined as:
            beta2(w) = d^2k(w)/dw^2

        where k(w) is the wavevector computed as k(w) = w * neff(w) / C0.

        Parameters
        ----------
        w0 : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) at which to evaluate the group velocity dispersion.

        Returns
        -------
        beta2_p(w0) : float or np.ndarray
            Group velocity dispersion values in units of 10^2 fs^2/µm.
            Returns `NaN` for frequencies below the pump cutoff (`wp_cutoff`), if provided.
            The output shape matches that of `w0`.
        """

        if any(self.wp_cutoff):
            pmin_i = np.argmin(self.wp < self.wp_cutoff[0])
            w0p = self.wp[pmin_i:]
        else:
            w0p = self.wp

        mask = ~np.isnan(self.kp(w0p))
        kpol = np.poly1d(np.polyfit(w0p[mask], self.kp(w0p)[mask], 8))
        beta2 = np.polyder(kpol, 2)
        return beta2(w0)

    def beta2_f(self, w0):
        """
        Calculates the group velocity dispersion (GVD) of the triplet mode at given frequency value(s) in units of 10^2 fs^2/µm. .

        The GVD is defined as:
            beta2(w) = d^2k(w)/dw^2

        where k(w) is the wavevector computed as k(w) = w * neff(w) / C0.

        Parameters
        ----------
        w0 : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) at which to evaluate the group velocity dispersion.

        Returns
        -------
        beta2_f(w0) : float or np.ndarray
            Group velocity dispersion values in units of 10^2 fs^2/µm.
            The output shape matches that of `w0`.
        """

        # Note: Considering no cutoff for the triplet mode
        w0p = self.wf
        mask = ~np.isnan(self.kf(w0p))
        kpol = np.poly1d(np.polyfit(w0p[mask], self.kf(w0p)[mask], 8))
        beta2 = np.polyder(kpol, 2)
        return beta2(w0)

    def beta3_p(self, w0):
        """
        Calculates the third order dispersion (TOD) of the pump mode at given frequency value(s) in units of 10^3 fs^3/µm. .

        The TOD is defined as:
            beta3(w) = d^3k(w)/dw^3

        where k(w) is the wavevector computed as k(w) = w * neff(w) / C0.

        Parameters
        ----------
        w0 : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) at which to evaluate the third order dispersion.

        Returns
        -------
        beta3_p(w0) : float or np.ndarray
            Third order dispersion values in units of 10^3 fs^3/µm.
            Returns `NaN` for frequencies below the pump cutoff (`wp_cutoff`), if provided.
            The output shape matches that of `w0`.
        """

        if any(self.wp_cutoff):
            pmin_i = np.argmin(self.wp < self.wp_cutoff[0])
            w0p = self.wp[pmin_i:]
        else:
            w0p = self.wp

        mask = ~np.isnan(self.kp(w0p))
        kpol = np.poly1d(np.polyfit(w0p[mask], self.kp(w0p)[mask], 8))
        beta3 = np.polyder(kpol, 3)
        return beta3(w0)

    def beta3_f(self, w0):
        """
        Calculates the third order dispersion (TOD) of the triplet mode at given frequency value(s) in units of 10^3 fs^3/µm.

        The TOD is defined as:
            beta3(w) = d^3k(w)/dw^3

        where k(w) is the wavevector computed as k(w) = w * neff(w) / C0.

        Parameters
        ----------
        w0 : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) at which to evaluate the third order dispersion.

        Returns
        -------
        beta3_f(w0) : float or np.ndarray
            Third order dispersion values in units of 10^3 fs^3/µm.
            The output shape matches that of `w0`.
        """

        # Note: Considering no cutoff for the triplet mode
        w0p = self.wf
        mask = ~np.isnan(self.kf(w0p))
        kpol = np.poly1d(np.polyfit(w0p[mask], self.kf(w0p)[mask], 8))
        beta3 = np.polyder(kpol, 3)
        return beta3(w0)

    def delk0(self, w0_f):
        """
        Computes the phase mismatch Delta_k0 for the frequency degenerate third order parametric down-conversion (TOPDC) process.

        The phase mismatch is calculated as:
            delk0(w0_f) = kp(3 * w0_f) - 3 * kf(w0_f)

        where:
            - w0_f is the frequency of the triplet photons, assuming degenerate generation:
              w1 = w2 = w3 = w0_f = w_pump / 3
            - k_p is the pump wavevector, evaluated at 3 * w0_f
            - k_f is the triplet wavevector, evaluated at w0_f

        Both wavevectors are computed as k(w) = w * neff(w) / C0 using cubic spline interpolation
        of the respective mode's effective refractive index.

        Parameters
        ----------
        w0_f : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) of the triplet photon at which to evaluate the phase-mismatch for the TOPDC process in the frequency degenerate configuration.

        Returns
        -------
        delk0(w0_f) : float or np.ndarray
            Phase mismatch delk0 for each value of `w0_f`. The shape of the output matches that of `w0_f`.
            If a pump cutoff frequency (`wp_cutoff`) is defined, values where 3 * w0_f < wp_cutoff
            will return NaN.
        """
        return self.kp(3 * w0_f) - 3 * self.kf(w0_f)

    def gvm(self, w0_f):
        """
        Computes the group velocity mismatch (GVM) between the pump and the triplet modes, for the frecuency degenerate configuration of third-order parametric downconversion (TOPDC).

        The GVM is defined as the difference in group velocities:
            GVM(w0_f) = vg_p(3 * w0_f) - vg_f(w0_f)

        where:
            - vg_p is the group velocity of the pump mode, evaluated at 3 * w0_f
            - vg_f is the group velocity of the triplet mode, evaluated at w0_f
            - w0_f is the frequency of the triplet photons for the degenerate TOPDC configuration

        Group velocities are computed using numerical differentiation of the wavevector
        expressions based on the interpolated effective refractive indices.

        Parameters
        ----------
        w0_f : float or np.ndarray
            Frequency value(s) (in 10^14 rad/s) of the triplet photon at which to evaluate the GVM for the TOPDC process in the frequency degenerate configuration.

        Returns
        -------
        gvm(w0_f) : float or np.ndarray
            Group velocity mismatch values for each input frequency. The shape matches that of `w0_f`.
            Returns NaN for input frequencies where 3 * w0_f is below the pump cutoff frequency (`wp_cutoff`), if provided.
        """
        return self.vg_p(3 * w0_f) - self.vg_f(w0_f)

    def w_pm(self, wi):
        """
        Determines the phase-matching frequency for the triplet mode by solving the equation delk0(w0_f) = 0.

        This method uses a numerical solver (`fsolve`) to find the root of the phase mismatch function
        delk0(w0_f), which corresponds to the condition for perfect phase matching in the degenerate
        TOPDC process.

        Parameters
        ----------
        wi : float
            Initial guess for the triplet central frequency (in 10^14 s^-1).
            Since the delk0 function can have multiple roots, this initial estimate is critical
            for guiding the solver to the correct solution. You can use the `delk0` method to
            plot the phase mismatch curve and choose an appropriate starting value.

        Returns
        -------
        w_pm : float
            Triplet mode frequency (in 10^14 rad/s) that satisfies delk0(w0_f) = 0.
            Returns a value close to the provided initial guess `wi`. If the solver does not
            converge, an error may be raised.
        """

        def delk(w0):
            delk = self.delk0(w0)
            if np.isnan(delk):
                delk = 1e3
            return delk

        root = fsolve(delk, wi, xtol=1e-06, maxfev=500)
        omf0 = root[0]
        return omf0

    def w_gvm(self, wi):
        """
        Determines the frequency at which group velocity matching (GVM) occurs between the pump and triplet modes.

        This method uses a numerical solver (`fsolve`) to find the root of the group velocity mismatch function
        gvm(w0_f), corresponding to the condition where:
            gvm(w0_f) = vg_p(3 * w0_f) - vg_f(w0_f) = 0

        Parameters
        ----------
        wi : float
            Initial guess for the triplet central frequency (in 10^14 rad/s).
            Since gvm(w0_f) can have multiple roots, the initial guess `wi` is critical
            for guiding the solver to the correct solution. You can use the `gvm` method to
            plot the phase mismatch curve and choose an appropriate starting value.

        Returns
        -------
        w_gvm : float
            Triplet mode frequency (in 10^14 rad/s) where group velocity matching is achieved.
            Returns a value close to the provided initial guess `wi`. If the solver does not
            converge, an error may be raised.
        """

        # Find a root using a numerical solver
        # wi: Initial guess
        def gvm0(w0):
            gvm0 = self.gvm(w0)
            if np.isnan(gvm0):
                gvm0 = 1e3
            return gvm0

        root = fsolve(gvm0, wi, xtol=1e-06, maxfev=500)
        omf0 = root[0]
        return omf0

    def pm_bw(self, L, w_i):
        """
        Estimates the bandwidth of the phase-matching function (PMF) for the degenerate TOPDC process
        if the next conditions are fulfilled at the triplet mode central frequency w0_f:
            -Phase matching: delk0(w0_f) = 0
            -Group velocity matching: gvm(w0_f) = 0
            -beta2_f(w0_f) >> beta2_p(3*w0_f)
        If the previous conditions are fulfilled, the PMF function will be symmetrical and
        its bandwidth will be completely determined by the length and value of beta2_f(w0_f).

        The bandwidth is defined as the width of the central lobe of the PMF and is approximated using:
            pm_bw = sqrt((4 * pi) / (L * |beta2_f|))

        where:
            - L is the length of the optical fiber or waveguide
            - beta2_f is the group velocity dispersion (GVD) of the triplet mode, evaluated at the
            triplet phase-matching frequency

        The triplet phase-matching frequency is found by solving delk0(w0_f) = 0 using `w_i` as the initial guess.

        Parameters
        ----------
        L : float
            Length of the optical fiber or waveguide (in micrometers-um).

        w_i : float
            Initial guess for the triplet phase-matching frequency (in 10^14 rad/s).
            This is used to solve delk0(w0_f) = 0 and find the frequency at which beta2_f is evaluated.

        Returns
        -------
        pm_bw : float
            Estimated bandwidth of the PMF (in 10^14 rad/s).
            The result corresponds to the width of the central lobe of the phase-matching function.
        """
        # Phase-matching bandwidth calculation
        # w_i: Approximate photon triplet phase-matching frequency
        b2f = self.beta2_f(self.w_pm(w_i))
        pm_bw = np.sqrt((4 * np.pi) / (L * np.abs(b2f)))
        return pm_bw

    def jsa(self, w1, w2, w3, L, wp0, sigmap):
        """
        Compute the joint spectral amplitude (JSA) for triplet generation
        via degenerate third-order parametric down-conversion (TOPDC).

        This method performs the numerical computations using the class's
        frequency and effective index data. It supports both single-point and full-grid calculations of the JSA.

        The 3D mesh should be constructed as follows:
            omf = np.linspace(wf0 - domg, wf0 + domg, nps)
            w1 = omf[:, None, None]
            w2 = omf[None, :, None]
            w3 = omf[None, None, :]

        Parameters
        ----------
        w1 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Can be 1D for a single-point calculation or
            3D for a full JSA grid computation.

        w2 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        w3 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        L : float
            Length of the optical fiber or waveguide (in micrometers).

        wp0 : float
            Central frequency of the pump (in 10^14 s^-1).

        sigmap : float
            Pump bandwidth (in 10^14 s^-1), defined as the half-width at the 1/e intensity point.

        Returns
        -------
        jsa : np.ndarray
            3D Joint Spectral Amplitude function over the frequency mesh (if 3D input).
        """

        # JSA calculation
        jsa = np.sinc(
            (L / (2 * np.pi))
            * (self.kp(w1 + w2 + w3) - self.kf(w1) - self.kf(w2) - self.kf(w3))
        ) * np.exp(-(((w1 + w2 + w3) - wp0) ** 2) / (2 * sigmap**2))

        return jsa

    def jsa_filter(self, w1, w2, w3, L, wp0, sigmap, wlow, whigh):
        """
        Compute the joint spectral amplitude (JSA) for triplet generation
        via degenerate third-order parametric down-conversion (TOPDC) and
        applies a passband rectangular filter.

        This method performs the numerical computations using the class's
        frequency and effective index data. It supports both single-point and full-grid calculations of the JSA.

        The 3D mesh should be constructed as follows:
            omf = np.linspace(wf0 - domg, wf0 + domg, nps)
            w1 = omf[:, None, None]
            w2 = omf[None, :, None]
            w3 = omf[None, None, :]

        Parameters
        ----------
        w1 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Can be 1D for a single-point calculation or
            3D for a full JSA grid computation.

        w2 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        w3 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        L : float
            Length of the optical fiber or waveguide (in micrometers).

        wp0 : float
            Central frequency of the pump (in 10^14 s^-1).

        sigmap : float
            Pump bandwidth (in 10^14 s^-1), defined as the half-width at the 1/e intensity point.

        wlow : float
            Filter low cutoff frequency (in 10^14 s^-1).

        whigh : float
            Filter high cutoff frequency (in 10^14 s^-1).


        Returns
        -------
        jsa : np.ndarray
            3D Joint Spectral Amplitude function over the frequency mesh (if 3D input).
        """

        # JSA calculation

        def filter(w, wlow, whigh):
            fw = np.heaviside(w - wlow, 0) * np.heaviside(whigh - w, 0)
            return fw

        jsa = (
            filter(w1, wlow, whigh)
            * filter(w2, wlow, whigh)
            * filter(w3, wlow, whigh)
            * np.sinc(
                (L / (2 * np.pi))
                * (self.kp(w1 + w2 + w3) - self.kf(w1) - self.kf(w2) - self.kf(w3))
            )
            * np.exp(-(((w1 + w2 + w3) - wp0) ** 2) / (2 * sigmap**2))
        )

        return jsa

    def rho(self, w1, w2, w3, L, wp0, sigmap):
        """
        Compute the joint spectral amplitude (JSA) and related quantities for triplet generation
        via degenerate third-order parametric down-conversion (TOPDC).

        This method calls the `jsa_calc` class to perform the numerical computations using the class's
        frequency and effective index data. It supports both single-point and full-grid calculations of the JSA.

        The 3D mesh should be constructed as follows:
            omf = np.linspace(wf0 - domg, wf0 + domg, nps)
            w1 = omf[:, None, None]
            w2 = omf[None, :, None]
            w3 = omf[None, None, :]

        Parameters
        ----------
        w1 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Can be 1D for a single-point calculation or
            3D for a full JSA grid computation.

        w2 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        w3 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        L : float
            Length of the optical fiber or waveguide (in micrometers).

        wp0 : float
            Central frequency of the pump (in 10^14 s^-1).

        sigmap : float
            Pump bandwidth (in 10^14 s^-1), defined as the half-width at the 1/e intensity point.

        Returns
        -------
        jsa : np.ndarray
            3D Joint Spectral Amplitude function over the frequency mesh (if 3D input).

        jsa_proj : np.ndarray
            A 2D projection of the JSA onto the (w2, w3) plane, used for visualization.

        density_matrix : np.ndarray
            The single-photon reduced density matrix, calculated by performing a partial
            trace over two parties of the tri-photon density matrix .

        eigen_modes : np.ndarray
            The eigenmodes  of the reduced density matrix. Each column corresponds to a mode over the
            frequency vector `omf` used to generate the w1, w2, w3 grid.

        eigen_values : np.ndarray
            The eigenvalues associated with the mode decomposition, representing the mode weights.

        purity : float
            The spectral purity of the state.
        """
        return rho_calc(
            w1,
            w2,
            w3,
            L,
            wp0,
            sigmap,
            self.wf,
            self.neff_triplet,
            self.wp,
            self.neff_pump,
            *self.wp_cutoff
        )

    def rho_filter(self, w1, w2, w3, L, wp0, sigmap, wlow, whigh):
        """
        Compute the filtered joint spectral amplitude (JSA) and related quantities for triplet generation
        via degenerate third-order parametric down-conversion (TOPDC).

        This method calls the `rho_filter_calc` class to perform the numerical computations using the class's
        frequency and effective index data. It supports both single-point and full-grid calculations of the JSA.

        The 3D mesh should be constructed as follows:
            omf = np.linspace(wf0 - domg, wf0 + domg, nps)
            w1 = omf[:, None, None]
            w2 = omf[None, :, None]
            w3 = omf[None, None, :]

        Parameters
        ----------
        w1 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Can be 1D for a single-point calculation or
            3D for a full JSA grid computation.

        w2 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        w3 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        L : float
            Length of the optical fiber or waveguide (in micrometers).

        wp0 : float
            Central frequency of the pump (in 10^14 s^-1).

        sigmap : float
            Pump bandwidth (in 10^14 s^-1), defined as the half-width at the 1/e intensity point.

        wlow : float
            Filter low cutoff frequency (in 10^14 s^-1).

        whigh : float
            Filter high cutoff frequency (in 10^14 s^-1).

        Returns
        -------
        jsa : np.ndarray
            3D Joint Spectral Amplitude function over the frequency mesh (if 3D input).

        jsa_proj : np.ndarray
            A 2D projection of the JSA onto the (w2, w3) plane, used for visualization.

        density_matrix : np.ndarray
            The single-photon reduced density matrix, calculated by performing a partial
            trace over two parties of the tri-photon density matrix .

        eigen_modes : np.ndarray
            The eigenmodes  of the reduced density matrix. Each column corresponds to a mode over the
            frequency vector `omf` used to generate the w1, w2, w3 grid.

        eigen_values : np.ndarray
            The eigenvalues associated with the mode decomposition, representing the mode weights.

        purity : float
            The spectral purity of the state.
        """
        return rho_filter_calc(
            w1,
            w2,
            w3,
            L,
            wp0,
            sigmap,
            wlow,
            whigh,
            self.wf,
            self.neff_triplet,
            self.wp,
            self.neff_pump,
            *self.wp_cutoff
        )

    def rho_filter_auto(self, L, wp0, wlow, whigh):
        """
        Automatically compute the filtered joint spectral amplitude (JSA) and related quantities for triplet generation
        via degenerate third-order parametric down-conversion (TOPDC).

        This method calls the `rho_filter_calc_auto` class to perform the numerical computations using the class's
        frequency and effective index data.

        Parameters
        ----------
        L : float
            Length of the optical fiber or waveguide (in micrometers).

        wp0 : float
            Central frequency of the pump (in 10^14 s^-1).

        wlow : float
            Filter low cutoff frequency (in 10^14 s^-1).

        whigh : float
            Filter high cutoff frequency (in 10^14 s^-1).

        Returns
        -------
        omega : np.ndarray
            1D array of evaluation frequencies (in 10^14 rad/s).

        jsa : np.ndarray
            3D Joint Spectral Amplitude function over the frequency mesh (if 3D input).

        jsa_proj : np.ndarray
            A 2D projection of the JSA onto the (w2, w3) plane, used for visualization.

        density_matrix : np.ndarray
            The single-photon reduced density matrix, calculated by performing a partial
            trace over two parties of the tri-photon density matrix .

        eigen_modes : np.ndarray
            The eigenmodes  of the reduced density matrix. Each column corresponds to a mode over the
            frequency vector `omf` used to generate the w1, w2, w3 grid.

        eigen_values : np.ndarray
            The eigenvalues associated with the mode decomposition, representing the mode weights.

        purity : float
            The spectral purity of the state.
        """
        return rho_filter_calc_auto(
            L,
            wp0,
            wlow,
            whigh,
            self.wf,
            self.neff_triplet,
            self.wp,
            self.neff_pump,
            *self.wp_cutoff
        )

    def topdc_scaled_auto(self, L, wp0, wlow, whigh):
        """
        Automatically compute the filtered joint spectral amplitude (JSA) and related quantities for triplet generation
        via degenerate third-order parametric down-conversion (TOPDC) in units scaled to the phase-matching bandwidth.

        This method calls the `topdc_scaled_calc_auto` class to perform the numerical computations using the class's
        frequency and effective index data.

        Parameters
        ----------
        L : float
            Length of the optical fiber or waveguide (in micrometers).

        wp0 : float
            Central frequency of the pump (in 10^14 s^-1).

        wlow : float
            Filter low cutoff frequency (in 10^14 s^-1).

        whigh : float
            Filter high cutoff frequency (in 10^14 s^-1).

        Returns
        -------
        omega : np.ndarray
            1D array of evaluation frequencies (in 10^14 rad/s).

        jsa : np.ndarray
            3D Joint Spectral Amplitude function over the frequency mesh (if 3D input).

        jsa_proj : np.ndarray
            A 2D projection of the JSA onto the (w2, w3) plane, used for visualization.

        density_matrix : np.ndarray
            The single-photon reduced density matrix, calculated by performing a partial
            trace over two parties of the tri-photon density matrix .

        eigen_modes : np.ndarray
            The eigenmodes  of the reduced density matrix. Each column corresponds to a mode over the
            frequency vector `omf` used to generate the w1, w2, w3 grid.

        eigen_values : np.ndarray
            The eigenvalues associated with the mode decomposition, representing the mode weights.

        purity : float
            The spectral purity of the state.
        """
        return topdc_scaled_calc_auto(
            L,
            wp0,
            wlow,
            whigh,
            self.wf,
            self.neff_triplet,
            self.wp,
            self.neff_pump,
            *self.wp_cutoff
        )


class rho_calc:
    def __init__(
        self, w1, w2, w3, L, wp0, sigmap, wf, neff_triplet, wp, neff_pump, *wp_cutoff
    ):
        """
        Class to compute the Joint Spectral Amplitude (JSA) and related quantities for triplets generated by
        degenerate third-order parametric down-conversion (TOPDC).

        This class supports both single-point and full-grid evaluation. When the inputs w1, w2, and w3 are 3D arrays,
        they form a full coordinate grid over the frequency space used for numerical computation of the JSA.

        The 3D mesh should be constructed as follows:
            omf = np.linspace(wf0 - domg, wf0 + domg, nps)
            w1 = omf[:, None, None]
            w2 = omf[None, :, None]
            w3 = omf[None, None, :]

        Parameters
        ----------
        w1 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Can be 1D or 3D.

        w2 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        w3 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        L : float
            Length of the optical fiber or waveguide (in micrometers (um)).

        wp0 : float
            Central frequency of the pump (in 10^14 rad/s).

        sigmap : float
            Bandwidth of the pump (in 10^14 rad/s). Defined as the half-width at the 1/e intensity point.

         Attributes
        ----------
        jsa : np.ndarray
            The calculated 3D Joint Spectral Amplitude function.

        jsa_proj : np.ndarray
            A 2D projection of the JSA onto the (w2, w3) plane, used for visualization.

        density_matrix : np.ndarray
            The single-photon reduced density matrix, calculated by performing a partial trace over two parties of the tri-photon density matrix .

        eigen_modes : np.ndarray
            The eigenmodes  of the reduced density matrix. Each column corresponds to a mode over the
            frequency vector `omf` used to generate the w1, w2, w3 grid.

        eigen_values : np.ndarray
            The eigenvalues associated with the mode decomposition, representing the mode weights.

        purity : float
            The spectral purity of the state.
        """

        disp_data = topdc_calc(wf, neff_triplet, wp, neff_pump, wp_cutoff[0])

        # JSA calculation
        jsa = np.sinc(
            (L / (2 * np.pi))
            * (
                disp_data.kp(w1 + w2 + w3)
                - disp_data.kf(w1)
                - disp_data.kf(w2)
                - disp_data.kf(w3)
            )
        ) * np.exp(-(((w1 + w2 + w3) - wp0) ** 2) / (2 * sigmap**2))

        if (
            isinstance(w1, (int, float))
            or isinstance(w2, (int, float))
            or isinstance(w3, (int, float))
            or w1.ndim < 3
            or w2.ndim < 3
            or w3.ndim < 3
        ):
            jsa_proj = None
            rho = None
            U = None
            S = None
            Vh = None
            purity = None
        else:
            # Projection of the JSA calculation
            jsa_proj = np.sum(np.square(np.abs(jsa)), axis=0)

            jsan = np.nan_to_num(jsa) / np.linalg.norm(np.nan_to_num(jsa))

            # Calculation of the reduced density matrix
            rho = np.einsum("ijk,ljk->il", jsan, np.conj(jsan))

            # Calculation of the eigenvalues of rho
            U, S, Vh = np.linalg.svd(rho, full_matrices=True, hermitian=True)

            # Calculation of purity
            purity = np.sum(S) ** 2 / np.sum(S**2)

        self.jsa = jsa
        self.jsa_proj = jsa_proj
        self.density_matrix = rho
        self.eigen_modes = U
        self.eigen_values = S
        self.purity = purity


class rho_filter_calc:
    def __init__(
        self,
        w1,
        w2,
        w3,
        L,
        wp0,
        sigmap,
        wlow,
        whigh,
        wf,
        neff_triplet,
        wp,
        neff_pump,
        *wp_cutoff
    ):
        """
        Class to compute the filtered Joint Spectral Amplitude (JSA) and related quantities for triplets generated by
        degenerate third-order parametric down-conversion (TOPDC).

        This class supports both single-point and full-grid evaluation. When the inputs w1, w2, and w3 are 3D arrays,
        they form a full coordinate grid over the frequency space used for numerical computation of the JSA.

        The 3D mesh should be constructed as follows:
            omf = np.linspace(wf0 - domg, wf0 + domg, nps)
            w1 = omf[:, None, None]
            w2 = omf[None, :, None]
            w3 = omf[None, None, :]

        Parameters
        ----------
        w1 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Can be 1D or 3D.

        w2 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        w3 : np.ndarray
            Array of evaluation frequencies (in 10^14 rad/s). Same shape as w1.

        L : float
            Length of the optical fiber or waveguide (in micrometers (um)).

        wp0 : float
            Central frequency of the pump (in 10^14 rad/s).

        sigmap : float
            Bandwidth of the pump (in 10^14 rad/s). Defined as the half-width at the 1/e intensity point.

        wlow : float
            Filter low cutoff frequency (in 10^14 s^-1).

        whigh : float
            Filter high cutoff frequency (in 10^14 s^-1).


         Attributes
        ----------
        jsa : np.ndarray
            The calculated 3D Joint Spectral Amplitude function.

        jsa_proj : np.ndarray
            A 2D projection of the JSA onto the (w2, w3) plane, used for visualization.

        density_matrix : np.ndarray
            The single-photon reduced density matrix, calculated by performing a partial trace over two parties of the tri-photon density matrix .

        eigen_modes : np.ndarray
            The eigenmodes  of the reduced density matrix. Each column corresponds to a mode over the
            frequency vector `omf` used to generate the w1, w2, w3 grid.

        eigen_values : np.ndarray
            The eigenvalues associated with the mode decomposition, representing the mode weights.

        purity : float
            The spectral purity of the state.
        """

        disp_data = topdc_calc(wf, neff_triplet, wp, neff_pump, wp_cutoff[0])

        def filter(w, wlow, whigh):
            fw = np.heaviside(w - wlow, 0) * np.heaviside(whigh - w, 0)
            return fw

        # JSA calculation
        jsa = (
            filter(w1, wlow, whigh)
            * filter(w2, wlow, whigh)
            * filter(w3, wlow, whigh)
            * np.sinc(
                (L / (2 * np.pi))
                * (
                    disp_data.kp(w1 + w2 + w3)
                    - disp_data.kf(w1)
                    - disp_data.kf(w2)
                    - disp_data.kf(w3)
                )
            )
            * np.exp(-(((w1 + w2 + w3) - wp0) ** 2) / (2 * sigmap**2))
        )

        if (
            isinstance(w1, (int, float))
            or isinstance(w2, (int, float))
            or isinstance(w3, (int, float))
            or w1.ndim < 3
            or w2.ndim < 3
            or w3.ndim < 3
        ):
            jsa_proj = None
            rho = None
            U = None
            S = None
            Vh = None
            purity = None
        else:
            # Projection of the JSA calculation
            jsa_proj = np.sum(np.square(np.abs(jsa)), axis=0)

            jsan = np.nan_to_num(jsa) / np.linalg.norm(np.nan_to_num(jsa))

            # Calculation of the reduced density matrix
            rho = np.einsum("ijk,ljk->il", jsan, np.conj(jsan))

            # Calculation of the eigenvalues of rho
            U, S, Vh = np.linalg.svd(rho, full_matrices=True, hermitian=True)

            # Calculation of purity
            purity = np.sum(S) ** 2 / np.sum(S**2)

        self.jsa = jsa
        self.jsa_proj = jsa_proj
        self.density_matrix = rho
        self.eigen_modes = U
        self.eigen_values = S
        self.purity = purity


class rho_filter_calc_auto:
    def __init__(
        self, L, wp0, wlow, whigh, wf, neff_triplet, wp, neff_pump, *wp_cutoff
    ):
        """
        Class to automatically compute the filtered Joint Spectral Amplitude (JSA) and related quantities for triplets generated by
        degenerate third-order parametric down-conversion (TOPDC).

        Parameters
        ----------
        L : float
            Length of the optical fiber or waveguide (in micrometers (um)).

        wp0 : float
            Central frequency of the pump (in 10^14 rad/s).

        sigmap : float
            Bandwidth of the pump (in 10^14 rad/s). Defined as the half-width at the 1/e intensity point.

        wlow : float
            Filter low cutoff frequency (in 10^14 s^-1).

        whigh : float
            Filter high cutoff frequency (in 10^14 s^-1).


         Attributes
        ----------
        omega : np.ndarray
            1D array of evaluation frequencies (in 10^14 rad/s).

        jsa : np.ndarray
            The calculated 3D Joint Spectral Amplitude function.

        jsa_proj : np.ndarray
            A 2D projection of the JSA onto the (w2, w3) plane, used for visualization.

        density_matrix : np.ndarray
            The single-photon reduced density matrix, calculated by performing a partial trace over two parties of the tri-photon density matrix .

        eigen_modes : np.ndarray
            The eigenmodes  of the reduced density matrix. Each column corresponds to a mode over the
            frequency vector `omf` used to generate the w1, w2, w3 grid.

        eigen_values : np.ndarray
            The eigenvalues associated with the mode decomposition, representing the mode weights.

        purity : float
            The spectral purity of the state.
        """

        disp_data = topdc_calc(wf, neff_triplet, wp, neff_pump, wp_cutoff[0])

        wf0 = disp_data.w_pm(
            wp0 / 3
        )  # Triplet phase-matching frequency in 10^14 rad/s (The input parameter 13  is the initial guess for the triplet central frequency--> You can check this by plotting the dispersion of the pump and triplet modes)

        print(
            "Zeroth-order phase-matching (L/2)\N{GREEK CAPITAL LETTER DELTA}k="
            + str((L / 2) * disp_data.delk0(wf0))
        )
        print(
            "Relative slowness s=1/vp-1/vf="
            + str(1 / disp_data.vg_p(wp0) - 1 / disp_data.vg_f(wf0))
            + " [1/(10^8 m/s)]"
        )
        print(
            "Group velocity dispersion of the triplet mode: "
            + str(disp_data.beta2_f(wf0) * 1e2)
            + " [fs^2/um]"
        )
        print(
            "Group velocity dispersion of the pump mode: "
            + str(disp_data.beta2_p(wp0) * 1e2)
            + " [fs^2/um]"
        )

        # Spectral resolution
        sigmap = disp_data.pm_bw(
            L, wp0 / 3
        )  # Approximate HWHM of the phase-matching function, measured in terms of the sinc's main lobe width
        delw = (
            sigmap / 60
        )  # The spectral resolution is defined such that a minimum of 60 discrete points are sampled within each frequency vector over the bandwidth of the phase-matching function, regardless of the specific width of the evaluation function.

        # Spectral window
        domg = (
            5 * sigmap
        )  # We set the half-width of the frequency window wide enough to sample correctly both the jsa and reduced density matrix
        omf = np.arange(
            wf0 - domg, wf0 + domg, delw
        )  # Frequency evaluation vector centered at the photon triplet phase-matching frequency with units 10^14 rad/s

        w1 = omf[:, None, None]
        w2 = omf[None, :, None]
        w3 = omf[None, None, :]

        def filter(w, wlow, whigh):
            fw = np.heaviside(w - wlow, 0) * np.heaviside(whigh - w, 0)
            return fw

        # JSA calculation
        jsa = (
            filter(w1, wlow, whigh)
            * filter(w2, wlow, whigh)
            * filter(w3, wlow, whigh)
            * np.sinc(
                (L / (2 * np.pi))
                * (
                    disp_data.kp(w1 + w2 + w3)
                    - disp_data.kf(w1)
                    - disp_data.kf(w2)
                    - disp_data.kf(w3)
                )
            )
            * np.exp(-(((w1 + w2 + w3) - wp0) ** 2) / (2 * sigmap**2))
        )

        if (
            isinstance(w1, (int, float))
            or isinstance(w2, (int, float))
            or isinstance(w3, (int, float))
            or w1.ndim < 3
            or w2.ndim < 3
            or w3.ndim < 3
        ):
            jsa_proj = None
            rho = None
            U = None
            S = None
            Vh = None
            purity = None
        else:
            # Projection of the JSA calculation
            jsa_proj = np.sum(np.square(np.abs(jsa)), axis=0)

            jsan = np.nan_to_num(jsa) / np.linalg.norm(np.nan_to_num(jsa))

            # Calculation of the reduced density matrix
            rho = np.einsum("ijk,ljk->il", jsan, np.conj(jsan))

            # Calculation of the eigenvalues of rho
            U, S, Vh = np.linalg.svd(rho, full_matrices=True, hermitian=True)

            # Calculation of purity
            purity = np.sum(S) ** 2 / np.sum(S**2)

        self.wf0 = wf0
        self.sigmap = sigmap
        self.omega = omf
        self.jsa = jsa
        self.jsa_proj = jsa_proj
        self.density_matrix = rho
        self.eigen_modes = U
        self.eigen_values = S
        self.purity = purity

    def plotter(self):
        fig, axs = plt.subplots(
            2, 2, sharex=False, sharey=False, constrained_layout=True
        )

        # JSA projection plot
        c1 = axs[0, 0].pcolor(
            self.omega,
            self.omega,
            self.jsa_proj / np.nanmax(self.jsa_proj),
            rasterized=True,
        )
        axs[0, 0].set_box_aspect(1)
        axs[0, 0].set_xlabel(r"$\omega_{1}$  [$10^{14} $ rad/s]")
        axs[0, 0].set_ylabel(r"$\omega_{2}$  [$10^{14} $ rad/s]")
        axs[0, 0].set_title(r"$s(\omega_1,\omega_2)$", fontsize=10)
        # Automatically set ticks using MaxNLocator
        axs[0, 0].xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For x-axis
        axs[0, 0].yaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For y-axis
        fig.colorbar(c1, ax=axs[0, 0])

        # Reduced density matrix plot
        cr1 = axs[0, 1].pcolor(
            self.omega, self.omega, self.density_matrix / np.max(self.density_matrix)
        )
        axs[0, 1].set_box_aspect(1)
        axs[0, 1].set_xlabel(r"$\omega_{1}$  [$10^{14} $ rad/s]")
        axs[0, 1].set_ylabel(r"$\omega_{1}'$  [$10^{14} $ rad/s]")
        axs[0, 1].set_title(r"$\rho_1(\omega_1,\omega_{1}')$", fontsize=10)
        axs[0, 1].xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For x-axis
        axs[0, 1].yaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For y-axis
        fig.colorbar(cr1, ax=axs[0, 1])

        # Weighted eigenmodes plot
        Sn = self.eigen_values / np.sum(self.eigen_values)  # Normalized eigenvalues
        Un = self.eigen_modes / abs(self.eigen_modes).max(
            axis=0
        )  # Normalized eigenmodes

        axs[1, 0].plot(self.omega, Sn[0] * Un[:, 0], label="n=0", linewidth=2.5)
        axs[1, 0].plot(self.omega, Sn[1] * Un[:, 1], label="n=1", linewidth=2.5)
        axs[1, 0].plot(self.omega, Sn[2] * Un[:, 2], label="n=2", linewidth=2.5)
        axs[1, 0].plot(self.omega, Sn[3] * Un[:, 3], label="n=3", linewidth=2.5)
        axs[1, 0].set_xlabel(r"$\omega_F$ [$10^{14} $ rad/s]")
        axs[1, 0].set_ylabel("Amplitude [arb. u]")
        axs[1, 0].set_title(r"Weighted modes", fontsize=10)
        axs[1, 0].legend(fontsize="small")

        # Mode distribution
        x = np.arange(0, len(self.eigen_values))
        plt.bar(x, Sn)

        axs[1, 1].set_xlabel(r"Mode number", fontsize=12)
        axs[1, 1].set_ylabel(r"Mode fraction ", fontsize=12)
        axs[1, 1].set_xlim(-0.75, 10)

        fig.set_edgecolor("none")
        # plt.gca().set_aspect('equal')
        fig.suptitle(r"Purity $\kappa$= {:.2f}".format(self.purity), fontsize=16)

        plt.show()

    def plotter_scaled(self):

        fig, axs = plt.subplots(
            2, 2, sharex=False, sharey=False, constrained_layout=True
        )

        # JSA projection plot
        c1 = axs[0, 0].pcolor(
            (self.omega - self.wf0) / self.sigmap,
            (self.omega - self.wf0) / self.sigmap,
            self.jsa_proj / np.nanmax(self.jsa_proj),
            rasterized=True,
        )
        axs[0, 0].set_box_aspect(1)
        axs[0, 0].set_xlabel(r"$\delta\omega_{1}/\sigma_{PM}$")
        axs[0, 0].set_ylabel(r"$\delta\omega_{2}/\sigma_{PM}$")
        axs[0, 0].set_title(r"JSI projection", fontsize=10)
        # Automatically set ticks using MaxNLocator
        axs[0, 0].xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For x-axis
        axs[0, 0].yaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For y-axis
        fig.colorbar(c1, ax=axs[0, 0])

        # Reduced density matrix plot
        cr1 = axs[0, 1].pcolor(
            (self.omega - self.wf0) / self.sigmap,
            (self.omega - self.wf0) / self.sigmap,
            self.density_matrix / np.max(self.density_matrix),
        )
        axs[0, 1].set_box_aspect(1)
        axs[0, 1].set_xlabel(r"$\delta\omega_{1}/\sigma_{PM}$")
        axs[0, 1].set_ylabel(r"$\delta\omega_{1}'/\sigma_{PM}$")
        axs[0, 1].set_title(r"$\rho_1$", fontsize=10)
        axs[0, 1].xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For x-axis
        axs[0, 1].yaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For y-axis
        fig.colorbar(cr1, ax=axs[0, 1])

        # Weighted eigenmodes plot
        Sn = self.eigen_values / np.sum(self.eigen_values)  # Normalized eigenvalues
        Un = self.eigen_modes / abs(self.eigen_modes).max(
            axis=0
        )  # Normalized eigenmodes

        axs[1, 0].plot(
            (self.omega - self.wf0) / self.sigmap,
            Sn[0] * Un[:, 0],
            label="n=0",
            linewidth=2.5,
        )
        axs[1, 0].plot(
            (self.omega - self.wf0) / self.sigmap,
            Sn[1] * Un[:, 1],
            label="n=1",
            linewidth=2.5,
        )
        axs[1, 0].plot(
            (self.omega - self.wf0) / self.sigmap,
            Sn[2] * Un[:, 2],
            label="n=2",
            linewidth=2.5,
        )
        axs[1, 0].plot(
            (self.omega - self.wf0) / self.sigmap,
            Sn[3] * Un[:, 3],
            label="n=3",
            linewidth=2.5,
        )
        axs[1, 0].set_xlabel(r"$\delta\omega_F/\sigma_{PM}$ ")
        axs[1, 0].set_ylabel("Amplitude [arb. u]")
        axs[1, 0].set_title(r"Weighted modes", fontsize=10)
        axs[1, 0].legend(fontsize="small")

        # Mode distribution
        x = np.arange(0, len(self.eigen_values))
        plt.bar(x, Sn)

        axs[1, 1].set_xlabel(r"Mode number", fontsize=12)
        axs[1, 1].set_ylabel(r"Mode fraction ", fontsize=12)
        axs[1, 1].set_xlim(-0.75, 10)

        fig.set_edgecolor("none")
        # plt.gca().set_aspect('equal')
        fig.suptitle(r"Purity $\kappa$= {:.2f}".format(self.purity), fontsize=16)

        plt.show()


class topdc_scaled_calc_auto:
    def __init__(
        self, L, wp0, wlow, whigh, wf, neff_triplet, wp, neff_pump, *wp_cutoff
    ):
        """
        Class to automatically compute the filtered Joint Spectral Amplitude (JSA) and related quantities for triplets generated by
        degenerate third-order parametric down-conversion (TOPDC).

        Parameters
        ----------
        L : float
            Length of the optical fiber or waveguide (in micrometers (um)).

        wp0 : float
            Central frequency of the pump (in 10^14 rad/s).

        sigmap : float
            Bandwidth of the pump (in 10^14 rad/s). Defined as the half-width at the 1/e intensity point.

        wlow : float
            Filter low cutoff frequency (in 10^14 s^-1).

        whigh : float
            Filter high cutoff frequency (in 10^14 s^-1).


         Attributes
        ----------
        omega : np.ndarray
            1D array of evaluation frequencies (in 10^14 rad/s).

        jsa : np.ndarray
            The calculated 3D Joint Spectral Amplitude function.

        jsa_proj : np.ndarray
            A 2D projection of the JSA onto the (w2, w3) plane, used for visualization.

        density_matrix : np.ndarray
            The single-photon reduced density matrix, calculated by performing a partial trace over two parties of the tri-photon density matrix .

        eigen_modes : np.ndarray
            The eigenmodes  of the reduced density matrix. Each column corresponds to a mode over the
            frequency vector `omf` used to generate the w1, w2, w3 grid.

        eigen_values : np.ndarray
            The eigenvalues associated with the mode decomposition, representing the mode weights.

        purity : float
            The spectral purity of the state.
        """

        disp_data = topdc_calc(wf, neff_triplet, wp, neff_pump, wp_cutoff[0])

        wf0 = disp_data.w_pm(
            wp0 / 3
        )  # Triplet phase-matching frequency in 10^14 rad/s (The input parameter 13  is the initial guess for the triplet central frequency--> You can check this by plotting the dispersion of the pump and triplet modes)

        print(
            "Zeroth-order phase-matching (L/2)\N{GREEK CAPITAL LETTER DELTA}k="
            + str((L / 2) * disp_data.delk0(wf0))
        )
        print(
            "Relative slowness s=1/vp-1/vf="
            + str(1 / disp_data.vg_p(wp0) - 1 / disp_data.vg_f(wf0))
            + " [1/(10^8 m/s)]"
        )
        print(
            "Group velocity dispersion of the triplet mode: "
            + str(disp_data.beta2_f(wf0) * 100)
            + " [fs^2/um]"
        )
        print(
            "Group velocity dispersion of the pump mode: "
            + str(disp_data.beta2_p(wp0) * 100)
            + " [fs^2/um]"
        )

        # Spectral resolution
        sigmap = disp_data.pm_bw(
            L, wp0 / 3
        )  # Approximate HWHM of the phase-matching function, measured in terms of the sinc's main lobe width
        delw = (
            1 / 60
        )  # The spectral resolution is defined such that a minimum of 60 discrete points are sampled within each frequency vector over the bandwidth of the phase-matching function, regardless of the specific width of the evaluation function.

        # Spectral window
        domg = 5  # We set the half-width of the frequency window wide enough to sample correctly both the jsa and reduced density matrix. In this case, the half width is 5 times the phase-matching bandwidth
        omf = np.arange(
            -domg, domg, delw
        )  # Frequency evaluation vector centered at the photon triplet phase-matching frequency with units 10^14 rad/s

        w1 = sigmap * omf[:, None, None] + wf0
        w2 = sigmap * omf[None, :, None] + wf0
        w3 = sigmap * omf[None, None, :] + wf0

        def filter(w, wlow, whigh):
            fw = np.heaviside(w - wlow, 0) * np.heaviside(whigh - w, 0)
            return fw

        # JSA calculation
        jsa = (
            filter(w1, wlow, whigh)
            * filter(w2, wlow, whigh)
            * filter(w3, wlow, whigh)
            * np.sinc(
                (L / (2 * np.pi))
                * (
                    disp_data.kp(w1 + w2 + w3)
                    - disp_data.kf(w1)
                    - disp_data.kf(w2)
                    - disp_data.kf(w3)
                )
            )
            * np.exp(-(((w1 + w2 + w3) - wp0) ** 2) / (2 * sigmap**2))
        )

        if (
            isinstance(w1, (int, float))
            or isinstance(w2, (int, float))
            or isinstance(w3, (int, float))
            or w1.ndim < 3
            or w2.ndim < 3
            or w3.ndim < 3
        ):
            jsa_proj = None
            rho = None
            U = None
            S = None
            Vh = None
            purity = None
        else:
            # Projection of the JSA calculation
            jsa_proj = np.sum(np.square(np.abs(jsa)), axis=0)

            jsan = np.nan_to_num(jsa) / np.linalg.norm(np.nan_to_num(jsa))

            # Calculation of the reduced density matrix
            rho = np.einsum("ijk,ljk->il", jsan, np.conj(jsan))

            # Calculation of the eigenvalues of rho
            U, S, Vh = np.linalg.svd(rho, full_matrices=True, hermitian=True)

            # Calculation of purity
            purity = np.sum(S) ** 2 / np.sum(S**2)

        self.wf0 = wf0
        self.sigmap = sigmap
        self.omega = omf
        self.jsa = jsa
        self.jsa_proj = jsa_proj
        self.density_matrix = rho
        self.eigen_modes = U
        self.eigen_values = S
        self.purity = purity

    def plotter_scaled(self):

        fig, axs = plt.subplots(
            2, 2, sharex=False, sharey=False, constrained_layout=True
        )

        # JSA projection plot
        c1 = axs[0, 0].pcolor(
            (self.omega),
            (self.omega),
            self.jsa_proj / np.nanmax(self.jsa_proj),
            rasterized=True,
        )
        axs[0, 0].set_box_aspect(1)
        axs[0, 0].set_xlabel(r"$\delta\omega_{1}/\sigma_{PM}$")
        axs[0, 0].set_ylabel(r"$\delta\omega_{2}/\sigma_{PM}$")
        axs[0, 0].set_title(r"JSI projection", fontsize=10)
        # Automatically set ticks using MaxNLocator
        axs[0, 0].xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For x-axis
        axs[0, 0].yaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For y-axis
        fig.colorbar(c1, ax=axs[0, 0])

        # Reduced density matrix plot
        cr1 = axs[0, 1].pcolor(
            (self.omega),
            (self.omega),
            self.density_matrix / np.max(self.density_matrix),
        )
        axs[0, 1].set_box_aspect(1)
        axs[0, 1].set_xlabel(r"$\delta\omega_{1}/\sigma_{PM}$")
        axs[0, 1].set_ylabel(r"$\delta\omega_{1}'/\sigma_{PM}$")
        axs[0, 1].set_title(r"$\rho_1$", fontsize=10)
        axs[0, 1].xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For x-axis
        axs[0, 1].yaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=6)
        )  # For y-axis
        fig.colorbar(cr1, ax=axs[0, 1])

        # Weighted eigenmodes plot
        Sn = self.eigen_values / np.sum(self.eigen_values)  # Normalized eigenvalues
        Un = self.eigen_modes / abs(self.eigen_modes).max(
            axis=0
        )  # Normalized eigenmodes

        axs[1, 0].plot(
            (self.omega),
            Sn[0] * Un[:, 0],
            label="n=0",
            linewidth=2.5,
        )
        axs[1, 0].plot(
            (self.omega),
            Sn[1] * Un[:, 1],
            label="n=1",
            linewidth=2.5,
        )
        axs[1, 0].plot(
            (self.omega),
            Sn[2] * Un[:, 2],
            label="n=2",
            linewidth=2.5,
        )
        axs[1, 0].plot(
            (self.omega),
            Sn[3] * Un[:, 3],
            label="n=3",
            linewidth=2.5,
        )
        axs[1, 0].set_xlabel(r"$\delta\omega_F/\sigma_{PM}$ ")
        axs[1, 0].set_ylabel("Amplitude [arb. u]")
        axs[1, 0].set_title(r"Weighted modes", fontsize=10)
        axs[1, 0].legend(fontsize="small")

        # Mode distribution
        x = np.arange(0, len(self.eigen_values))
        plt.bar(x, Sn)

        axs[1, 1].set_xlabel(r"Mode number", fontsize=12)
        axs[1, 1].set_ylabel(r"Mode fraction ", fontsize=12)
        axs[1, 1].set_xlim(-0.75, 10)

        fig.set_edgecolor("none")
        # plt.gca().set_aspect('equal')
        fig.suptitle(r"Purity $\kappa$= {:.2f}".format(self.purity), fontsize=16)

        plt.show()

import numpy as np

def dM_SN(m_core, a_SN = -0.9, b_SN = 13.9, m_thresh = 14.8, f_sn=None, dM=0.5):
    """Mass lost from SN explosion.
    
    Parameters
    ----------
    a_sn : float
        Linear function parameter (slope)
    b_sn : float
        Linear function parameter (y-intercept)
    m_thresh : float
        Cutoff mass after which dM_SN becomes constant. Before this cutoff, dM_SN 
        is a linear function.
    f_sn : float or None
        If none, dM_SN is a linear function with a cutoff at m_thresh. If a float
        between 0 and 1 is provided, dM_SN is that fraction of the core mass provided.
    dM : float
        dM_SN is equal to dM above the cutoff instead of zero.
        
    Notes
    -----
    If both f_sn and dM are provided, f_sn will override dM.
    """
    
    if (f_sn is not None) and (f_sn < 1) and (f_sn > 0):
        return f_sn * m_core
    
    threshold = (m_core <= m_thresh)
    dM_SN = (a_SN * m_core + b_SN) * threshold
    
    if (dM > 0):
        dM_SN += dM * (1-threshold)
        
    return dM_SN


# Updated, more flexible functions

def mass_BH_a(m_zams_a, f_core = 0.34, f_winds=None, f_ppi=None, ppi_thresh=40, **kwargs):
    """Mass of black hole formed from primary star of a given mass. 

    Parameters
    ----------
    m_zams_a : float
        The mass of the primary star in Msun.
    f_acc : float
        Fraction of mass that is accreted by the secondary star during the first
        phase of mass transfer.
    f_core : float
        Fraction of star's mass that is in the core.
    f_winds : float
        Fraction of mass lost due to winds.
    f_ppi : float
        Linear slope of piecewise function describing the mass loss due to 
        pulsational pair instability, where the x-intercept is at ppi_thresh.
    ppi_thresh : float
        Lower threshold for pulsational pair instability mass loss.

    Return
    ------
    float
        Mass in Msun
    """
    
    m_core_a = f_core * m_zams_a
    
    if (f_winds is not None) and (f_winds < 1) and (f_winds > 0):
        m_core_a *= (1 - f_winds)
        
    if (f_ppi is not None) and (f_ppi < 1) and (f_ppi > 0) and (ppi_thresh > 0):
        dm_ppi = f_ppi * (m_core_a - ppi_thresh) * (m_core_a > ppi_thresh)
        m_core_a -= dm_ppi
        
    return m_core_a - dM_SN(m_core_a, **kwargs)


def mass_BH_b(m_zams_a, m_zams_b, f_acc, f_core = 0.34, f_winds=None, f_ppi=None, ppi_thresh=40, **kwargs):
    """Mass of black hole formed from secondary star, given a pair of masses.

    Parameters
    ----------
    m_zams_a : float
        The mass of the primary star in Msun.
    m_zams_b : float
        The mass of the secondary star in Msun.
    q_crit_2 : float
        Critical mass ratio at which mass transfer becomes unstable. Assumed to
        be M_b / M_a, where a and be refer to the objects that were the 
        primary and secondary star, respectively.
    f_acc : float
        Fraction of mass that is accreted by the secondary star during the first
        phase of mass transfer.

    Return
    ------
    float
        Mass in Msun
    """
    
    m_post_mt1_b = m_zams_b + f_acc * (1 - f_core) * m_zams_a
    m_core_b = f_core * m_post_mt1_b
    
    if (f_winds is not None) and (f_winds < 1) and (f_winds > 0):
        m_core_b *= (1 - f_winds)
        
    if (f_ppi is not None) and (f_ppi < 1) and (f_ppi > 0) and (ppi_thresh > 0):
        dm_ppi = f_ppi * (m_core_b - ppi_thresh) * (m_core_b > ppi_thresh)
        m_core_b -= dm_ppi
        
    return m_core_b - dM_SN(m_core_b, **kwargs)


def m_zams_b_limit(m_zams_a, q_crit_2, f_acc, f_core = 0.34, f_winds=None, f_ppi=None, ppi_thresh=40, **kwargs):
    """Maximum possible ZAMS mass of the secondary star that can stay stable.

    Parameters
    ----------
    m_zams_a : float
        The mass of the primary star in Msun.
    q_crit_2 : float
        Critical mass ratio at which mass transfer becomes unstable. Assumed to
        be M_b / M_a, where a and be refer to the objects that were the 
        primary and secondary star, respectively.
    f_acc : float
        Fraction of mass that is accreted by the secondary star during the first
        phase of mass transfer.

    Return
    ------
    float
        Mass in Msun
    """
    
    m_bh_a = mass_BH_a(m_zams_a, f_core, f_winds, f_ppi, ppi_thresh, **kwargs)
    m_accreted = f_acc * (1 - f_core) * m_zams_a
    return q_crit_2 * m_bh_a - m_accreted


def interp_min_bh_masses(q_zams, q_crit_2, f_acc, f_core, **kwargs):
    """For a given array of q_zams values, return the minimum BH masses."""
    # Calculate everything as a function of m_zams,a
    m_zams_a = np.linspace(5, 150)
    m_zams_b = m_zams_b_limit(m_zams_a, q_crit_2, f_acc, f_core, **kwargs)
    min_a = mass_BH_a(m_zams_a, f_core, **kwargs)
    min_b = mass_BH_b(m_zams_a, m_zams_b, f_acc, f_core, **kwargs)
    q = min_b / min_a
    
    # Interpolate so it's a function of q_zams
    m_a = np.interp(q_zams, q, min_a)
    m_b = np.interp(q_zams, q, min_b)
    return m_a, m_b

def piecewise_f_core(M_star, m_turn=58, f_turn=0.41, slope_1=(0.1/55.), slope_2=(0.07 / 95.)):
    return f_turn + slope_1 * (M_star - m_turn) * (M_star < m_turn) + slope_2 * (M_star - m_turn) * (M_star >= m_turn)



# Original NPWV prescriptions

# def min_zams_a(q_crit_2, f_acc, q_zams, f_core = 0.34, a_SN = -0.9, b_SN = 13.9):
#     """Minimum ZAMS mass of primary that still forms a BH, from NPWV.
#     """
    
#     numerator = b_SN * q_crit_2
#     denominator = q_crit_2 * f_core * (1 - a_SN) - f_acc * (1 - f_core) - q_zams
#     return numerator / denominator


# def min_BH_a(q_crit_2, f_acc, q_zams, f_core = 0.34, **kwargs):
#     """Minimum mass of black hole formed from primary star.

#     Parameters
#     ----------
#     q_crit_2 : float
#         Critical mass ratio at which mass transfer becomes unstable. Assumed to
#         be M_b / M_a, where a and be refer to the objects that were the 
#         primary and secondary star, respectively.
#     f_acc : float
#         Fraction of mass that is retained by the secondary star during the first
#         phase of mass transfer.
#     q_zams : float
#         Mass ratio of the binary at formation (ZAMS).

#     Return
#     ------
#     float
#         Mass in Msun
#     """
#     m_zams_a_val = min_zams_a(q_crit_2, f_acc, q_zams, f_core, **kwargs)
#     m_core_a = f_core * m_zams_a_val
#     return m_core_a - dM_SN(m_core_a, **kwargs)

# def min_BH_b(q_crit_2, f_acc, q_zams, f_core = 0.34, **kwargs):
#     """Minimum mass of black hole formed from secondary star.

#     Parameters
#     ----------
#     q_crit_2 : float
#         Critical mass ratio at which mass transfer becomes unstable. Assumed to
#         be M_b / M_a, where a and be refer to the objects that were the 
#         primary and secondary star, respectively.
#     f_acc : float
#         Fraction of mass that is retained by the secondary star during the first
#         phase of mass transfer.
#     q_zams : float
#         Mass ratio of the binary at formation (ZAMS).

#     Return
#     ------
#     float
#         Mass in Msun
#     """
#     m_post_mt1 = q_crit_2 * min_BH_a(q_crit_2, f_acc, q_zams, f_core, **kwargs)
#     m_core_b = m_post_mt1 * f_core
#     return m_core_b - dM_SN(m_core_b, **kwargs)
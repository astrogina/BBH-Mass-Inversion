import numpy as np
import pandas as pd

def get_mass_transfer_df(file_name):
    """
    Get BBH mergers and information about their mass transfer history.

    Parameters
    ----------
    file_name : string
        path to COSMIC dat file

    Returns 
    -------
    pandas DataFrame
        Returns a dataframe where each row corresponds to one BBH merger.
        Columns are the COSMIC bin_num, number of mass transfer events, number 
        of common envelope events, number of times the primary star initiates 
        mass transfer, number of times the secondary star initiates mass 
        transfer, the kstar of the first primary RLOF, the kstar of the first 
        secondary RLOF, mass of BH_1, mass of BH_2, time to BBH formation, time 
        to BBH merger, M_1 at ZAMS, M_2 at ZAMS, if the binary is mass flipped, 
        mass of BBH primary, mass of BBH secondary, and mass ratio at ZAMS 
    """
    
    # Grab relevant bin_nums from bcm
    bcm = pd.read_hdf(file_name, key='bcm', columns=['bin_num', 'merger_type'])
    
    mass_transfer_df = pd.DataFrame(index=bcm.loc[bcm.merger_type == '1414'].bin_num.values, columns=['bin_num', 'num_mt', 'num_ce', 'num_rlof_1', 'num_rlof_2', 'mt_kstar_1', 'mt_kstar_2', 'COSMIC_mass_1', 'COSMIC_mass_2','formation_time', 'merger_time', 'zams_mass_1', 'zams_mass_2', 'mass_flipped','bbh_pri_mass', 'bbh_sec_mass', 'q_zams'])
    
    mass_transfer_df['bin_num'] = bcm.loc[bcm.merger_type == '1414'].bin_num.values.astype(float)
    
    # Set count columns to zero. Not sure if this is necessary
    count_cols = ['num_mt', 'num_ce', 'num_rlof_1', 'num_rlof_2']
    for col in count_cols:
        mass_transfer_df[col].values[:] = 0
    
    # Grab bpp
    bpp = pd.read_hdf(file_name, key='bpp', 
                      columns=['bin_num', 'tphys', 'kstar_1', 'kstar_2', 'evol_type', 'mass_1', 'mass_2', 'RRLO_1', 'RRLO_2'])
    
    # Make sure we're matching bin_nums
    # Also not sure if this is necessary who knows
    bpp = bpp.loc[bpp.bin_num.isin(mass_transfer_df.bin_num)]
    # bin_num_mask = mass_transfer_df.bin_num.isin(bpp.bin_num.unique())

    # Store ZAMS masses
    is_zams = (bpp.evol_type == 1)
    mass_transfer_df.loc[bpp.loc[is_zams].bin_num.values, 'zams_mass_1'] = bpp.loc[is_zams].mass_1.values.astype(float)
    mass_transfer_df.loc[bpp.loc[is_zams].bin_num.values, 'zams_mass_2'] = bpp.loc[is_zams].mass_2.values.astype(float)

    # Store BBH masses 
    is_bbh = (bpp.kstar_1 == 14) & (bpp.kstar_2 == 14)
    bbh_formation = (bpp.evol_type == 2) | (bpp.evol_type == 4)
    bbh_formation = bbh_formation & is_bbh
    mass_transfer_df.loc[bpp.loc[bbh_formation].bin_num.values, 'COSMIC_mass_1'] = bpp.loc[bbh_formation].mass_1.values.astype(float)
    mass_transfer_df.loc[bpp.loc[bbh_formation].bin_num.values, 'COSMIC_mass_2'] = bpp.loc[bbh_formation].mass_2.values.astype(float)

    # Store delay time
    mass_transfer_df.loc[bpp.loc[bbh_formation].bin_num.values, 'formation_time'] = bpp.loc[bbh_formation].tphys.values.astype(float)

    # Store merger time
    bbh_merger = bpp.evol_type == 6
    mass_transfer_df.loc[bpp.loc[bbh_merger].bin_num.values, 'merger_time'] = bpp.loc[bbh_merger].tphys.values.astype(float)

    # Counting mass transfers, but complicated
    # There's definitely a way to do this in a loop I'm just dumb and tired
    counts = bpp.loc[bpp.evol_type == 4].value_counts('bin_num')
    mass_transfer_df.loc[counts.index, 'num_mt'] += counts.values.astype(float)

    counts = bpp.loc[bpp.evol_type == 7].value_counts('bin_num')
    mass_transfer_df.loc[counts.index, 'num_ce'] += counts.values.astype(float)

    # Count number of times each star transfers mass (initiates RLOF) and record kstar of donor 
    is_not_bbh = np.invert(is_bbh)
    primary_rlof = (bpp.evol_type == 3) & (bpp.RRLO_1 > 1)
    secondary_rlof = (bpp.evol_type == 3) & (bpp.RRLO_2 > 1)

    counts = bpp.loc[primary_rlof & is_not_bbh].value_counts('bin_num')
    mass_transfer_df.loc[counts.index, 'num_rlof_1'] += counts.values.astype(float)
    mass_transfer_df.loc[counts.index, 'mt_kstar_1'] = bpp.loc[primary_rlof & is_not_bbh].groupby('bin_num').nth(0).astype(float)

    counts = bpp.loc[secondary_rlof & is_not_bbh].value_counts('bin_num')
    mass_transfer_df.loc[counts.index, 'num_rlof_2'] += counts.values.astype(float)
    mass_transfer_df.loc[counts.index, 'mt_kstar_2'] = bpp.loc[secondary_rlof & is_not_bbh].groupby('bin_num').nth(0).astype(float)
    
        
    # Store BBH masses in easier to use form
    mass_transfer_df['bbh_pri_mass'] = mass_transfer_df[['COSMIC_mass_1', 'COSMIC_mass_2']].max(axis=1).astype(float)
    mass_transfer_df['bbh_sec_mass'] = mass_transfer_df[['COSMIC_mass_1', 'COSMIC_mass_2']].min(axis=1).astype(float)
    
    # Check if binary is mass_flipped
    mass_transfer_df['mass_flipped'] = mass_transfer_df.COSMIC_mass_2 > mass_transfer_df.COSMIC_mass_1.astype(float)
    
    # Get q_zams
    mass_transfer_df['q_zams'] = mass_transfer_df.zams_mass_2.values / mass_transfer_df.zams_mass_1.values.astype(float)
    
    return mass_transfer_df

# Analytic equations of minimum black hole mass assuming 2 mass transfer events.
# Based on section 2 of No Peaks Without Valleys (van Son 2022).

a_SN = -0.9
b_SN = 13.9
f_core = 0.34
m_thresh = 14.8 # Msun

def dM_SN(m_core):
    """
    Mass lost from SN explosion.
    """
    threshold = (m_core <= m_thresh)
    dM_SN = (a_SN * m_core + b_SN) * threshold
    return dM_SN

def min_zams_a(q_crit_2, f_acc, q_zams):
    """
    Minimum ZAMS mass of primary that still forms a BH.
    """
    numerator = b_SN * q_crit_2
    denominator = q_crit_2 * f_core * (1 - a_SN) - f_acc * (1 - f_core) - q_zams
    return numerator / denominator

def min_BH_a(q_crit_2, f_acc, q_zams):
    """
    Minimum mass of black hole formed from primary star.

    Parameters
    ----------
    q_crit_2 : float
        Critical mass ratio at which mass transfer becomes unstable. Assumed to
        be M_b / M_a, where a and be refer to the objects that were the 
        primary and secondary star, respectively.
    f_acc : float
        Fraction of mass that is retained by the secondary star during the first
        phase of mass transfer.
    q_zams : float
        Mass ratio of the binary at formation (ZAMS).

    Return
    ------
    float
        Mass in Msun
    """
    m_zams_a_val = min_zams_a(q_crit_2, f_acc, q_zams)
    m_core_a = f_core * m_zams_a_val
    return m_core_a - dM_SN(m_core_a)

def min_BH_b(q_crit_2, f_acc, q_zams):
    """
    Minimum mass of black hole formed from secondary star.

    Parameters
    ----------
    q_crit_2 : float
        Critical mass ratio at which mass transfer becomes unstable. Assumed to
        be M_b / M_a, where a and be refer to the objects that were the 
        primary and secondary star, respectively.
    f_acc : float
        Fraction of mass that is retained by the secondary star during the first
        phase of mass transfer.
    q_zams : float
        Mass ratio of the binary at formation (ZAMS).

    Return
    ------
    float
        Mass in Msun
    """
    m_post_mt1 = q_crit_2 * min_BH_a(q_crit_2, f_acc, q_zams)
    m_core_b = m_post_mt1 * f_core
    return m_core_b - dM_SN(m_core_b)
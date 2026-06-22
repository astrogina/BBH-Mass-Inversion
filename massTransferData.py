import numpy as np
import pandas as pd


# TODO: make it possible to enter a list of optional column names
def get_mass_transfer_data(dat_file, merging_bbhs_only=True, chunksize=None, optional_columns=None):
    """Get mass transfer information for all systems in a given COSMIC 
    dat file.

    Parameters
    ----------
    file_name : string
        path to COSMIC dat file
    merging_bbhs_only : bool
        If True, only consider BBHs that merge within COSMIC simulation. 
        If False, will collect data on all systems in the dat file.
    chunk_size : int
        Maximum chunk size of bpp to load at a time. If None, will load 
        the whole bpp at once.
    optional_columns : list of strings
        A list of optional columns to include in the dataframe. 
        #TODO: list the options here

    Returns 
    -------
    pandas DataFrame
        Returns a dataframe where each row corresponds to one BBH merger.
        The default are the COSMIC bin_num, number of mass transfer events, number 
        of common envelope events, number of times the primary star initiates 
        mass transfer, number of times the secondary star initiates mass 
        transfer, the kstar of the first primary RLOF, the kstar of the first 
        secondary RLOF, mass of BH_a, mass of BH_b, time to BBH formation, time 
        to BBH merger, M_a at ZAMS, M_b at ZAMS, if the binary is mass ratio reversed, 
        mass of BBH primary, mass of BBH secondary, and mass ratio at ZAMS.
        # TODO: update description
    """

    mass_transfer_df = None
    
    if chunksize is not None and chunksize >= 25:
        start_line = 0
        stop_line = start_line + chunksize
        # Grab bpp
        bpp = pd.read_hdf(dat_file, key='bpp', start=start_line, stop=stop_line)

        while len(bpp) > 0:
            # Adjust to only contain up to the last complete evolution
            bpp.set_index(np.arange(0, len(bpp)), inplace=True)
            last_row = bpp.loc[bpp.evol_type == 10].index[-1]
            bpp = bpp.iloc[0:last_row+1]

            if merging_bbhs_only:
                merging_bbhs = bpp.loc[(bpp.evol_type == 3) & (bpp.kstar_1 == 14) & (bpp.kstar_2 == 14)].bin_num
                bpp = bpp.loc[bpp.bin_num.isin(merging_bbhs)]

            # there may not be anything in this chunk after cuts, so need to check
            if len(bpp) > 0:
                mt_df = process_bpp(bpp, optional_columns)
                
                if mass_transfer_df is None: 
                    mass_transfer_df = mt_df
                else:
                    mass_transfer_df = pd.concat((mass_transfer_df, mt_df))

            # set next chunk
            start_line = start_line + last_row + 1
            stop_line = start_line + chunksize
            bpp = pd.read_hdf(dat_file, key='bpp', start=start_line, end=stop_line)
        
        return mass_transfer_df

    if chunksize is None:
        bpp = pd.read_hdf(dat_file, key='bpp')
        if merging_bbhs_only:
            merging_bbhs = bpp.loc[(bpp.evol_type == 3) & (bpp.kstar_1 == 14) & (bpp.kstar_2 == 14)].bin_num
            bpp = bpp.loc[bpp.bin_num.isin(merging_bbhs)]
            mass_transfer_df = process_bpp(bpp, optional_columns)
            return mass_transfer_df

    if chunksize < 25:
        print('chunksize is too small--may not be enough to load an entire binary evolution')


def process_bpp(bpp, optional_columns):
    """
    Gets mass transfer data for a given bpp.
    """
    # Make sure the index is equal to the bin_num
    bpp.set_index('bin_num', drop=False, inplace=True)
      
    mass_transfer_df = pd.DataFrame(index=bpp.bin_num.unique(), columns=['bin_num', 'num_mt', 'num_ce', 'num_rlof_1', 'num_rlof_2', 'mt_kstar_1', 'mt_kstar_2', 'mt_kstar_1_change', 'mt_kstar_2_change','COSMIC_mass_1', 'COSMIC_mass_2','formation_time', 'merger_time', 'mass_1_zams', 'mass_2_zams', 'mass_flipped','bbh_pri_mass', 'bbh_sec_mass', 'q_zams', 'sn_mass_loss_1', 'sn_mass_loss_2', 'smt_to_ce1', 'smt_to_ce2','a_pre_mt1', 'a_pre_mt2', 'a_kstar_change_mt1', 'a_kstar_change_mt2', 'a_pre_ce1', 'a_pre_ce2', 'a_post_mt1', 'a_post_mt2', 'porb_pre_mt1', 'porb_pre_mt2', 'porb_pre_ce1', 'porb_pre_ce2', 'porb_post_mt1', 'porb_post_mt2', 'r_don_pre_mt1', 'r_don_pre_mt2', 'r_don_kstar_change_mt1', 'r_don_kstar_change_mt2', 'r_don_pre_ce1', 'r_don_pre_ce2', 'time_to_ce1', 'time_to_ce2', 'mass_1_pre_mt1', 'mass_2_pre_mt1', 'mass_1_post_mt1', 'mass_2_post_mt1', 'mass_1_pre_ce1', 'mass_2_pre_ce1', 'mass_1_post_ce1', 'mass_2_post_ce1', 'mass_1_pre_mt2', 'mass_2_pre_mt2', 'mass_1_post_mt2', 'mass_2_post_mt2', 'mass_1_pre_ce2', 'mass_2_pre_ce2', 'mass_1_post_ce2', 'mass_2_post_ce2', 'sep_zams', 'porb_zams', 'ecc_zams', 'rad_1_zams', 'rad_2_zams', 'sep_bbh_form', 'porb_bbh_form','tphys_bbh_form', 'ecc_bbh_form', 'bbh_merger'])
    mass_transfer_df.loc[:, 'bin_num'] = bpp.bin_num.unique()
    
    # Set count columns to zero. Not sure if this is necessary
    count_and_true_false_cols = ['num_mt', 'num_ce', 'num_rlof_1', 'num_rlof_2', 
                                 'smt_to_ce1', 'smt_to_ce2']
                                # , 'bbh_merger']
    for col in count_and_true_false_cols:
        mass_transfer_df.loc[:, col] = 0

    # Store ZAMS masses
    is_zams = (bpp.evol_type == 1)
    mass_transfer_df['mass_1_zams'] = bpp.loc[is_zams].mass_1
    mass_transfer_df['mass_2_zams'] = bpp.loc[is_zams].mass_2
    mass_transfer_df['q_zams'] = mass_transfer_df.mass_2_zams / mass_transfer_df.mass_1_zams
    mass_transfer_df['sep_zams'] = bpp.loc[is_zams].sep
    mass_transfer_df['porb_zams'] = bpp.loc[is_zams].porb
    mass_transfer_df['ecc_zams'] = bpp.loc[is_zams].ecc
    mass_transfer_df['rad_1_zams'] = bpp.loc[is_zams].rad_1
    mass_transfer_df['rad_2_zams'] = bpp.loc[is_zams].rad_2

    # Store BBH info 
    is_bbh = (bpp.kstar_1 == 14) & (bpp.kstar_2 == 14)
    bbh_formation = bpp.loc[is_bbh][~bpp.loc[is_bbh].index.duplicated(keep='first')]
    mass_transfer_df['COSMIC_mass_1'] = bbh_formation.mass_1
    mass_transfer_df['COSMIC_mass_2'] = bbh_formation.mass_2
    mass_transfer_df['sep_bbh_form'] = bbh_formation.sep
    mass_transfer_df['porb_bbh_form'] = bbh_formation.porb
    mass_transfer_df['tphys_bbh_form'] = bbh_formation.tphys
    mass_transfer_df['ecc_bbh_form'] = bbh_formation.ecc
    mass_transfer_df['bbh_pri_mass'] = mass_transfer_df[['COSMIC_mass_1', 'COSMIC_mass_2']].max(axis=1)
    mass_transfer_df['bbh_sec_mass'] = mass_transfer_df[['COSMIC_mass_1', 'COSMIC_mass_2']].min(axis=1)
    mass_transfer_df['mass_flipped'] = mass_transfer_df.COSMIC_mass_2 > mass_transfer_df.COSMIC_mass_1
    mass_transfer_df['formation_time'] = bbh_formation.tphys

    bbh_merger = is_bbh & (bpp.evol_type == 3)
    mass_transfer_df.loc[mass_transfer_df.bin_num.isin(bpp.loc[bbh_merger].bin_num), 'bbh_merger'] = 1.
    mass_transfer_df['merger_time'] = bpp.loc[bbh_merger].tphys

    #SN mass loss
    mass_transfer_df['sn_mass_loss_1'] = bpp.loc[(bpp.evol_type == 15)].mass_1 - bpp.loc[np.roll(bpp.evol_type == 15, 1)].mass_1
    mass_transfer_df['sn_mass_loss_2'] = bpp.loc[(bpp.evol_type == 16)].mass_2 - bpp.loc[np.roll(bpp.evol_type == 16, 1)].mass_2

    # Counting mass transfers, but complicated
    # There's definitely a way to do this in a loop I'm just dumb and tired
    counts = bpp.loc[bpp.evol_type == 4].bin_num.value_counts()
    mass_transfer_df.loc[counts.index, 'num_mt'] += counts.values.astype(float)

    counts = bpp.loc[bpp.evol_type == 7].bin_num.value_counts()
    mass_transfer_df.loc[counts.index, 'num_ce'] += counts.values.astype(float)

    # Count number of times each star transfers mass (initiates RLOF)
    is_not_bbh = np.invert(is_bbh)
    primary_rlof = (bpp.evol_type == 3) & (bpp.RRLO_1 > 1)
    secondary_rlof = (bpp.evol_type == 3) & (bpp.RRLO_2 > 1)

    counts = bpp.loc[primary_rlof & is_not_bbh].bin_num.value_counts()
    mass_transfer_df.loc[counts.index, 'num_rlof_1'] += counts.values.astype(float)
    counts = bpp.loc[secondary_rlof & is_not_bbh].bin_num.value_counts()
    mass_transfer_df.loc[counts.index, 'num_rlof_2'] += counts.values.astype(float)

    # Get kstar of donor for first time that star initiates mass transfer 
    mt_1 = primary_rlof & is_not_bbh
    mass_transfer_df['mt_kstar_1'] = bpp.loc[mt_1][~bpp.loc[mt_1].index.duplicated(keep='first')].kstar_1
    mt_2 = secondary_rlof & is_not_bbh
    mass_transfer_df['mt_kstar_2'] = bpp.loc[mt_2][~bpp.loc[mt_2].index.duplicated(keep='first')].kstar_2

    # Setting index b/c it helps me visualize, not sure if this is necessary
    bpp.set_index(np.arange(0, len(bpp)), inplace=True)
    # Grab general mass transfer info, yes this is really sloppy, sue me
    smt1_start = (bpp.evol_type == 3) & (bpp.RRLO_1 > 1) & (bpp.kstar_1 < 14)
    indexes = bpp.loc[smt1_start].index
    if len(indexes) > 0:
        bpp_subset = pd.concat([bpp.iloc[indexes[i]:indexes[i]+7] for i in range(len(indexes))])
        bpp_subset.set_index('bin_num', drop=False, inplace=True)
        
        bpp_subset_start = bpp_subset.loc[bpp_subset.evol_type == 3]
        bpp_subset_end = bpp_subset.loc[bpp_subset.evol_type == 4]
        bpp_subset_kstar_change = bpp_subset.loc[(bpp_subset.evol_type == 2) & (bpp_subset.RRLO_1 > 1)]
        
        mass_transfer_df['mt_kstar_1_change'] = bpp_subset_kstar_change.loc[(~bpp_subset_kstar_change.index.duplicated(keep='first'))].kstar_1
        mass_transfer_df['mass_1_pre_mt1'] = bpp_subset_start.loc[(~bpp_subset_start.index.duplicated(keep='first'))].mass_1
        mass_transfer_df['mass_1_post_mt1'] = bpp_subset_end.loc[(~bpp_subset_end.index.duplicated(keep='first'))].mass_1
        mass_transfer_df['mass_2_pre_mt1'] = bpp_subset_start.loc[(~bpp_subset_start.index.duplicated(keep='first'))].mass_2
        mass_transfer_df['mass_2_post_mt1'] = bpp_subset_end.loc[(~bpp_subset_end.index.duplicated(keep='first'))].mass_2
        mass_transfer_df['a_pre_mt1'] = bpp_subset_start.loc[(~bpp_subset_start.index.duplicated(keep='first'))].sep
        mass_transfer_df['a_kstar_change_mt1'] = bpp_subset_kstar_change.loc[(~bpp_subset_kstar_change.index.duplicated(keep='first'))].sep
        mass_transfer_df['a_post_mt1'] = bpp_subset_end.loc[(~bpp_subset_end.index.duplicated(keep='first'))].sep
        mass_transfer_df['porb_pre_mt1'] = bpp_subset_start.loc[(~bpp_subset_start.index.duplicated(keep='first'))].porb
        mass_transfer_df['porb_post_mt1'] = bpp_subset_end.loc[(~bpp_subset_end.index.duplicated(keep='first'))].porb
        mass_transfer_df['r_don_pre_mt1'] = bpp_subset_start.loc[(~bpp_subset_start.index.duplicated(keep='first'))].rad_1
        mass_transfer_df['r_don_kstar_change_mt1'] = bpp_subset_kstar_change.loc[(~bpp_subset_kstar_change.index.duplicated(keep='first'))].rad_1
        
    smt2_start = (bpp.evol_type == 3) & (bpp.RRLO_2 > 1) & (bpp.kstar_2 < 14)
    indexes = bpp.loc[smt2_start].index
    if len(indexes) > 0:
        bpp_subset = pd.concat([bpp.iloc[indexes[i]:indexes[i]+7] for i in range(len(indexes))])
        bpp_subset.set_index('bin_num', drop=False, inplace=True)
        
        bpp_subset_start = bpp_subset.loc[bpp_subset.evol_type == 3]
        bpp_subset_end = bpp_subset.loc[bpp_subset.evol_type == 4]
        bpp_subset_kstar_change = bpp_subset.loc[(bpp_subset.evol_type == 2) & (bpp_subset.RRLO_2 > 1)]

        mass_transfer_df['mt_kstar_2_change'] = bpp_subset_kstar_change.loc[(~bpp_subset_kstar_change.index.duplicated(keep='first'))].kstar_1
        mass_transfer_df['mass_1_pre_mt2'] = bpp_subset_start.loc[(~bpp_subset_start.index.duplicated(keep='first'))].mass_1
        mass_transfer_df['mass_1_post_mt2'] = bpp_subset_end.loc[(~bpp_subset_end.index.duplicated(keep='first'))].mass_1
        mass_transfer_df['mass_2_pre_mt2'] = bpp_subset_start.loc[(~bpp_subset_start.index.duplicated(keep='first'))].mass_2
        mass_transfer_df['mass_2_post_mt2'] = bpp_subset_end.loc[(~bpp_subset_end.index.duplicated(keep='first'))].mass_2
        mass_transfer_df['a_pre_mt2'] = bpp_subset_start.loc[(~bpp_subset_start.index.duplicated(keep='first'))].sep
        mass_transfer_df['a_kstar_change_mt2'] = bpp_subset_kstar_change.loc[(~bpp_subset_kstar_change.index.duplicated(keep='first'))].sep
        mass_transfer_df['a_post_mt2'] = bpp_subset_end.loc[(~bpp_subset_end.index.duplicated(keep='first'))].sep
        mass_transfer_df['porb_pre_mt2'] = bpp_subset_start.loc[(~bpp_subset_start.index.duplicated(keep='first'))].porb
        mass_transfer_df['porb_post_mt2'] = bpp_subset_end.loc[(~bpp_subset_end.index.duplicated(keep='first'))].porb
        mass_transfer_df['r_don_pre_mt2'] = bpp_subset_start.loc[(~bpp_subset_start.index.duplicated(keep='first'))].rad_2
        mass_transfer_df['r_don_kstar_change_mt2'] = bpp_subset_kstar_change.loc[(~bpp_subset_kstar_change.index.duplicated(keep='first'))].rad_2
    
    # Grabbing all SMT -> CE binaries
    evol_to_ce = (bpp['evol_type'] == 3) & (bpp['evol_type'].shift(-1) == 2)
    evol_to_ce &= (bpp['evol_type'].shift(-2) == 7) & (bpp['evol_type'].shift(-3) == 8)
    
    # MT1 is SMT -> CE
    evol_to_ce1 = evol_to_ce & (bpp.RRLO_1 > 1)
    indexes = bpp.loc[evol_to_ce1].index
    if len(indexes) > 0:
        bpp_subset = pd.concat([bpp.iloc[indexes[i]:indexes[i]+4] for i in range(len(indexes))])
        bpp_subset.set_index('bin_num', drop=False, inplace=True)
        df_subset = mass_transfer_df.bin_num.isin(bpp.loc[evol_to_ce1].bin_num)
        
        mass_transfer_df.loc[df_subset, 'smt_to_ce1'] = np.full(len(indexes), 1.)
        mass_transfer_df.loc[df_subset, 'a_post_ce1'] = bpp_subset.loc[bpp_subset.evol_type == 8].sep
        mass_transfer_df.loc[df_subset, 'a_pre_ce1'] = bpp_subset.loc[bpp_subset.evol_type == 7].sep
        mass_transfer_df.loc[df_subset, 'porb_post_ce1'] = bpp_subset.loc[bpp_subset.evol_type == 8].porb
        mass_transfer_df.loc[df_subset, 'porb_pre_ce1'] = bpp_subset.loc[bpp_subset.evol_type == 7].porb
        mass_transfer_df.loc[df_subset, 'time_to_ce1'] = bpp_subset.loc[bpp_subset.evol_type == 7].tphys - bpp_subset.loc[bpp_subset.evol_type == 3].tphys
        mass_transfer_df.loc[df_subset, 'r_don_pre_ce1'] = bpp_subset.loc[bpp_subset.evol_type == 7].rad_1
        mass_transfer_df.loc[df_subset, 'mass_1_pre_ce1'] = bpp_subset.loc[bpp_subset.evol_type == 7].mass_1
        mass_transfer_df.loc[df_subset, 'mass_2_pre_ce1'] = bpp_subset.loc[bpp_subset.evol_type == 7].mass_2
    
    # MT2 is SMT -> CE
    evol_to_ce2 = evol_to_ce & (bpp.RRLO_2 > 1)
    indexes = bpp.loc[evol_to_ce2].index  
    if len(indexes) > 0:
        bpp_subset = pd.concat([bpp.iloc[indexes[i]:indexes[i]+4] for i in range(len(indexes))])
        bpp_subset.set_index('bin_num', drop=False, inplace=True)
        df_subset = mass_transfer_df.bin_num.isin(bpp.loc[evol_to_ce2].bin_num)

        mass_transfer_df.loc[df_subset, 'smt_to_ce2'] = np.full(len(indexes), 1.)
        mass_transfer_df.loc[df_subset, 'a_post_ce2'] = bpp_subset.loc[bpp_subset.evol_type == 8].sep
        mass_transfer_df.loc[df_subset, 'a_pre_ce2'] = bpp_subset.loc[bpp_subset.evol_type == 7].sep
        mass_transfer_df.loc[df_subset, 'porb_post_ce2'] = bpp_subset.loc[bpp_subset.evol_type == 8].porb
        mass_transfer_df.loc[df_subset, 'porb_pre_ce2'] = bpp_subset.loc[bpp_subset.evol_type == 7].porb
        mass_transfer_df.loc[df_subset, 'time_to_ce2'] = bpp_subset.loc[bpp_subset.evol_type == 7].tphys - bpp_subset.loc[bpp_subset.evol_type == 3].tphys
        mass_transfer_df.loc[df_subset, 'r_don_pre_ce2'] = bpp_subset.loc[bpp_subset.evol_type == 7].rad_2
        mass_transfer_df.loc[df_subset, 'mass_1_pre_ce2'] = bpp_subset.loc[bpp_subset.evol_type == 7].mass_1
        mass_transfer_df.loc[df_subset, 'mass_2_pre_ce2'] = bpp_subset.loc[bpp_subset.evol_type == 7].mass_2
    
    return mass_transfer_df


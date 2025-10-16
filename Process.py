#!/usr/bin/python3
import numpy as np
import pandas as pd


def filter_gene(data):
    # filter out genes not existing in any cells or locations
    data0 = np.array(data.drop(labels='Genes', axis='columns'))
    sum_gene = np.sum(data0, axis=1)
    idx = np.where(sum_gene ==0 )[0]
    data.drop(idx, axis='index', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # filter out locations and cells without genes
    
    
    return data



def overlap_gene(st, sc, st_loc):
    # select overlapped genes between ST and SC
    overlap = list(set(st['Genes'].tolist()) & set(sc['Genes'].tolist()))
    sc_overlap = sc.loc[sc['Genes'].isin(overlap)]
    st_overlap = st.loc[st['Genes'].isin(overlap)]

    
    # reorder the Genes column
    st_overlap = st_overlap.sort_values('Genes')
    sc_overlap = sc_overlap.sort_values('Genes')
    
    # reorder the cells of ST data
    st_loc.sort_values('CellID', inplace=True)
    st_overlap.sort_index(axis=1, inplace=True)

    st_overlap.reset_index(drop=True, inplace=True)
    sc_overlap.reset_index(drop=True, inplace=True)

    return st_overlap, sc_overlap, st_loc



def get_sc_ave(st, sc, sc_meta, vmr_threshold=0.99, log2fold=1.25):
    # reorder cells of SC data
    sc_meta.sort_values('CellID', inplace=True)
    sc.sort_index(axis=1, inplace=True)

    sc_list = sc.keys().tolist()
    sc_meta_list = sc_meta['CellID'].tolist()
    overlap_cell = list(set(sc_meta_list) & set(sc_list))
    sc_meta = sc_meta.loc[sc_meta['CellID'].isin(overlap_cell)]
    sc = sc.loc[:, sc.keys().isin(overlap_cell)]
    sc['Genes'] = st['Genes']

    sc0 = np.array( sc.drop( labels='Genes', axis=1) )
    keys, key_num =  np.unique(sc_meta['CellType'], return_counts=True)
    key_num = dict(dict(zip(sorted(keys), key_num)))

    print(key_num)

    sc_ave = pd.DataFrame(columns=keys)
    sc_vmr = pd.DataFrame(columns=keys)

    for key in keys:
        ids = sc_meta['CellID'][sc_meta['CellType']==key]
        temp = sc.loc[:, ids]
        sc_ave[key] = temp.mean(axis=1)
        sc_vmr[key] = temp.var(axis=1) / temp.mean(axis=1)

    sc_ave.reset_index(drop=True, inplace=True)
    sc_vmr.reset_index(drop=True, inplace=True)

    sc_ave['max_idx'] = sc_ave.idxmax(axis=1)
    sc_ave['max_val'] = sc_ave.drop(labels='max_idx', axis='columns').max(axis=1)
    # sc_ave['Genes'] = sc['Genes']
    sc_ave['remove'] = 0 # sc['Genes']
    sc_vmr = sc_vmr.fillna(0)
    sc_vmr['meanvmr'] = sc_vmr.mean(axis=1)
    sc_vmr_threshold = sc_vmr['meanvmr'].max() * vmr_threshold
    log2foldlist = []

    for i in range(len(sc_ave['remove'])):
        sc_i = sc0[i, :]
        # choose the max values
        max_val = sc_ave['max_val'][i]
        n_max = key_num[sc_ave['max_idx'][i]]
        # compute log2fold
        log2fold_i = np.log2(max_val * (len(sc.keys()) - n_max) / (sc_i.sum() - max_val * n_max + 1e-6))
        log2foldlist.append(log2fold)
        if log2fold_i < log2fold or sc_vmr['meanvmr'][i] >= sc_vmr_threshold:
            sc_ave.loc[i, 'remove'] = 1
        else:
            sc_ave.loc[i, 'remove'] = 0
    sc_vmr['log2fold'] = log2foldlist

    # drop the genes
    drop_idx = np.where(sc_ave['remove'] == 1)[0]
    sc_ave.drop(drop_idx.tolist(), axis=0, inplace=True)
    st.drop(drop_idx.tolist(), axis=0, inplace=True)
    sc.drop(drop_idx.tolist(), axis=0, inplace=True)
    sc_vmr.drop(drop_idx.tolist(), axis=0, inplace=True)

    sc_ave.drop(labels='max_idx', axis=1, inplace=True)
    sc_ave.drop(labels='max_val', axis=1, inplace=True)
    sc_ave.drop(labels='remove', axis=1, inplace=True)
    sc_ave['Genes'] = sc['Genes']
    # sc_ave['Genesst'] = st['Genes']

    return st, sc, sc_ave, sc_meta, sc_vmr



def process_main(st, st_loc, sc, sc_meta, vmr_threshold=0.90, log2fold=1.25):


    print('# of genes in st and sc: ', len(st['Genes']), len(sc['Genes']))
    print('\n')

    print('Now filtering out zero count genes')
    st_filter_zero_gene = filter_gene(st)
    sc_filter_zero_gene = filter_gene(sc)


    print('# of genes in st and sc: ', len(st_filter_zero_gene['Genes']), len(sc_filter_zero_gene['Genes']))
    print('\n')

    print('Now sorting the data and selecting genes expressed in both sc and st')
    st_overlap, sc_overlap, st_loc_sort = overlap_gene(st_filter_zero_gene, sc_filter_zero_gene, st_loc)

    keys, keynum = np.unique(st_overlap['Genes'], return_counts=True)
    print('Repeated genes in st:', keys[keynum>1])
    keys, keynum = np.unique(sc_overlap['Genes'], return_counts=True)
    print('Repeated genes in sc:', keys[keynum>1])


    print('# of genes in st and sc: ', len(st_overlap['Genes']), len(sc_overlap['Genes']))
    print('\n')

    print('Now selecting genes and calculating cell type mean expression')
    st_selected, sc_selected, sc_ave, sc_meta_sort, sc_vmr = get_sc_ave(st_overlap, sc_overlap, sc_meta, vmr_threshold=vmr_threshold, log2fold=log2fold)

    print('# of genes in st and sc: ', len(st_selected['Genes']), len(sc_selected['Genes']))

    return st_selected, sc_selected, sc_ave, st_loc_sort, sc_meta_sort


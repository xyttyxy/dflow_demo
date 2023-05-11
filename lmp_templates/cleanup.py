import numpy as np
from ase.atoms import Atoms

def find_O2(atoms_in):
    """try to find the O2 molecules throughout the entire structure
    ------
    Returns:
    indices of O atoms belonging to some O2 molecule.
    Always returns both atoms in a molecule
    """
    
    atoms_o_lst = [a.index for a in atoms_in if a.symbol == 'O']
    
    atoms_cu = Atoms([a for a in atoms_in if a.index not in atoms_o_lst], cell=atoms_in.cell, pbc=True)
    zmax_cu = np.max(atoms_cu.get_positions()[:, 2])
    z_buffer = 0 # angstrom
    atoms_o = Atoms([atoms_in[a] for a in atoms_o_lst],
                    cell=atoms_in.cell, pbc=True)
    
    dists = atoms_o.get_all_distances(mic=False) # ase/np bug, cannot use mic=True
    dists_mask = (dists < 1.5) & (dists > 0)
    o2_pairs = np.argwhere(dists_mask)
    def to_original_index(o2_pairs, o_lst):
        o2_pairs_original_index = []
        for pair in o2_pairs:
            pair_original_index = [atoms_o_lst[pair[0]], atoms_o_lst[pair[1]]]
            o2_pairs_original_index.append(pair_original_index)
        return np.array(o2_pairs_original_index, dtype=int)
    

    return to_original_index(o2_pairs, atoms_o_lst)
    
def cleanup_o2(lmpptr):
    from lammps import lammps
    from mpi4py import MPI

    from ase.cell import Cell
    def master_print(rank, msg):
        if rank == 0:
            print(msg)
        
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nworkers = MPI.COMM_WORLD.Get_size()
    lmp = lammps(ptr=lmpptr)
    
    master_print(rank, '----- cleanup.py start -----')
    
    nglobal = lmp.get_natoms()
    
    # ensure atom ids consecutive
    lmp.command('reset_atom_ids sort yes')
    # sorted by atoms id
    x = lmp.gather_atoms("x", dtype=1, count=3)
    types = lmp.gather_atoms("type", dtype=0, count=1)
    box = lmp.extract_box()
    box = np.array(box[0]+box[1])
    cell_x = [box[3],0,0]
    cell_y = [0,box[4],0]
    cell_z = [0, 0, box[5]]
    cell = Cell([cell_x, cell_y, cell_z])
    
    positions = np.array(x).reshape(nglobal, 3)
    symbols = ['Cu' if elm == 1 else 'O' for elm in types]
    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=True)
    master_print(rank, f'CLEANUP: {atoms} reconstructed')
    o2_pairs = find_O2(atoms)

    npairs = len(o2_pairs)
    master_print(rank, f'CLEANUP: found {npairs} o2 molecules')
    master_print(rank, o2_pairs)
    
    # must ensure all ranks use same seed
    np.random.seed(0)
    
    if npairs > 10:
        remove_pairs = o2_pairs[np.random.choice(npairs, np.floor(npairs/3).astype(int))]
    elif npairs < 5 and npairs > 2:
        remove_pairs = o2_pairs[np.random.choice(npairs, 2)]
    else:
        master_print(rank, 'CLEANUP: no o2 molecules to be removed')
        remove_pairs = np.array([])

    if remove_pairs.any():
        nremove = len(remove_pairs)
        master_print(rank, f'CLEANUP: removing {nremove} o2 molecules')
        remove_indices = np.unique(remove_pairs.flatten()).tolist()
        # lammps atom id start from 1, ase starts from 0
        master_print(rank, remove_pairs)
        master_print(rank, remove_indices)
        remove_string = ' '.join([str(int(elm+1)) for elm in remove_indices])
        master_print(rank, remove_string)
        lmp.command(f'group to_remove id {remove_string}')
        lmp.command('delete_atoms group to_remove')
        lmp.command('group to_remove delete')
    else:
        master_print(rank, 'CLEANUP: no o2 molecules removed')

    master_print(rank, '----- cleanup.py stop -----')


# if __name__ == '__main__':
#     from ase.io import read

#     atoms_in = read('test_cleanup.vasp')
#     o2_pairs = find_O2(atoms_in)
#     o2_indices = np.unique(o2_pairs.flatten())
#     atoms_out = atoms_in.copy()
#     del atoms_out[o2_indices]
#     from ase.visualize import view
#     view([atoms_in, atoms_out])

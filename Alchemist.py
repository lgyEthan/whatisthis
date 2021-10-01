import numpy as np
from tqdm import tqdm
from ase import Atom, Atoms
from collections import Counter

from ase.io.vasp import read_vasp, write_vasp
# from ase.io.trajectory import Trajectory
# from ase.io.lammpsdata import write_lammps_data

from random import randint # numpy.random.randint는 [low,high) 인 반면 random.randint는 [low, high]
from numpy.random import rand

from ase.data import atomic_numbers, atomic_names, atomic_masses
from ase.data import vdw_radii, covalent_radii
from ase import units

from ase.calculators.lammps import Prism, convert

__version__ = None
__date__ = "2021.09.13"
__author__ = "Giyeok Lee, Gunyoung Heo, Hyun Jun Kim"
__maintainer__ = "Giyeok Lee, Gunyoung Heo, Hyun Jun Kim"
__email__ = "lgy4230@yonsei.ac.kr"
__copyright__ = "Copyright (c) Materials Theory Group @ Yonsei University (2021)"


def expand_cell(
    model:Atoms,
    x=None, y=None, z=None,
    a=None, b=None, c=None,
    result=False, return_cell=False, scale_atoms=False
    ):
    '''
    model : Atoms

    [input]
    * x, y, z = cartesian axis x, y, z (has priority then a, b, c)
    * a, b, c = lattice vector a, b, c
    * result : If True, selected value is set to the final value
    * return_cell : If True, Cell object will be returned instead of Atoms

    [output]
    Atoms if not return_cell else 3x3 shape ndarray
    '''
    # from ase.cell import Cell
    temp_model = model.copy()
    cell = temp_model.cell

    cartesian_axis = [x, y, z]
    lattice_param  = [a, b, c]

    unit_vecs = []
    result_lattice_param = []
    for i in range(3):
        cur_a = cell.cellpar()[i]

        X, A = cartesian_axis[i], lattice_param[i]
        ua = cell[i]/np.linalg.norm(cell[i])
        unit_vecs.append(ua)
        dx_a = ua[i]

        if X is not None:
            A = X/dx_a if result else X/dx_a+cur_a

        elif A is not None:
            A = A if result else cur_a+A

        else:
            A = cur_a

        result_lattice_param.append(A)

    wanted_cell = np.array(unit_vecs) * np.array(result_lattice_param).reshape(3,1)

    if return_cell:
        return wanted_cell
    else:
        temp_model.set_cell(wanted_cell, scale_atoms=scale_atoms)
        return temp_model

def get_surroundings(model:Atoms, index:int, rcut=2.5, only_count=None):
    '''
    [input]
    model: Atoms
    index: index of selected ion
    ion: str or list of wanted ions
    rcut: radius cutoff
    only_count: only count given atoms (ex. when only_count='O', only oxygens among neighboring atoms will counted)
    '''
    without_center_atom = [0]*len(model)
    without_center_atom[index] = rcut*2

    if only_count is not None:
        if isinstance(only_count, str):
            masking = np.array(model.get_chemical_symbols())!=only_count
        else:
            masking = np.ones(len(model))
            for oc in only_count:
                masking *= np.array(model.get_chemical_symbols())!=oc
        masked = masking * rcut*2
    else:
        masked = np.zeros(len(without_center_atom))

    return model[(np.array(without_center_atom) + masked + model.get_distances(index, slice(None), mic=True))<rcut]

def layer_indexing(model, al_tol=0.5, return_interlayer_spacing=False):
    this_model = model.copy()
    coord_along_axis = this_model.get_positions()[:,2]
    interlayer_spacing = {}
    t = dict(enumerate(np.argsort(coord_along_axis)))
    for i in sorted(t):
        if i == 0:
            al = 0 # 1이 낫나? 가장 아래 layer의 tag를 0 아니면 1로 두는거야. 통일성을 위해 0으로 하자
        else:
            if coord_along_axis[t[i]] - coord_along_axis[t[i-1]] > al_tol:
                al += 1
        this_model[t[i]].tag = al
        if al not in interlayer_spacing:
            interlayer_spacing[al] = [coord_along_axis[t[i]]]
        else:
            interlayer_spacing[al].append(coord_along_axis[t[i]])
    return this_model, interlayer_spacing if return_interlayer_spacing else this_model


def parse_symbol(neighbors:Atoms, clt:str, oneighbors:Atoms=None, Tri:bool=False, current_element=None) -> str:

    if clt == "Ot":
        # obts if st substituted to at. else, ob
        return "Ts" if "Al" in neighbors.get_chemical_symbols() else "B"

    elif clt == "H":
        return "Ho"

    elif clt in ["Oh", "Oc"]:
        assert oneighbors is not None
        if current_element=="O":
            if Tri:
                # ohs or obss
                return "Hs" if "H" in oneighbors.get_chemical_symbols() else "S"
            else:
                # oh or ob
                return "Rh" if "H" in oneighbors.get_chemical_symbols() else "B"
        else:
            assert current_element == "H"
            return "Ho"

    elif clt == "Tetra":
        return "At" if current_element=="Al" else "Si"

    elif clt == "Octa":
        assert current_element == "Mg" if Tri else "Al"
        return current_element
    else:
        raise IOError(f"Report this. which clt is missing?: {clt}")


def layer_info(total_model:Atoms, layer_index:int, recursion=False):
    indexes = total_model.get_tags()==layer_index
    layer = total_model[indexes]
    counter = list(Counter(layer.get_chemical_symbols()))

    if len(counter)==1:
        if counter[0] in ["O"]:
            if not recursion:
                before_layer = layer_info(total_model, layer_index-1, recursion=True) if layer_index!=0 else None
                after_layer = layer_info(total_model, layer_index+1, recursion=True) if layer_index!=np.max(total_model.get_tags()) else None
            else:
                return "O"
                
            if "Octa" in [before_layer, after_layer]:
                layer_tag = "Oc"
            elif "Tetra" in [before_layer, after_layer]:
                layer_tag = "Ot"
            else:
                raise IOError("Neither T nor O layer is near in this O layer. What is this case?")

        elif counter[0] in ["Si"]:
            layer_tag = "Tetra"

        elif counter[0] in ["Al", "Mg"]:
            # We will only consider Al or Mg in octahedral layer
            nn = [len(get_surroundings(total_model, i, 2.5, only_count="O")) for i in np.where(indexes)[0]]
            if 6 in nn:
                layer_tag = "Octa" # Octahedral layer

            elif 4 in nn:
                layer_tag  = "Tetra" # Tetrahedral layer. in case of Al_Si

            else:
                print(counter)
                raise IOError(f"What is this case? The number of nearest neighbors : {nn},\
                    layer_index : {layer_index}")

        else:
            layer_tag = counter[0]
            if layer_tag != "H":
                print(f"Oh? Are you trying to add ions in interlayer regions? Else Report GY")

    elif len(counter)==2:
        if sorted(counter) == ["H", "O"]:
            layer_tag = "Oh"
        elif sorted(counter) == ["Al", "Si"]:
            nn = [len(get_surroundings(total_model, i, 2.5, only_count="O")) for i in np.where(indexes)[0]]
            if 4 in nn:
                layer_tag = "Tetra"
            else:
                raise IOError(f"What is this case? This layer:{layer_index}\
                which composed of Al and Si seems weird")
        else:
            # If substitution occurs in Octahedral Layer
            # Or Si <-> O distance along the z axis is too small
            if sorted(counter) == ["O", "Si"]:
                raise IOError("Try the smaller al_tol. Ex.Alchemist(Atoms, al_tol=0.3). default=0.4")
            raise NotImplementedError(f"The layer consists of {counter} is not supported.")
    else:
        # when len(counter)>=3
        raise NotImplementedError("Too many atoms in one layer")

    return layer_tag


class Alchemist:

    # symbol to element
    sym2elem = {
    "Al" : "Al",
    "At" : "Al",
    "B" : "O",
    "Ca" : "Ca",
    "Cl" : "Cl",
    "Fe" : "Fe",
    "H" : "H",
    "Ho" : "H",
    "Hs" : "O",
    "Li" : "Li",
    "Mg" : "Mg",
    "Mn" : "Mg",
    "Na" : "Na",
    "O" : "O",
    "Os" : "O",
    "Rh" : "O",
    "S" : "O",
    "Si" : "Si",
    "Ts" : "O"
    }

    # symbol to atom_type
    sym2at = {
    "Al" : "ao",
    "At" : "at",
    "B" : "ob",
    "Ca" : "cao",
    "Cl" : "Cl",
    "Fe" : "feo",
    "H" : "h*",
    "Ho" : "ho",
    "Hs" : "ohs",
    "Li" : "lio",
    "Mg" : "mgo",
    "Mn" : "mgh",
    "Na" : "Na",
    "O" : "o*",
    "Os" : "obos",
    "Rh" : "oh",
    "S" : "obss",
    "Si" : "st",
    "Ts" : "obts",
    }

    # symbol to mass
    sym2mass = {
    "Al" : 26.98154,
    "At" : 26.98154,
    "B" : 15.9994,
    "Ca" : 40.08,
    "Cl" : 35.453,
    "Fe" : 55.847,
    "H" : 1.00797,
    "Ho" : 1.00797,
    "Hs" : 15.9994,
    "Li" : 6.941,
    "Mg" : 24.305,
    "Mn" : 24.305,
    "Na" : 22.99,
    "O" : 15.9994,
    "Os" : 15.9994,
    "Rh" : 15.9994,
    "S" : 15.9994,
    "Si" : 28.0855,
    "Ts" : 15.9994,
    }

    # symbol to charge
    sym2q = {
    "Al" : 1.575,
    "At" : 1.575,
    "B" : -1.05,
    "Ca" : 1.36,
    "Cl" : -1,
    "Fe" : 1.575,
    "H" : 0.41,
    "Ho" : 0.425,
    "Hs" : -1.0808,
    "Li" : 0.525,
    "Mg" : 1.36,
    "Mn" : 1.05,
    "Na" : 1,
    "O" : -0.82,
    "Os" : -1.1808,
    "Rh" : -0.95,
    "S" : -1.2996,
    "Si" : 2.1,
    "Ts" : -1.1688,
    }

    # symbol to LJ potential coefficient.
    # [epsilon, sigma]
    sym2lj = {
    "Al" : [1.32974601391015E-06, 4.27132193158393E+00],
    "At" : [1.84017954628169E-06, 3.30195662519835E+00],
    "B" : [1.55416412439343E-01, 3.16552008792968E+00],
    "Ca" : [5.03013082432347E-06, 5.56666386355447E+00],
    "Cl" : [1.00099893331851E-01, 4.39997098045992E+00],
    "Fe" : [9.03920061491161E-07, 4.90576514556967E+00],
    "H" : [0.00000000000000E+00, 0], # 이거 sigma 0 맞는지 계산 한 번 더 해
    "Ho" : [0.00000000000000E+00, 0], # 이거 sigma 0 맞는지 계산 한 번 더 해
    "Hs" : [1.55416412439343E-01, 3.16552008792968E+00],
    "Li" : [9.01727524328185E-07, 4.21059486645003E+00],
    "Mg" : [9.03018804621205E-07, 5.26432586884668E+00],
    "Mn" : [9.03018804621205E-07, 5.26432586884668E+00],
    "Na" : [1.30099987146773E-01, 2.35001266393092E+00],
    "O" : [1.55416412439343E-01, 3.16552008792968E+00],
    "Os" : [1.55416412439343E-01, 3.16552008792968E+00],
    "Rh" : [1.55416412439343E-01, 3.16552008792968E+00],
    "S" : [1.55416412439343E-01, 3.16552008792968E+00],
    "Si" : [1.84017954628169E-06, 3.30195662519835E+00],
    "Ts" : [1.55416412439343E-01, 3.16552008792968E+00],  
    }

    def __init__(self, clay_minerals:Atoms, Tri=False, al_tol=0.4):
        '''
        Input clay minerals.
        parse atom_type to structures

        [input]
        * Tri : True=Trioctahedral, False=Dioctahedral phyllosilicates
        * TO : True=TO, False=TOT (TOT O is not supported)
        '''

        # Note. 우리의 경우는 Mg 있으면 Tri로 하자
        if "Mg" in clay_minerals.get_chemical_symbols():
            Tri = True
        self.Tri = Tri

        if "H" not in clay_minerals.get_chemical_symbols():
            clay_minerals = self.attach_oh(clay_minerals, al_tol=al_tol, Tri=Tri)

        # ------------------------------------------------------------
        # TODO 물이 있다면, 현재는 이 코드 사용 불가능. 현재는 물 없는 clay를 가정
        # 물을 지우는 코드는 여기 위치에 두어야 함.
        # parsed_atoms의 물은 remove_water로 지울 수 있으나,
        # 그건 atom_type에 맞게 H랑 O를 다 바꿔둬서 가능한 것. (예. H->Ho, O->B)
        # input_atoms 에서부터 물이 있는 경우에는 다른 구분짓는 방법이 필요함.
        # ------------------------------------------------------------

        # parsing atom_types along z directions (other directions are unavailable for now)
        # Do we have to consider the curve-shaped clay minerals (ex. antigorite)?
        self.al_tol = al_tol
        self.parsed_atoms = clay_minerals.copy()
        self.update_height()

        tags = self.parsed_atoms.get_tags()
        layer_tags = [layer_info(self.parsed_atoms, i) for i in range(np.max(tags)+1)]
        self.input_atoms = self.parsed_atoms.copy()


        # N_ML = 1 + np.sum(self.dheights>2.5) # number of Monolayer. cutoff=2.5
        ML_bottoms = [0] + list(np.where(self.dheights>2.5)[0]+1)
        N_ML = len(ML_bottoms) # number of Monolayer. cutoff=2.5
        NT = Counter(layer_tags).get("Tetra", 0)
        TO = False if (NT/N_ML)%2==0 else True

        cursor_db = np.argsort(self.input_atoms.get_tags())

        for ind, ii in enumerate(ML_bottoms):
            st = ii
            et = ML_bottoms[ind+1]-1 if ind!=len(ML_bottoms)-1 else np.max(tags)
            current_layer_tags = layer_tags[st:et+1]

            for atom, atom_index in zip(self.input_atoms[cursor_db], cursor_db):
                if atom.tag < st:
                    continue
                elif atom.tag > et:
                    break
                else:
                    current_tag = atom.tag - st
                    clt = current_layer_tags[current_tag]

                neighbors = get_surroundings(self.input_atoms, atom_index, rcut=2.5)
                if clt in ["Oh", "Oc"]:
                    oneighbors = get_surroundings(self.input_atoms, atom_index, rcut=1.2)
                else:
                    oneighbors = None

                self.parsed_atoms[atom_index].symbol = parse_symbol(neighbors=neighbors, clt = clt, oneighbors=oneighbors, Tri=Tri, current_element=atom.symbol)
        
        self.parsed_tags = layer_tags
        self.TO = TO

        self.N_ML = N_ML
        self.ML_bottoms = ML_bottoms # for the separation of 
        self.solvated = False

        # parsing charges
        # FYI) you can get it using atoms.get_initial_charges() method
        self.update_charge()


    @staticmethod
    def attach_oh(clay_minerals_without_H:Atoms, al_tol=0.5, bond_length=1, Tri=False) -> Atoms:
        source_model, interlayer_spacing = layer_indexing(clay_minerals_without_H, al_tol, return_interlayer_spacing=True)
        tags = source_model.get_tags()
        layer_tags = [layer_info(source_model, i) for i in range(np.max(tags)+1)]

        heights = []
        for als in interlayer_spacing:
            heights.append(np.mean(interlayer_spacing[als]))
        dheights = np.diff(heights)

        ML_bottoms = [0] + list(np.where(dheights>2.5)[0]+1)
        N_ML = len(ML_bottoms)

        cursor_db = np.argsort(tags)

        ohed_model = source_model.copy()
        for ind, ii in enumerate(ML_bottoms):
            st = ii
            et = ML_bottoms[ind+1]-1 if ind!=len(ML_bottoms)-1 else np.max(tags)
            current_layer_tags = layer_tags[st:et+1]

            for atom, atom_index in zip(source_model[cursor_db], cursor_db):
                if atom.tag < st:
                    continue
                elif atom.tag > et:
                    break
                else:
                    current_tag = atom.tag - st
                    clt = current_layer_tags[current_tag]

                if clt != "Oc":
                    continue
                prev_label = current_layer_tags[current_tag - 1] if current_tag != 0 else None
                after_label = current_layer_tags[current_tag + 1] if current_tag != et-st else None

                if "Octa" not in [prev_label, after_label]:
                    raise IOError("Report GY. Since this layer is already parsed as Oc, Octahedral layer should be existed")
                elif prev_label == "Octa":
                    direc = +1
                else:
                    direc = -1

                nn = len(get_surroundings(source_model, atom_index, rcut=2.5))
                if Tri:
                    if nn==2:
                        ohed_model.append(Atom(symbol="H", position=atom.position+direc*np.array([0,0,bond_length])))
                    elif nn==3:
                        continue
                    else:
                        raise IOError(f"{nn} neighboring bonds found in Oc layer of Trioctahedral phyllosilicate")
                else:
                    if nn==3:
                        ohed_model.append(Atom(symbol="H", position=atom.position+direc*np.array([0,0,bond_length])))
                    elif nn==4:
                        continue
                    else:
                        raise IOError(f"{nn} neighboring bonds found in Oc layer of Dioctahedral phyllosilicate")
        return ohed_model

    def _atoms(self):
        model = self.parsed_atoms.copy()
        for atom in model:
            atom.symbol = self.sym2elem[atom.symbol]
        return model

    @property
    def atoms(self):
        return self._atoms()


    def _dspacing(self):
        z = self.parsed_atoms.cell[2,2] # different from lattice param, 'c'
        cur_bots = np.array(self.heights)[self.ML_bottoms]
        prev_tops = np.array(self.heights)[np.array(self.ML_bottoms)-1]
        prev_tops[0] -= z
        return np.mean(cur_bots - prev_tops)

    @property
    def dspacing(self):
        return self._dspacing()


    def update_height(self):
        self.parsed_atoms, interlayer_spacing = layer_indexing(self.parsed_atoms, self.al_tol, return_interlayer_spacing=True)

        heights = []
        for als in interlayer_spacing:
            heights.append(np.mean(interlayer_spacing[als]))

        self.al_heights = interlayer_spacing
        self.heights = heights
        self.dheights = np.diff(heights)     


    def update_charge(self):
        for atom in self.parsed_atoms:
            atom.charge = self.sym2q[atom.symbol]


    def widen(self, dspacing=7.5, n_h2o=None, widening=None):
        '''
        set the interlayer region as 'dspacing'
        when you just want to widen the interlayer region, choose widening
        '''
        if self.solvated:
            raise IOError("For now, solvated model is not possible to widen the interlayer region.\
                Plz remove water first using remove_water method.\
                i.e.) self.remove_water()")

        ML_bottoms = self.ML_bottoms
        N_ML = self.N_ML

        # TODO
        # if n_h2o is not None:
            # dspacing = Calculated_c_axis_which_mathing_the_density

        widening = dspacing - self.dspacing if widening is None else widening

        # self.parsed_atoms.cell[2] += np.array([0,0,N_ML*widening])
        self.parsed_atoms = expand_cell(self.parsed_atoms, z = N_ML*widening, result=False)
        model = self.parsed_atoms.copy()      

        cursor_db = np.argsort(model.get_tags())

        for ind, ii in enumerate(ML_bottoms):
            st = ii
            et = ML_bottoms[ind+1]-1 if ind!=len(ML_bottoms)-1 else np.max(model.get_tags())

            for atom, atom_index in zip(model[cursor_db], cursor_db):
                if atom.tag < st:
                    continue
                elif atom.tag > et:
                    break
                else:
                    uc = model.cell[2]/np.linalg.norm(model.cell[2])
                    dz_c = uc[2]
                    self.parsed_atoms[atom_index].position += uc * widening/dz_c * ind
                    # self.parsed_atoms[atom_index].position += np.array([0, 0, widening])*ind
        self.update_height()

    def remove_water(self):
        '''
        You can use this only after parsing the atoms.
        This means, you can only remove the water which generated from the '.add_water' method
        '''
        index_for_remove = []
        for atom in self.parsed_atoms:
            if atom.symbol in ["H", "O"]:
                index_for_remove.append(atom.index)
        del self.parsed_atoms[index_for_remove]
        self.solvated = False


    def add_water(self, n_h2o=None, density=0.997, add=True, prohibited_region = None, max_iter = 10000):
        '''
        #TODO: add annotation

        If you want to explictly put the number of water molecules, use n_h2o
        If yo do not want to use the dafault density, use density parameter

        * Prohibited_region : Don't let water (oxygen's position) come into certain region. [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi].
                              For partially select, use None
            |---> ex) [None, None, 3, 5, 4, 10] will delete the oxygen in y=(3, 5), z=(4,10) region. Doesn't care about x coordinates.        
        '''
        def mm(symbol):
            return atomic_masses[atomic_numbers[symbol]]

        atoms = self.parsed_atoms.copy()

        # Basic data setup (from literature)
        ### water density = 997kg/m3 == 0.997g/cm3
        ### 1cm = 1E8 Angstrom
        if n_h2o is None:
            base_density = float(density) / 1E8**3 * units.mol / (mm("H")*2 + mm("O")) # H20/A**3

        vdw_h2o = 1.7 # van der Waals radius in (Angstrom) / coulombic radii=1.4
        b = 0.957 # O-H bond lengths
        omega = np.radians(104.4776) # H-O-H bond angle (degrees)    

        atom_index = len(atoms)
        sub_vol = 0
        
        for atom in atoms:
            an = atomic_numbers[self.sym2elem[atom.symbol]]
            sub_vol += 4/3*np.pi*(covalent_radii[an])**3

        if n_h2o is not None:
            water_approx = int(n_h2o)
        else:
            water_approx = (atoms.get_volume() - sub_vol) * base_density
        solv_atoms = atoms.copy()
        
        if not add:
            # Only return the number of water molecules
            return water_approx

        if prohibited_region is not None:
            x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = prohibited_region
        else:
            x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = None, None, None, None, None, None

        for i in tqdm(np.arange(1, water_approx+1), desc=f"Attaching {int(water_approx):d} H2O"):
            iteration = 0
            while True:
                iteration += 1
                if iteration > max_iter:
                    raise RuntimeError("iteration exceeds max_iter.. Maybe the region is too tight. Plz reduce the density")
                x, y, z = rand(3, ) * atoms.cell.cellpar()[:3]

                chpr = [(x_lo, x_hi, x), (y_lo, y_hi, y), (z_lo, z_hi, z)]
                # Prohibited region
                for lo, hi, value in chpr:
                    prohibited = False
                    if lo is None and hi is None:
                        pass
                    elif lo is None and hi is not None:
                        if value <= hi:
                            prohibited = True
                            break
                    elif lo is not None and hi is None:
                        if value >= lo:
                            prohibited = True
                            break
                    else: # both selected
                        if value >= lo and value <= hi:
                            prohibited = True
                            break

                if prohibited:
                    continue

                temp_atoms = solv_atoms.copy()
                o_temp = Atom(symbol="O", position=(x, y, z))
                temp_atoms.append(o_temp) # put o atoms to last
                odists = temp_atoms.get_distances(-1, indices=range(len(temp_atoms)-1), mic=True)
                if np.min(odists[:]) > vdw_h2o:
                    break

            # Random angle for placement of first H atom
            phi = rand() * 2 * np.pi
            theta = rand() * np.pi

            # First H position
            xh1 = x + b * np.sin(theta) * np.cos(phi)
            yh1 = y + b * np.sin(theta) * np.sin(phi)
            zh1 = z + b * np.cos(theta)

            # Second H position
            sign = randint(0, 1)
            xh2 = x + b * np.sin(theta-(-1)**sign*omega) * np.cos(phi)
            yh2 = y + b * np.sin(theta-(-1)**sign*omega) * np.sin(phi)
            zh2 = z + b * np.cos(theta-(-1)**sign*omega)
            
            # put Atoms to solvated model
            o1 = Atom("O", position = np.array([x, y, z]))
            h1 = Atom("H", position = np.array([xh1, yh1, zh1]))
            h2 = Atom("H", position = np.array([xh2, yh2, zh2]))

            solv_atoms.append(o1)
            solv_atoms.append(h1)
            solv_atoms.append(h2)
            solv_atoms.wrap()

        self.parsed_atoms = solv_atoms
        self.update_charge()
        self.solvated = True


    def left_chamber(self, n_ion=6, concentration=1):
        '''
        []
        '''
        # TODO
        # for adding ion, use add_ion method
        # when adding ions, use prohibited_region

        self.parsed_atoms = SOMETHING
        self.update_charge()
        pass

    def right_chamber(self, ):
        self.parsed_atoms = SOMETHING
        self.update_charge()
        pass


    def add_ion(self, symbol="Na", number=1, prohibited_region=None):
        # TODO
        self.parsed_atoms = SOMETHING
        self.update_charge()
        pass

    def write_data(self, data_file:str="clay.data", unit:str="metal"):
        '''
        atom_style은 현재 full만 지원
        '''

        model = self.parsed_atoms.copy()
        # charges = model.get_initial_charges() # It will automatically parsed when using write_lammps_data, with atom_style="charge/full"
        def list_sorting(list):
            temp = []
            for item in list:
                if item not in temp:
                    temp.append(item)
            return temp

        p = Prism(model.cell)
        xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance", "ASE", unit)
        pos = p.vector_to_lammps(model.get_positions(), wrap=False)
        
        #number of atoms
        atom_number = len(model)

        #list of atom_types -> index maintained
        symbols = model.get_chemical_symbols()
        sorted_symbols = list_sorting(symbols)

        #list of mass -> index maintained
        masses = model.get_masses()
        sorted_masses = list_sorting(masses)

        #dict of symbol and index number {'B': 1, 'Ho': 2, 'Mg': 3, 'Rh': 4, 'Si': 5}
        symbols_dict = {string : i+1 for i,string in enumerate(sorted_symbols)}

        #number of atom_type
        elements_number = len(sorted_symbols)

        with open(data_file, 'w') as fileobj:
            fileobj.write(f'LAMMPS data file \n \n')

            fileobj.write(f'{atom_number} atoms \n')
            fileobj.write(f'{0} bonds \n')
            fileobj.write(f'{0} angles \n')
            fileobj.write(f'{0} dihedrals \n')
            fileobj.write(f'{0} impropers \n\n')

            fileobj.write(f'{len(sorted_symbols)} atom types \n')
            fileobj.write(f'{0} bond types \n')
            fileobj.write(f'{0} angle types \n\n')


            fileobj.write(f'0 {xhi} xlo xhi \n')
            fileobj.write(f'0 {yhi} ylo yhi \n')
            fileobj.write(f'0 {zhi} zlo zhi \n')
            
            fileobj.write(f'{xy} {xz} {yz} xy xz yz \n')

            fileobj.write('\nMasses \n \n')
            for symbol in sorted_symbols:
                fileobj.write(f'{symbols_dict[symbol]} {self.sym2mass[symbol]} #{self.sym2at[symbol]} \n')

            fileobj.write('\nPair Coeffs # lj/cut/coul/long \n \n')
            for symbol in sorted_symbols:
                fileobj.write(f'{symbols_dict[symbol]} {self.sym2lj[symbol][0]} {self.sym2lj[symbol][1]} #{self.sym2at[symbol]} \n')

            fileobj.write('\nBond Coeffs # morse \n \n')
            for symbol in sorted_symbols:
                fileobj.write(f'{symbols_dict[symbol]} {self.sym2lj[symbol][0]} {self.sym2lj[symbol][1]} #{self.sym2at[symbol]} \n')

            fileobj.write('\nAngle Coeffs # 뭐로 해야될까 \n \n')
            for symbol in sorted_symbols:
                fileobj.write(f'{symbols_dict[symbol]} {self.sym2lj[symbol][0]} {self.sym2lj[symbol][1]} #{self.sym2at[symbol]} \n')

            i = 0
            fileobj.write('\nAtoms # full \n \n')
            for atom,xyz in zip(model,pos):
                i += 1
                fileobj.write(f'{i} 1 {symbols_dict[atom.symbol]} {self.sym2q[atom.symbol]} {xyz[0]} {xyz[1]} {xyz[2]} #{self.sym2at[atom.symbol]} \n')


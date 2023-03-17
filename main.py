from __future__ import annotations

from enum import Enum
import math
import random
from collections import deque
import numpy as np
from typing import TypeVar, Any

try:
    import threading
    USE_THREADING = True
except ModuleNotFoundError:
    USE_THREADING = False
    print("WARNING: Module 'threading' not found.",
          "The program will still run but Blender will crash upon exiting the GUI")

try:
    import bpy
    import bmesh
except ModuleNotFoundError:
    raise Exception("BPY or BMESH not found: This program must be run in Blender in order to work")

try:
    import dill
    USE_DILL = True
    USE_PICKLE = False
except ModuleNotFoundError:
    try:
        import pickle
        USE_PICKLE = True
        USE_DILL = False
        print("WARNING: Module 'dill' not found. Using 'pickle' instead.")
    except ModuleNotFoundError:
        USE_PICKLE = False
        USE_DILL = False
        print("WARNING: Modules 'dill' and 'pickle' not found. Saving capability will be disabled.")

try:
    from PyNite.Visualization import Renderer
    from PyNite.FEModel3D import FEModel3D
    USE_PYNITE = True
except ModuleNotFoundError:
    USE_PYNITE = False
    print("WARNING: Module 'PyNite' not found. Model analysis capability will be disabled.")

VectorType = TypeVar("VectorType", bound="VectorTup")
MaterialType = TypeVar("MaterialType", bound="Material")
ForceObjType = TypeVar("ForceObjType", bound="ForceObject")
ForceVertType = TypeVar("ForceVertType", bound="ForceVertex")
BlendObjectType = TypeVar("BlendObjectType", bound="BlendObject")


# A vector, representable as a tuple
class VectorTup:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __mul__(self, other: float | VectorType) -> VectorType:  # Other int / float
        if isinstance(other, float):
            return VectorTup(self.x * other, self.y * other, self.z * other)
        else:
            return VectorTup(self.x * other.x, self.y * other.y, self.z * other.z)

    def __rmul__(self, other: float | VectorType) -> VectorType:
        if isinstance(other, float):
            return VectorTup(self.x * other, self.y * other, self.z * other)
        else:
            return VectorTup(self.x * other.x, self.y * other.y, self.z * other.z)

    def __truediv__(self, other: float) -> VectorType:
        return VectorTup(self.x / other, self.y / other, self.z / other)

    def __add__(self, other: VectorType) -> VectorType:
        return VectorTup(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: VectorType) -> VectorType:
        return VectorTup(self.x - other.x, self.y - other.y, self.z - other.z)

    def __repr__(self) -> str:
        return f"Vector: ({self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

    def __bool__(self) -> bool:
        return not (self.x or self.y or self.z)  # If all numbers are 0 or invalid return false (via de morgan's laws)

    def __neg__(self) -> VectorType:
        return VectorTup(-self.x, -self.y, -self.z)

    def __lt__(self, other: VectorType) -> bool:  # Comparator definitions
        return self.get_magnitude() < other.get_magnitude()

    def __gt__(self, other: VectorType) -> bool:
        return self.get_magnitude() > other.get_magnitude()

    def __le__(self, other: VectorType) -> bool:
        return self.get_magnitude() <= other.get_magnitude()

    def __ge__(self, other: VectorType) -> bool:
        return self.get_magnitude() >= other.get_magnitude()

    def __ne__(self, other: VectorType) -> bool:  # Tests for inequality of entire vector, not magnitude inequality
        return not (self.x == other.x and self.y == other.y and self.z == other.z)

    def __eq__(self, other: VectorType) -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __getitem__(self, key: int) -> float:
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.z
        else:
            raise IndexError("VectorTup: Index out of bounds")

    def __setitem__(self, key: int, value: float) -> None:
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.z = value
        else:
            raise IndexError("VectorTup: Index out of bounds")

    def __getstate__(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    def __setstate__(self, state: dict) -> None:
        self.x, self.y, self.z = state["x"], state["y"], state["z"]

    def normalise(self) -> None:
        """ Calculates normalised version of vector without returning
        :return: None
        """
        magnitude = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.x = self.x / magnitude
        self.y = self.y / magnitude
        self.z = self.z / magnitude

    def get_normalised(self) -> VectorType:
        """ Calculates and returns normalised version of vector
        :return: Normalised vector
        """
        magnitude = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        x_temp = self.x / magnitude
        y_temp = self.y / magnitude
        z_temp = self.z / magnitude
        return VectorTup(x_temp, y_temp, z_temp)

    def cross(self, other: VectorType) -> VectorType:
        return VectorTup(self.y * other.z - self.z * other.y,
                         self.z * other.x - self.x * other.z,
                         self.x * other.y - self.y * other.x)

    def set_magnitude(self, magnitude: int) -> None:
        """ Normalises vector then multiplies it by given magnitude
        :param magnitude: Magnitude to set vector to
        :return: None
        """
        ini_magnitude = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.x = (self.x / ini_magnitude) * magnitude
        self.y = (self.y / ini_magnitude) * magnitude
        self.z = (self.z / ini_magnitude) * magnitude

    def get_magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def as_tup(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


class Material:
    def __init__(self, name: str, density: float, E: float, G: float, rad: float = 10) -> None:
        """These parameters are all specific named properties of a material
        'Members' refers to the edges from vertex to vertex in the object
        :param name: String
        :param density: Float: Density of elements, used to calculate their mass
        :unit: Kg/M3
        :param E: Float: Modulus of elasticity of material members
        :unit: Pascals (N / M2)
        :param G: Float: Shear modulus of material members
        :unit: Pascals (N / M2)
        :param rad: Float: Radius of elements, treated as circles
        :unit: Meters
        :Iy: Float: Moment of inertia of material's members about their local y-axis
        :Iz: Float: Moment of inertia of material's members about their local z-axis
        :J: Float: Polar moment of inertia of the material's members
        :A: Float: Cross-sectional area of material's members (Internal beam areas)
        """
        self.name = name
        self.density = density
        self.E = E
        self.G = G
        self.Iy = (math.pi * (rad ** 4)) / 4
        self.Iz = 2 * self.Iy
        self.J = (math.pi * ((2 * rad) ** 4)) / 32
        self.A = math.pi * (rad ** 2)

    def __repr__(self) -> str:
        return f"Material: {self.name} [{self.density}, {self.E}, {self.G}, {self.Iy}, {self.Iz}, {self.J}, {self.A}]"

    def __str__(self) -> str:
        return f"""{self.name} 
        [Density: {self.density}, E: {self.E}, G: {self.G}, Iy: {self.Iy}, Iz: {self.Iz}, J: {self.J}, A: {self.A}]"""

    def __getitem__(self, key) -> float:
        if key == 0 or key.lower() == "density":
            return self.density
        elif key == 1 or key == "E":
            return self.E
        elif key == 2 or key == "G":
            return self.G
        elif key == 3 or key == "Iy":
            return self.Iy
        elif key == 4 or key == "Iz":
            return self.Iz
        elif key == 5 or key == "J":
            return self.J
        elif key == 6 or key == "A":
            return self.A
        raise Exception("Invalid key: Material")

    def __setitem__(self, key, value: float) -> None:
        if key == 0 or key.lower() == "density":
            self.density = value
        elif key == 1 or key == "E":
            self.E = value
        elif key == 2 or key == "G":
            self.G = value
        elif key == 3 or key == "Iy":
            self.Iy = value
        elif key == 4 or key == "Iz":
            self.Iz = value
        elif key == 5 or key == "J":
            self.J = value
        elif key == 6 or key == "A":
            self.A = value
        raise Exception("Invalid Key: Material")

    def __len__(self) -> int:
        return 7

    def __getstate__(self) -> dict:
        return {"density": self.density, "E": self.E,
                "G": self.G, "Iy": self.Iy,
                "Iz": self.Iz, "J": self.J,
                "A": self.A}

    def __setstate__(self, state: dict) -> None:
        self.density = state["density"]
        self.E = state["E"]
        self.G = state["G"]
        self.Iy = state["Iy"]
        self.Iz = state["Iz"]
        self.J = state["J"]
        self.A = state["A"]

    def as_tup(self) -> tuple[float, float, float, float, float, float, float]:
        return self.density, self.E, self.G, self.Iy, self.Iz, self.J, self.A

    def recalc_radius(self, rad: float = 0.01) -> None:
        self.Iy = (math.pi * (rad ** 4)) / 4
        self.Iz = 2 * self.Iy
        self.J = (math.pi * ((2 * rad) ** 4)) / 32
        self.A = math.pi * (rad ** 2)

    @staticmethod
    def return_recalc_radius(rad: float) -> tuple[float, float, float, float]:
        """
        :param rad: radius of object formed with material
        :return: (Iy, Iz, J, A)
        """
        return (math.pi * (rad ** 4)) / 4, (math.pi * (rad ** 4)) / 2, \
               (math.pi * ((2 * rad) ** 4)) / 32, math.pi * (rad ** 2)


class MaterialEnum(Enum):
    """
    Enum class containing pre-definitions for materials to be used
    Material format is: Name, Density, Modulus of Elasticity, Shear Modulus
    """
    # Steel material from:
    # https://www.metalformingmagazine.com/article/?/materials/high-strength-steel/metal-properties-elastic-modulus
    STEEL = Material("STEEL", 7900, 2.1e11, 7.93e10)

    # Birch material from: https://www.matweb.com/search/datasheet_print.aspx?matguid=c499c231f20d4284a4da8bea3d2644fc
    BIRCH = Material("WOOD_BIRCH", 640, 1.186e10, 8.34e6)

    # Oak material from: https://www.matweb.com/search/DataSheet.aspx?MatGUID=ea505704d8d343f2800de42db7b16de8&ckck=1
    # Green oak specifically
    OAK = Material("WOOD_OAK", 750, 7.86e9, 6.41e6)

    # Granite material from: https://www.matweb.com/search/datasheet.aspx?matguid=3d4056a86e79481cb6a80c89caae1d90
    # and https://www.sciencedirect.com/science/article/pii/S1674775522000993
    GRANITE = Material("GRANITE", 1463, 4e10, 6.1e7)

    # Diamond material from: http://www.chm.bris.ac.uk/motm/diamond/diamprop.htm
    # and https://arxiv.org/ftp/arxiv/papers/1811/1811.09503.pdf
    DIAMOND = Material("DIAMOND", 3340, 1.22e12, 5.3e11)

    # Plastic material from: https://designerdata.nl/materials/plastics/thermo-plastics/low-density-polyethylene
    POLYETHYLENE = Material("PLASTIC_POLYETHYLENE", 955, 3e8, 2.25e8)

    # Plastic material from: https://www.matweb.com/search/datasheet_print.aspx?matguid=e19bc7065d1c4836a89d41ff23d47413
    PVC = Material("PLASTIC_PVC", 1300, 1.7e9, 6.35e7)

    # Glass material from: https://www.structuralglass.org/single-post/2016/11/26/glass-physical-properties
    # The model cannot account for the drastic difference in tensile and compressive strength for glass
    # As such glass simulation will be unrealistic for tension, but correct for compression
    GLASS = Material("GLASS", 2500, 7e10, 2.8e9)

    # Copper material from:http://www.mit.edu/~6.777/matprops/copper.htm
    # and https://www.azom.com/properties.aspx?ArticleID=597
    COPPER = Material("COPPER", 8960, 1.84e10, 6.74e9)

    # Aluminium material from: https://www.britannica.com/science/shear-modulus
    # and https://www.mit.edu/~6.777/matprops/aluminum.htm
    ALUMINIUM = Material("ALUMINIUM", 2700, 7e10, 2.4e10)

    # Brass material from: https://www.matweb.com/search/datasheet_print.aspx?matguid=d3bd4617903543ada92f4c101c2a20e5
    BRASS = Material("BRASS", 8890, 9.84e10, 3.55e10)

    # Lightweight concrete
    # Concrete material from: https://civiltoday.com/civil-engineering-materials/concrete/361-density-of-concrete
    # https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/05062/chapt2c.cfm
    CONCRETE_LIGHT = Material("CONCRETE_LIGHT", 2000, 2.3e10, 1e10)

    # Dense concrete
    # Concrete material from: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/05062/chapt2c.cfm
    CONCRETE_DENSE = Material("CONCRETE_DENSE", 2400, 3.28e10, 1.43e10)

    # C90/105 Reinforced concrete
    # This concrete material will act more accurately than the other two 
    # as it's compressive and tensile strengths are more similar than regular concrete
    # where tensile is lower than compressive normally
    # Concrete material from: https://eurocodeapplied.com/design/en1992/concrete-design-properties
    CONCRETE_REINFORCED = Material("CONCRETE_REINFORCED", 2400, 4.36e10, 1.82e10)

    






# Object populated with edges
class ForceObject:
    def __init__(self, verts: list[ForceVertType],
                 edges: list[list[int]], density: float) -> None:
        """
        :param verts: List[ForceVertex]
        :param edges: List[List[Int]] : Inner list of len 2
        :param density: float : kilograms / m^2
        """
        self.verts = verts
        self.edges = edges
        self.density = density
        self.mass = 1
        edgewise_mass = 1 / len(edges)
        self.edge_masses = [edgewise_mass] * len(edges)
        self.edge_rads = [10] * len(edges)
        self.base_nodes = self.find_base(tolerance=0.4)
        print(f"Force Object Initialised: {len(self.verts)} Verts, {len(self.edges)} Edges")

    def __repr__(self) -> str:
        return f"ForceObject: ({len(self.verts)} Verts) ({len(self.edges)} Edges)"

    def __str__(self) -> str:
        temp = ""
        i = 0
        for edge in self.edges:
            temp += f"{i} : "
            temp += [edge[0], edge[1]].__str__()
            temp += "\n"
            i += 1
        return temp

    def __len__(self) -> int:
        return len(self.verts)

    def __getstate__(self) -> dict:
        return {"verts": [v.__getstate__() for v in self.verts],
                "edges": self.edges,
                "mass": self.mass}

    def __setstate__(self, state: dict) -> None:
        self.verts = [VectorTup(v["x"], v["y"], v["z"]) for v in state["verts"]]
        self.edges = state["edges"]
        self.edges = state["mass"]

    def apply_random_forces(self, frange: tuple[float]) -> None:  # Tuple [2] specifying min and max values
        for vert in self.verts:
            temp_vec = make_random_vector(frange)
            vert.dir += temp_vec

    def apply_gravity(self) -> None:
        """
        Applies gravitational force to object vertex-wise based on the formula F = GMm/r2
        :return: None
        """
        grav_constant = 6.67e-11  # Newton's gravitational constant, with unit Nm2kg-2
        earth_mass = 5.972e24  # Mass of the earth in kg
        earth_rad = 6.371e6  # Average radius of the earth in m
        inverse_rad = 1 / (earth_rad ** 2)  # Compute division outside loop for performance reasons

        for vert_nums, element_mass in zip(self.edges, self.edge_masses):
            # The calculated force must be halved as it is assumed to be evenly split over both vertices
            gravitational_force = VectorTup(0, 0, - (grav_constant * earth_mass * element_mass * inverse_rad * 0.5))
            self.verts[vert_nums[0]].dir += gravitational_force
            self.verts[vert_nums[1]].dir += gravitational_force

    def find_base(self, tolerance: float = 0) -> set[int]:
        """ Finds nodes which the ForceObject would rest on if placed vertically downwards
        :param tolerance:
        :return:
        """
        min_height: float = math.inf
        base_nodes = set()
        for i, vert in enumerate(self.verts):
            if vert.loc.z + tolerance < min_height:
                min_height = vert.loc.z
                base_nodes = {i}
            elif vert.loc.z - tolerance <= min_height <= vert.loc.z + tolerance:
                base_nodes.add(i)
        return base_nodes

    # Creates n links from each vertex in object 1 to vertices in object two
    def mesh_link(self, other: ForceObjType, num_links: int = 2) -> None:
        """ Does not interact with object faces
        :param other: ForceObject
        :param num_links: Int: Defines how many links are created from each vertex
        :return: None
        """
        extracted = self.verts
        other_extracted = other.verts
        shift = len(extracted)
        # Shifts the indices of object two by the length of object one
        # This is done such that the index of mesh two's first vertex is n+1 for n = length of mesh one

        new_edges = deque([])  # Deque to contain pairs of vertices representing new edges to be made
        # Deque used here as the only way new_edges will be modified is by appending, which is faster for a deque

        if num_links < 1:  # Forces a positive amount of links
            num_links = 1
        for i, vert in enumerate(extracted):
            # Iterates through each vertex from the first object

            min_dist = [math.inf] * num_links  # Creates a list to carry currently found smallest distances
            temp_closest_nums = deque([None] * num_links)  # Creates a deque to carry indices of found closest vertices
            # The deque data structure is used here as it is optimised for adding and removing left and rightmost vals

            for j, vert2 in enumerate(other_extracted):
                # Iterates through each vertex from the other object

                temp_dist = vert.get_euclidean_distance(vert2)
                # Gets euclidean distance between initial and second vert
                min_dist, flag = min_add(min_dist, temp_dist)
                # Adds current distance to min_dist list in the correct position if it is less than any list values
                # flag boolean indicates whether a new distance has been added to min_dist

                if flag:
                    temp_closest_nums.appendleft(j + shift)
                    temp_closest_nums.pop()
                    # If a new smallest distance has been found, performs both:
                    # Removes largest distance edge of the temp_closest_nums list
                    # Adds new small distance edge to temp_closest_nums list

            if None not in temp_closest_nums:
                # Checks if appropriate closest nodes have been found
                # This will only fail if len(other_extracted) < num_links
                for vtc in temp_closest_nums:
                    # Iterates through the closest vertices that have been found for current vertex
                    new_edges.append([i, vtc])  # Adds an edge from node i to node vtc
            else:
                print("WARNING: ForceObject.mesh_link failed, meshes will not be linked")
                return

        self.verts.extend(other_extracted)
        # Adds second object's vertices to list of current object's vertices
        self.edges.extend(new_edges)
        # Adds newly formed edges to current object's edges
        self.edges.extend([[edge_new[0] + shift, edge_new[1] + shift] for edge_new in other.edges])
        # Adds second object's edges to first object
        # This operation shifts each edge value to point to the correct verts in the new combined object

    def mesh_link_chain(self, others: list[ForceObjType], num_links: int = 2) -> None:
        """Creates n links from each vertex of every object to vertices in other objects in the list
        Does not interact with object faces
        :param others: List[ForceObject]
        :param num_links: Int : Defines how many links are created from each vertex
        :return: None
        """
        MAX_DIST = math.inf  # Very large constant
        extracted = [self.verts]  # List containing lists of each object's vertices

        shifts = [0, len(extracted[0])]
        # List to shift the indices of all objects by the length of all objects that come before
        # This is done such that the index of mesh m first vertex is n+1 for n = length of combined previous meshes

        new_edges = deque([])  # Deque to contain pairs of vertices representing new edges to be made
        # Deque used here as the only way new_edges will be modified is by appending, which is faster for a deque

        for item in others:  # Iterate through other objects
            extracted.append(item.verts)  # Adds vertex information of each other object to extracted
            shifts.append(len(extracted[-1]) + shifts[-1])  # Adds new shifts for object m

        for mesh_num, active_mesh in enumerate(extracted):
            # Iterates over each mesh to check for its closest links in all other meshes

            for vert_num, active_vert in enumerate(active_mesh):
                # Iterates over each vertex in active mesh

                min_dist = [MAX_DIST] * num_links  # Creates a list to carry currently found smallest distances
                closest_indices = deque([None] * num_links)  # Creates a deque to hold indices of found closest vertices
                # Deque is used here as it is optimised for adding and removing left and rightmost vals

                for secondary_mesh_num in (n for n in range(len(extracted)) if n != mesh_num):
                    # Iterates through all meshes other than active mesh
                    # tuple comprehension used for the iterator for its performance over list

                    for secondary_vert_num, secondary_vert in enumerate(extracted[secondary_mesh_num]):
                        # Iterates through all vertices in secondary mesh

                        temp_dist = active_vert.get_euclidean_distance(secondary_vert)
                        # Gets euclidean distance between initial and second vert
                        min_dist, flag = min_add(min_dist, temp_dist)
                        # Inserts current distance to min_dist list if it is less than any list values
                        # flag boolean indicates whether a new distance has been added to min_dist

                        if flag:
                            closest_indices.appendleft(secondary_vert_num + shifts[secondary_mesh_num])
                            closest_indices.pop()
                            # If a new smallest distance has been found, performs both:
                            # Removes largest distance edge of the closest_indices list
                            # Adds new small distance edge to closest_indices list

                for final_ind in closest_indices:
                    # Loops through closest indices found for the active mesh
                    if [final_ind, vert_num + shifts[mesh_num]] not in new_edges:
                        # Checks if the reverse of edge has been already added to the edge list
                        new_edges.append([vert_num + shifts[mesh_num], final_ind])
                        # Adds edge from node vert_num with the correct shift to final_ind to the new_edges list

        self.edges.extend(new_edges)
        # Adds newly formed edges to current object's edges
        for i, other in enumerate(others):
            # Loops through all other objects and adds their information to the current object
            self.verts.extend(other.verts)
            # Adds other object's vertices to list of current object's vertices
            self.edges.extend([[new_edge[0] + shifts[i + 1], new_edge[1] + shifts[i + 1]] for new_edge in other.edges])
            # Adds other object's edges to first object
            # This operation shifts each edge value to point to the correct verts in the new combined object

    # Creates a finite element model from a mesh
    def to_finite(self, mat: MaterialType, lock_dict: dict, spring_constant: float) -> FEModel3D:  # Redo return typing
        """ Compiles ForceObject to FEA model via the following steps:
         - Loads in each node from the ForceObject
         - Adds each edge from the ForceObject
         - Adds each nodal force on the ForceObject
         - Adds spring supports
         - Adds standard supports
        :param mat: Material Object: Self defined material object, not blender material
        :param lock_dict: Dictionary containing directional lock parameters for each normal node
        :param spring_constant: Spring constant of supporting base springs
        :return: FEModel3D Object
        """
        final_finite = FEModel3D()
        density, E, G, Iy, Iz, J, A = mat.as_tup()
        for i, node in enumerate(self.verts):
            final_finite.add_node(str(i), node.loc.x, node.loc.y, node.loc.z)
            if i not in self.base_nodes:
                final_finite.def_support(
                    str(i),
                    support_DX=lock_dict["DX"],
                    support_DY=lock_dict["DY"],
                    support_DZ=lock_dict["DZ"],
                    support_RX=lock_dict["RX"],
                    support_RY=lock_dict["RY"],
                    support_RZ=lock_dict["RZ"]
                )
        for j, (edge, rad) in enumerate(zip(self.edges, self.edge_rads)):
            Iy, Iz, J, A = mat.return_recalc_radius(rad)
            final_finite.add_member(f"Edge{j}", str(edge[0]), str(edge[1]), E, G, Iy, Iz, J, A)
        for k, fnode in enumerate(self.verts):
            final_finite.add_node_load(str(k), Direction='FX', P=fnode.dir.x,
                                       case="Case 1")
            final_finite.add_node_load(str(k), Direction='FY', P=fnode.dir.y,
                                       case="Case 1")
            final_finite.add_node_load(str(k), Direction='FZ', P=fnode.dir.z,
                                       case="Case 1")

        use_tension = True
        use_compression = False

        # self.base_nodes is a list of indices of vertices within self.verts
        # where the nodes comprise the base of the object
        for l, base_node in enumerate(self.base_nodes):
            # Adds springs to the base nodes
            vert_literal = self.verts[base_node]
            spring_node_name = f"{l}s"
            final_finite.add_node(spring_node_name, vert_literal.loc.x, vert_literal.loc.y, vert_literal.loc.z - 1)
            # Adds node from which spring can be linked to corresponding base node

            final_finite.add_spring(f"Spring{l}", str(base_node), spring_node_name,
                                    spring_constant, tension_only=use_tension,
                                    comp_only=use_compression)

            final_finite.def_support(
                spring_node_name,
                support_DX=True,
                support_DY=True,
                support_DZ=True,
                support_RX=True,
                support_RY=True,
                support_RZ=True
            )  # Supports spring base node in all directions

            use_tension, use_compression = not use_tension, not use_compression
            # Alternates compression only and tension only springs to avoid model instability

            #Adds supports to the base nodes
            final_finite.def_support(str(base_node), support_DX=True, support_DY=True, support_RZ=True)

        return final_finite

    def get_net_moment(self) -> VectorType:
        """
        :return: VectorTup : (Moment X, Moment Y, Moment Z)
        :unit: NewtonMeters (Possibly NewtonInches depending on blender)
        """
        COG: VectorType = self.get_centre_of_gravity()
        final = VectorTup()
        for vert in self.verts:
            dist = COG - vert.loc
            final += dist.cross(vert.dir)
        return final

    def get_centre_of_gravity(self) -> VectorType:
        """ Gets the centre of gravity of an object,
        assumes uniform mass distribution,
        requires mass input and uses vertex locations as mass points
        :return: VectorTup : (x,y,z)
        """
        final = VectorTup()
        for vert in self.verts:
            final += vert.loc
        final /= len(self.verts)
        return final

    def to_blend_object(self) -> BlendObjectType:
        return BlendObject(self.verts, self.edges)

    @staticmethod
    def get_edge_mass(length: float, radius: float, density: float) -> float:
        """ Calculates and returns the "mass" of an edge E.
        Calculated as if the edge were a cylinder.
        :param length: length of cylinder
        :param radius: pre-defined cylinder radius
        :param density: cylinder material density
        :unit: Kg/M3
        :return: calculated mass via visible formula
        """
        return math.pi * (radius ** 2) * length * density

    @staticmethod
    def get_edge_rad(length: float, mass: float, density: float) -> float:
        """ Gets radius of cylindrical representation of an edge
        Must be given its length, mass and density
        :param length: length of cylinder
        :param mass: cylinder mass
        :param density: cylinder material density
        :unit: Kg/M3
        :return: calculated radius based on formula
        """
        if length == 0 or density == 0:  # Prevents divide by zero error
            return 0
        return math.sqrt(mass / (length * density * math.pi))


class ForceObjectUniformMass(ForceObject):
    def __init__(self, verts: list[ForceVertType], edges: list[list[int]], density: float, mass: float):
        """ Subclass of ForceObject which enforces uniform masses along each edge of the object
        This uniform mass means that the radii of each edge can vary given uniform density
        Each edge is treated as a cylinder
        :param verts: List of vertices
        :param edges: List of edges (pairs of vertex indices)
        :param density: Density of the material the object is made out of
        :param mass: Enforced mass of overall object, must be divided by edge number for edgewise mass
        """
        super().__init__(verts, edges, density)
        average_edge_mass = mass / len(edges)  # Divides overall mass for edgewise mass
        self.edge_masses = [average_edge_mass] * len(self.edges)

        # Gets a list of lengths of each edge (distances between point A and B in edge (A,B))
        distances = [self.verts[edge[0]].get_euclidean_distance(self.verts[edge[1]]) for edge in self.edges]

        # Gets the radii of each edge given fixed mass and the length of each edge
        self.edge_rads = [self.get_edge_rad(dist, average_edge_mass, density) for dist in distances]
        self.mass = mass


class ForceObjectUniformRad(ForceObject):
    def __init__(self, verts: list[ForceVertType], edges: list[list[int]], density: float, radius: float):
        """ Subclass of ForceObject which enforces uniform "radius" of each edge represented as a cylinder
        This uniform edge radius means that each edge gets to have different mass
        This also allows us to calculate the overall mass of the object given the information we have
        :param verts: List of vertices
        :param edges: List of edges (pairs of vertex indices)
        :param density: Density of the material the object is made out of
        :param radius: Enforced radius of cylindrical representation of each edge in object
        """
        super().__init__(verts, edges, density)
        self.edge_rads = [radius] * len(edges)

        # Gets a list of lengths of each edge (distances between point A and B in edge (A,B))
        distances = [self.verts[edge[0]].get_euclidean_distance(self.verts[edge[1]]) for edge in self.edges]

        # Gets the masses of each edge given fixed radius
        self.edge_masses = [self.get_edge_mass(dist, radius, density) for dist in distances]

        # Overall mass is sum of calculated edge masses
        self.mass = sum(self.edge_masses)


class ForceVertex:
    def __init__(self, loc: VectorType, direction: VectorType) -> None:
        self.loc: VectorType = loc
        self.dir: VectorType = direction

    def __repr__(self) -> str:
        return f"ForceVertex: (loc:{self.loc}, dir:{self.dir})"

    def __str__(self) -> str:
        return f"(loc:{self.loc}, dir:{self.dir})"

    def __add__(self, other: VectorType | ForceVertType) -> ForceVertType:
        if isinstance(other, VectorTup):
            return ForceVertex(self.loc, self.dir + other)
        elif isinstance(other, ForceVertex):
            return ForceVertex(self.loc, self.dir + other.dir)
        else:
            raise TypeError(f"Invalid ForceVertex addition: ForceVertex, {type(other)}")

    def __sub__(self, other: VectorType | ForceVertType) -> ForceVertType:
        if isinstance(other, VectorTup):
            return ForceVertex(self.loc, self.dir - other)
        elif isinstance(other, ForceVertex):
            return ForceVertex(self.loc, self.dir - other.dir)
        else:
            raise TypeError(f"Invalid ForceVertex addition: ForceVertex, {type(other)}")

    def __getstate__(self) -> dict:
        return {"loc": self.loc.__getstate__(),
                "dir": self.dir.__getstate__()}

    def __setstate__(self, state) -> None:
        self.loc = VectorTup(state["loc"]["x"], state["loc"]["y"], state["loc"]["z"])
        self.dir = VectorTup(state["dir"]["x"], state["dir"]["y"], state["dir"]["z"])

    def get_magnitude(self) -> float:
        return self.dir.get_magnitude()

    def apply_force(self, force: VectorType) -> None:
        self.dir = self.dir + force

    def get_euclidean_distance(self, other: ForceVertType) -> float:
        return math.dist(self.loc.as_tup(), other.loc.as_tup())


# Representation of a blender object to be rendered in the scene
class BlendObject:
    def __init__(self, verts: list[ForceVertType], edges: list[list[int]]) -> None:
        self.verts = [vert.loc for vert in verts]
        self.forces = [vert_force.dir for vert_force in verts]
        self.edges = edges  # Make sure these are of form bpy.context.object.data.edges
        self.materials = []
        for i in range(0, 5, 1):
            self.materials.append(create_new_shader(str(i), (i, 0, 5 - i, 1)))

    def __getstate__(self) -> dict:
        return {"name": self.name,
                "verts": [v.__getstate__() for v in self.verts],
                "forces": [f.__getstate__() for f in self.forces],
                "edges": self.edges}

    def __setstate__(self, state: dict) -> None:
        self.name = state["name"]
        self.verts = [VectorTup(v["x"], v["y"], v["z"]) for v in state["verts"]]
        self.forces = [VectorTup(f["x"], f["y"], f["z"]) for f in state["forces"]]
        self.edges = state["edges"]
        self.materials = []
        for i in range(0, 5, 1):  # Re-defines class materials from scratch as these cannot natively be pickled
            self.materials.append(create_new_shader(str(i), (i, 0, 5 - i, 1)))

    def make(self, fast: bool = False) -> None:
        """
        :param fast: boolean : Determines whether the fast rendering method is used
            Fast Rendering: Renders 0 dimensional mesh of edges and vertices
            Slow Rendering: Renders coloured edges based on the forces applied
        :return: None
        """
        if fast:  # https://blender.stackexchange.com/questions/100913/render-large-3d-graphs-millions-of-vertices-edges
            bpy.ops.mesh.primitive_cube_add()  # Creates a primitive cube so context object is at (0, 0, 0)
            context_dat = bpy.context.object.data
            bm = bmesh.new()
            vert_map = {}
            # Adds each vertex to a bmesh, transposes data to more efficient
            for i, pos in enumerate(self.verts):
                pos = np.array(pos.as_tup())
                vert = bm.verts.new(pos)
                vert_map[i] = vert

            for edge in self.edges:
                bm.edges.new((vert_map[edge[0]], vert_map[edge[1]]))

            bm.to_mesh(context_dat)
            bm.free()
        else:
            for edge in self.edges:
                self.create_cylinder((self.verts[edge[0]], self.verts[edge[1]]), 0.01)

    # Adapted from:
    # https://blender.stackexchange.com/questions/5898/how-can-i-create-a-cylinder-linking-two-points-with-python
    def create_cylinder(self, points: tuple[tuple[float]], cylinder_radius: float) -> None:
        """ Creates a cylinder
        Y axis has to be entirely flipped due to some earlier error which I will not be addressing yet
        :param points: list[Tuple[float]] : list of length 2 containing a start and end point for the given cylinder
        :param cylinder_radius: float : radius of created cylinder
        :return: None
        """
        x_dist = points[1][0] - points[0][0]
        y_dist = points[0][1] - points[1][1]
        z_dist = points[1][2] - points[0][2]
        distance = math.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
        if distance == 0:
            return
        bpy.ops.mesh.primitive_cylinder_add(
            radius=cylinder_radius,
            depth=distance,
            location=(x_dist / 2 + points[0][0],
                      y_dist / 2 + points[1][1],
                      z_dist / 2 + points[0][2]),
            vertices=3
        )
        phi_rotation = math.atan2(x_dist, y_dist)

        theta_rotation = math.acos(z_dist / distance)
        bpy.context.object.rotation_euler[0] = theta_rotation
        bpy.context.object.rotation_euler[2] = phi_rotation
        bpy.context.object.data.materials.append(random.choice(self.materials))


"""
Classless functions below, possibly aggregate them into a single class Ops at a later date
"""


# Utility function adding a value to a queue style iterable, maintaining sorting from smallest to largest
def min_add(iterable, val: float) -> tuple[Any, bool]:
    for i, item in enumerate(iterable):
        if val < item:
            iterable = iterable[:i] + [val] + iterable[i:(len(iterable) - 1)]
            return iterable, True
    return iterable, False


def get_selected(object_instance):
    return [x.select for x in object_instance.data.polygons]


def make_random_vector(frange: tuple[float, float]) -> VectorType:
    """
    :param frange: tuple[float, float]
    :return: VectorTup : VectorTup with randomised values
    """
    return VectorTup(random.uniform(frange[0], frange[1]), random.uniform(frange[0], frange[1]),
                     random.uniform(frange[0], frange[1]))


# https://vividfax.github.io/2021/01/14/blender-materials.html
# Creates and returns a new empty blender material with [name: material_name]
def create_new_material(material_name: str) -> object:
    """
    :param material_name: String
    :return: Blender Material Object: Empty
    """
    mat = bpy.data.materials.get(material_name)
    if mat is None:  # If material does not yet exist, creates it
        mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True  # Uses blender nodes
    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()
    return mat


# https://vividfax.github.io/2021/01/14/blender-materials.html
# Creates and returns a blender material with [name: material_name, emission colour: rgb: (r,g,b,1)]
def create_new_shader(material_name: str, rgb: tuple[float]) -> object:
    """
    :param material_name: String
    :param rgb: Tuple[float] (Len 3): Colour of material
    :return: Blender Material Object: Coloured, Type=Emission
    """
    mat = create_new_material(material_name)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type="ShaderNodeOutputMaterial")
    shader = nodes.new(type="ShaderNodeEmission")
    nodes["Emission"].inputs[0].default_value = rgb
    nodes["Emission"].inputs[1].default_value = 0.2
    links.new(shader.outputs[0], output.inputs[0])  # Links output of emission shader to input of the material output
    return mat


# Creates a force object simply using raw vertex and edge data
def force_obj_from_raw_mass(obj: str | object, mass: float,
                            obj_material: MaterialType = MaterialEnum.STEEL.value) -> ForceObjType:
    """ Creates a ForceObject from raw blender object data
    :param obj: Blender object or String [Object Name]
    :param mass: Enforced mass of entire object
    :param obj_material: Material object passed in to determine the density of final force object
    :return: ForceObject(Vertices: List[ForceVertex], Edges: List[Vert1, Vert2], Mass: float)
    """
    if isinstance(obj, str):
        temp_obj = bpy.data.objects[obj]
    else:
        temp_obj = obj

    temp_dat = temp_obj.data
    vert_num = len(temp_dat.vertices)
    edge_num = len(temp_dat.edges)
    global_verts = []  # Array of ForceVertex objects translated to global coordinates from local
    global_edges = []
    for i in range(vert_num):
        temp_glob = temp_obj.matrix_world @ temp_dat.vertices[i].co  # Translation to global coords
        global_verts.append(ForceVertex(VectorTup(temp_glob[0], temp_glob[1], temp_glob[2]), VectorTup(0, 0, 0)))

    for j in range(edge_num):  # LOOK INTO THIS, MAY BE SOURCE OF INEFFICIENCY
        edge_verts = temp_dat.edges[j].vertices
        global_edges.append([edge_verts[0], edge_verts[1]])

    final = ForceObjectUniformMass(global_verts, global_edges, obj_material.density, mass)
    final.apply_gravity()
    return final


# Creates a force object simply using raw vertex and edge data
def force_obj_from_raw_rad(obj: str | object, radius: float,
                           obj_material: MaterialType = MaterialEnum.STEEL.value) -> ForceObjType:
    """ Creates a ForceObject from raw blender object data
    :param obj: Blender object or String [Object Name]
    :param radius: Enforced radius for each edge's cylinder representation
    :param obj_material: Material object passed in to determine the density of final force object
    :return: ForceObject(Vertices: List[ForceVertex], Edges: List[Vert1, Vert2], Mass: float)
    """
    if isinstance(obj, str):
        temp_obj = bpy.data.objects[obj]
    else:
        temp_obj = obj

    temp_dat = temp_obj.data
    vert_num = len(temp_dat.vertices)
    edge_num = len(temp_dat.edges)
    global_verts = []  # Array of ForceVertex objects translated to global coordinates from local
    global_edges = []
    for i in range(vert_num):
        temp_glob = temp_obj.matrix_world @ temp_dat.vertices[i].co  # Translation to global coords
        global_verts.append(ForceVertex(VectorTup(temp_glob[0], temp_glob[1], temp_glob[2]), VectorTup(0, 0, 0)))

    for j in range(edge_num):  # LOOK INTO THIS, MAY BE SOURCE OF INEFFICIENCY
        edge_verts = temp_dat.edges[j].vertices
        global_edges.append([edge_verts[0], edge_verts[1]])

    final = ForceObjectUniformRad(global_verts, global_edges, obj_material.density, radius)
    final.apply_gravity()
    return final


def save_obj(obj: object, file_name: str) -> None:
    """
    Uses dill library to save an object to a file name with .pkl suffix
    :param obj: Any object
    :param file_name: String filename
    :return: None
    """
    if ".pkl" not in file_name:
        try:
            with open(f"{file_name}.pkl", "wb") as f:
                dill.dump(obj, f)
        except PermissionError:
            print(f"Could not open file {file_name}.pkl due to: PermissionError")
        except FileNotFoundError:
            print(f"Could not open {file_name}.pkl due to: FileNotFoundError")
        except NameError:
            print(f"Could not open {file_name}.pkl due to: NameError, dill not installed")
    else:
        try:
            with open(file_name, "wb") as f:
                dill.dump(obj, f)
        except PermissionError:
            print(f"Could not open file {file_name} due to: PermissionError")
        except FileNotFoundError:
            print(f"Could not open {file_name} due to: FileNotFoundError")
        except NameError:
            print(f"Could not open {file_name} due to: NameError, dill not installed")


def save_obj_pickle(obj: object, file_name: str) -> None:
    """
    Uses pickle library to save an object to a file name with .pkl suffix
    Used as an alternative to the dill saving structure
    :param obj: Any object
    :param file_name: String filename
    :return: None
    """
    if ".pkl" not in file_name:
        try:
            with open(f"{file_name}.pkl", "wb") as f:
                pickle.dump(obj, f)
        except PermissionError:
            print(f"Could not open file {file_name}.pkl due to: PermissionError")
        except FileNotFoundError:
            print(f"Could not open {file_name}.pkl due to: FileNotFoundError")
        except NameError:
            print(f"Could not open {file_name}.pkl due to: NameError, pickle not installed")
    else:
        try:
            with open(file_name, "wb") as f:
                pickle.dump(obj, f)
        except PermissionError:
            print(f"Could not open file {file_name} due to: PermissionError")
        except FileNotFoundError:
            print(f"Could not open {file_name} due to: FileNotFoundError")
        except NameError:
            print(f"Could not open {file_name} due to: NameError, pickle not installed")


def load_obj(file_name: str) -> object:
    """
    Uses dill library to load an object from a file with .pkl suffix
    :param file_name: String filename
    :return: Unpickled object
    """
    if ".pkl" not in file_name:
        try:
            with open(f"{file_name}.pkl", "rb") as f:
                final = dill.load(f)
        except PermissionError:
            print(f"Could not open file {file_name}.pkl due to: PermissionError")
            return None
        except FileNotFoundError:
            print(f"Could not open {file_name}.pkl due to: FileNotFoundError")
            return None
        except NameError:
            print(f"Could not open {file_name}.pkl due to: NameError, dill not installed")
            return None
    else:
        try:
            with open(file_name, "rb") as f:
                final = dill.load(f)
        except PermissionError:
            print(f"Could not open file {file_name} due to: PermissionError")
            return None
        except FileNotFoundError:
            print(f"Could not open {file_name} due to: FileNotFoundError")
            return None
        except NameError:
            print(f"Could not open {file_name} due to: NameError, dill not installed")
            return None
    return final


def load_obj_pickle(file_name: str) -> object:
    """
    Uses pickle library to load an object from a file with .pkl suffix
    Used as an alternative to the dill saving structure
    :param file_name: String filename
    :return: Unpickled object
    """
    if ".pkl" not in file_name:
        try:
            with open(f"{file_name}.pkl", "rb") as f:
                final = pickle.load(f)
        except PermissionError:
            print(f"Could not open file {file_name}.pkl due to: PermissionError")
            return None
        except FileNotFoundError:
            print(f"Could not open {file_name}.pkl due to: FileNotFoundError")
            return None
        except NameError:
            print(f"Could not open {file_name}.pkl due to: NameError, pickle not installed")
            return None
    else:
        try:
            with open(file_name, "rb") as f:
                final = pickle.load(f)
        except PermissionError:
            print(f"Could not open file {file_name} due to: PermissionError")
            return None
        except FileNotFoundError:
            print(f"Could not open {file_name} due to: FileNotFoundError")
            return None
        except NameError:
            print(f"Could not open {file_name} due to: NameError, pickle not installed")
            return None
    return final


def render_finite(model: FEModel3D, deform: bool = False, save_path: str = "") -> None:
    """ Analyzes a FEModel3D then renders them to a custom output
    :param model: FEModel3D : Generated and correctly loaded finite element model
    :param deform: Boolean : Determines whether force based deformation will be displayed upon render
    :param save_path: Determines the path to which the analyzed model will be saved to. No save if unspecified.
    :return: None
    """
    # Performs Finite Element Analysis on loaded model
    force_finite.analyze(log=True, check_statics=True)

    # Brings global variables USE_DILL and USE_PICKLE into the function scope
    global USE_DILL, USE_PICKLE

    # Saves FEM analysis results to specified filepath
    if save_path != "":
        if USE_DILL:
            try:
                save_obj(force_finite, save_path)
            except PermissionError:
                print(f"Object could not be saved due to: PermissionError on filepath {save_path}")
            except FileNotFoundError:
                print(f"Object could not be saved due to: FileNotFoundError on filepath {save_path}")
        elif USE_PICKLE:
            try:
                save_obj_pickle(force_finite, save_path)
            except PermissionError:
                print(f"Object could not be saved due to: PermissionError on filepath {save_path}")
            except FileNotFoundError:
                print(f"Object could not be saved due to: FileNotFoundError on filepath {save_path}")
        else:
            print("Object could not be saved due to: Neither Pickle nor Dill could be found")

    # Creates Renderer object which takes in analyzed FEM Model
    finite_renderer = Renderer(model)

    # Sets Renderer parameters
    finite_renderer.color_map = "Mx"
    finite_renderer.annotation_size = 0.1
    finite_renderer.deformed_shape = deform
    finite_renderer.labels = False
    finite_renderer.scalar_bar = True
    finite_renderer.scalar_bar_text_size = 10

    # Renders the model. This creates a new window that will lock Blender as a program in order to
    finite_renderer.render_model()


def render_finite_from_file(file_path: str, deform: bool = False) -> None:
    """ Renders a pre-analyzed FEM model from a file
    :param file_path: Filepath from which the finite element model will be loaded. This must include file name.
    :param deform: Determines whether force based deformation will be displayed upon render
    :return: None
    """
    finite_model = None

    # Brings global variables USE_DILL and USE_PICKLE into the function scope
    global USE_DILL, USE_PICKLE

    try:
        # Attempts to load the file from a specified filepath
        if USE_DILL:
            # Attempts to use dill as the first option to load from
            finite_model = load_obj(file_path)
        elif USE_PICKLE:
            # Attempts to use pickle as a backup if dill is not found
            finite_model = load_obj_pickle(file_path)
        else:
            # If neither pickle nor dill are found the file cannot be loaded
            print("Object could not be loaded due to: No save options available (Please install Pickle or Dill)")
            return
    except FileNotFoundError:
        print(f"Object could not be loaded due to: FileNotFoundError for filepath {file_path}")
        return
    except PermissionError:
        print(f"Object could not be loaded due to: PermissionError for filepath {file_path}")
        return

    if finite_model:
        # Creates Renderer object which takes in analyzed FEM Model
        finite_renderer = Renderer(finite_model)

        # Sets Renderer parameters
        finite_renderer.color_map = "Mx"
        finite_renderer.annotation_size = 0.2
        finite_renderer.deformed_shape = deform
        finite_renderer.labels = False
        finite_renderer.scalar_bar = True
        finite_renderer.scalar_bar_text_size = 15

        # Renders the model. This creates a new window that will lock Blender as a program in order to
        finite_renderer.render_model()

    else:
        # finite_model is either None or some other un-analyzable data
        print("Object not loaded properly, possibly empty")


def unify_to_fobject_mass(mass: float) -> ForceObjType:
    """
    Takes all objects in the scene, links them together, forms them into a blender mesh
    Then all other objects are deleted and the blender mesh is loaded into the scene
    This blender mesh is then re-loaded into a ForceObject to be analysed
    A little inefficient and roundabout however it is the simplest way to fix mesh instability
    :arg mass: mass of the force object result
    :return: ForceObject representation of linked scene meshes
    """

    if mass == 0:
        print("WARNING: Objects cannot have 0 mass, automatically adjusting to a mass of 1 kilogram")
        mass = 1

    bpy_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    force_objects = [force_obj_from_raw_mass(ob, mass) for ob in bpy_objects]

    if len(force_objects) > 1:
        force_objects[0].mesh_link_chain(force_objects[1:])
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        force_objects[0].to_blend_object().make(fast=True)
        bpy_object = bpy.data.objects[0]
        return force_obj_from_raw_mass(bpy_object, mass)
    elif len(force_objects) == 1:
        return force_objects[0]
    else:
        raise IndexError("IndexError: force_objects has zero length, please add objects to the blender scene")


def unify_to_fobject_rad(radius: float) -> ForceObjType:
    """
    Takes all objects in the scene, links them together, forms them into a blender mesh
    Then all other objects are deleted and the blender mesh is loaded into the scene
    This blender mesh is then re-loaded into a ForceObject to be analysed
    A little inefficient and roundabout however it is the simplest way to fix mesh instability
    :arg radius: radius of the finite elements of the object
    :return: ForceObject representation of linked scene meshes
    """

    if radius == 0:
        print("WARNING: An object cannot have element radius of 0, automatically adjusting to 1 meter")
        radius = 1

    bpy_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    force_objects = [force_obj_from_raw_rad(ob, radius) for ob in bpy_objects]

    if len(force_objects) > 1:
        force_objects[0].mesh_link_chain(force_objects[1:])
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        force_objects[0].to_blend_object().make(fast=True)
        bpy_object = bpy.data.objects[0]
        return force_obj_from_raw_rad(bpy_object, radius)
    elif len(force_objects) == 1:
        return force_objects[0]
    else:
        raise IndexError("IndexError: force_objects has zero length, please add objects to the blender scene")


def vert_locks(key: str = "NONE") -> dict:
    """
    Gets the dictionary of booleans representing vertex locks for certain combinations of directions
    Vertex locks are a PyNite feature which defines the directions in which vertices cannot move or rotate
    These are important because they counteract model instability
    This is designed for interfacing with a UI
    :param key: String containing the vertex degrees of freedom to lock
    :return: dictionary of booleans defining whether that particular degree of freedom is locked
    """
    key = key.upper()

    if key in ["ALL", "ALL_LOCKS", "A"]:
        # Locks all degrees of freedom
        return {"DX": True, "DY": True, "DZ": True, "RX": True, "RY": True, "RZ": True}
    elif key in ["POS", "ALL_POSITION", "POSITION", "P"]:
        # Locks all translational but no rotational degrees of freedom
        return {"DX": True, "DY": True, "DZ": True, "RX": False, "RY": False, "RZ": False}
    elif key in ["ROT", "ALL_ROTATION", "ROTATION", "R"]:
        # Locks all rotational but no translational degrees of freedom
        return {"DX": False, "DY": False, "DZ": False, "RX": True, "RY": True, "RZ": True}
    elif key in ["NONE", "N", ""]:
        # Locks no degrees of freedom
        return {"DX": False, "DY": False, "DZ": False, "RX": False, "RY": False, "RZ": False}

    final = {"DX": False, "DY": False, "DZ": False, "RX": False, "RY": False, "RZ": False}
    if "DX" in key:
        final["DX"] = True
    if "DY" in key:
        final["DY"] = True
    if "DZ" in key:
        final["DZ"] = True
    if "RX" in key:
        final["RX"] = True
    if "RY" in key:
        final["RY"] = True
    if "RZ" in key:
        final["RZ"] = True
    return final


if __name__ == "__main__":
    force_object_final = unify_to_fobject_mass(0.1)
    # If USE_PYNITE is false, the newly linked mesh is still rendered to the scene
    if USE_PYNITE:
        default_lock_dict = vert_locks("DYRX")  # Get user input for this
        force_finite = force_object_final.to_finite(MaterialEnum.STEEL.value, default_lock_dict, 1e6)
        bpy.ops.wm.save_mainfile()  # Saves the file before rendering via PyNite visualiser
        if USE_THREADING:
            render_thread = threading.Thread(target=render_finite, args=(force_finite, True))
            render_thread.start()
            render_thread.join()
        else:
            render_finite(force_finite, deform=True, save_path="C:\\Users\\Gabriel\\Documents\\finite_mesh.pkl")

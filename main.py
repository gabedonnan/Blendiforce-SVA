import bpy
import bmesh
from enum import Enum
import dill
import math
import random
from collections import deque
from typing import TypeVar, Type, Optional, Any, Iterable

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

    def normalise(self) -> None:
        magnitude = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.x = self.x / magnitude
        self.y = self.y / magnitude
        self.z = self.z / magnitude

    def get_normalised(self) -> VectorType:
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
        ini_magnitude = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.x = (self.x / ini_magnitude) * magnitude
        self.y = (self.y / ini_magnitude) * magnitude
        self.z = (self.z / ini_magnitude) * magnitude

    def get_magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def as_tup(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


class Material:
    def __init__(self, name: str, E: float, G: float, Iy: float, Iz: float, J: float, A: float) -> None:
        """These parameters are all specific named properties of a material
        'Members' refers to the edges from vertex to vertex in the object
        :param name: String
        :param E: Float: Modulus of elasticity of material members
        :param G: Float: Shear modulus of material members
        :param Iy: Float: Moment of inertia of material's members about their local y-axis
        :param Iz: Float: Moment of inertia of material's members about their local z-axis
        :param J: Float: Polar moment of inertia of the material's members
        :param A: Float: Cross-sectional area of material's members (Internal beam areas)
        """
        self.name = name
        self.E = E
        self.G = G
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.A = A

    def __repr__(self) -> str:
        return f"Material: {self.name} [{self.E}, {self.G}, {self.Iy}, {self.Iz}, {self.J}, {self.A}]"

    def __str__(self) -> str:
        return f"{self.name} [E: {self.E}, G: {self.G}, Iy: {self.Iy}, Iz: {self.Iz}, J: {self.J}, A: {self.A}]"

    def __getitem__(self, key) -> float:
        if key == 0 or key == "E":
            return self.E
        elif key == 1 or key == "G":
            return self.G
        elif key == 2 or key == "Iy":
            return self.Iy
        elif key == 3 or key == "Iz":
            return self.Iz
        elif key == 4 or key == "J":
            return self.J
        elif key == 5 or key == "A":
            return self.A
        raise Exception("Invalid key: Material")

    def __setitem__(self, key, value: float) -> None:
        if key == 0 or key == "E":
            self.E = value
        elif key == 1 or key == "G":
            self.G = value
        elif key == 2 or key == "Iy":
            self.Iy = value
        elif key == 3 or key == "Iz":
            self.Iz = value
        elif key == 4 or key == "J":
            self.J = value
        elif key == 5 or key == "A":
            self.A = value
        raise Exception("Invalid Key: Material")

    def __len__(self) -> int:
        return 6

    def as_list(self) -> list[float]:
        return [self.E, self.G, self.Iy, self.Iz, self.J, self.A]


class MaterialEnum(Enum):
    STEEL = Material("STEEL", 0, 0, 0, 0, 0, 0)


# Object populated with edges
class ForceObject:
    def __init__(self, obj: object, verts: list[ForceVertType],
                 edges: list[list[int]], faces: list[list[int]],
                 mass: float) -> None:
        """
        :param obj: Blender Object
        :param verts: List[VectorTup]
        :param edges: List[List[Int]] : Inner list of len 2
        :param faces: List[List[Int]] : Inner list of len n (faces of any shape)
        :param mass: float : kilograms
        """
        self.obj = obj  # Bound blender object
        self.verts = verts
        self.edges = edges
        self.faces = faces
        self.mass = mass
        print(f"Force Object Initialised: {len(self.verts)} Verts, {len(self.edges)} Edges, {len(self.faces)} Faces")

    def __repr__(self) -> str:
        return f"ForceObject: ({len(self.verts)} Verts) ({len(self.edges)} Edges) ({len(self.faces)} Faces)"

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

    def apply_random_forces(self, frange: tuple[float]) -> None:  # Tuple [2] specifying min and max values
        for vert in self.verts:
            temp_vec = make_random_vector(frange)
            vert.dir += temp_vec

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
        new_edges = []
        num_links = int(num_links)
        if num_links < 1:
            num_links = 1
        for i, vert in enumerate(extracted):
            min_dist = [9999] * num_links
            temp_closest = [None] * num_links
            temp_closest_nums = [None] * num_links
            for j, vert2 in enumerate(other_extracted):
                temp_dist = vert.get_euclidean_distance(
                    vert2)  # Gets euclidean distance between initial and second vert
                min_dist, flag = min_add(min_dist, temp_dist)
                if flag:
                    temp_closest = [vert2] + temp_closest[:-1]  # add to beginning and pop last
                    temp_closest_nums = [j + shift] + temp_closest_nums[:-1]
            if None not in temp_closest_nums:
                for vtc in temp_closest_nums:
                    new_edges.append([i, vtc])
            else:
                print("ERROR")

        self.verts.extend(other_extracted)
        self.edges.extend(new_edges)
        self.edges.extend([[edge_new[0] + shift, edge_new[1] + shift] for edge_new in other.edges])

    def mesh_link_chain(self, others: list[ForceObjType], num_links: int = 2) -> None:
        """Creates n links from each vertex of every object to vertices in other objects in the list
        Does not interact with object faces
        :param others: List[ForceObject]
        :param num_links: Int : Defines how many links are created from each vertex
        :return: None
        """
        MAX_DIST = 99999  # Constant
        extracted = [self.verts]
        shifts = [0, len(extracted[0])]
        new_edges = []
        for item in others:  # Iterate through other objects
            extracted.append(item.verts)
            shifts.append(len(extracted[-1]) + shifts[-1])
        for mesh_num, active_mesh in enumerate(extracted):
            for vert_num, active_vert in enumerate(active_mesh):
                min_dist = [MAX_DIST] * num_links
                closest_indices = deque([None] * num_links)
                for secondary_mesh_num in (n for n in range(len(extracted)) if n != mesh_num):
                    for secondary_vert_num, secondary_vert in enumerate(extracted[secondary_mesh_num]):
                        temp_dist = active_vert.get_euclidean_distance(
                            secondary_vert)
                        min_dist, flag = min_add(min_dist, temp_dist)
                        if flag:
                            closest_indices.appendleft(secondary_vert_num + shifts[secondary_mesh_num])
                            closest_indices.pop()
                for final_ind in closest_indices:
                    if [final_ind, vert_num + shifts[mesh_num]] not in new_edges:
                        new_edges.append([vert_num + shifts[mesh_num], final_ind])

        self.edges.extend(new_edges)
        for i, other in enumerate(others):
            self.verts.extend(other.verts)
            self.edges.extend([[new_edge[0] + shifts[i + 1], new_edge[1] + shifts[i + 1]] for new_edge in other.edges])

    # Creates a finite element model from a mesh
    def to_finite(self, mat: MaterialType) -> object:  # Redo return typing
        """ Works only for single material objects
        :param mat: Material Object: Self defined material object, not blender material
        :return: FEModel3D Object
        """
        final_finite = FEModel3D()
        extracted = self.verts
        E, G, Iy, Iz, J, A = mat.as_list()
        for node in extracted:
            final_finite.add_node(str(node), node.loc[0], node.loc[1], node.loc[2])
        for j, edge in enumerate(self.edges):
            final_finite.add_member("C" + str(j), str(edge[0]), str(edge[1]), E, G, Iy, Iz, J, A)
        for k, fnode in enumerate(extracted):
            final_finite.add_node_load(str(fnode), Direction='FX', P=fnode.dir.x,
                                       case="Case " + str(k))
            final_finite.add_node_load(str(fnode), Direction='FY', P=fnode.dir.y,
                                       case="Case " + str(k))
            final_finite.add_node_load(str(fnode), Direction='FZ', P=fnode.dir.z,
                                       case="Case " + str(k))
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

    def get_magnitude(self) -> float:
        return self.dir.get_magnitude()

    def apply_force(self, force: VectorType) -> None:
        self.dir = self.dir + force

    def get_euclidean_distance(self, other: ForceVertType) -> float:
        return math.dist(self.loc.as_tup(), other.loc.as_tup())


# Representation of a blender object to be rendered in the scene
class BlendObject:
    def __init__(self, name: str, verts: list[ForceVertType], edges: list[list[int]], faces: list[list[int]]) -> None:
        self.name = name
        self.verts = [vert.loc for vert in verts]
        self.forces = [vert_force.dir for vert_force in verts]
        self.edges = edges  # Make sure these are of form bpy.context.object.data.edges
        self.faces = faces  # Faces should only be visible faces
        self.materials = []
        for i in range(0, 5, 1):
            self.materials.append(create_new_shader(str(i), (i, 0, 5 - i, 1)))

    def make(self, collection_name: str = "Collection") -> None:
        """
        :param collection_name: string defining the name of the collection that is currently active
        :return: None
        """
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
def force_obj_from_raw(obj: str | object) -> ForceObjType:  # Obj is object identifier
    """
    :param obj: Blender object or String [Object Name]
    :return: ForceObject(Object Reference: Blender Object, Vertices: List[ForceVertex], Edges: List[Vert1, Vert2])
    """
    if isinstance(obj, str):
        temp_obj = bpy.data.objects[obj]
    else:
        temp_obj = obj
    obj_mass = obj["MASS"]  # Accesses object's custom property "MASS"
    temp_dat = temp_obj.data
    vert_num = len(temp_dat.vertices)
    edge_num = len(temp_dat.edges)
    face_num = len(temp_dat.polygons)
    global_verts = []  # Array of ForceVertex objects translated to global coordinates from local
    global_edges = []
    global_faces = []
    for i in range(vert_num):
        temp_glob = temp_obj.matrix_world @ temp_dat.vertices[i].co  # Translation to global coords
        global_verts.append(ForceVertex(VectorTup(temp_glob[0], temp_glob[1], temp_glob[2]), VectorTup(0, 0, 0)))

    for j in range(edge_num):
        edge_verts = temp_dat.edges[j].vertices
        global_edges.append([edge_verts[0], edge_verts[1]])

    for k in range(face_num):
        face_verts = temp_dat.polygons[k].vertices
        global_faces.append(face_verts)

    return ForceObject(temp_obj, global_verts, global_edges, global_faces, obj_mass)


def save_obj(obj: object, file_name: str) -> None:
    """
    Uses dill library to save an object to a file name with .pkl suffix
    :param obj: Any object
    :param file_name: String filename
    :return: None
    """
    if ".pkl" not in file_name:
        with open(file_name + ".pkl", "wb") as f:
            dill.dump(obj, f)
    else:
        with open(file_name, "wb") as f:
            dill.dump(obj, f)


def load_obj(file_name: str) -> object:
    """
    Uses dill library to load an object from a file with .pkl suffix
    :param file_name: String filename
    :return: Unpickled object
    """
    if ".pkl" not in file_name:
        with open(file_name + ".pkl", "rb") as f:
            final = dill.load(f)
    else:
        with open(file_name, "rb") as f:
            final = dill.load(f)
    return final


if __name__ == "__main__":
    # Convenient constants
    CTX = bpy.context
    DAT = bpy.data
    OPS = bpy.ops

    obj = CTX.active_object
    bpy_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    force_objects = []
    for ob in bpy_objects:
        print(ob.name)
        force_objects.append(force_obj_from_raw(ob))
        force_objects[-1].apply_random_forces((-4, 7))
    print(len(force_objects[0]))

    force_objects[0].mesh_link_chain(force_objects[1:])
    x = BlendObject("ham", force_objects[0].verts, force_objects[0].edges, force_objects[0].faces)
    x.make()

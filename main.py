import bpy
import bmesh
import math
import random


class ApplyForceDialogueOperator(bpy.types.Operator):
    bl_idname = "object.apply_force_dialogue"
    bl_label = "Choose forces to apply"

    name = bpy.props.StringProperty(name="Object Selected:", default="Cube")  # CHANGE DEFAULT LATER
    force = bpy.props.FloatVectorProperty(name="Force:", default=(1.0, 1.0, 1.0))

    def execute(self, context):
        ForceObject(bpy.context.active_object, self.force)
        self.report({'INFO'}, "Made Force Object")
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


# A vector, representable as a tuple
class VectorTup:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __mul__(self, other):  # Other int / float
        return VectorTup(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return VectorTup(self.x * other, self.y * other, self.z * other)

    def __div__(self, other):
        return VectorTup(self.x / other, self.y / other, self.z / other)

    def __add__(self, other):
        return VectorTup(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return VectorTup(self.x - other.x, self.y - other.y, self.z - other.z)

    def __repr__(self):
        return f"Vector: ({self.x}, {self.y}, {self.z})"

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __bool__(self):
        return False if (not (
                    self.x or self.y or self.z)) else True  ##If all numbers are 0 or invalid return false (via demorgans laws)

    def __neg__(self):
        return VectorTup(-self.x, -self.y, -self.z)

    def __lt__(self, other):  # Comparator definitions
        return self.get_magnitude() < other.get_magnitude()

    def __gt__(self, other):
        return self.get_magnitude() > other.get_magnitude()

    def __le__(self, other):
        return self.get_magnitude() <= other.get_magnitude()

    def __ge__(self, other):
        return self.get_magnitude() >= other.get_magnitude()

    def __ne__(self, other):  ##Tests for inequality of entire vector, not magnitude inequality
        return not (self.x == other.x and self.y == other.y and self.z == other.z)

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y and self.z == other.z)

    def normalise(self):
        magnitude = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.x = self.x / magnitude
        self.y = self.y / magnitude
        self.z = self.z / magnitude

    def get_normalised(self):
        magnitude = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        x_temp = self.x / magnitude
        y_temp = self.y / magnitude
        z_temp = self.z / magnitude
        return VectorTup(x_temp, y_temp, z_temp)

    def set_magnitude(self, magnitude):
        ini_magnitude = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.x = (self.x / ini_magnitude) * magnitude
        self.y = (self.y / ini_magnitude) * magnitude
        self.z = (self.z / ini_magnitude) * magnitude

    def get_magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def as_tup(self):
        return (self.x, self.y, self.z)


# Object populated with edges
class ForceObject:
    def __init__(self, obj, edges):  # Blender Obj, Str, [ForceVertex]
        self.obj = obj  # Bound blender object
        self.edges = edges  # array of VertPair objects making up force object

    def __add__(self, other):  # Adds to list of edges of object
        temp = ForceObject(self.obj, self.name, self.edges)
        temp.edges.extend(other.edges)
        return temp

    def __repr__(self):
        return f"ForceObject: ({len(self.edges)} Edges)"

    def __str__(self):
        temp = ""
        i = 0
        for edge in self.edges:
            temp += f"{i} : "
            temp += edge.__str__()
            temp += "\n"
            i += 1
        return temp

    def __len__(self):
        return len(self.edges)

    def apply_random_forces(self, frange):  # Tuple [2] specifying min and max values
        for edge in self.edges:
            temp_vec = make_random_vector(frange)
            temp_loc = random.random()  # Will never generate 1 but will not matter because it will be close
            edge.apply_force(temp_vec, temp_loc)

    def extract_verts(self):
        temp_verts = []
        for edge in self.edges:
            if edge.one not in temp_verts:
                temp_verts.append(edge.one)
            if edge.two not in temp_verts:
                temp_verts.append(edge.two)
        return temp_verts

    # Creates n links from each vertex in object 1 to vertices in object two
    def mesh_link(self, other, link_type="CLOSEST", num_links=2):
        extracted = self.extract_verts()
        other_extracted = other.extract_verts()
        new_edges = []
        num_links = int(num_links)
        if num_links < 1:
            num_links = 1
        temp_dist = 0
        if link_type == "CLOSEST":
            for vert in extracted:
                min_dist = [9999] * num_links
                temp_closest = [None] * num_links
                for vert2 in other_extracted:
                    temp_dist = vert.get_euclidean_distance(
                        vert2)  # Gets euclidean distance between initial and second vert
                    min_dist, flag = min_add(min_dist, temp_dist)
                    if flag:
                        temp_closest = [vert2] + temp_closest[:-1]  # add to beginning and pop last
                if None not in temp_closest:
                    for vtc in temp_closest:
                        new_edges.append(VertPair(vert, vtc))
                else:
                    print("ERROR")
            # print(other.edges)
            self.edges.extend(new_edges)
            self.edges.extend(other.edges)

    # Creates n links from each vertex of every object to vertices in other objects in the list
    def mesh_link_chain(self, others, link_type="CLOSEST", num_links=2):
        extracted = [self.extract_verts()]
        for item in others:
            extracted.append(item.extract_verts())
        new_edges = []
        num_links = int(num_links)
        if num_links < 1:
            num_links = 1
        temp_dist = 0
        if link_type == "CLOSEST":
            for i, mesh in enumerate(extracted):  # Extracted formatted as such[[], [], [], []]
                for vert in mesh:
                    min_dist = [9999] * num_links
                    temp_closest = [None] * num_links
                    for j in (n for n in range(len(extracted)) if n != i):  # All indices of list other than i
                        for vert2 in extracted[j]:  # Iterates through all vertices from non-active objects
                            temp_dist = vert.get_euclidean_distance(
                                vert2)  # Gets euclidean distance between initial and second vert
                            min_dist, flag = min_add(min_dist, temp_dist)
                            if flag:
                                temp_closest = [vert2] + temp_closest[:-1]  # add to beginning and pop last
                    for vtc in temp_closest:
                        if VertPair(vtc,
                                    vert) not in new_edges:  # Checks if the link has already been created in the other direction
                            new_edges.append(VertPair(vert, vtc))

            self.edges.extend(new_edges)
            for other in others:
                self.edges.extend(other.edges)

    # Creates a finite element model from a mesh
    def to_finite(self, mat):
        final_finite = FEModel3D()
        extracted = self.extract_verts()
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


# Utility function adding a value to a queue style iterable, maintaining sorting from smallest to largest
def min_add(iterable, val):
    for i, item in enumerate(iterable):
        if val < item:
            iterable = iterable[:i] + [val] + iterable[i:(len(iterable) - 1)]
            return iterable, True
    return iterable, False


class Material:
    def __init__(self, name, E, G, Iy, Iz, J, A):
        self.name = name
        self.E = E
        self.G = G
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.A = A

    def __repr__(self):
        return ("Material: " + self.name + " [" + str(self.E) + ", " + str(self.G) +
                ", " + str(self.Iy), + ", " + str(self.Iz) + ", " +
                str(self.J) + ", " + str(self.A) + "]")

    def __str__(self):
        return (self.name + " [E:" + str(self.E) + ", G:" + str(self.G) +
                ", Iy:" + str(self.Iy), + ", Iz:" + str(self.Iz) + ", J:" +
                str(self.J) + ", A:" + str(self.A) + "]")

    def __getitem__(self, key):
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

    def __setitem__(self, key, value):
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

    def __len__(self):
        return 6

    def as_list(self):
        return [self.E, self.G, self.Iy, self.Iz, self.J, self.A]


class ForceVertex:
    def __init__(self, loc, dir):
        self.loc = loc  # (x, y, z) tuple
        self.dir = dir  # VectorTup object

    def get_magnitude(self):
        return self.dir.get_magnitude()

    def __repr__(self):
        return f"ForceVertex: (loc:{self.loc}, dir:{self.dir})"

    def __str__(self):
        return f"(loc:{self.loc}, dir:{self.dir})"

    def apply_force(self, force):
        self.dir = self.dir + force

    def get_euclidean_distance(self, other):
        return math.dist(self.loc.as_tup(), other.loc.as_tup())


# For holding a pair of ForceVertex objects,
# these pairs indicate a link between two vertices used in the final lattice
class VertPair:
    def __init__(self, one, two):
        self.one = one
        self.two = two

    # Equality checks if contains same two elements in either slot
    def __eq__(self, other):
        return (self.one == other.one and self.two == other.two) or (self.one == other.two and self.two == other.one)

    def __ne__(self, other):
        return not ((self.one == other.one and self.two == other.two) or (
                    self.one == other.two and self.two == other.one))

    def __str__(self):
        return f"({self.one.__str__()}, {self.two.__str__()})"

    def __repr__(self):
        return f"VertPair: ({self.one}, {self.two})"

    def __getitem__(self, key):
        if key == 0:
            return self.one
        if key == 1:
            return self.two
        raise Exception("Invalid Key: VertPair")

    # loc is a value between 0 and 1 specifying where along the edge the force is applied, 0 being vert 1 and 1 being vert 2
    def apply_force(self, force, loc):
        self.one.apply_force(force * loc)
        self.two.apply_force(force * (1 - loc))


# Creates a force object simply using raw vertex and edge data
def force_obj_from_raw(obj):  # Obj is object identifier
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

    for j in range(edge_num):
        edge_verts = temp_dat.edges[j].vertices
        global_edges.append(VertPair(global_verts[edge_verts[0]], global_verts[edge_verts[1]]))

    return ForceObject(temp_obj, global_edges)


def make_random_vector(frange):
    return VectorTup(random.uniform(frange[0], frange[1]), random.uniform(frange[0], frange[1]),
                     random.uniform(frange[0], frange[1]))


# Conveinient constants
CTX = bpy.context
DAT = bpy.data
OPS = bpy.ops

# OPS.object.mode_set(mode = 'OBJECT')
# OPS.object.select_all(action = 'SELECT')
# OPS.object.delete(use_global = False)
# OPS.mesh.primitive_cube_add(size = 2, enter_editmode = False, align = 'WORLD', location = (0,0,0), rotation=(0.2,1.2,1))
# obj = bpy.data.objects["CUBE"]

obj = CTX.active_object


# OPS.object.mode_set(mode = 'EDIT')
# OPS.mesh.select_mode(type = 'FACE')
# OPS.mesh.select_all(action = 'DESELECT')
# OPS.object.mode_set(mode = 'OBJECT')
# obj.data.polygons[0].select = True
# OPS.object.mode_set(mode = 'EDIT')

# def getSelected(obj): #Gets selected faces of object INPUTS: bpy.context.obj object (for current object use bpy.context.active_object)  RETURNS: [Boolean]
#    return [x.select for x in obj.data.polygons]

def getSelected(obj):
    return [x.select for x in (obj).data.polygons]


# bpy.utils.register_class(ApplyForceDialogueOperator)
# bpy.ops.object.apply_force_dialogue('INVOKE_DEFAULT')
fobjects = []
for ob in bpy.data.objects:
    print(ob.name)
    fobjects.append(force_obj_from_raw(ob))
    fobjects[-1].apply_random_forces((-4, 7))
# fobj1 = force_obj_from_raw(bpy.context.view_layer.objects.active)
# fobj1.apply_random_forces((-1,12))
# fobj1.extract_verts()
# print(fobj1)
print(len(fobjects[0]))

# fobjects[0].mesh_link(fobjects[1])
fobjects[0].mesh_link_chain(fobjects[1:])
print(fobjects[0])

# Creates and returns a new empty blender material with [name: material_name]
def create_new_material(material_name): #https://vividfax.github.io/2021/01/14/blender-materials.html#:~:text=Assign%20a%20material%20to%20an%20object%20in%20Blender%20using%20Python&text=The%20function%20takes%20a%20string%20as%20the%20name%20for%20the%20new%20material.&text=Then%20add%20a%20shader%20to,glossy)%20and%20the%20rgb%20colour.&text=Then%20create%20the%20object%2C%20assign%20the%20material%20and%20call%20the%20function.
    #Creates new material
    mat = bpy.data.materials.get(material_name)
    if mat is None: #If material does not yet exist, creates it
        mat = bpy.data.materials.new(name = material_name)
    mat.use_nodes = True #Uses blender nodes
    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()
    return mat

# Creates and returns a blender material with [name: material_name, emission colour: rgb: (r,g,b,1)]
def create_new_shader(material_name, rgb):
    mat = create_new_material(material_name)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type = "ShaderNodeOutputMaterial")
    shader = nodes.new(type = "ShaderNodeEmission")
    nodes["Emission"].inputs[0].default_value = rgb
    nodes["Emission"].inputs[1].default_value = 1
    links.new(shader.outputs[0], output.inputs[0]) #Links output of emission shader to input of the material output
    return mat


# Representation of a blender object to be rendered in the scene
class Blobject:
    def __init__(self, name, verts, edges, faces):
        self.name = name
        self.verts = verts
        self.edges = edges
        self.faces = faces  # Faces should only be visible faces
        self.materials = []
        for i in range(0, 1.1, 0.1):
            self.materials.append(create_new_shader(str(i), (i,i,i,1)))


    # upon calling the object like x() where x is of type Blobject, runs this function
    def __call__(self):
        obj_mesh = bpy.data.meshes.new(self.name + "Mesh")
        obj_final = bpy.data.objects.new(self.name, obj_mesh)
        # Verts, edges and faces MUST be corrected
        # Good to remove all useless internal vertices + edges resulting from them
        obj_mesh.from_pydata(self.verts, self.edges, self.faces)
        obj_final.show_name = True
        obj_mesh.update()

        return obj_final  # Returns the final object
        # bpy.context.collection.objects.link(obj_final) is also an option if I dont want to return
        # Returns a final object to be rendered in the scene

    def create_colour_map(self, obj_mesh):
        # Assigns appropriate colours to each face of an object based on forces
        pass


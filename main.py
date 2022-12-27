import bpy
import bmesh
import math



class ApplyForceDialogueOperator(bpy.types.Operator):
    bl_idname = "object.apply_force_dialogue"
    bl_label = "Choose forces to apply"
    
    name = bpy.props.StringProperty(name = "Object Selected:", default = "Cube") #CHANGE DEFAULT LATER
    force = bpy.props.FloatVectorProperty(name = "Force:", default = (1.0,1.0,1.0))
    
    def execute(self,context):
        ForceObject(bpy.context.active_object, self.name, self.force)
        self.report({'INFO'}, "Made Force Object")
        return {'FINISHED'}
    
    def invoke(self,context,event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

"""class ForceFace:
    def __init__(self, force):
        self.force = force"""
        
class VectorTup:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __mul__(self, other): #Other int / float
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
        return False if (not (self.x or self.y or self.z)) else True ##If all numbers are 0 or invalid return false (via demorgans laws)

    def __neg__(self):
        return VectorTup(-self.x, -self.y, -self.z)
    
    def __lt__(self, other): #Comparator definitions
        return self.get_magnitude() < other.get_magnitude()
    
    def __gt__(self, other):
        return self.get_magnitude() > other.get_magnitude()
    
    def __le__(self, other):
        return self.get_magnitude() <= other.get_magnitude()
    
    def __ge__(self, other):
        return self.get_magnitude() >= other.get_magnitude()
    
    def __ne__(self, other): ##Tests for inequality of entire vector, not magnitude inequality
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
    
    

class ForceObject:
    def __init__(self, obj, edges): #Blender Obj, Str, [ForceVertex]
        self.obj = obj #Bound blender object
        self.edges = edges #array of VertPair objects making up force object

    def __add__(self, other): #Adds to list of edges of object
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
        
class ForceVertex:
    def __init__(self, loc, dir):
        self.loc = loc #(x, y, z) tuple
        self.dir = dir #VectorTup object
        
    def get_magnitude(self):
        return self.dir.get_magnitude()
    
class VertPair: # For holding a pair of ForceVertex objects, these pairs indicate a link between two vertices used in the final lattice
    def __init__(self, one, two):
        self.one = one
        self.two = two
        
    def __eq__(self, other):
        return (self.one == other.one and self.two == other.two)
    
    def __ne__(self, other):
        return not(self.one == other.one and self.two == other.two)
    
    def __str__(self):
        return f"({self.one.__str__()}, {self.two.__str__()})"
    
    def __repr__(self):
        return f"VertPair: ({self.one}, {self.two})"
    
#Creates a force object simply using raw vertex and edge data 
def force_obj_from_raw(obj): #Obj is object identifier
    temp_obj = bpy.data.objects[obj]
    temp_dat = temp_obj.data
    vert_num = len(temp_dat.vertices)
    edge_num = len(temp_dat.edges)
    #face_num = len(temp_dat.polygons)
    
    global_verts = [] #Array of ForceVertex objects translated to global coordinates from local
    global_edges = []
    for i in range(vert_num):
        temp_glob = temp_obj.matrix_world @ temp_dat.vertices[i].co #Translation to global coords
        global_verts.append(  ForceVertex(  VectorTup(temp_glob[0], temp_glob[1], temp_glob[2]), VectorTup(0,0,0))  )
    
    for j in range(edge_num):
        edge_verts = temp_dat.edges[j].vertices
        global_edges.append(VertPair(global_verts[edge_verts[0]], global_verts[edge_verts[1]]))
        
    return ForceObject(temp_obj, global_edges)
    


#Conveinient constants
CTX = bpy.context
DAT = bpy.data
OPS = bpy.ops


OPS.object.mode_set(mode = 'OBJECT')
OPS.object.select_all(action = 'SELECT')
OPS.object.delete(use_global = False)
OPS.mesh.primitive_cube_add(size = 2, enter_editmode = False, align = 'WORLD', location = (0,0,0), rotation=(0.2,1.2,1))
#obj = bpy.data.objects["CUBE"]

obj = CTX.active_object
OPS.object.mode_set(mode = 'EDIT')
OPS.mesh.select_mode(type = 'FACE')
OPS.mesh.select_all(action = 'DESELECT')
OPS.object.mode_set(mode = 'OBJECT')
obj.data.polygons[0].select = True
OPS.object.mode_set(mode = 'EDIT')

#def getSelected(obj): #Gets selected faces of object INPUTS: bpy.context.obj object (for current object use bpy.context.active_object)  RETURNS: [Boolean]
#    return [x.select for x in obj.data.polygons]

def getSelected(obj):
        return [x.select for x in (obj).data.polygons]


bpy.utils.register_class(ApplyForceDialogueOperator)
bpy.ops.object.apply_force_dialogue('INVOKE_DEFAULT')

print(getSelected(obj))


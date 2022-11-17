import bpy
import bmesh

bpy.types.Object.custom_float_vector_temp = bpy.props.FloatVectorProperty(name = "Force Application Temp")

bpy.data.objects[0].custom_float_vector_temp = (3.0,1.0,0.0)

class ApplyForceDialogueOperator(bpy.types.Operator):
    bl_idname = "objects.apply_force_dialogue"
    bl_label = "Choose forces to apply"
    
    name = bpy.props.StringProperty(name = "Object Selected:", default = "Cube") #CHANGE DEFAULT LATER
    #forces = [bpy.props.FloatProperty(name = "Force:", default = 1.0) for n in len(getSelected(bpy.context.active_object))]
    force = bpy.props.FloatVectorProperty(name = "Force:", default = (1.0, 0.0, 0.0)) #Default force is 1 along x axis
    
    def execute(self,context):
        createForceObject(force = self.force)
        self.report({'INFO'}, "Added Force Object")
        return {'FINISHED'}
        
    def invoke(self,context,event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

"""class ForceFace:
    def __init__(self, force):
        self.force = force"""

class ForceObject:
    def __init__(self, obj, force):
        self.obj = obj
        self.forces = {} #Dictionary mapping force amounts and faces
        
    def getSelected(self):
        return [x.select for x in (self.obj).data.polygons]
    
    def applyInitialForces(self):
        forceBools = self.getSelected()
        for i in range(len(forceBools)):
            if (forceBools[i]):
                self.forces[i] = 1 #Applies an initial force of 1 to selected surface of object

def createForceObject(force, name = bpy.context.active_object):
    fo = ForceObject(name, force)
    
    

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



print(getSelected(obj))

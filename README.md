# Gabriel Donnan's final year project for Computer Science at The University of Bath

Creating Custom Properties:
`bpy.types.Object.custom_float_vector_temp = bpy.props.FloatVectorProperty(name = "Force Application Temp")`

Set to Obj/Edit mode:
`bpy.ops.object.mode_set(mode = 'OBJECT')`

Select all objects: (Can replace SELECT with DESELECT for deselecting all, works in edit mode too for verts/faces/edges)
`bpy.ops.object.select_all(action = 'SELECT')`

Set edit mode (face/vert/edge):
`bpy.ops.mesh.select_mode(type = 'FACE')`

Get active object:
`bpy.context.active_object`

Flowchart for diss @ https://app.diagrams.net/#G1wQDt6sTcFuspLHn4elXabmbRPKbr9Xtv

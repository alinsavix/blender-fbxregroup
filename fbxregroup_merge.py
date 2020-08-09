#!/usr/bin/env python3
# call via `blender --background --factory-startup --python thisfile.py -- -m <file>.fbx
#

import getopt
import math
from pprint import pprint
import re
import sys
import os

print("argv: %s" % (sys.argv[1:]))
# Sadly, can't define this later on, so it ends up having to sit right
# in the middle of our imports!
#
# ? Should we redirect stdout/stderr before execing blender?
#
# ? Why does VSCode always want to add two blank lines before function?


def execBlender(reason: str):
    blender_bin = "blender"

    from pathlib import Path
    mypath = Path(__file__).resolve()

    print("Not running under blender (%s)" % (reason))
    print("Re-execing myself under blender (blender must exist in path)...")

    blender_args = [
        blender_bin,   # argv[0] -- called program name
        "--background",
        "--factory-startup",
        "--python",
        str(mypath),
        "--",
    ]

    print("executing: %s" % " ".join((blender_args) + sys.argv[1:]))
    try:
        os.execvp(blender_bin, blender_args + sys.argv[1:])
    except OSError as e:
        print("Couldn't exec blender: %s" % (e))
        sys.exit(1)


# Check if we're running under Blender ... and if not, fix that.
# We both have to check to make sure we can import bpy, *and* check
# to make sure there's something meaningful inside that module (like
# an actual context) because there exist 'stub' bpy modules for
# developing outside of blender, that will still import just fine...)
try:
    import bpy
except ImportError as e:
    execBlender("no bpy available")

# It imported ok, so now check to see if we have a context object
if bpy.context is None:
    execBlender("no context available")

# We have to do the above before trying to import other things,
# because some other things might not be installed on system
# python, and the 'utils' module tries to import bpy stuff (which
# might not exist outside of the blender context)
from mathutils import Vector
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import colorize
import utils


# This script will take an fbx file (generally created by the fbxregroup.py
# script, which generates one fbx file per kitbash 'object"), and:
#   1. imports the fbx (duh)
#   2. Sets up a vertex group on each sub-object, named after the sub-object
#   3. Combines all the sub-objects into a single object (the assigned vertex
#      groups can be used to peel them back apart if desired)
#   4. Reset the object's origin to the middle of the object
#   5. Move the object to global (0,0,0)
#   6. Saves a blend file of the results
#
# In the future, this script may also:
#   - Assign basic materials
#   - Create preview renders
#   - Perhaps add kitops metadata
#


#
# Actual code follows
#

# FIXME: wtf is a good name for this? Both 'get' and 'create' are
# not quite right...
def materialGetIndex(obj: bpy.types.Object, matname: str) -> int:
    index = obj.material_slots.find(matname)
    if index != -1:
        # print("found mat: %s (index %d)" % (matname, index))
        return index

    mat = utils.add_material(matname, use_nodes=True,
                             make_node_tree_empty=True)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')

    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    utils.set_principled_node(
        principled_node=principled_node,
        base_color=colorize.colorNext(),
        metallic=0.5,
        specular=0.5,
        roughness=0.1,
    )

    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    obj.data.materials.append(mat)

    # FIXME: Can this fail?
    index = obj.material_slots.find(matname)
    # print("returning new mat %s, index: %d" % (matname, index))
    return index


# Delete an object using a context override
# FIXME: Can this just be `bpy.data.objects.remove()`?
def deleteObj(obj: bpy.types.Object) -> None:
    c = {}
    c["object"] = c["active_object"] = obj

    c["selected_objects"] = [obj]
    c["selected_editable_objects"] = c["selected_objects"]

    x = bpy.ops.object.delete(c, use_global=True, confirm=False)


# Merge some objects into one object using a fake context object
def mergeObjs(active: bpy.types.Object, selected: List[bpy.types.Object]) -> None:
    c = {}
    c["object"] = c["active_object"] = active

    c["selected_objects"] = selected
    c["selected_editable_objects"] = c["selected_objects"]

    x = bpy.ops.object.join(c)


def getObjectBounds(object_name):
    """
    returns the corners of the bounding box of an object in world coordinates
    """
    # bpy.context.scene.update()
    ob = bpy.context.scene.objects[object_name]
    bbox_corners = [ob.matrix_world @
                    Vector(corner) for corner in ob.bound_box]

    # And now compute min/max values for those, which is what we really
    # need for most uses
    bbox_minmax = [
        min([b[0] for b in bbox_corners]),  # x_min
        max([b[0] for b in bbox_corners]),  # x_max
        min([b[1] for b in bbox_corners]),  # y_min
        max([b[1] for b in bbox_corners]),  # y_max
        min([b[2] for b in bbox_corners]),  # z_min
        max([b[2] for b in bbox_corners]),  # z_max
    ]

    return bbox_minmax


# Figure out where the origin of our object should be, which should
# generally be the center of the bottom face of the bounding box.
def getObjectNewOrigin(object_name):
    bb = getObjectBounds(object_name)
    x = (bb[0] + bb[1]) / 2.0
    y = (bb[2] + bb[3]) / 2.0
    z = bb[4]

    return [x, y, z]


# Figure out how big the object bounding box is along the diagonal,
# to help with fitting it on-camera. We only care about X and Y (really
# tall objects go home)
def getDiagonalLength(object_name):
    bbox = getObjectBounds(object_name)

    # PYTHAGORAS TIME!
    # FIXME: There's probably a better way
    a = abs(bbox[1] - bbox[0])
    b = abs(bbox[3] - bbox[2])

    c = math.sqrt(math.pow(a, 2) + math.pow(b, 2))
    return c


def main(argstr):
    input_name = ""

    print(argstr)

    if (("--" in argstr) == False):
        print("Usage: blender --background --python thisfile.py -- -m <file>.fbx")
        return 1

    argsStartPos = argstr.index("--")
    argsStartPos += 1
    myArgs = argstr[argsStartPos:]

    try:
        opts, args = getopt.getopt(myArgs, "hm:", ["help", "model-file="])
    except getOpt.GetoptError:
        print("Options error")
        sys.exit(1)

    for opt, arg in opts:
        if (opt in ("-h", "--help")):
            print("Need help? Read the source.")
            sys.exit(0)
        elif (opt == "-m"):
            input_name = arg

    if (input_name == ""):
        print("No input file(s) given")

    file_re = re.compile("(.*).fbx", re.IGNORECASE)
    m = file_re.match(input_name)
    if not m:
        print("Filename pattern not matched, did you specify an input file?")
        sys.exit(1)
    basename = m.groups(1)

    # print("importing %s.fbx ..." % (basename))
    # bpy.ops.import_scene.fbx(filepath=input_name)

    # print("saving %s.blend ..." % (basename))
    # bpy.ops.wm.save_mainfile(filepath="%s.blend" % (basename))

    # Make sure we've got nothing
    for obj in bpy.context.scene.objects:
        deleteObj(obj)

    # load us up some bits
    print("importing %s.fbx ..." % (basename))
    bpy.ops.import_scene.fbx(filepath=input_name)

    to_merge = []

    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue

        # Apply some transforms (mostly scale)
        # FIXME: Do we need this? Should make sure our scale isn't
        # changing between stages
        obj.select_set(True)
        bpy.ops.object.transform_apply(
            location=False, rotation=True, scale=True)
        obj.select_set(False)

        to_merge.append(obj)

    # Set up vertex groups
    for obj in to_merge:
        vg = obj.vertex_groups.new(name=obj.name)
        verts = [v.index for v in obj.data.vertices]
        vg.add(verts, 0.0, "REPLACE")

    mergeObjs(to_merge[0], to_merge)

    obj = to_merge[0]
    new_origin = getObjectNewOrigin(obj.name)
    bpy.context.scene.cursor.location = Vector(new_origin)

    # FIXME: Figure out the right fields to set to do this with a context
    # override.
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    obj.select_set(False)

    bpy.context.scene.cursor.location = Vector((0.0, 0.0, 0.0))
    obj.location = Vector((0.0, 0.0, 0.0))

    print("saving %s.blend ..." % (basename))
    bpy.ops.wm.save_mainfile(filepath="%s.blend" % (basename))

    # Remember the object name, so we can pull it in later
    obj_name = obj.name

    # Build a scene and render
    # scene_path = "scenes/fbxregroup_render.blend"
    scene_path = "scenes/matcap_library_demo_scene.blend"
    bpy.ops.wm.open_mainfile(
        filepath=scene_path, load_ui=False, use_scripts=False)

    scene = bpy.data.scenes["Scene"]
    world = scene.world

    # hdri_path = "hdris/green_point_park_2k.hdr"
    # hdri_path = "hdris/kloppenheim_03_2k.hdr"
    # hdri_path = "hdris/photo_studio_01_1k.hdr"
    # utils.build_environment_texture_background(world=world, hdri_path=hdri_path,
    #    brightness=1.4)

    # floor_object = utils.create_plane(size=12.0, name="Floor")

    # Link in the thing we just wrote out
    # This strange selection of variables per stackexchange, at
    # https://blender.stackexchange.com/a/38061/98544 . We should check to
    # see if it could potentially be simplified.
    blendfile = "%s.blend" % (basename)
    section = "\\Object\\"
    object = obj_name

    filepath = blendfile + section + object
    directory = blendfile + section
    filename = object

    bpy.ops.wm.append(filepath=filepath, filename=filename,
                      directory=directory)

    # Re-find our object
    obj = bpy.data.objects[obj_name]
    # obj.location = bpy.data.objects["Placeholder"].location
    obj.location = Vector((0.0, 0.0, 0.005))
    obj.rotation_euler = Vector((0.0, 0.0, math.pi / 4))

    # FIXME: Should probably put this in a subroutine
    for p in obj.data.polygons:
        # Get all the vertex groups of all the vertices of this polygon
        verts_vertexGroups = [
            g.group for v in p.vertices for g in obj.data.vertices[v].groups]

        # Find the most frequent (mode) of all vertex groups
        counts = [verts_vertexGroups.count(idx) for idx in verts_vertexGroups]
        modeIndex = counts.index(max(counts))
        mode = verts_vertexGroups[modeIndex]

        groupName = obj.vertex_groups[mode].name

        # Now find the material slot with the same VG's name
        # ms_index = obj.material_slots.find(groupName)
        ms_index = materialGetIndex(obj, groupName)

        # Set material to polygon
        if ms_index != -1:  # material found
            p.material_index = ms_index
        else:
            print("no material for %s (shouldn't happen)" % (groupName))

    # mat = add_material_simple(name="MaterialSimple",
    #                           diffuse_color=(0.8, 0.0, 0.0, 1.0))

    if False:
        mat = utils.add_material("Material", use_nodes=True,
                                 make_node_tree_empty=True)

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        output_node = nodes.new(type='ShaderNodeOutputMaterial')

        principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        utils.set_principled_node(
            principled_node=principled_node,
            base_color=colorize.colorNext(),
            metallic=0.5,
            # specular=0.5,
            roughness=0.1,
        )

        links.new(principled_node.outputs['BSDF'],
                  output_node.inputs['Surface'])

        if obj.data.materials:
            # assign to 1st material slot
            obj.data.materials[0] = mat
        else:
            # no slots
            obj.data.materials.append(mat)

    # print(obj.data.materials.keys())
    # sys.exit(0)
    # utils.create_camera(location=Vector((2.0, 2.0, 3.5)))
    # camera_object = bpy.context.object

    # utils.set_camera_params(camera_object.data, obj)
    # utils.add_track_to_constraint(camera_object, obj)
    camera = bpy.data.objects["Camera"]

    output_file = "%s.png" % (basename)
    utils.set_output_properties(scene=scene, resolution_percentage=100,
                                output_file_path=output_file)

    num_samples = 16
    utils.set_cycles_renderer(scene, camera,
                              num_samples, use_denoising=False)

    # At 85mm, a 0.7944-blender-unit diagonal, you get a 290px wide
    # object (with a 512px wide image). Lets use that as kind of a
    # target.
    #
    # FIXME: Round out these normals
    diag = getDiagonalLength(obj.name)
    target_diag = 0.7944
    diffpct = diag / target_diag
    camera_distance = 5.0

    # FIXME: What is a good min/max for this?
    # camera.data.lens = 85.0 / diffpct
    camera.data.lens = max(85.0 / diffpct, 20.0)
    camera.data.lens = min(camera.data.lens, 500.0)
    print("diagonal length is %f" % (getDiagonalLength(obj.name)))
    print("lens length divisor is %f, final length %f" %
          (diffpct, camera.data.lens))

    # set_camera_params(camera=cam, focus_target_object=obj)
    # camera.data.lens = 85.0
    bpy.ops.render.render(animation=False, write_still=True,
                          use_viewport=False)
    # bpy.ops.object.delete(use_global=False, confirm=True)

    # print("Opening %s.fbx ..." % (basename))
    # bpy.ops.wm.open_mainfile(filepath=input_name, load_ui=False)

    # print("preparing...")
    # scene_objects = load()
    # start_objects = len(scene_objects)
    # print("processing...")
    # cycles = 0

    # print("done (%d --> %d)" % (start_objects, end_objects))
    # bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    sys.exit(main(sys.argv))

# call via `blender --background --python thisfile.py -- -m <file>.fbx
#
# Loads a file with some things. Makes vertex groups. Combines things.
import getopt
import re
import sys
import os
import bpy

from mathutils import Vector
from typing import Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
def create_camera(location: Tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=location)

    return bpy.context.object


def set_camera_params(camera: bpy.types.Camera,
                      focus_target_object: bpy.types.Object,
                      lens: float = 85.0,
                      fstop: float = 1.4) -> None:
    # Simulate Sony's FE 85mm F1.4 GM
    # camera.sensor_fit = 'HORIZONTAL'
    camera.sensor_width = 36.0
    camera.sensor_height = 24.0
    camera.lens = lens
    camera.dof.use_dof = True
    camera.dof.focus_object = focus_target_object
    camera.dof.aperture_fstop = fstop
    camera.dof.aperture_blades = 11


# Merge some objects into one object. Do this by creating a 'fake'
# context object which will then be passed to join()
def deleteObj(obj):
    c = {}
    c["object"] = c["active_object"] = obj

    c["selected_objects"] = [obj]
    c["selected_editable_objects"] = c["selected_objects"]

    x = bpy.ops.object.delete(c, use_global=True, confirm=False)


def mergeObjs(active, selected):
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


def main(argstr):
    input_name = ""

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
        print("No input blend file(s) given")

    file_re = re.compile("(.*).fbx", re.IGNORECASE)
    m = file_re.match(input_name)
    if not m:
        print("Filename pattern not matched, did you specify an fbx file?")
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

    # cam = create_camera(location=Vector((0.0, 0.0, 2.0)))
    utils.create_camera(location=Vector((2.0, 2.0, 5.0)))

    # Current context object is the camera
    camera_object = bpy.context.object

    set_camera_params(camera_object.data, obj)
    utils.add_track_to_constraint(camera_object, obj)

    scene = bpy.data.scenes["Scene"]
    num_samples = 16
    utils.set_output_properties(scene, 100, "test.png")
    utils.set_cycles_renderer(scene, camera_object,
                              num_samples, use_denoising=False)

    # set_camera_params(camera=cam, focus_target_object=obj)

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

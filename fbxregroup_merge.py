# call via `blender --background --python thisfile.py -- -m <file>.fbx
#
# Loads a file with some things. Makes vertex groups. Combines things.
import getopt
import re
import sys
import os
import bpy

from mathutils import Vector


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

    # FIXME: Figure out the right fields to set to do this with a context
    # override.
    to_merge[0].select_set(True)
    bpy.ops.object.origin_set(center='BOUNDS')
    to_merge[0].select_set(False)

    to_merge[0].location = (0, 0, 0)

    print("saving %s.blend ..." % (basename))
    bpy.ops.wm.save_mainfile(filepath="%s.blend" % (basename))

    # bpy.ops.object.delete(use_global=False, confirm=True)

    # print("Opening %s.fbx ..." % (basename))
    # bpy.ops.wm.open_mainfile(filepath=input_name, load_ui=False)

    # print("preparing...")
    # scene_objects = load()
    # start_objects = len(scene_objects)
    # print("processing...")
    # cycles = 0

    # print("done (%d --> %d)" % (start_objects, end_objects))
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    sys.exit(main(sys.argv))

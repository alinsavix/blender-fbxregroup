#!/usr/bin/env python3
# call via `blender --background --factory-startup --python thisfile.py -- -m <file>.fbx
#

import argparse
import math
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


# This script will take an fbx file, presumably from a kitbash set that
# has a lot of components laid out in some kind of arrangement (generally
# a grid), and:
#   1. Import the fbx file
#   2. Apply whatever transforms come in as part of the import
#   3. Figure out which objects in the file have overlapping bounding
#      boxes (because many kitbash kits have 'objects' that are actually
#      made up of multiple objects)
# and
#   4. Write out one fbx file per 'combined' object, containing all of
#      the smaller objects that make up the greater objects.
#
# You can then (once it's coded in) take the resulting fbx files, which
# takes those individual fbx files (each of which should be one 'object',
# though potentially/likely in separate pieces) and:
#   1. imports the fbx (duh)
#   2. Sets up a vertex group on each sub-object, named after the sub-object
#   3. Combines all the sub-objects into a single object (the assigned vertex
#      groups can be used to peel them back apart if desired)
#   4. Reset the object's origin to the middle of the bottom of of the object
#   5. Move the object to global (0,0,0)
#   6. Assigns basic materials, one color per vertex group
#   7. Creates a preview render
#   8. Saves a blend file of the results


#
# Actual code follows
#

def debug(s):
    if False:
        print("DEBUG: " + s)


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


def getObjectBoundsMulti(objs=[]):
    # try to catch pieces that are adjacent but not actually intersecting
    slop = 0.001

    boxes = []
    for o in objs:
        boxes.append(getObjectBounds(o))

    # print("::::".join([str(b[0]) for b in boxes.values()]))
    bbox_combined = [
        min([b[0] for b in boxes]) - slop,
        max([b[1] for b in boxes]) + slop,
        min([b[2] for b in boxes]) - slop,
        max([b[3] for b in boxes]) + slop,
        min([b[4] for b in boxes]) - slop,
        max([b[5] for b in boxes]) + slop,
    ]

    return bbox_combined


def isOverlapping1D(box1min, box1max, box2min, box2max):
    if (box1max >= box2min and box2max >= box1min):
        return True
    else:
        return False


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


def check_Collision(box1, box2):
    """
    Check Collision of 2 Bounding Boxes (using min/max values)
    """

    # If it don't exist, it don't collide
    if box1 is None or box2 is None:
        return False

    # slop = 0.001
    x = isOverlapping1D(
        box1[0], box1[1], box2[0], box2[1]
    )

    y = isOverlapping1D(
        box1[2], box1[3], box2[2], box2[3]
    )

    z = isOverlapping1D(
        box1[4], box1[5], box2[4], box2[5]
    )

    isColliding = (x and y and z)

    # print("x: %s   y: %s   z: %s   final: %s" % (x, y, z, isColliding))
    return isColliding


# find collissions -- recursive
#
# w = object to investigate (name)
# objs = dictionary of objects to check for intersection
# seen = objects we've already processed
# depth = current depth (for debugging purposes)
def fc_recurse(w, objs, seen, depth=0):
    if depth == 0:
        debug("-----")
    debug("recursively checking: %s (depth=%d, seen=%d)" %
          (w, depth, len(seen)))

    # We've definitely seen ourself
    seen.update({w: True})

    # 'seen' changed, so get the latest all-encompassing box
    seen_bbox = getObjectBoundsMulti(seen)

    skip = 0
    count = 0
    match = 0
    for k in list(objs.keys()):
        # Have we already processed the object we want to test?
        #
        if seen.get(k) is not None:
            debug("  Already saw %s, skipping" % (k))
            skip += 1
            continue

        debug("  Haven't seen %s, continuing intersect test" % (k))
        k_bbox = getObjectBoundsMulti([k])

        # Check to see if the new object we're checking intersects
        count += 1
        if check_Collision(seen_bbox, k_bbox):
            # print("intersection: %s + %s" % (w, k))
            match += 1
            debug("  Matched object %s" % (k))
            fc_recurse(k, objs, seen, depth + 1)
        # else:
            # print("no intersection: %s + %s" % (w, k))

    debug("recurse for '%s' result:  depth=%d   c=%d, s=%d, m=%d" %
          (w, depth, count, skip, match))

    if depth == 0:
        debug("-----")

    return


# Merge some objects into one object. Do this by creating a 'fake'
# context object which will then be passed to join()
def merge_obj(active, selected):
    c = {}
    c["object"] = c["active_object"] = bpy.context.scene.objects[active]

    c["selected_objects"] = []
    for s in selected:
        c["selected_objects"].append(bpy.context.scene.objects[s])

    c["selected_editable_objects"] = c["selected_objects"]

    from ppretty import ppretty
    debug("calling join, args: " + ppretty(c))
    x = bpy.ops.object.join(c)
    debug("join called, result: %s" % (x))


# load up our initial starting dict
def load():
    scene_objects = {}

    # Start with nothing selected
    bpy.ops.object.select_all(action='DESELECT')

    # Cycle through what's in the scene, and for each thing do some stuff.
    # This is probably going to be stupidly expensive, but since it's only
    # intended to be run once, it's prooooobably ok. Maybe.
    for obj in bpy.context.scene.objects:
        # print(obj.name, obj, obj.type)
        if obj.type != 'MESH':
            continue

        # Apply some transforms (mostly scale)
        obj.select_set(True)
        bpy.ops.object.transform_apply(
            location=False, rotation=True, scale=True)
        obj.select_set(False)

        # Cache bounding box sizes
        # Can/should we just pass the entire object here?
        # bb = get_BoundBox(obj.name)

        # A group of objects always includes itself
        scene_objects[obj.name] = [obj.name]

    return scene_objects


# File is loaded, process everything. scene_objects is edited in-place
def process(scene_objects):
    start_objects = len(scene_objects)
    merged = 0
    print("processing %d objects..." % (start_objects))

    # Start with the full list of objects in the file. As each are tested,
    # they'll be removed, so our future searches have fewer things to
    # check.
    full = list(scene_objects.keys())
    for k in full:
        # Have we already processed an object (i.e. it's missing now)?
        if scene_objects.get(k, None) is None:
            continue

        print("working: %s..." % (k), end='')

        # Check object k vs. every remaining thing in bboxes, and return in
        # 'seen' a dict of intersecting objects (including itself)
        seen = {}
        fc_recurse(k, scene_objects, seen)

        if len(seen) == 1:
            # Intersect with ourself, and just ourself
            print(" standalone object")
        else:
            # Intersect with other things ... merge 'em!
            print(" merging %d objects (%s)" %
                  (len(seen), " ".join(list(seen.keys()))))
            merged += 1

            for s in seen:
                if s not in k:
                    scene_objects[k].append(s)

                    # Anything part of a join is no longer elgible for merging
                    scene_objects.pop(s, None)

    return merged


# FIXME: should really use something from os.path here instead of a regex
file_re = re.compile("(.*).(fbx|blend)", re.IGNORECASE)

# Load a file, be it blend or fbx (or others, at some point)


def loadfile(filename):
    m = file_re.match(filename)
    if not m:
        print("Filename pattern not matched, did you specify a valid file type?")
        sys.exit(1)

    basename = m.group(1)
    filetype = str(m.group(2)).lower()
    print(filetype)

    print("importing %s ..." % (filename))

    if filetype == "fbx":
        # FIXME: Does this actually have a return code?
        bpy.ops.import_scene.fbx(filepath=filename)

    elif filetype == "blend":
        # FIXME: Again, does this actually have a return code?
        bpy.ops.wm.open_mainfile(
            filepath=filename, load_ui=False, use_scripts=False)
    else:
        print("I accepted filetype %s but don't know how to process it?!" % (filetype))
        sys.exit(1)

    return [basename, filetype]


# Split a file into multiple pieces, based on bounding box overlaps.
# Much of the code here (anything for loading (and possibly saving) for
# example) should really be refactored out.
def cmd_split(args):
    print("cmd_split")

    if len(args.files) == 0:
        print("command 'split' requires a filename argument")
        sys.exit(1)

    # Does the 'utils' library we borrowe give us an easier 'clean' function?
    for obj in bpy.context.scene.objects:
        deleteObj(obj)

    # FIXME: Handle multiple files
    basename, filetype = loadfile(args.files[0])

    print("preparing...")
    scene_objects = load()
    start_objects = len(scene_objects)
    print("processing...")
    cycles = 0

    # Something in my "process" logic is sometimes still leaving things not
    # grouped (I think because the size of the overall bounding box is larger
    # than the individual pieces), so run until everything converges.
    while True:
        print("Starting cycle %d..." % (cycles))
        cycles += 1
        count = process(scene_objects)
        print("cycle %d merged %d objects" % (cycles, count))
        if count == 0:
            break

    # Actually merge things
    for k, v in scene_objects.items():
        # don't actually merge ... write as fbx instead
        # if len(v) > 1:
        #     merge_obj(k, v)

        bpy.ops.object.select_all(action='DESELECT')
        for obj in v:
            bpy.context.scene.objects[obj].select_set(True)

        bpy.ops.export_scene.fbx(
            filepath='%s.fbx' % (k),
            check_existing=False,
            use_selection=True,
            apply_scale_options='FBX_SCALE_NONE',
            use_mesh_modifiers=True,   # Should this be false?
            mesh_smooth_type='OFF',
            add_leaf_bones=True,
            path_mode='AUTO',  # or ABSOLUTE or RELATIVE
        )

    # bpy.ops.wm.save_as_mainfile(filepath="%s_new.blend" %
    #                             (basename), check_existing=False)
    end_objects = len(scene_objects)
    print("done (%d --> %d)" % (start_objects, end_objects))

    # FIXME: should probably handle exiting better
    bpy.ops.wm.quit_blender()


# FIXME: Probably needs refactoring
# FIXME: Probably needs a better name
def cmd_finalize(args):
    print("cmd_finalize")

    if len(args.files) == 0:
        print("command 'finalize' requires filenames to process")
        sys.exit(1)

    # print("importing %s.fbx ..." % (basename))
    # bpy.ops.import_scene.fbx(filepath=input_name)

    # print("saving %s.blend ..." % (basename))
    # bpy.ops.wm.save_mainfile(filepath="%s.blend" % (basename))

    # Make sure we've got nothing
    # FIXME: refactor this and/or use something/add something to utils module
    for obj in bpy.context.scene.objects:
        deleteObj(obj)

    # load us up some bits
    # FIXME: Process more than one file
    basename, filetype = loadfile(args.files[0])

    to_merge = []

    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue

        # Apply some transforms (mostly scale)
        # FIXME: Do we need this?
        # FIXME: Make sure our scale isn't changing between stages
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
    diag = getDiagonalLength(obj.name)
    target_diag = 0.7944
    diffpct = diag / target_diag
    # camera_distance = 5.0

    # Have a min and max lens length so we don't end up -too- far
    # out in the weeds
    # FIXME: What is a good min/max for this?
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


def main(argv):
    input_name = ""

    print(argv)

    # When we get called from blender, the entire blender command line is
    # passed to us as argv. Arguments for us specifically are separated
    # with a double dash, which makes blender stop processing arguments.
    # If there's not a double dash in argv, it means we can't possibly
    # have any arguments, in which case, we should blow up.
    if (("--" in argv) == False):
        print("Usage: blender --background --python thisfile.py -- -m <file>.fbx")
        return 1

    # chop argv down to just our arguments
    args_start = argv.index("--") + 1
    argv = argv[args_start:]

    parser = argparse.ArgumentParser(
        # FIXME: We need to specify this, because we have no meaningful
        # argv[0], but we probably shouldn't just hardcode it
        prog='fbxregroup.py',
        description='Toss some kitbash bits around (and more?)',
    )

    subparsers = parser.add_subparsers(help="sub-command help")
    subparser_split = subparsers.add_parser(
        "split", help="split into subfiles")
    subparser_split.set_defaults(func=cmd_split)

    subparser_split.add_argument(
        "files",
        help="specify files to process",
        metavar="files",
        type=str,  # FIXME: Is there a 'file' type arg?
        nargs="+",
    )

    subparser_finalize = subparsers.add_parser(
        "finalize",
        help="finalize split objects",
    )

    subparser_finalize.set_defaults(func=cmd_finalize)

    subparser_finalize.add_argument(
        "files",
        help="specify files to process",
        metavar="files",
        type=str,  # FIXME: Is there a 'file' type arg?
        nargs="+",
    )

    # parser.add_argument(
    #     "command",
    #     help="specifies what fbxregroup command to execute",
    #     type=str,
    #     nargs="?",
    #     choices=commands.keys(),
    #     const="split",
    #     default="split",
    # )

    # parser.add_argument(
    #     "--modelfile", "-m", action="store", type=str, help="what model file to operate on"
    # )

    args = parser.parse_args(argv)
    print(args)

    args.func(args)

    sys.exit(0)

    if args.command == "split":
        print("split mode")
        sys.exit(0)

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


if __name__ == "__main__":
    sys.exit(main(sys.argv))

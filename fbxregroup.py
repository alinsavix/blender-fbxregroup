# call via `blender --background --python thisfile.py -- -m <file>.fbx
import getopt
import re
import sys
import os
import bpy

from mathutils import Vector

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
# If desired, these results can then be passed to fbxregroup_merge.py
# for more magic.

#
# Actual code follows
#


def debug(s):
    if False:
        print("DEBUG: " + s)


def deleteObj(obj):
    c = {}
    c["object"] = c["active_object"] = obj

    c["selected_objects"] = [obj]
    c["selected_editable_objects"] = c["selected_objects"]

    x = bpy.ops.object.delete(c, use_global=True, confirm=False)


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


# FIXME: cache the bounds? Make take boxes instead of names? Not sure.
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
    # x = check_Collision(bboxes.get("Hard_Surface_Mesh_2887"),
    #                     bboxes.get("Hard_Surface_Mesh_2897"))
    # print(x)
    # sys.exit(0)

    start_objects = len(scene_objects)
    merged = 0
    print("processing %d objects..." % (start_objects))

    # x = check_Collision(bboxes.get("Hard_Surface_Mesh_3095"),
    #                     bboxes.get("Hard_Surface_Mesh_3096"))
    # print(x)

    # x = check_Collision(bboxes.get("Hard_Surface_Mesh_3096"),
    #                     bboxes.get("Hard_Surface_Mesh_3095"))
    # print(x)

    # sys.exit(0)

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

    for obj in bpy.context.scene.objects:
        deleteObj(obj)

    print("importing %s.fbx ..." % (basename))
    bpy.ops.import_scene.fbx(filepath=input_name)

    # print("saving %s.blend ..." % (basename))
    # bpy.ops.wm.save_mainfile(filepath="%s.blend" % (basename))

    # print("Opening %s.blend ..." % (basename))
    # bpy.ops.wm.open_mainfile(filepath=input_name, load_ui=False)

    print("preparing...")
    scene_objects = load()
    start_objects = len(scene_objects)
    print("processing...")
    cycles = 0

    # Something in my logic is sometimes still leaving things not grouped
    # (I think because the size of the overall bounding box is larger than
    # the individual pieces), so run until everything converges.
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
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    # sys.stdin.close()
    # try:
    sys.exit(main(sys.argv))
    # sys.exit(
    #     main(
    #         ["--", "-m", "/Users/jg378k/Documents/Item_combiner_fbx_test.blend"]
    #     )
    # )
    # except Exception as e:
    #     print(e)

#!/usr/bin/env python3
# call via `blender --background --factory-startup --python thisfile.py -- -m <file>.fbx
#

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile

from typing import Dict, List, Tuple

_debug = 0
# temp things that won't work if directly executed, just so we get the types
# from mathutils import Vector, kdtree
# from typing import List, Tuple
# end temp things

print("argv: %s" % (sys.argv[1:]))
# Sadly, can't define this later on, so it ends up having to sit right
# in the middle of our imports!
#
# ? Should we redirect stdout/stderr before execing blender?
def execBlender(reason: str):
    blender_bin = "blender"

    from pathlib import Path
    mypath = Path(__file__).resolve()

    print("Not running under blender (%s)" % (reason))
    print("Re-execing myself under blender (blender must exist in path)...")

    # Or could use e.g:
    # import subprocess
    # subprocess.Popen([bpy.app.binary_path, '-b', path, '--python', os.path.join(addon.path(), 'addon', 'utility', 'save.py')])

    blender_args = [
        blender_bin,
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

from bpy.props import StringProperty
# We have to do the above before trying to import other things,
# because some other things might not be installed on system
# python, and the 'utils' module tries to import bpy stuff (which
# might not exist outside of the blender context)
from mathutils import Vector, kdtree

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.colorize import ColorSequence
from utils.materials import add_material, set_principled_node
from utils import render
from utils.utils import clean_objects
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
    if _debug > 0:
        print("DEBUG: " + s)


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd, cwd=None):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(
            ['git', 'describe', '--always', '--dirty'], os.path.dirname(__file__))
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "unknown"

    return GIT_REVISION


# FIXME: Is there a better way than this global?
colorseq = ColorSequence()

# FIXME: wtf is a good name for this? Both 'get' and 'create' are
# not quite right...
def materialGetIndex(obj: bpy.types.Object, matname: str) -> int:
    index = obj.material_slots.find(matname)
    if index != -1:
        # print("found mat: %s (index %d)" % (matname, index))
        return index

    mat = add_material(matname, use_nodes=True, make_node_tree_empty=True)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')

    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_principled_node(
        principled_node=principled_node,
        base_color=colorseq.next(),
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


def getObjectBoundsMulti(objnames: List[str] = [], slop: float = 0.0):
    boxes = []
    for o in objnames:
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
#
# FIXME: Make this use actual objects instead of names
def getObjectNewOrigin(object_name):
    bb = getObjectBounds(object_name)
    x = (bb[0] + bb[1]) / 2.0
    y = (bb[2] + bb[3]) / 2.0
    z = bb[4]

    return [x, y, z]


# And the same, but for multiple objects
def getObjectNewOriginMulti(objnames: List[str] = []) -> Tuple[float, float, float]:
    bb = getObjectBoundsMulti(objnames, slop=0.0)
    x = (bb[0] + bb[1]) / 2.0
    y = (bb[2] + bb[3]) / 2.0
    z = bb[4]

    return (x, y, z)


def setObjOrigin(obj: bpy.types.Object, origin: Vector):
    c = bpy.context.copy()
    c["active_object"] = obj

    # AFAIK the 3D cursor isn't part of the context
    old_origin = bpy.context.scene.cursor.location
    bpy.context.scene.cursor.location = origin
    bpy.ops.object.origin_set(c, type='ORIGIN_CURSOR')
    bpy.context.scene.cursor.location = old_origin


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


### START OF CODE WE STOLE FROM ASHER ###
def find_shortest_distance(obj_a, obj_b):
    """
    Given two mesh objects and an iteration limit,
    this function finds and refines a shortest-distance line
    between the objects' evaluated geometry.
    """

    if not ((obj_a.type == 'MESH') and (obj_b.type == 'MESH')):
        print("ERROR: find_shortest_distance called w/o two mesh objects")
        return False, obj_a.location, obj_b.location

    # Location of obj_b in obj_a's local space
    b_loc = obj_a.matrix_world.inverted() @ obj_b.location

    # Get closest point on obj_a, in obj_a's local space
    success_a, point_a, normal_a, index_a = obj_a.closest_point_on_mesh(b_loc)
    # Convert point_a to world space
    point_a = obj_a.matrix_world @ point_a
    point_a_out = point_a
    # convert point_a to obj_b's local space
    point_a = obj_b.matrix_world.inverted() @ point_a

    # Get closest point on obj_b, in obj_b's local space
    success_b, point_b, normal_b, index_b = obj_b.closest_point_on_mesh(
        point_a)
    # Convert point_b to world space
    point_b = obj_b.matrix_world @ point_b
    point_b_out = point_b
    # convert point_b to obj_a's local space
    point_b = obj_a.matrix_world.inverted() @ point_b

    success = (success_a and success_b)

    return success, point_a_out, point_b_out


def calc_distance(vec_a, vec_b):
    """
    Returns the Euclidian distance between two points in 3D Space.\n

    How is there not a built-in function for this?
    """
    return (vec_b - vec_a).magnitude


def refine_tree(targetobj, tree, allobjs, search_radius, distance, refine=True):
    """
    Takes in a KD Tree of objects and performs a two-step distance filter. \n

    First it gets the objects with origins within (float) radius distance of (object) obj,
    then it (optionally) refines that list to those with geometry that is within (float) dist distance of obj's geometry.
    """

    if targetobj.type != 'MESH':
        print("refine_tree: WARNING object is {targetobj.type}, not MESH")
        return []

    # Lazy bounding box center, so objects with offset origins don't produce weird results
    vec_sum = Vector((0.0, 0.0, 0.0))
    for vec in targetobj.bound_box:
        vec_sum += Vector(vec)

    bbox_center = targetobj.location + (vec_sum / 8.0)

    # The tree only contains vectors and indicies.
    # This gives us a list of objects within the search radius.
    objs = []
    for (co, index, rad) in tree.find_range(bbox_center, search_radius):
        ob = allobjs[index]
        if (ob.type == 'MESH') and not (ob == targetobj):
            objs.append(ob)

    if not refine:
        print("SKIP")
        return objs

    filtered_objs = []

    for ob in objs:
        if ob == targetobj:
            continue

        # For some reason (we should maybe figure out why, sometime) the
        # results of find_shortest_distance depends on ordering. Run it
        # both ways and pick the shortest distance, with the assumption
        # that one or the other is at least close to correct
        flag, point_a1, point_b1 = find_shortest_distance(targetobj, ob)
        flag, point_a2, point_b2 = find_shortest_distance(ob, targetobj)

        if not flag:
            print("refine_tree: ERROR from find_shortest_distance")
            continue

        raw_dist = min(calc_distance(point_a1, point_b1),
                       calc_distance(point_a2, point_b2))
        converted_dist = bpy.utils.units.to_string(
            'METRIC', 'LENGTH', raw_dist)

        disposition = "discard"
        if raw_dist <= distance:
            disposition = "keep"
            filtered_objs.append(ob)

        debug(
            f"  {ob.name} is {converted_dist} from {targetobj.name} ({disposition})")

        if raw_dist <= distance:
            filtered_objs.append(ob)

    return filtered_objs

### END OF CODE WE STOLE FROM ASHER ###


def kdFindNearby(obj, tree, allobjs, distance, search_radius=1):
    raw_bounds = obj.bound_box

    bounds = []
    for point in raw_bounds:
        bounds.append(Vector(point).magnitude)

    # FIXME: can we get rid of the magic numbers? What *are* these numbers?
    radius = max(bounds) + search_radius

    objs = refine_tree(obj, tree, allobjs, radius, distance)
    return objs


def kdCreate(objs):
    tree = kdtree.KDTree(len(objs))

    for i, obj in enumerate(objs):
        tree.insert(obj.location, i)

    tree.balance()
    return tree


# find collissions -- recursive
#
# w = object to investigate
# kdtree = kdtree of all objects in scene
# objs = dictionary of objects to check for intersection
# seen = objects we've already processed
# depth = current depth (for debugging purposes)
#
# This should only beb called for objects that aren't in "seen"
def fc_recurse(w, tree, allobjs, distance, processed, seen, depth=0):
    # We've definitely seen ourself
    seen.update({w.name: True})

    if w.type != 'MESH':
        debug(f"fc_recurse not checking {w.name}, type {w.type} != 'MESH'")
        return

    if depth == 0:
        debug("-----")

    debug(f"recursively checking: {w.name} (depth={depth}, seen={len(seen)})")

    # 'seen' changed, so get the latest all-encompassing box
    # seen_bbox = getObjectBoundsMulti(seen)

    nearby = kdFindNearby(w, tree, allobjs, distance)
    debug(f"  Found {len(nearby)} nearby matches")

    skip = 0
    match = 0

    for ob in nearby:
        if seen.get(ob.name) is not None:
            debug(f"  Already saw {ob.name}, skipping")
            skip += 1
            continue

        if processed.get(ob.name) is not None:
            debug(f"  Already processed {ob.name}, skipping")
            skip += 1
            continue

        match += 1
        debug(f"  Matched {ob.name}, recursing")
        fc_recurse(ob, tree, allobjs, distance, processed, seen, depth)

    debug(
        f"recurse for '{w.name}' result:  depth={depth}   s={skip}, m={match}")

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


# Given an object, merge its children, after setting up vertex groups for
# each, so we don't lose their original identity.
#
# FIXME: Should standalone objects have vertex groups created?
def mergeChildren(obj: bpy.types.Object):
    name = obj.name
    if obj.display_type == 'WIRE':
        all = list(obj.children)
        deleteObj(obj)
    else:
        all = [obj] + list(obj.children)

    for o in all:
        vg = o.vertex_groups.new(name=o.name)
        verts = [v.index for v in o.data.vertices]
        vg.add(verts, 0.0, "REPLACE")

    source_objects = ",".join([x.name for x in all])

    combined_obj = all[0]
    mergeObjs(combined_obj, all)
    combined_obj.name = name
    new_origin = getObjectNewOrigin(name)
    setObjOrigin(combined_obj, Vector(new_origin))

    # it would be better if this was an actual list rather than just a CSV,
    # but blender makes that really hard, so we just get a CSV. This is mostly
    # intended for humans, so that's probably ok.
    combined_obj.fbxregroup.origin_merged = source_objects
    # print(f"{combined_obj.fbxregroup.origin_file}")

    return combined_obj


# make a plane that sits under our object
def createBasePlane(objname: str, location, xdim: float, ydim: float) -> bpy.types.Object:
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_plane_add(
        size=1, location=location, calc_uvs=False, enter_editmode=False)
    bpy.ops.transform.resize(value=[xdim, ydim, 1.0])
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    obj = bpy.context.object
    bpy.ops.object.select_all(action='DESELECT')

    obj.hide_render = True
    obj.display_type = 'WIRE'
    obj.display.show_shadows = False

    obj.name = objname

    return obj


# go through and make sure our scene is ready for processing
def sceneprep():
    # scene_objects = {}
    scene_objects = []

    # Make sure we're in a known/consistent mode (i.e. object mode)
    if bpy.context.active_object is not None:
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

    # Start with nothing selected
    bpy.ops.object.select_all(action='DESELECT')

    # Cycle through what's in the scene, and for each thing do some stuff.
    # This is probably going to be stupidly expensive, but since it's only
    # intended to be run once, it's prooooobably ok. Maybe.
    for obj in bpy.context.scene.objects:
        # print(obj.name, obj, obj.type)
        if obj.type not in ['MESH', 'VOLUME', 'ARMATURE', 'EMPTY', 'LIGHT']:
            # if obj.type not in ['MESH']:
            debug(
                f"load discarded '{obj.name}' (improper object type {obj.type}")
            continue

        # Apply some transforms (mostly we care about scalescale). Some
        # object types aren't supported ("no data to transform"), but I
        # doun't know the entire list. Certainly light types.
        if obj.type not in ['LIGHT']:
            obj.select_set(True)
            bpy.ops.object.transform_apply(
                location=False, rotation=True, scale=True)
            # Set the origin, just since some of them are really wacky when we
            # import them.
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
            obj.select_set(False)

        scene_objects.append(obj)

    return scene_objects


# File is loaded, process everything. scene_objects is edited in-place
def findclusters(objects):
    start_count = len(objects)
    clustered = 0
    clusterobjs = {}
    print(f"finding clusters for {start_count} objects...")

    # ? Do we want to just create the kdtree once, and filter out things
    # ? we've already seen, or do we want to recreate it each time to remove
    # ? what we've already seen?
    tree = kdCreate(objects)

    processed = {}
    for obj in objects:
        # Have we already processed an object?
        if obj.name in processed:
            debug(f"already processed: {obj.name}")
            continue

        debug(f"proximity testing: {obj.name}...")

        # 'seen' (for lack of a better term) is objects we've already touched
        # for this particular top-level object
        seen = {}
        fc_recurse(obj, tree, objects, 0.025, processed, seen)

        clusterobjs[obj.name] = seen
        if len(seen) == 1:
            debug("...standalone object")
        else:
            # need to merge these, so mark 'em as such
            debug("...merging %d objects (%s)" %
                  (len(seen), " ".join(list(seen.keys()))))
            clustered += 1

        for k in seen.keys():
            processed.update({k: True})

    # Try to have a deterministic way to name things -- in this case, just
    # take the object in a group that sorts first alphabetically, and use
    # that as the main object name.
    clusterobjs_sorted = {}
    for k in clusterobjs:
        ordered = sorted(list(clusterobjs[k].keys()))
        clusterobjs_sorted.update({ordered[0]: ordered})

    return clusterobjs_sorted


# FIXME: should really use something from os.path here instead of a regex
file_re = re.compile("(.*).(fbx|blend)", re.IGNORECASE)

# Load a file, be it blend or fbx (or others, at some point)
def loadfile(filename):
    m = file_re.match(filename)
    if not m:
        raise ValueError("Bad filename specified")

    basename = m.group(1)
    filetype = str(m.group(2)).lower()

    print("importing %s ..." % (filename))

    # FIXME: Figure out if these have return codes, or throw exceptions,
    # or ... what.
    if filetype == "fbx":
        bpy.ops.import_scene.fbx(filepath=filename)

    elif filetype == "blend":
        bpy.ops.wm.open_mainfile(
            filepath=filename, load_ui=False, use_scripts=False)
    else:
        # FIXME: What kind of exception is actually right here?
        raise Exception(f"accepted {filetype} file, but can't process it?")

    return [basename, filetype]


def cleanload(filename: str) -> (str, str):
    clean_objects()
    basename, filetype = loadfile(filename)
    return(basename, filetype)


def writefbx_batch(clusters: Dict[str, List[str]], filepattern: str = "%s.fbx") -> None:
    for k, v in clusters.items():
        bpy.ops.object.select_all(action='DESELECT')

        for o in v:
            bpy.data.objects[o].select_set(True)

        bpy.context.view_layer.objects.active = bpy.data.objects[v[0]]

        # ? Should we use obj instead of fbx?
        bpy.ops.export_scene.fbx(
            filepath=filepattern % (k),
            check_existing=False,
            use_selection=True,
            apply_scale_options='FBX_SCALE_NONE',
            use_mesh_modifiers=True,   # ? Should this be optional?
            use_subsurf=True,
            mesh_smooth_type='OFF',
            use_custom_props=True,   # ? Should this befalse?
            add_leaf_bones=True,
            path_mode='COPY',  # or ABSOLUTE or RELATIVE
            embed_textures=True,    # embed textures
            # axis_forward=?,    # ? Should we set this?
            # axis_up=?,  # ? Should we set this?
            use_metadata=True,   # FIXME: What does this actually add?
        )
        bpy.ops.object.select_all(action='DESELECT')


class FbxRegroupSourceInfo(bpy.types.PropertyGroup):
    origin_file: StringProperty(
        name="Kitbash Object Origin Filename",
        description="The name of the file that originally contained this object",
        subtype='FILE_NAME',   # FIXME: Do we really need this set?
    )

    origin_merged: StringProperty(
        name="Kitbash Merged Object Origin Object Names",
        description="Comma-separated names of the objects originally merged to make this object",
    )

    origin_object: StringProperty(
        name="Kitbash Object Origin Original Name",
        description="Name of original parent object that created this object",
    )

    cmd_version: StringProperty(
        name="fbxregroup Version",
        description="The fbxregroup version that created this object",
    )


# Split a file into multiple pieces, based on bounding box overlaps.
# Much of the code here (anything for loading (and possibly saving) for
# example) should really be refactored out.
def cmd_split(args):
    # print("cmd_split")

    # if len(args.files) == 0:
    #     print("command 'split' requires a filename argument")
    #     sys.exit(1)

    # clean_objects()

    # # FIXME: Handle multiple files
    # basename, filetype = loadfile(args.files[0])
    basename, filetype = cleanload(args.files[0])

    print("preparing...")
    scene_objects = sceneprep()
    start_objects = len(scene_objects)

    print("processing...")
    clusters = findclusters(scene_objects)
    # bpy.ops.preferences.addon_enable(module="kitops")

    # Actually group and write things
    writefbx_batch(clusters, "%s.fbx")

    # bpy.ops.ko.create_insert()
    # sys.modules['kitops.addon.utility.smart'].save_insert(
    #     path=f"{k}.blend", objects=bpy.context.selected_objects,
    # )
    # sys.modules['kitops.addon.utility.smart'].create_insert()
    # sys.modules['kitops.addon.utility.smart'].save_insert(
    #     path=f"{k}.blend", objects=bpy.context.selected_objects,
    # )
    # bpy.ops.ko.save_insert()
    # bpy.ops.ko.create_insert('INVOKE_REGION_WIN')
    # bpy.context.window_manager.kitops.insert_name = f"fbxrg_{k}"
    # bpy.ops.ko.save_as_insert('INVOKE_REGION_WIN', check_existing=False)
    # bpy.ops.ko.close_factory_scene()

    # bpy.ops.wm.save_as_mainfile(filepath="%s_new.blend" %
    #                             (basename), check_existing=False)
    end_objects = len(clusters)
    print("done (%d --> %d)" % (start_objects, end_objects))

    return


# Do something similar to cmd_split, except write out a blend file in
# a format that kitops-batch can process correctly into inserts. (basically,
# for things that are more than one obbject, make a wireframe plane as the
# 'base' and parent the rest of the cluster to it)
#
# Though as of at least 2020-12-04, kitops batch fucks it up... sigh.
def cmd_kitops(args):
    # FIXME: Handle multiple files
    basename, filetype = cleanload(args.files[0])

    # FIXME: Should we pass this in, instead of makingn this call twice?
    # (The other time is during startup)
    fbxregroup_version = git_version()

    print("preparing...")
    scene_objects = sceneprep()
    start_objects = len(scene_objects)

    print("processing...")
    for obj in scene_objects:
        # print(f"origin object: {obj.fbxregroup.origin_object}")
        if obj.fbxregroup.origin_object == "":
            obj.fbxregroup.origin_object = obj.name
        if obj.fbxregroup.origin_file == "":
            obj.fbxregroup.origin_file = os.path.basename(args.files[0])
        obj.fbxregroup.cmd_version = fbxregroup_version

    clusters = findclusters(scene_objects)

    for k, v in clusters.items():
        if len(v) == 1:
            continue

        new_origin = getObjectNewOriginMulti(v)
        dims = getObjectBoundsMulti(v)
        xdim = (dims[1] - dims[0]) * 1.1
        ydim = (dims[3] - dims[2]) * 1.1
        base = createBasePlane(f"{k}_base", new_origin, xdim, ydim)

        for n in v:
            obj = bpy.data.objects[n]
            obj.name = "obj_" + obj.name
            obj.parent = base
            obj.matrix_parent_inverse = base.matrix_world.inverted()

        base.name = k

    print("saving %s.blend ..." % (basename))
    # bpy.context.view_layer.update()

    bpy.ops.wm.save_mainfile(
        filepath=f"kitops_{basename}.blend", compress=True)

    #     getObjectBoundsMulti(v, slop=0)
    #     bpy.ops.object.select_all(action='DESELECT')

    #     for o in v:
    #         bpy.data.objects[o].select_set(True)

    #     bpy.context.view_layer.objects.active = bpy.data.objects[v[0]]

    # # Actually group and write things
    # writefbx_batch(clusters, "%s.fbx")

    # bpy.ops.ko.create_insert()
    # sys.modules['kitops.addon.utility.smart'].save_insert(
    #     path=f"{k}.blend", objects=bpy.context.selected_objects,
    # )
    # sys.modules['kitops.addon.utility.smart'].create_insert()
    # sys.modules['kitops.addon.utility.smart'].save_insert(
    #     path=f"{k}.blend", objects=bpy.context.selected_objects,
    # )
    # bpy.ops.ko.save_insert()
    # bpy.ops.ko.create_insert('INVOKE_REGION_WIN')
    # bpy.context.window_manager.kitops.insert_name = f"fbxrg_{k}"
    # bpy.ops.ko.save_as_insert('INVOKE_REGION_WIN', check_existing=False)
    # bpy.ops.ko.close_factory_scene()

    # bpy.ops.wm.save_as_mainfile(filepath="%s_new.blend" %
    #                             (basename), check_existing=False)
    end_objects = len(clusters)
    print("done (%d --> %d)" % (start_objects, end_objects))

    return


# Probably need a better name, or something
#
# Take a blend file (probably created from cmd_kitops) and merge the mergabble
# items, adding vertex groups for the original components, since kitops can't
# deal with multiple objects sanely.
#
# Every object without a parent gets merged. Objects without a parent have
# their children merged, if any. Objects that are wireframes are removed and
# have their children merged.
def cmd_kitops_merge(args):
    # FIXME: Handle multiple files
    basename, filetype = cleanload(args.files[0])

    # FIXME: Should we pass this in, instead of makingn this call twice?
    # (The other time is during startup)
    fbxregroup_version = git_version()

    print("preparing...")
    scene_objects = sceneprep()
    start_objects = len(scene_objects)

    merge_parents = []
    for obj in scene_objects:
        if obj.type != 'MESH':
            debug(f"skipping non-mesh {obj.name}")
            continue

        if obj.parent is not None:
            debug(f"object {obj.name} has parent {obj.parent.name}, skipping")
            continue

        if len(obj.children) > 0:
            merge_parents.append(obj)

        if obj.fbxregroup.origin_object == "":
            obj.fbxregroup.origin_object = obj.name
        if obj.fbxregroup.origin_file == "":
            obj.fbxregroup.origin_file = os.path.basename(args.files[0])
        obj.fbxregroup.cmd_version = fbxregroup_version
        # ideally transforms are already applied, but may as well make sure
        # bpy.ops.object.select_all(action='DESELECT')
        # obj.select_set(True)
        # bpy.ops.object.transform_apply(
        #     location=False, rotation=True, scale=True)
        # obj.select_set(False)

    for obj in merge_parents:
        mergeChildren(obj)

    # Everything should be merged, just save
    print(f"saving merged_{basename}.blend ...")
    bpy.ops.wm.save_mainfile(
        filepath=f"merged_{basename}.blend", compress=True)

    # FIXME: this is probably wrong
    # FIXME: wrong thing to look atf
    end_objects = len(bpy.context.scene.objects)
    print(f"done ({start_objects} --> {end_objects})")

    return


# FIXME: Needs better name?
def cmd_kitops_batch(args):
    # FIXME: Handle multiple files
    # print(args.outdir)
    basename, filetype = cleanload(args.files[0])

    # FIXME: Should we pass this in, instead of makingn this call twice?
    # (The other time is during startup)
    fbxregroup_version = git_version()

    # Don't actually need prep -- just use what's there
    # print("preparing...")
    # scene_objects = sceneprep()
    # start_objects = len(scene_objects)

    # make sure our textures are bundled, save into the output
    # directory, and then do the export. Wish these things could
    # be provided directly to kitops, but it doesn't like to play
    # well with others.
    #
    # ? Do we want to be able to bypass this?
    with tempfile.NamedTemporaryFile(dir=args.outdir, prefix="fbxregroup_",
                                     suffix=".blend", delete=False) as f:
        workfile = f.name

    bpy.ops.file.pack_all()

    print(f"fbxregroup: saving to temporary working file {workfile}")
    versions = bpy.context.preferences.filepaths.save_version
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_mainfile(filepath=workfile, compress=True)
    bpy.context.preferences.filepaths.save_version = versions

    bpy.ops.preferences.addon_enable(module="kitops")
    bpy.ops.preferences.addon_enable(module="kitops-batch")
    bpy.ops.kob.batch_export_blend()

    print(f"fbxregroup: removing temporary file {workfile}")
    # ?! This may happen before the batch export happens -- doublecheck
    os.remove(workfile)

    return


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
        # FIXME: What other types should we ignore?
        if obj.type in ['CAMERA']:
            continue

        # FIXME: Make sure our scale isn't changing between stages
        if obj.type not in ['LIGHT']:
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
        mat = add_material("Material", use_nodes=True,
                           make_node_tree_empty=True)

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        output_node = nodes.new(type='ShaderNodeOutputMaterial')

        principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        set_principled_node(
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

    # The blender output path seems to need to bbe absolute
    # FIXME: This will probably break if the user uses an absolute path
    output_file = os.path.join(os.path.realpath("."), f"{basename}.png")
    render.set_output_properties(scene=scene, resolution_percentage=100,
                                 output_file_path=output_file)

    num_samples = 16
    render.set_cycles_renderer(scene, camera,
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


def writable_dir(d: str) -> str:
    if not os.path.isdir(d) or not os.access(d, os.W_OK):
        raise argparse.ArgumentError(f"'{d}' is not a writable directory")

    return d


def main(argv):
    input_name = ""
    print("fbxregroup version: %s" % (git_version()))

    print(argv)

    # When we get called from blender, the entire blender command line is
    # passed to us as argv. Arguments for us specifically are separated
    # with a double dash, which makes blender stop processing arguments.
    # If there's not a double dash in argv, it means we can't possibly
    # have any arguments, in which case, we should blow up.
    if (("--" in argv) == False):
        print("Usage: blender --background --python thisfile.py -- <file>.fbx")
        return 1

    # Set our custom props so everything has them
    bpy.utils.register_class(FbxRegroupSourceInfo)
    bpy.types.Object.fbxregroup = bpy.props.PointerProperty(
        type=FbxRegroupSourceInfo)

    # chop argv down to just our arguments
    args_start = argv.index("--") + 1
    argv = argv[args_start:]

    parser = argparse.ArgumentParser(
        # FIXME: We need to specify this, because we have no meaningful
        # argv[0], but we probably shouldn't just hardcode it
        prog='fbxregroup.py',
        description='Toss some kitbash bits around (and more?)',
    )

    parser.add_argument("--debug", "-d", action="count", default=0)

    subparsers = parser.add_subparsers(help="sub-command help")

    ## SPLIT ##
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

    ## KITOPS ##
    subparser_kitops = subparsers.add_parser(
        "kitops",
        help="file prepared for kitops batch",
    )

    subparser_kitops.set_defaults(func=cmd_kitops)

    subparser_kitops.add_argument(
        "files",
        help="specify files to process",
        metavar="files",
        type=str,  # FIXME: Is there a 'file' type arg?
        nargs="+",
    )

    ## FINALIZE ##
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

    ## KITOPS MERGE ##
    subparser_kitops_merge = subparsers.add_parser(
        "kitops-merge",
        help="merge kitbash clusters to importable objects",
    )

    subparser_kitops_merge.set_defaults(func=cmd_kitops_merge)

    subparser_kitops_merge.add_argument(
        "files",
        help="specify files to process",
        metavar="files",
        type=str,  # FIXME: Is there a 'file' type arg?
        nargs="+",
    )

    ## KITOPS BATCH ##
    subparser_kitops_batch = subparsers.add_parser(
        "kitops-batch",
        help="batch export prepared file to inserts",
    )

    subparser_kitops_batch.set_defaults(func=cmd_kitops_batch)

    subparser_kitops_batch.add_argument(
        "files",
        help="specify files to process",
        metavar="files",
        type=str,  # FIXME: Is there a 'file' type arg?
        nargs="+",
    )

    subparser_kitops_batch.add_argument(
        "--outdir", "-o",
        help="output directory for kitops inserts",
        metavar="outdir",
        type=writable_dir,
        default=".",
        nargs='?',
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

    global _debug
    _debug = args.debug

    args.func(args)

    bpy.ops.wm.quit_blender()
    sys.exit(0)  # Shouldn't be reached

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

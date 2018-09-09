def load(f, existing_mesh=None, with_quads=False):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _load, mode='rb', existing_mesh=existing_mesh)

def _load(f, existing_mesh=None, with_quads=False):
    import os
    from lace.serialization import ply, obj, wrl, stl, xyz, bsf
    ext = os.path.splitext(f.name)[1].lower()
    if ext == ".obj":
        return obj.load(f, existing_mesh=existing_mesh, with_quads=with_quads)
    if with_quads:
        raise ValueError('Only the OBJ loader supports with_quads=True')
    if ext == ".ply":
        return ply.load(f, existing_mesh=existing_mesh)
    elif ext == ".wrl":
        return wrl.load(f, existing_mesh=existing_mesh)
    elif ext == ".ylp":
        return ply.load_encrypted(f, existing_mesh=existing_mesh)
    elif ext == ".stl":
        return stl.load(f, existing_mesh=existing_mesh)
    elif ext == ".xyz":
        return xyz.load(f, existing_mesh=existing_mesh)
    elif ext == ".bsf":
        return bsf.load(f, existing_mesh=existing_mesh)
    else:
        raise NotImplementedError("Unknown mesh file format: %s" % f.name)

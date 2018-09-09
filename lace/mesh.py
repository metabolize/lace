# NB: setter_property incorrectly trigger's pylint's method-hidden error
# pylint: disable=method-hidden, unused-argument, redefined-outer-name
import numpy as np
from blmath.numerics import as_numeric_array
from blmath.util.decorators import setter_property

from lace import serialization
from lace import color
from lace import geometry
from lace import landmarks
from lace import topology
from lace import visibility
from lace import texture
from lace import search
from lace import visualization

class Mesh(
        serialization.MeshMixin,
        color.MeshMixin,
        geometry.MeshMixin,
        landmarks.MeshMixin,
        topology.MeshMixin,
        visibility.MeshMixin,
        texture.MeshMixin,
        search.MeshMixin,
        visualization.MeshMixin,
        object
    ):
    """3d Triangulated Mesh class

    Attributes:
        v: Vx3 array of vertices
        f4: Fx4 array of vert indices for each quad face (with_quads=True)
        f: Fx3 array of vert indices for each tri face (with_quads=False)

    Optional attributes:
        vn: VNx3 array of vertex normals
        vt: VTx3 array of texture vertices
        vc: Vx3 array of each vertex's color
        segm: dictionary of part names to face indices (f or f4)
        e: Ex2 array of edges

    Optional attributes (with_quads=True):
        fc4: Fx4 array of colors for each face
        fn4: Fx4 array of vertex normal indices for each face
        ft4: Fx4 array of texture vertex indices for each face

    Optional attributes (with_quads=False):
        fc: Fx3 array of colors for each face
        fn: Fx3 array of vertex normal indices for each face
        ft: Fx3 array of texture vertex indices for each face

    """
    def __init__(self,
                 filename=None,
                 v=None, f=None, f4=None,
                 vc=None, fc=None, fc4=None,
                 vn=None, fn=None, fn4=None,
                 vt=None, ft=None, ft4=None,
                 e=None, segm=None,
                 basename=None,
                 landmarks=None, ppfilename=None, lmrkfilename=None,
                 with_quads=False):
        self._with_quads = with_quads
        self._blocked_properties = [
            prop3 if with_quads else prop3 + '4'
            for prop3 in ['f', 'fc', 'fn', 'ft']
        ]

        _properties = ['v', 'f', 'f4', 'vc', 'fc', 'fc4', 'vn', 'fn', 'fn4', 'vt', 'ft', 'ft4', 'e', 'segm']
        # First, set initial values, so that unset v doesn't override the v
        # from a file. We do this explicitly rather than using _properties
        # so that pylint will recognize the attributes and make
        # attribute-defined-outside-init warnings useful.
        self.v = self.f = self.f4 = self.vc = self.fc = self.fc4 = self.vn = self.fn = self.fn4 = self.vt = self.ft = self.ft4 = self.e = self.segm = None
        self.basename = basename

        # Load from a file, if one is specified
        if filename is not None:
            if isinstance(filename, basestring):
                # Support `Mesh(filename)`
                from lace.serialization import mesh
                mesh.load(filename, existing_mesh=self)
                if self.basename is None:
                    import os
                    self.basename = os.path.splitext(os.path.basename(filename))[0]
                self.filename = filename
            elif isinstance(filename, Mesh) or hasattr(filename, 'v'):
                # Support `Mesh(mesh)`
                import copy
                other_mesh = filename
                if with_quads != other_mesh.with_quads:
                    raise ValueError('Must use with_quads={} when copying a mesh with_quads={}'.format(
                        other_mesh.with_quads, other_mesh.with_quads))
                # A deep copy with all of the numpy arrays copied:
                for a in ['v', 'f', 'f4'] + other_mesh.__dict__.keys(): # NB: v and f[4] first as vc[4] and fc[4] need them
                    if a == 'landm_raw_xyz':
                        # We've deprecated landm_raw_xyz and it raises an error to access it now, but some
                        # older pickled meshes (don't pickle meshes!) still have it as a property and they
                        # raise an error on the default getattr call (in addition to not converting the
                        # data to the current format), so we special case this.
                        self.landm_xyz = other_mesh.__dict__['landm_raw_xyz']
                    else:
                        setattr(self, a, copy.deepcopy(getattr(other_mesh, a)))
            elif v is None:
                # Historically, meshes were crated with Mesh(verts)
                try:
                    v = np.array(filename, dtype=np.float64)
                except ValueError:
                    pass

        # And then override whatever came from the file or the other
        # mesh with explicitly given values
        for a in _properties:
            if locals()[a] is not None:
                setattr(self, a, locals()[a])

        # The following should probably be cleaned up at some point...
        if landmarks is not None:
            self.landm = landmarks
        if ppfilename is not None:
            self.landm = ppfilename
        if lmrkfilename is not None:
            self.landm = lmrkfilename

    def __del__(self):
        if hasattr(self, 'textureID'):
            # This may be set by MeshViewer
            from OpenGL.GL import glDeleteTextures
            glDeleteTextures([self.textureID])

    # These control pickling and unpickling:
    # We need this extra little tweak to handle loading pickled meshes that have member variables
    # that have since been changed to @properties. Without this, the member just doesn't get loaded
    # from the pickle.
    def __getstate__(self):
        # Same as the default implementation, but if it's not defined then __setstate__ is not called
        return self.__dict__

    def __setstate__(self, d):
        MEMBERS_CHANGED_TO_PROPERTIES = ['texture_filepath']
        for name in MEMBERS_CHANGED_TO_PROPERTIES:
            if name in d:
                d['_' + name] = d[name]
                del d[name]
        self.__dict__ = d # come on pylint, __dict__ is defined *before* __init__ pylint: disable=attribute-defined-outside-init

    def copy(self, only=None):
        '''
        Returns a deep copy with all of the numpy arrays copied
        If only is a list of strings, i.e. ['f', 'v'], then only those properties will be copied
        '''
        if only is None:
            return Mesh(self, with_quads=self.with_quads)
        else:
            import copy
            m = Mesh(with_quads=self.with_quads)
            for a in only:
                setattr(m, a, copy.deepcopy(getattr(self, a)))
            return m

    def copy_fv(self):
        '''Sugar on top of copy to only copy the basic f & v attributes'''
        return self.copy(only=['f', 'v'])

    def copy_fvs(self):
        '''Sugar on top of copy to only copy the basic f, v, & segm
        attributes
        '''
        return self.copy(only=['f', 'v', 'segm'])

    def _ensure_property_set_is_allowed(self, prop):
        if prop in self._blocked_properties:
            alternate_prop = prop + '4' if self.with_quads else prop.rstrip('4')
            raise ValueError("When with_quads={}, setting {} is not allowed. Use {} instead.".format(
                self.with_quads, prop, alternate_prop))

    def _set_property(self, prop, val):
        if val is not None:
            self._ensure_property_set_is_allowed(prop)
        self.__dict__[prop] = val

    @property
    def with_quads(self):
        return self._with_quads

    # Rather than use a private _x varaible and boilerplate getters for
    # these, we'll use the actual var name and just override the setter.
    @setter_property
    def v(self, val):
        # cached properties that are dependent on v
        self._clear_cached_properties('vertices_to_edges_matrix', 'vertices_to_edges_matrix_single_axis')
        self._set_property('v', as_numeric_array(val, dtype=np.float64, shape=(-1, 3), allow_none=True, empty_as_none=True))

    @setter_property
    def f(self, val):
        # cached properties that are dependent on f
        self._clear_cached_properties('faces_per_edge', 'vertices_per_edge', 'vertices_to_edges_matrix', 'vertices_to_edges_matrix_single_axis')
        self._set_property('f', as_numeric_array(val, dtype=np.uint64, shape=(-1, 3), allow_none=True, empty_as_none=True))

    @setter_property
    def f4(self, val):
        self._set_property('f4', as_numeric_array(val, dtype=np.uint64, shape=(-1, 4), allow_none=True, empty_as_none=True))

    @setter_property
    def fc(self, val):
        self._set_property('fc', color.colors_like(val, self.f))

    @setter_property
    def fc4(self, val):
        self._set_property('fc4', color.colors_like(val, self.f4))

    @setter_property
    def vc(self, val):
        self._set_property('vc', color.colors_like(val, self.v))

    @setter_property
    def fn(self, val):
        '''
        Note that in some applications, face normals are vectors, in others they are indexes into the vertex normal array.
        Someday we should refactor things so that we have seperate fn (face normal vectors) & fvn (face vertex normal indicies).
        Today is not that day.
        '''
        self._set_property('fn', as_numeric_array(val, dtype=np.float64, shape=(-1, 3), allow_none=True, empty_as_none=True))

    @setter_property
    def vn(self, val):
        self._set_property('vn', as_numeric_array(val, dtype=np.float64, shape=(-1, 3), allow_none=True, empty_as_none=True))

    @setter_property
    def ft(self, val):
        self._set_property('ft', as_numeric_array(val, dtype=np.uint32, shape=(-1, 3), allow_none=True, empty_as_none=True))

    @setter_property
    def ft4(self, val):
        self._set_property('ft4', as_numeric_array(val, dtype=np.uint32, shape=(-1, 4), allow_none=True, empty_as_none=True))

    @setter_property
    def fn4(self, val):
        self._set_property('fn4', as_numeric_array(val, dtype=np.uint32, shape=(-1, 4), allow_none=True, empty_as_none=True))

    @setter_property
    def vt(self, val):
        self._set_property('vt', as_numeric_array(val, dtype=np.float64, shape=(-1, 2), allow_none=True, empty_as_none=True))

    @setter_property
    def e(self, val):
        self._set_property('e', as_numeric_array(val, dtype=np.uint64, shape=(-1, 2), allow_none=True, empty_as_none=True))

    @setter_property
    def segm(self, val):
        self._set_property('segm', val)

    def _clear_cached_properties(self, *keys):
        for cached_property_key in keys:
            if cached_property_key in self.__dict__:
                del self.__dict__[cached_property_key]

    @classmethod
    def concatenate(cls, *args):
        """Concatenates an arbitrary number of meshes.

        Currently supports vertices, vertex colors, and faces.
        """
        nargs = len(args)
        if nargs == 1:
            return args[0]

        with_quads = args[0].with_quads
        if any([a.with_quads != with_quads for a in args]):
            raise ValueError('Expected `with_quads` to match for all args.')

        vs = [a.v for a in args if a.v is not None]
        vcs = [a.vc for a in args if a.vc is not None]
        fs = [a.f for a in args if a.f is not None]
        f4s = [a.f4 for a in args if a.f4 is not None]

        if vs and len(vs) != nargs:
            raise ValueError('Expected `v` for all args or none.')
        if vcs and len(vcs) != nargs:
            raise ValueError('Expected `vc` for all args or none.')
        if fs and len(fs) != nargs:
            raise ValueError('Expected `f` for all args or none.')
        if f4s and len(f4s) != nargs:
            raise ValueError('Expected `f4` for all args or none.')

        # Offset face indices by the cumulative vertex count.
        face_offsets = np.cumsum([v.shape[0] for v in vs[:-1]])
        for offset, f in zip(face_offsets, fs[1:]):  # https://bitbucket.org/logilab/pylint/issues/603/operator-generates-false-positive-unused pylint: disable=unused-variable
            f += offset

        return Mesh(
            v=np.vstack(vs) if vs else None,
            vc=np.vstack(vcs) if vcs else None,
            f=np.vstack(fs) if fs else None,
            with_quads=with_quads
        )

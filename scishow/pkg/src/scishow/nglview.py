"""Extended NGLWidget with additional representations and features.

References
----------
- [NGL documentation](https://nglviewer.org/ngl/api/index.html)
- [NGL source code](https://github.com/nglviewer/ngl)
- [NGLView source code](https://github.com/nglviewer/nglview)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field, asdict, fields

import nglview as nv
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence, Any
    from typing import Literal, NoReturn
    from scishow.typing import Vector3, Matrix3x3


class NGLWidget(nv.NGLWidget):
    """Extended nglview.NGLWidget with additional representations and features.

    References
    ----------
    - [NGLWidget source code](https://github.com/nglviewer/nglview/blob/master/nglview/widget.py)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def add_points(
        self,
        coords: list[float],
        colors: list[float],
        opacity: float = 0.7,
    ):
        self._js(
            f"""
            var point_buffer = new NGL.PointBuffer(
                {{
                    position: new Float32Array({coords}),
                    color: new Float32Array({colors}),
                }},
                {{
                    sizeAttenuation: true,
                    pointSize: 2,
                    opacity: 0.1,
                    useTexture: true,
                    alphaTest: 0.0,
                    edgeBleach: 0.7,
                    forceTransparent: true,
                    sortParticles: true
                }}
            )
            var shape = new NGL.Shape('grid');

            shape.addBuffer(point_buffer);
            var shapeComp = this.stage.addComponentFromObject(shape);
            shapeComp.addRepresentation("buffer", {{ opacity: {opacity} }});
            """
        )
        return

    def add_spheres(
        self,
        coords: Sequence[float],
        colors: Sequence[float] = (0.5, 0.5, 0.5),
        radii: Sequence[float] | float = 0.2,
        name: str | None = "spheres",
        representation_params: RepresentationParameters | None = None,
    ):
        coords = np.asarray(coords, dtype=np.single)
        colors = np.asarray(colors, dtype=np.half)
        radii = np.asarray(radii, dtype=np.half)

        num_coords = coords.size
        if num_coords % 3 != 0:
            raise ValueError(
                "There must be 3n values in `coords`, corresponding to n (x,y,z)-coordinates."
            )
        num_points = num_coords // 3
        num_rgbs = colors.size
        if num_rgbs < 3:
            raise ValueError("`colors` must at least be a single RGB value.")
        if num_rgbs == 3:
            colors = np.tile(colors, reps=num_points)
        elif num_rgbs != num_coords:
            raise ValueError("Size of `colors` and `coords` do not match.")
        num_radii = radii.size
        if num_radii == 1:
            radii = np.tile(radii, reps=num_points)
        elif num_radii != num_points:
            raise ValueError("Size of `radii` and `coords` do not match.")
        add_repr_args = [f'"buffer"']
        if representation_params:
            add_repr_args.append(str(representation_params))
        self._js(
            f"""
            var params = {
                dict(
                    position=coords.ravel().tolist(),
                    color=colors.ravel().tolist(),
                    radius=radii.ravel().tolist(),
                )
            };
            var shape = new NGL.Shape('{name}');
            var buffer = new NGL.SphereBuffer(params);

            shape.addBuffer(buffer);
            var shapeComp = this.stage.addComponentFromObject(shape);
            shapeComp.addRepresentation({", ".join(add_repr_args)});
            """
        )
        return

    def add_representation_within_radius_of_selection(
        self,
        component_id: int = 0,
        selection: str = "ligand",
        radius: float = 4.0,
        representation_type: str = "licorice",
    ) -> NoReturn:
        """
        Add a representation to all residues that have atoms within a certain radius of an atom in a
        selection.

        Parameters
        ----------
        view : nglview.widget.NGLWidget
        component_id : int
        selection : str
        radius : float
        representation_type : str

        Returns
        -------
        None
        """
        self._execute_js_code(
            f"""
            var component = this.stage.compList[{component_id}];
            var selection = new NGL.Selection("{selection}");
            var atomsWithinRad = component.structure.getAtomSetWithinSelection(selection, {radius});
            // Expand selection to all atoms within residues of selected atoms.
            var residuesWithinRad = component.structure.getAtomSetWithinGroup(atomsWithinRad);
            component.addRepresentation(
                "{representation_type}",
                {{sele: residuesWithinRad.toSeleString()}}
            );
            """
        )
        return

    def add_representation_to_structure(
        self,
        component_id: int = 0,
        selection: str = "protein",
        representation: str = "cartoon",
        aspect_ratio: float = 1,
        scale: float = 1,
        multiple_bond: bool = True,
    ):
        self._execute_js_code(
            f"""
            // Get the component
            var component = this.stage.compList[{component_id}];
            // exit if the component does not contain a structure
            if(component.type !== "structure") return;
            // add representation
            component.addRepresentation(
                "{representation}",
                {{
                    sele: {selection},
                    aspectRatio: {aspect_ratio},
                    scale: {scale},
                    multipleBond: {multiple_bond}
                }}
            );
            """
        )
        return

    def remove_component_by_name(self, name: str) -> NoReturn:
        """
        Given a component's name, remove it from the stage of a given NGLWidget.

        Parameters
        ----------
        view : nglview.widget.NGLWidget
            The widget to remove the component from.
        name : str
            Name of the component to remove.

        Returns
        -------
        None
        """
        self._execute_js_code(
            f"""this.stage.removeComponent(this.stage.getComponentsByName("{name}").first)"""
        )
        return

    def add_volume(
        self,
        data: np.ndarray,
        basis: Matrix3x3 | np.ndarray = np.eye(3),
        origin: Vector3 | np.ndarray = np.zeros(3),
        representation_type: Literal["surface", "dot", "slice"] = "surface",
        representation_params: SurfaceRepresentationParameters | None = None,
        name: str = "Volume",
        path: str = "memory",
    ):
        """Add a volume to the NGLWidget.

        A volume is a 3D grid of values,
        which can be visualized in different ways.

        Parameters
        ----------
        data
            A 3D array of values.
        basis
            Basis vectors (a.k.a. grid spacing matrix) for the volume.
            This is a 3x3 array where each row (i.e. basis[i])
            is a vector from one point to the next point in the i-th dimension.
            For example, for an orthogonal grid, the basis is a diagonal matrix
            where the diagonal elements are the grid spacing in each dimension.
        origin
            Coordinates of the first grid point.
        representation_type
            The representation type for the volume.
            Can be one of "surface", "dot", or "slice".
            This can also be changed later through the GUI.
        representation_params
            Represenation parameters for the volume representation type.
        name
            Name of the volume component.
            This is used to identify the component in the GUI or programmatically.
        path
            Path of the volume file.
            This is used to identify the component in the GUI or programmatically.

        References
        ----------
        - [NGL documentation](https://nglviewer.org/ngl/api/classes/volume.html)
        """
        shape = data.shape
        nx_ny_nz = ", ".join(map(str, shape))
        affine_map = np.eye(4)
        affine_map[:3, :3] = basis.transpose()
        affine_map[:3, 3] = origin
        matrix_args = ", ".join(map(str, affine_map.flat))
        add_repr_args = [f'"{representation_type}"']
        if representation_params:
            add_repr_args.append(str(representation_params))
        command = f"""
        var vol = new NGL.Volume("{name}", "{path}", {self._to_js_array(data)}, {nx_ny_nz})
        var m = new NGL.Matrix4()
        m.set({matrix_args})
        vol.setMatrix(m)
        var comp = this.stage.addComponentFromObject(vol)
        comp.addRepresentation({", ".join(add_repr_args)})
        """
        self._js(command)
        return self

    def add_origin(self):
        self._js(
            """
            var shape = new NGL.Shape("axes", { disableImpostor: true });
            shape.addArrow([ 0, 0, 0 ], [ 10, 0, 0 ], [ 1, 0, 0 ], 0.2);
            shape.addArrow([ 0, 0, 0 ], [ 0, 10, 0 ], [ 0, 1, 0 ], 0.2);
            shape.addArrow([ 0, 0, 0 ], [ 0, 0, 10 ], [ 0, 0, 1 ], 0.2);
            shape.addText([10, 0, 0], [0, 0, 0], 9, "x")
            shape.addText([0, 10, 0], [0, 0, 0], 9, "y")
            shape.addText([0, 0, 10], [0, 0, 0], 9, "z")
            var shapeComp = this.stage.addComponentFromObject(shape, {visible: true});
            shapeComp.addRepresentation("axes");
            """
        )
        return

    def add_bounding_box(
        self,
        component_id: int = 0,
        selection: str | None = None,
        color=(0, 0, 0),
        radius=0.1,
        name="bbox"
    ):
        """Draw an axis-aligned bounding box around a structure.

        Parameters
        ----------
        component_id
            ID of the component to draw the bounding box around.
        selection
            Selection string to specify the atoms to include in the bounding box.
            If None, the entire structure is used.
        color
            RGB color of the box edges as a tuple (r, g, b).
        radius
            Radius of the box edges.
        name
            Name of the shape component.
        """
        r, g, b = map(repr, color)
        js_code = f"""
        const component = this.stage.compList[{component_id}];
        if (!component || !component.structure) {{
            console.error("No structure in component {component_id}");
        }} else {{
            const selection = {f"new NGL.Selection('{selection}')" if selection else "undefined"};
            const box = component.structure.getBoundingBox(selection);
            const min = [box.min.x, box.min.y, box.min.z];
            const max = [box.max.x, box.max.y, box.max.z];

            const corners = [
                [min[0], min[1], min[2]],  // 0
                [max[0], min[1], min[2]],  // 1
                [min[0], max[1], min[2]],  // 2
                [max[0], max[1], min[2]],  // 3
                [min[0], min[1], max[2]],  // 4
                [max[0], min[1], max[2]],  // 5
                [min[0], max[1], max[2]],  // 6
                [max[0], max[1], max[2]]   // 7
            ];

            const edges = [
                [0, 1], [0, 2], [1, 3], [2, 3],  // bottom
                [4, 5], [4, 6], [5, 7], [6, 7],  // top
                [0, 4], [1, 5], [2, 6], [3, 7]   // verticals
            ];

            const shape = new NGL.Shape("{name}");
            for (const [i, j] of edges) {{
                shape.addCylinder(corners[i], corners[j], [{r}, {g}, {b}], {radius});
            }}

            const shapeComp = this.stage.addComponentFromObject(shape);
            shapeComp.addRepresentation("buffer");
        }}
        """
        self._js(js_code)
        return

    def add_box(
        self,
        lower_bounds,
        upper_bounds,
        color=(1, 0, 0),
        radius=0.1,
        name="box"
    ):
        """
        Draw a 3D box defined by lower and upper bounds using 12 cylinders (edges).

        Parameters:
            view (nglview.NGLWidget): The NGL viewer instance.
            lower_bounds (tuple or list of 3 floats): (x_min, y_min, z_min)
            upper_bounds (tuple or list of 3 floats): (x_max, y_max, z_max)
            color (tuple): RGB color as 3 floats (0–1)
            radius (float): Radius of the cylinder edges
            name (str): Name of the shape component
        """
        if not (len(lower_bounds) == len(upper_bounds) == 3):
            raise ValueError("lower_bounds and upper_bounds must be 3-element tuples/lists.")

        x0, y0, z0 = lower_bounds
        x1, y1, z1 = upper_bounds
        r, g, b = color

        js_code = f"""
        const min = [{x0}, {y0}, {z0}];
        const max = [{x1}, {y1}, {z1}];

        const corners = [
            [min[0], min[1], min[2]],  // 0
            [max[0], min[1], min[2]],  // 1
            [min[0], max[1], min[2]],  // 2
            [max[0], max[1], min[2]],  // 3
            [min[0], min[1], max[2]],  // 4
            [max[0], min[1], max[2]],  // 5
            [min[0], max[1], max[2]],  // 6
            [max[0], max[1], max[2]]   // 7
        ];

        const edges = [
            [0, 1], [0, 2], [1, 3], [2, 3],  // bottom
            [4, 5], [4, 6], [5, 7], [6, 7],  // top
            [0, 4], [1, 5], [2, 6], [3, 7]   // verticals
        ];

        const shape = new NGL.Shape("{name}");
        for (const [i, j] of edges) {{
            shape.addCylinder(corners[i], corners[j], [{r}, {g}, {b}], {radius});
        }}

        const shapeComp = this.stage.addComponentFromObject(shape);
        shapeComp.addRepresentation("buffer");
        console.log("[SUCCESS] Box '{name}' rendered.");
        """
        self._js(js_code)
        return

    @staticmethod
    def _to_js_array(array: np.ndarray) -> str:
        """Convert a numpy array to a JavaScript array."""
        array = np.asarray(array)
        array_flat = array.ravel("F")
        array_sanitized = array_flat.astype(str)
        array_sanitized[np.isnan(array_flat)] = "NaN"
        array_sanitized[np.isposinf(array_flat)] = "Infinity"
        array_sanitized[np.isneginf(array_flat)] = "-Infinity"
        return f"[{", ".join(array_sanitized)}]"

@dataclass(kw_only=True)
class RepresentationParameters:
    """General representation parameters for NGLWidget.

    Attributes
    ----------
    name
        Name of the representation.
    lazy
        Only build and update the representation when visible.
    clip_near
        Position of camera near/front clipping plane in percent of scene bounding box.
    clip_radius
        Radius of the clipping sphere.
    clip_center
        Position for spherical clipping.
    flat_shaded
        Render with flat shading.
    opacity
        Translucency: 1 is fully opaque, 0 is fully transparent.
    depth_write
        Whether depth writing is enabled.
    side
        Which triangle sides to render. One of "front", "back", "double".
    wireframe
        Render as wireframe.
    color_data
        Atom or bond indexed data for coloring.
    color_scheme
        Color scheme identifier.
    color_scale
        Color scale, either a predefined name or array of colors.
    color_reverse
        Whether to reverse the color scale.
    color_value
        Static color value to use.
    color_domain
        Value range for the color scale, must have two integers [min, max].
    color_mode
        Color mode, one of 'rgb', 'hsv', 'hsl', 'hsi', 'lab', or 'hcl'.
    roughness
        Material roughness between 0 and 1.
    metalness
        Material metalness between 0 and 1.
    diffuse
        Diffuse color for lighting.
    diffuse_interior
        Ignore normal when rendering interior surfaces.
    use_interior_color
        Whether to use a different interior color.
    interior_color
        Color to apply to interior surfaces.
    interior_darkening
        How much to darken interior surfaces, from 0 to 1.
    disable_picking
        Disable object picking (e.g., for interactivity).

    References
    ----------
    - [NGL source code](https://github.com/nglviewer/ngl/blob/60be69b5fe0e9c43cb3a06fe1cb691fa9478c790/src/representation/representation.ts#L18-L88)
    - [NGL source code](https://github.com/nglviewer/ngl/blob/60be69b5fe0e9c43cb3a06fe1cb691fa9478c790/src/representation/representation.ts#L155-L249)
    """
    name: str | None = None
    lazy: bool | None = None
    clip_near: int | None = None
    clip_radius: int | None = None
    clip_center: Vector3 | None = None
    flat_shaded: bool | None = None
    opacity: float | None = None
    depth_write: bool | None = None
    side: str | None = None
    wireframe: bool | None = None
    color_data: str | None = None
    color_scheme: str | None = None
    color_scale: str | list[str | Color] | None = None
    color_reverse: bool | None = None
    color_value: Color | str | int | None = None
    color_domain: list[int] | None = None
    color_mode: str | None = None
    color_space: Literal["sRGB", "linear"] = None
    roughness: float | None = None
    metalness: float | None = None
    diffuse: Color | str | int | None = None
    diffuse_interior: bool | None = None
    use_interior_color: bool | None = None
    interior_color: Color | str | int | None = None
    interior_darkening: float | None = None
    disable_picking: bool | None = None
    matrix: Matrix4 | None = None
    quality: str | None = None
    visible: bool | None = None
    color: Color | str | int | None = None
    sphere_detail: int | None = None
    radial_segments: int | None = None
    open_ended: bool | None = None
    disable_impostor: bool | None = None

    def __str__(self) -> str:
        js_fields = []
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                js_key = _to_camel_case(f.name)
                js_val = _js_repr(Color(val) if f.name.endswith('color') and not isinstance(val, Color) and isinstance(val, (str, int)) else val)
                js_fields.append(f"{js_key}: {js_val}")
        return f"{{{', '.join(js_fields)}}}"


@dataclass(kw_only=True)
class SurfaceRepresentationParameters(RepresentationParameters):
    """Surface representation parameters.

    Attributes
    ----------
    isolevel_type
        Meaning of the isolevel value.
        Either 'value' for the literal value or
        'sigma' as a factor of the sigma of the data.
        Only applies to volume data.
    isolevel
        The value at which to create the isosurface.
        Only applies to volume data.
    negate_isolevel
        Whether to negate the isolevel value.
        Only applies to volume data.
    isolevel_scroll
        Whether to show a slider to change the isolevel value.
        Only applies to volume data.
    smooth
        Number of laplacian smoothing iterations
        after surface triangulation.
        Only applies to volume data.
    background
        Whether to render the surface in the background, unlit.
    opaque_back
        Whether to render the back-faces (where normals point away from the camera)
        of the surface opaque, ignoring the transparency parameter.
    box_size
        Size of the box to triangulate volume data in.
        Set to zero to triangulate the whole volume.
        Only applies to volume data.
    contour
        Whether to show the contour lines of the isosurface.
        Only applies to volume data.
    use_worker
        Weather to triangulate the volume asynchronously in a Web Worker.
        Only applies to volume data.
    wrap
        Whether to wrap volume data around the edges;
        use in conjuction with `box_size`
        but not larger than the volume dimension.
        Only applies to volume data.

    References
    ----------
    - [NGL source code](https://github.com/nglviewer/ngl/blob/60be69b5fe0e9c43cb3a06fe1cb691fa9478c790/src/representation/surface-representation.ts#L27-L36)
    - [NGL source code](https://github.com/nglviewer/ngl/blob/60be69b5fe0e9c43cb3a06fe1cb691fa9478c790/src/representation/surface-representation.ts#L92-L134)
    """
    isolevel_type: Literal["value", "sigma"] | None = None
    isolevel: float | None = None
    negate_isolevel: bool | None = None
    isolevel_scroll: bool | None = None
    smooth: int | None = None
    background: bool | None = None
    opaque_back: bool | None = None
    box_size: int | None = None
    contour: bool | None = None
    use_worker: bool | None = None
    wrap: bool | None = None


@dataclass
class Color:
    """A JavaScript `Color` object from three.js."""
    value: int | str

    def __str__(self):
        if isinstance(self.value, int):
            return f"new Color(0x{self.value:06x})"
        return f"new Color('{self.value}')"


@dataclass
class Vector3:
    """A JavaScript `Vector3` object from three.js."""
    x: float
    y: float
    z: float

    def __str__(self):
        return f"new Vector3({self.x}, {self.y}, {self.z})"


@dataclass
class Matrix4:
    """A JavaScript `Matrix4` object from three.js."""
    elements: list[float]

    def __str__(self):
        return f"new Matrix4().fromArray({_js_repr(self.elements)})"


def _to_camel_case(snake_str: str) -> str:
    parts = snake_str.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])


def _js_repr(value: Any) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        return '[' + ', '.join(_js_repr(v) for v in value) + ']'
    elif isinstance(value, dict):
        return '{' + ', '.join(f"{k}: {_js_repr(v)}" for k, v in value.items()) + '}'
    elif hasattr(value, '__str__'):
        return str(value)
    return 'null'
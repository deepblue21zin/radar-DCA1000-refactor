from __future__ import annotations

import site
from pathlib import Path
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


def _candidate_opengl_site_dirs():
    candidates = []
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site))

    roaming_python_root = Path.home() / 'AppData' / 'Roaming' / 'Python'
    if roaming_python_root.exists():
        for site_dir in sorted(roaming_python_root.glob('Python*/site-packages'), reverse=True):
            candidates.append(site_dir)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        unique_candidates.append(candidate)
    return unique_candidates


def build_heatmap_lookup_table():
    position = np.arange(64) / 64
    position[0] = 0
    position = np.flip(position)
    colors = np.flip(
        [
            [62, 38, 168, 255], [63, 42, 180, 255], [65, 46, 191, 255],
            [67, 50, 202, 255], [69, 55, 213, 255], [70, 60, 222, 255],
            [71, 65, 229, 255], [70, 71, 233, 255], [70, 77, 236, 255],
            [69, 82, 240, 255], [68, 88, 243, 255], [68, 94, 247, 255],
            [67, 99, 250, 255], [66, 105, 254, 255], [62, 111, 254, 255],
            [56, 117, 254, 255], [50, 123, 252, 255], [47, 129, 250, 255],
            [46, 135, 246, 255], [45, 140, 243, 255], [43, 146, 238, 255],
            [39, 150, 235, 255], [37, 155, 232, 255], [35, 160, 229, 255],
            [31, 164, 225, 255], [28, 129, 222, 255], [24, 173, 219, 255],
            [17, 177, 214, 255], [7, 181, 208, 255], [1, 184, 202, 255],
            [2, 186, 195, 255], [11, 189, 188, 255], [24, 191, 182, 255],
            [36, 193, 174, 255], [44, 195, 167, 255], [49, 198, 159, 255],
            [55, 200, 151, 255], [63, 202, 142, 255], [74, 203, 132, 255],
            [88, 202, 121, 255], [102, 202, 111, 255], [116, 201, 100, 255],
            [130, 200, 89, 255], [144, 200, 78, 255], [157, 199, 68, 255],
            [171, 199, 57, 255], [185, 196, 49, 255], [197, 194, 42, 255],
            [209, 191, 39, 255], [220, 189, 41, 255], [230, 187, 45, 255],
            [239, 186, 53, 255], [248, 186, 61, 255], [254, 189, 60, 255],
            [252, 196, 57, 255], [251, 202, 53, 255], [249, 208, 50, 255],
            [248, 214, 46, 255], [246, 220, 43, 255], [245, 227, 39, 255],
            [246, 233, 35, 255], [246, 239, 31, 255], [247, 245, 27, 255],
            [249, 251, 20, 255],
        ],
        axis=0,
    )
    color_map = pg.ColorMap(position, colors)
    return color_map.getLookupTable(0.0, 1.0, 256)


class SpatialViewController:
    def __init__(
        self,
        *,
        roi_lateral_m: float,
        roi_forward_m: float,
        roi_min_forward_m: float,
        view_y: int,
        view_height: int,
        point_base_z_m: float,
        point_confidence_scale_m: float,
    ):
        self.roi_lateral_m = float(roi_lateral_m)
        self.roi_forward_m = float(roi_forward_m)
        self.roi_min_forward_m = float(roi_min_forward_m)
        self.view_y = int(view_y)
        self.view_height = int(view_height)
        self.point_base_z_m = float(point_base_z_m)
        self.point_confidence_scale_m = float(point_confidence_scale_m)

        self.available = False
        self.import_error = None
        self.gl = None
        self.view = None
        self.scatter = None
        self.stems = None
        self.tentative_scatter = None
        self.tentative_stems = None
        self._load_opengl()

    def _load_opengl(self):
        try:
            import pyqtgraph.opengl as gl
            self.gl = gl
            self.available = True
            self.import_error = None
        except ModuleNotFoundError as exc:
            self.import_error = exc
            if exc.name != 'OpenGL':
                return
            for candidate_site in _candidate_opengl_site_dirs():
                candidate_str = str(candidate_site)
                if candidate_str not in sys.path and candidate_site.exists():
                    sys.path.append(candidate_str)
                try:
                    import pyqtgraph.opengl as gl
                    self.gl = gl
                    self.available = True
                    self.import_error = None
                    return
                except ModuleNotFoundError as retry_exc:
                    self.import_error = retry_exc

    def attach(self, central_widget, label_widget):
        label_widget.setAlignment(QtCore.Qt.AlignCenter)
        if not self.available or self.gl is None:
            label_widget.setText('3D Spatial View (OpenGL unavailable)')
            return

        gl = self.gl
        label_widget.setText('3D Spatial View')
        self.view = gl.GLViewWidget(central_widget)
        self.view.setGeometry(QtCore.QRect(30, self.view_y, 741, self.view_height))
        self.view.setCameraPosition(distance=7.5, elevation=20, azimuth=-92)
        self.view.opts['center'] = pg.Vector(0.0, self.roi_forward_m / 2.0, 0.3)

        ground_grid = gl.GLGridItem()
        ground_grid.setSize(x=self.roi_lateral_m * 2.2, y=self.roi_forward_m * 1.1, z=0.0)
        ground_grid.setSpacing(x=0.5, y=0.5, z=0.5)
        ground_grid.translate(0.0, self.roi_forward_m / 2.0, 0.0)
        self.view.addItem(ground_grid)

        axis_item = gl.GLAxisItem()
        axis_item.setSize(0.7, 0.7, 0.7)
        axis_item.translate(-self.roi_lateral_m - 0.25, 0.0, 0.0)
        self.view.addItem(axis_item)

        roi_outline = np.array(
            [
                [-self.roi_lateral_m, self.roi_min_forward_m, 0.0],
                [self.roi_lateral_m, self.roi_min_forward_m, 0.0],
                [self.roi_lateral_m, self.roi_forward_m, 0.0],
                [-self.roi_lateral_m, self.roi_forward_m, 0.0],
                [-self.roi_lateral_m, self.roi_min_forward_m, 0.0],
            ],
            dtype=np.float32,
        )
        roi_outline_item = gl.GLLinePlotItem(
            pos=roi_outline,
            color=(0.65, 0.65, 0.65, 1.0),
            width=1.5,
            antialias=True,
            mode='line_strip',
        )
        self.view.addItem(roi_outline_item)

        self.stems = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=(1.0, 0.45, 0.20, 0.85),
            width=2.0,
            antialias=True,
            mode='lines',
        )
        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
            size=np.zeros((0,), dtype=np.float32),
        )
        self.tentative_stems = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=(1.0, 0.88, 0.25, 0.65),
            width=1.5,
            antialias=True,
            mode='lines',
        )
        self.tentative_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
            size=np.zeros((0,), dtype=np.float32),
        )
        self.view.addItem(self.stems)
        self.view.addItem(self.scatter)
        self.view.addItem(self.tentative_stems)
        self.view.addItem(self.tentative_scatter)

    def update(self, display_tracks, tentative_display_tracks):
        if not self.available or self.scatter is None or self.stems is None:
            return

        if not display_tracks and not tentative_display_tracks:
            self.scatter.setData(
                pos=np.zeros((0, 3), dtype=np.float32),
                color=np.zeros((0, 4), dtype=np.float32),
                size=np.zeros((0,), dtype=np.float32),
            )
            self.stems.setData(pos=np.zeros((0, 3), dtype=np.float32))
            if self.tentative_scatter is not None:
                self.tentative_scatter.setData(
                    pos=np.zeros((0, 3), dtype=np.float32),
                    color=np.zeros((0, 4), dtype=np.float32),
                    size=np.zeros((0,), dtype=np.float32),
                )
            if self.tentative_stems is not None:
                self.tentative_stems.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        positions = []
        colors = []
        sizes = []
        stems = []
        for track in display_tracks:
            z_m = self.point_base_z_m + (self.point_confidence_scale_m * float(track.confidence))
            positions.append([track.x_m, track.y_m, z_m])
            colors.append([
                1.0,
                float(max(0.20, 0.85 - (0.55 * track.confidence))),
                0.20,
                0.95,
            ])
            sizes.append(10.0 + (8.0 * float(track.confidence)))
            stems.extend([
                [track.x_m, track.y_m, 0.0],
                [track.x_m, track.y_m, z_m],
            ])

        self.scatter.setData(
            pos=np.asarray(positions, dtype=np.float32) if positions else np.zeros((0, 3), dtype=np.float32),
            color=np.asarray(colors, dtype=np.float32) if colors else np.zeros((0, 4), dtype=np.float32),
            size=np.asarray(sizes, dtype=np.float32) if sizes else np.zeros((0,), dtype=np.float32),
        )
        self.stems.setData(
            pos=np.asarray(stems, dtype=np.float32) if stems else np.zeros((0, 3), dtype=np.float32),
            color=(1.0, 0.45, 0.20, 0.85),
            width=2.0,
            mode='lines',
        )

        if self.tentative_scatter is None or self.tentative_stems is None:
            return

        tentative_positions = []
        tentative_colors = []
        tentative_sizes = []
        tentative_stems = []
        for track in tentative_display_tracks:
            z_m = self.point_base_z_m + (0.75 * self.point_confidence_scale_m * float(track.confidence))
            tentative_positions.append([track.x_m, track.y_m, z_m])
            tentative_colors.append([1.0, 0.88, 0.25, 0.72])
            tentative_sizes.append(8.0 + (6.0 * float(track.confidence)))
            tentative_stems.extend([
                [track.x_m, track.y_m, 0.0],
                [track.x_m, track.y_m, z_m],
            ])

        self.tentative_scatter.setData(
            pos=np.asarray(tentative_positions, dtype=np.float32) if tentative_positions else np.zeros((0, 3), dtype=np.float32),
            color=np.asarray(tentative_colors, dtype=np.float32) if tentative_colors else np.zeros((0, 4), dtype=np.float32),
            size=np.asarray(tentative_sizes, dtype=np.float32) if tentative_sizes else np.zeros((0,), dtype=np.float32),
        )
        self.tentative_stems.setData(
            pos=np.asarray(tentative_stems, dtype=np.float32) if tentative_stems else np.zeros((0, 3), dtype=np.float32),
            color=(1.0, 0.88, 0.25, 0.65),
            width=1.5,
            mode='lines',
        )

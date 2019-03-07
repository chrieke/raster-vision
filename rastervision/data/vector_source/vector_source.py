from abc import ABC, abstractmethod

from shapely.geometry import shape, mapping
import shapely

from rastervision.data.vector_source.class_inference import (
    ClassInference, ClassInferenceOptions)


def transform_geojson(geojson,
                      line_bufs=None,
                      point_bufs=None,
                      crs_transformer=None):
    new_features = []
    for f in geojson['features']:
        # This was added to handle empty geoms which appear when using
        # OSM vector tiles.
        if (not f.get('geometry')) or (not f['geometry'].get('coordinates')):
            continue

        geom = shape(f['geometry'])

        # Split GeometryCollection into list of geoms.
        geoms = [geom]
        if geom.geom_type == 'GeometryCollection':
            geoms = list(geom)

        # Split any MultiX to list of X.
        new_geoms = []
        for g in geoms:
            if geom.geom_type in ['MultiPolygon', 'MultiPoint', 'MultiLineString']:
                new_geoms.extend(list(g))
            else:
                new_geoms.append(g)
        geoms = new_geoms

        # Buffer geoms.
        class_id = f['properties']['class_id']
        new_geoms = []
        for g in geoms:
            if g.geom_type == 'LineString':
                line_buf = 1
                if line_bufs is not None:
                    line_buf = line_bufs.get(class_id, 1)
                # If line_buf for the class_id was explicitly set as None, then
                # don't buffer.
                if line_buf is not None:
                    g = g.buffer(line_buf)
                new_geoms.append(g)
            elif g.geom_type == 'Point':
                point_buf = 1
                if point_bufs is not None:
                    point_buf = point_bufs.get(class_id, 1)
                # If point_buf for the class_id was explicitly set as None, then
                # don't buffer.
                if point_buf is not None:
                    g = g.buffer(point_buf)
                new_geoms.append(g)
            else:
                # Use buffer trick to handle self-intersecting polygons. Buffer returns
                # a MultiPolygon if there is a bowtie, so we have to convert it to a
                # list of Polygons.
                poly_buf = g.buffer(0)
                if poly_buf.geom_type == 'MultiPolygon':
                    new_geoms.extend(list(poly_buf))
                else:
                    new_geoms.append(poly_buf)
        geoms = new_geoms

        if crs_transformer is not None:
            # Convert map to pixel coords.
            def transform_shape(x, y, z=None):
                return crs_transformer.map_to_pixel((x, y))

            geoms = [shapely.ops.transform(transform_shape, g) for g in geoms]

        for g in geoms:
            new_f = {
                'type': 'Feature',
                'geometry': mapping(g),
                'properties': f['properties']
            }
            new_features.append(new_f)

    return {'type': 'FeatureCollection', 'features': new_features}


class VectorSource(ABC):
    """A source of vector data.

    Uses GeoJSON as its internal representation of vector data.
    """

    def __init__(self,
                 crs_transformer,
                 line_bufs=None,
                 point_bufs=None,
                 class_inf_opts=None):
        """Constructor.

        Args:
            class_inf_opts: (ClassInferenceOptions)
        """
        self.crs_transformer = crs_transformer
        self.line_bufs = line_bufs
        self.point_bufs = point_bufs
        if class_inf_opts is None:
            class_inf_opts = ClassInferenceOptions()
        self.class_inference = ClassInference(class_inf_opts)

        self.geojson = None

    def get_geojson(self, to_pixel=True):
        if self.geojson is None:
            self.geojson = self._get_geojson()
        return transform_geojson(
            self.geojson,
            self.line_bufs,
            self.point_bufs,
            crs_transformer=(self.crs_transformer if to_pixel else None))

    @abstractmethod
    def _get_geojson(self):
        pass

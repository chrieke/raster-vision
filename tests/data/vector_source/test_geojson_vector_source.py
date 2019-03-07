import unittest
import os

from rastervision.data.vector_source import (GeoJSONVectorSourceConfigBuilder,
                                             GeoJSONVectorSourceConfig)
from rastervision.core.class_map import ClassMap
from rastervision.utils.files import json_to_file
from rastervision.rv_config import RVConfig
from rastervision.data.crs_transformer import IdentityCRSTransformer


class TestGeoJSONVectorSource(unittest.TestCase):
    """This also indirectly tests the ClassInference class."""

    def setUp(self):
        self.temp_dir = RVConfig.get_tmp_dir()
        self.uri = os.path.join(self.temp_dir.name, 'vectors.json')

    def tearDown(self):
        self.temp_dir.cleanup()

    def props_to_geojson(self, props):
        return {
            'type':
            'FeatureCollection',
            'features': [{
                'properties': props,
                'geometry': {'type': 'Point', 'coordinates': [1, 1]}
            }]
        }

    def _test_class_inf(self, props, exp_class_ids, default_class_id=None):
        geojson = self.props_to_geojson(props)
        json_to_file(geojson, self.uri)

        class_map = ClassMap.construct_from(['building', 'car', 'tree'])
        class_id_to_filter = {
            1: ['==', 'type', 'building'],
            2: ['any', ['==', 'type', 'car'], ['==', 'type', 'auto']]
        }
        b = GeoJSONVectorSourceConfigBuilder() \
            .with_class_inference(class_id_to_filter=class_id_to_filter,
                                  default_class_id=default_class_id) \
            .with_uri(self.uri) \
            .build()
        msg = b.to_proto()
        config = GeoJSONVectorSourceConfig.from_proto(msg)
        source = config.create_source(
            crs_transformer=IdentityCRSTransformer(), class_map=class_map)
        trans_geojson = source.get_geojson()
        class_ids = [f['properties']['class_id'] for f in trans_geojson['features']]
        self.assertEqual(class_ids, exp_class_ids)

    def test_class_inf_class_id(self):
        self._test_class_inf({'class_id': 3}, [3])

    def test_class_inf_label(self):
        self._test_class_inf({'label': 'car'}, [2])

    def test_class_inf_filter(self):
        self._test_class_inf({'type': 'auto'}, [2])

    def test_class_inf_default(self):
        self._test_class_inf({}, [4], default_class_id=4)

    def test_class_inf_no_default(self):
        self._test_class_inf({}, [])

    def test_transform_geojson(self):
        geojson = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'geometry': {
                        'type': 'Point',
                        'coordinates': []
                    }
                },
                {
                    'geometry': {
                        'type': 'GeometryCollection',
                        'geometries': [
                            {
                                'type': 'MultiPoint',
                                'coordinates': [[10, 10], [20, 20]]
                            }
                        ]
                    }
                },
                {
                    'geometry': {
                        'type': 'MultiLineString',
                        'coordinates': [
                            [[10, 10], [20, 20]],
                            [[20, 20], [30, 30]]]
                    }
                },
                {
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [4, 4],
                    },
                    'properties': {
                        'class_id': 2
                    }
                },
                {
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[[10, 10], [10, 20], [20, 20], [10, 10]]]
                    }
                }
            ]
        }
        json_to_file(geojson, self.uri)

        b = GeoJSONVectorSourceConfigBuilder() \
            .with_uri(self.uri) \
            .build()

        msg = b.to_proto()
        config = GeoJSONVectorSourceConfig.from_proto(msg)
        source = config.create_source(
            crs_transformer=IdentityCRSTransformer(), class_map=self.class_map)

        trans_geojson = source.get_geojson()
        import pprint
        pprint.pprint(trans_geojson, indent=4)


if __name__ == '__main__':
    unittest.main()

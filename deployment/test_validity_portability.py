import unittest
from verify_validity_artifacts import portable_summary
class PortableSummary(unittest.TestCase):
    def test_platform_separators_only(self):
        windows={'raw_hashes':{'run\\input.json':'abc'},'score':8}
        linux={'raw_hashes':{'run/input.json':'abc'},'score':8}
        self.assertEqual(portable_summary(windows),portable_summary(linux))
        self.assertNotEqual(portable_summary(windows),portable_summary(dict(linux,score=7)))
        self.assertNotEqual(portable_summary(windows),portable_summary({'raw_hashes':{'run/input.json':'different'},'score':8}))
    def test_collision_rejected(self):
        with self.assertRaises(ValueError):portable_summary({'raw_hashes':{'a\\b':'x','a/b':'x'}})
if __name__=='__main__':unittest.main()

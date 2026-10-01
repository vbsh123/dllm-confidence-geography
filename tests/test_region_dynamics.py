import unittest

from confidence_geography.region_dynamics import regions_at, candidate_regions, direction, leftmost_unresolved


class RegionDynamicsTests(unittest.TestCase):
    def test_four_masks_means_index_distance_five(self):
        self.assertEqual(regions_at({0, 4}, 4), [[0, 4]])
        self.assertEqual(regions_at({0, 5}, 4), [[0], [5]])

    def test_one_mask_separates_contiguous_components(self):
        self.assertEqual(regions_at({0, 1, 3, 4, 7}, 1), [[0, 1], [3, 4], [7]])

    def test_candidate_can_bridge_two_regions(self):
        regions = regions_at({0, 7}, 4)
        ids = candidate_regions(3, regions, 4)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(direction(3, ids, regions), 'bridge_multiple_regions')

    def test_internal_mask_is_in_region(self):
        regions = regions_at({0, 3}, 4)
        self.assertEqual(candidate_regions(1, regions, 4), [0])
        self.assertEqual(direction(1, [0], regions), 'internal_hole')

    def test_direction_distinguishes_append_from_skipping(self):
        regions = [[3, 4, 5]]
        self.assertEqual(direction(6, [0], regions), 'immediate_right')
        self.assertEqual(direction(7, [0], regions), 'right_skip')
        self.assertEqual(direction(2, [0], regions), 'immediate_left')
        self.assertEqual(direction(1, [0], regions), 'left_skip')

    def test_isolated_requires_full_mask_gap(self):
        self.assertEqual(candidate_regions(4, [[0]], 4), [0])
        self.assertEqual(candidate_regions(5, [[0]], 4), [])

    def test_region_counts_only_merge_as_gap_increases(self):
        occupied = {0, 3, 8, 9, 17, 25, 45}
        counts = [len(regions_at(occupied, x)) for x in [1, 2, 3, 4, 8, 16]]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_leftmost_unresolved_prefers_internal_hole_over_right_extension(self):
        rows = {p: {'eligible': True} for p in [1, 3, 5]}
        self.assertEqual(leftmost_unresolved([0, 2, 4], rows, 20), 1)
        self.assertEqual(leftmost_unresolved([0, 1, 2, 3, 4], {5: rows[5]}, 20), 5)
        self.assertIsNone(leftmost_unresolved([0, 1, 2, 3, 4], {5: rows[5]}, 5))


if __name__ == '__main__':
    unittest.main()

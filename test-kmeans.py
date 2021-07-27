import torch
import unittest
from kmeans import NaiveKmeans

class TestNaiveKmeans(unittest.TestCase):
    def test_seeder(self):
        x = torch.randn(256, 3)
        test_nkmeans = NaiveKmeans(3)
        test_centers, test_indices  = test_nkmeans._seeder(x)
        self.assertTrue(test_centers.size(0) == test_centers.size(1) == 3)
        self.assertTrue(test_indices[test_indices<x.size(0)].size(0)==3)

    def test_cost(self):
        tds_example = [[2, 3],
                       [6, 1],
                       [1, 2],
                       [3, 0]]
        centers = [[4, 2], [2, 1]]
        centers = [torch.tensor(row) for row in centers]
        tds_example = [torch.tensor(row) for row in tds_example]
        x = torch.stack(tds_example).float()
        test_nkmeans = NaiveKmeans(2)
        centers = torch.stack(centers).float()
        test_costs, indices = test_nkmeans._cost(x, centers)
        self.assertTrue(test_costs.size(0) == 4)
        #print(test_costs.sum().cpu().item(), test_costs)
        self.assertTrue(test_costs.sum().cpu().item() == 13)

    def test_naive_kmeans(self):
        tds_example = [[2, 3],
                       [6, 1],
                       [1, 2],
                       [3, 0]]
        centers = [[4, 2], [2, 1]]
        centers = [torch.tensor(row) for row in centers]
        tds_example = [torch.tensor(row) for row in tds_example]
        x = torch.stack(tds_example).float()
        centers = torch.stack(centers).float()
        test_nkmeans = NaiveKmeans(2)
        test_nkmeans(x, centers)

        


if __name__ == '__main__':
  unittest.main()

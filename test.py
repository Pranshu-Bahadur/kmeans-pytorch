import unittest
from pytorch_kmeans import uniform_seeder, uniform_d_squared_seeder 
from torch import tensor, rand

class TestSeeders(unittest.TestCase):

    def test_uniform_seeder(self):
        test_x = rand(16, 3)
        print(test_x.size())
        seeder_splits = uniform_seeder(test_x, 4)
        self.assertEqual(seeder_splits[0].size(0), 4)
        self.assertEqual(seeder_splits[1].size(0), 12)

    def test_uniform_d_squared_seeder(self):
        test_x = rand(16, 3)
        seeder_splits = uniform_d_squared_seeder(test_x, 4)
        self.assertEqual(seeder_splits[0].size(0), 4)
        self.assertEqual(seeder_splits[1].size(0), 12)

if __name__ == '__main__':
    unittest.main()
        
        

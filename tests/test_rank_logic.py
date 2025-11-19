import os
import unittest
from hudes.high_scores import get_rank, insert_high_score, init_db

DB_PATH = "test_rank.sqlite3"


class TestRankLogic(unittest.TestCase):
    def setUp(self):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db(DB_PATH)

    def tearDown(self):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

    def test_get_rank_empty(self):
        # Rank for a score in empty DB should be 1
        rank, total = get_rank(1.0, path=DB_PATH)
        self.assertEqual(rank, 1)
        self.assertEqual(total, 0)

    def test_get_rank_first_score(self):
        # Insert one score
        insert_high_score("USER", 2.0, 2.0, 60, 0, b"", path=DB_PATH)

        # Check rank for a better score (1.0)
        rank, total = get_rank(1.0, path=DB_PATH)
        self.assertEqual(rank, 1)
        self.assertEqual(total, 1)

        # Check rank for a worse score (3.0)
        rank, total = get_rank(3.0, path=DB_PATH)
        self.assertEqual(rank, 2)
        self.assertEqual(total, 1)

        # Check rank for same score (2.0)
        # Logic: less than 2.0 is 0. So rank is 1.
        rank, total = get_rank(2.0, path=DB_PATH)
        self.assertEqual(rank, 1)
        self.assertEqual(total, 1)

    def test_get_rank_multiple(self):
        insert_high_score("U1", 1.0, 1.0, 60, 0, b"", path=DB_PATH)
        insert_high_score("U2", 2.0, 2.0, 60, 0, b"", path=DB_PATH)
        insert_high_score("U3", 3.0, 3.0, 60, 0, b"", path=DB_PATH)

        # Better than all
        rank, total = get_rank(0.5, path=DB_PATH)
        self.assertEqual(rank, 1)
        self.assertEqual(total, 3)

        # Between 1 and 2
        rank, total = get_rank(1.5, path=DB_PATH)
        self.assertEqual(rank, 2)
        self.assertEqual(total, 3)

        # Worse than all
        rank, total = get_rank(3.5, path=DB_PATH)
        self.assertEqual(rank, 4)
        self.assertEqual(total, 3)


if __name__ == "__main__":
    unittest.main()

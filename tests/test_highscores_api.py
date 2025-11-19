import os
import pytest
from hudes.high_scores import (
    init_db,
    insert_high_score,
    get_all_scores,
    delete_high_score,
    get_rank,
)

DB_PATH = "test_high_scores_api.sqlite3"


@pytest.fixture
def setup_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db(DB_PATH)
    yield
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


@pytest.mark.asyncio
async def test_api_functions(setup_db):
    # Insert a dummy score
    insert_high_score(
        name="TEST",
        score=1.0,
        best_val_loss=1.0,
        duration=10,
        request_idx=0,
        log_bytes=b"",
        path=DB_PATH,
    )

    # Test get_all_scores (used by /api/highscores)
    scores = get_all_scores(path=DB_PATH)
    assert len(scores) == 1
    assert scores[0][2] == "TEST"

    # Test get_rank (used by /api/rank)
    rank, total = get_rank(0.5, path=DB_PATH)
    assert total == 1
    assert (
        rank == 1
    )  # 0.5 is better than 1.0 (lower is better? wait, code says score < ?)

    # Check logic:
    # SELECT COUNT(*) FROM high_scores WHERE score < ?
    # if score is 0.5, and existing is 1.0. 0.5 < 1.0 is True? No, existing is 1.0.
    # We are checking how many are strictly better (lower) than our score.
    # If I have 0.5, and existing is 1.0. Is 1.0 < 0.5? False. So 0 better. Rank 1.

    # Wait, get_rank(score):
    # cur = conn.execute("SELECT COUNT(*) FROM high_scores WHERE score < ?", (float(score),))
    # less = int(cur.fetchone()[0])
    # return less + 1

    # If I query for 1.5. Existing is 1.0. 1.0 < 1.5 is True. So less=1. Rank=2.
    rank, total = get_rank(1.5, path=DB_PATH)
    assert rank == 2

    # Test delete
    row_id = scores[0][0]
    success = delete_high_score(row_id, path=DB_PATH)
    assert success

    scores = get_all_scores(path=DB_PATH)
    assert len(scores) == 0

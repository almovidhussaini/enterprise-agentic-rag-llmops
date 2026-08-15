def test_project_imports():
    from app.agents.graph import graph

    assert graph is not None
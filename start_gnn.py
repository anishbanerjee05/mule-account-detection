# start_gnn.py - GNN service launcher. Called by demo.ps1.
import uvicorn
uvicorn.run(
    "gnn_service:app",
    host="0.0.0.0",
    port=8002,
    workers=1,
    timeout_worker_healthcheck=30
)
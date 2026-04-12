import uvicorn
from openenv.core.env_server import create_fastapi_app
from models import MilSupplyAction, MilSupplyObservation
from environment import MilSupplyEnvironment

env = MilSupplyEnvironment()
app = create_fastapi_app(env, MilSupplyAction, MilSupplyObservation)


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
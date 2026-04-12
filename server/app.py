import uvicorn
from openenv_core.env_server import create_app
from models import MilSupplyAction, MilSupplyObservation
from environment import MilSupplyEnvironment

env = MilSupplyEnvironment()
app = create_app(env, MilSupplyAction, MilSupplyObservation)


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
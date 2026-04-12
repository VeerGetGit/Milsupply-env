import uvicorn
from openenv.core.env_server import create_app
from models import MilSupplyAction, MilSupplyObservation
from environment import MilSupplyEnvironment

# Pass the CLASS not an instance
app = create_app(MilSupplyEnvironment, MilSupplyAction, MilSupplyObservation)


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
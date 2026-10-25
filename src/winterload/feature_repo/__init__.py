from feast import Project
from winterload.config import settings

project = Project(name=settings.feast_project, description="A project for the grid load in correlation with weather data")